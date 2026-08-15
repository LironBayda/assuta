"""
repo_test/batched_windowed.py

Fully batched (single-pass, all N windows at once) replacement for
Trainer._fit_window's per-window loop, built directly against the real
core/model.py, core/pinn.py (PhysicsInformedNN), and core/losses.py
(PINNLoss) -- not a reimplementation.

Composes three vmap'd pieces into one per-step scalar loss:
  1. trunk (f_x / f_x_kan): forward + dC/dt via nested vmap(time) + jacfwd,
     vmapped again over windows -- see batched_forward_with_dt.
  2. Ks_net: functional_call + vmap over windows (no time-vmap needed --
     its internal exp_conv_trap recurrence is a genuine sequential
     dependency over T, left untouched).
  3. PINNLoss: has no parameters to stack (nothing to functional_call) --
     vmapped directly as a plain function, since its config (weights,
     causality_eps) are closed-over Python floats, not traced tensors.
"""
from typing import List

import torch
from torch.func import functional_call, jacfwd, stack_module_state, vmap




def _submodule_state(stacked_params, stacked_buffers, prefix):
    plen = len(prefix)
    params = {k[plen:]: v for k, v in stacked_params.items() if k.startswith(prefix)}
    buffers = {k[plen:]: v for k, v in stacked_buffers.items() if k.startswith(prefix)}
    return params, buffers


def stack_trunks(trunk_modules: List[torch.nn.Module], hidden_attr: str):
    """hidden_attr: 'mlp' for f_x (sine/tanh), 'trunk' for f_x_kan."""
    base_model = trunk_modules[0]
    stacked_params, stacked_buffers = stack_module_state(trunk_modules)
    hidden_params, hidden_buffers = _submodule_state(stacked_params, stacked_buffers, f"{hidden_attr}.")
    final_params, final_buffers = _submodule_state(stacked_params, stacked_buffers, "final.")
    return {
        "base_model": base_model, "hidden_attr": hidden_attr,
        "hidden_params": hidden_params, "hidden_buffers": hidden_buffers,
        "final_params": final_params, "final_buffers": final_buffers,
    }


def batched_forward_with_dt(stacked, t: torch.Tensor):
    """t: (T,1) shared across all windows. Returns output, doutput_dt: (N,T,output_dim),
    matching f_x.forward_with_dt's existing pre-sigmoid convention exactly."""
    base_model = stacked["base_model"]
    hidden_module = getattr(base_model, stacked["hidden_attr"])

    def scalar_hidden(hidden_params, hidden_buffers, t_scalar):
        h = functional_call(hidden_module, (hidden_params, hidden_buffers), (t_scalar.view(1, 1),))
        return h.view(-1)

    def hidden_and_grad(hidden_params, hidden_buffers, t_scalar):
        h = scalar_hidden(hidden_params, hidden_buffers, t_scalar)
        dh_dt = jacfwd(lambda tt: scalar_hidden(hidden_params, hidden_buffers, tt))(t_scalar)
        return h, dh_dt.squeeze(-1)

    over_time = vmap(hidden_and_grad, in_dims=(None, None, 0))
    over_windows_and_time = vmap(over_time, in_dims=(0, 0, None))

    t_flat = t.view(-1)
    h_all, dh_dt_all = over_windows_and_time(stacked["hidden_params"], stacked["hidden_buffers"], t_flat)

    def final_fn(final_params, final_buffers, h):
        return functional_call(base_model.final, (final_params, final_buffers), (h,))

    output = vmap(final_fn, in_dims=(0, 0, 0))(stacked["final_params"], stacked["final_buffers"], h_all)
    weight = stacked["final_params"]["weight"]  # (N, output_dim, hidden_size)
    doutput_dt = torch.einsum("nth,noh->nto", dh_dt_all, weight)
    return output, doutput_dt


def stack_ks_nets(ks_net_modules: List[torch.nn.Module]):
    base_model = ks_net_modules[0]
    stacked_params, stacked_buffers = stack_module_state(ks_net_modules)
    return {"base_model": base_model, "params": stacked_params, "buffers": stacked_buffers}


def batched_ks_forward(stacked_ks, C0, C1, dC1, dC2, c_p_tensor):
    """
    C0     : (N, T) PER WINDOW -- this is f_x's own predicted c_p channel
             (output1[..., -1]), not raw shared data, so it legitimately
             differs per window (independent f_x weights per window) even
             though the true underlying arterial input is one shared signal.
    C1, dC1: (N, T, n_voxels) PER WINDOW.
    dC2    : same as C1/dC1 if 2TCM, or None if 1TCM.
    c_p_tensor : (T,) shared -- the RAW measured arterial input target used
             inside Ks_net.forward as `c_p_tensor` (fed to the closed-form
             convolution), distinct from C0 above.
    Returns whatever Ks_net.forward returns (a list of tensors), each with
    a new leading window dim N prepended by vmap.
    """
    base_model = stacked_ks["base_model"]

    def single_forward(params, buffers, C0_, C1_, dC1_, dC2_, cp_):
        return functional_call(base_model, (params, buffers), (C0_, C1_, dC1_, dC2_, cp_))

    dC2_in_dims = 0 if dC2 is not None else None
    fn = vmap(single_forward, in_dims=(0, 0, 0, 0, 0, dC2_in_dims, None))
    return fn(stacked_ks["params"], stacked_ks["buffers"], C0, C1, dC1, dC2, c_p_tensor)


def batched_loss(loss_module, output1, output2_list, tac_real, c_p_real):
    """
    loss_module : a single PINNLoss instance (no parameters -- config only,
        so it's used directly as a plain function, not stacked/functional_call'd).
    output1     : (N, T, tacs_num*C+1)
    output2_list: list of (N, T, n_voxels) tensors (Ks_net's return list)
    tac_real    : (N, T, n_voxels) -- PER WINDOW target data
    c_p_real    : (T,) shared arterial input target

    Returns per-window losses (N,) -- caller sums for the backward scalar,
    which is the exact algebraic equivalent of N independent backward() calls
    (every reduction inside PINNLoss is torch.sum, not mean).
    """
    in_dims_output2 = [0] * len(output2_list)

    def single_loss(o1, o2_list, tac_r, cp_r):
        return loss_module((o1, o2_list), (tac_r, cp_r))

    fn = vmap(single_loss, in_dims=(0, in_dims_output2, 0, None))
    return fn(output1, output2_list, tac_real, c_p_real)


def batched_train_step(trunk_stacked, ks_stacked, loss_module, t, c_p_tensor, tacs_tensor_stacked,
                        num_of_compartment, n_voxels_per_window, sigma_t=1.0):
    """
    One full batched forward+loss step across all N windows.

    t : (T,1) NORMALIZED time (same convention as Trainer.t) -- forward_with_dt
        computes d/dt_norm; sigma_t (= 1/t_std, matching Trainer.train_step)
        rescales it to d/dt_original, exactly mirroring the sequential path's
        `doutput1_dt = doutput1_dt * sigma_t` line.
    tacs_tensor_stacked : (N, T, n_voxels) -- PER WINDOW TAC targets.
    Returns: total scalar loss (sum over FINITE windows only, ready for
             .backward()), and the raw per-window losses (N,) for logging
             -- see mask_nonfinite_windows for how non-finite entries are
             excluded before the sum.
    """
    output1, doutput1_dt = batched_forward_with_dt(trunk_stacked, t)
    doutput1_dt = doutput1_dt * sigma_t
    N = output1.shape[0]

    if num_of_compartment == 2:
        dC1 = doutput1_dt[:, :, :n_voxels_per_window]
        dC2 = doutput1_dt[:, :, n_voxels_per_window:2 * n_voxels_per_window]
        C1 = output1[:, :, :n_voxels_per_window]
    else:
        dC1 = doutput1_dt[:, :, :-1]
        dC2 = None
        C1 = output1[:, :, :n_voxels_per_window]

    C0 = output1[:, :, -1]  # (N, T) -- f_x's own predicted c_p channel, genuinely
    # per-window (independent f_x weights per window); passed batched into
    # batched_ks_forward accordingly.
    output2 = batched_ks_forward(
        ks_stacked, C0, C1, dC1, dC2, c_p_tensor
    )

    per_window_losses = batched_loss(loss_module, output1, output2, tacs_tensor_stacked, c_p_tensor)
    return per_window_losses


def masked_total_loss(per_window_losses):
    """
    Sums per-window losses for .backward(), EXCLUDING any non-finite window
    for this step -- matching the sequential path's per-window isolation
    (one window blowing up doesn't corrupt every other window's gradient,
    which a naive .sum() over a batch containing a NaN/Inf would do).

    Returns (total_loss, finite_mask, n_nonfinite). finite_mask/n_nonfinite
    are for logging -- a window that is repeatedly non-finite across many
    steps gets NO gradient updates on those steps (effectively frozen at its
    last finite state) rather than being permanently removed from the batch
    the way the sequential path's `return epoch_counter, True` (stop this
    slice for good) does. This is a real, intentional simplification vs the
    sequential path -- flag it if a window needs true early-stop-and-drop
    semantics rather than freeze-on-bad-steps.
    """
    finite_mask = torch.isfinite(per_window_losses)
    n_nonfinite = (~finite_mask).sum().item()
    # detach the masked-out entries to exactly zero contribution (not just
    # small) -- torch.where with a zero literal, not multiplying by 0/1,
    # since 0 * inf = nan and would still poison the sum.
    safe_losses = torch.where(finite_mask, per_window_losses, torch.zeros_like(per_window_losses))
    total_loss = safe_losses.sum()
    return total_loss, finite_mask, n_nonfinite


def per_window_clip_grad_norm_(param_tensors, max_norm, eps=1e-6):
    """
    param_tensors : list of tensors, each shape (N, ...), with .grad already
        populated (call after .backward()).

    Clips each window's gradient INDEPENDENTLY, computing one combined norm
    across ALL given tensors for that window (matching
    torch.nn.utils.clip_grad_norm_'s existing "one shared norm across the
    whole given parameter list" semantics from Trainer._run_first_order_phase)
    -- NOT a single global norm across all N windows together, which would
    let one noisy window's gradient magnitude suppress every other window's
    update.
    """
    if not param_tensors:
        return
    N = param_tensors[0].shape[0]
    device = param_tensors[0].device
    total_sq = torch.zeros(N, device=device)
    for p in param_tensors:
        if p.grad is None:
            continue
        g = p.grad.reshape(N, -1)
        total_sq = total_sq + (g ** 2).sum(dim=1)
    total_norm = total_sq.sqrt()
    clip_coef = (max_norm / (total_norm + eps)).clamp(max=1.0)
    for p in param_tensors:
        if p.grad is None:
            continue
        shape = [N] + [1] * (p.grad.dim() - 1)
        p.grad.mul_(clip_coef.view(*shape))


# =====================================================================
# Full training loop over all N windows at once, mirroring
# Trainer._fit_window's per-window training but batched.
# =====================================================================
import logging

logger = logging.getLogger(__name__)


def build_stacked_pinns(pinn_builder, n_windows, arch):
    """
    pinn_builder : zero-arg callable returning a freshly-initialized
        PhysicsInformedNN (e.g. a closure over Trainer's hyperparams +
        a fixed image_shape for one window's voxel count). Called once
        per window so each gets its OWN random init -- same as the
        sequential path's per-window `_fit_window` construction.
    arch : "mlp" or "kan" -- selects which submodule name holds the
        trunk's hidden representation (f_x.mlp vs f_x_kan.trunk).

    Returns (pinns, trunk_stacked, ks_stacked, loss_module). `pinns` is
    kept alive only so its .loss_fn (shared, parameter-free) survives;
    all trainable state lives in trunk_stacked/ks_stacked from here on.
    """
    pinns = [pinn_builder() for _ in range(n_windows)]
    hidden_attr = "mlp" if arch == "mlp" else "trunk"
    trunk_stacked = stack_trunks([p.f_x for p in pinns], hidden_attr=hidden_attr)
    ks_stacked = stack_ks_nets([p.Ks_net for p in pinns])
    loss_module = pinns[0].loss_fn
    return pinns, trunk_stacked, ks_stacked, loss_module


def fit_windows_batched(
    pinn_builder, n_windows, arch, num_of_compartment, n_voxels_per_window,
    t_norm, sigma_t, c_p_tensor, tacs_tensor_stacked,
    epochs, lr, grad_clip, causality_eps_final, physics_weight_start, physics_weight_target,
    log_every=100,
):
    """
    Trains all n_windows window-PINNs in one batched AdamW loop -- the
    batched replacement for n_windows independent calls to
    Trainer._fit_window. NOT a replacement for the L-BFGS-schedule path
    (use_lbfgs_schedule=True): a single torch.optim.LBFGS instance over
    the stacked parameter tensor would do ONE shared line search across
    all windows' combined loss curvature, which is a materially different
    (and untested) algorithm from n_windows independent per-window L-BFGS
    runs -- callers should fall back to the sequential path when
    use_lbfgs_schedule=True. See Trainer._train_windowed's dispatch.

    Returns: (trunk_stacked, ks_stacked, history) where history contains
    per-epoch loss and any non-finite-window occurrences.
    """
    pinns, trunk_stacked, ks_stacked, loss_module = build_stacked_pinns(pinn_builder, n_windows, arch)

    f_x_params = list(trunk_stacked["hidden_params"].values()) + list(trunk_stacked["final_params"].values())
    ks_params = list(ks_stacked["params"].values())
    all_params = f_x_params + ks_params
    for p in all_params:
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(all_params, lr=lr)
    history = {"loss": [], "nonfinite_events": []}

    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        if causality_eps_final > 0:
            loss_module.causality_eps = causality_eps_final * progress
        if physics_weight_start is not None:
            loss_module.physics_weight = physics_weight_start + progress * (
                physics_weight_target - physics_weight_start
            )

        optimizer.zero_grad()
        per_window_losses = batched_train_step(
            trunk_stacked, ks_stacked, loss_module, t_norm, c_p_tensor, tacs_tensor_stacked,
            num_of_compartment, n_voxels_per_window, sigma_t=sigma_t,
        )
        total_loss, finite_mask, n_nonfinite = masked_total_loss(per_window_losses)

        if n_nonfinite > 0:
            bad_windows = (~finite_mask).nonzero(as_tuple=True)[0].tolist()
            logger.warning(
                f"[batched] epoch {epoch + 1}/{epochs}: {n_nonfinite} window(s) "
                f"non-finite this step (indices {bad_windows}); excluded from "
                f"backward, left at last finite parameter values for this step."
            )
            history["nonfinite_events"].append((epoch, bad_windows))

        if finite_mask.any():
            total_loss.backward()
            if grad_clip is not None:
                per_window_clip_grad_norm_(f_x_params, grad_clip)
                per_window_clip_grad_norm_(ks_params, grad_clip)
            optimizer.step()
        # if EVERY window is non-finite this step, skip optimizer.step()
        # entirely rather than stepping on an all-zero gradient.

        history["loss"].append(total_loss.item())
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            logger.info(f"[batched] epoch {epoch + 1}/{epochs} - total_loss: {total_loss.item():.6f}")

    return trunk_stacked, ks_stacked, history


def extract_ks_maps(ks_stacked, num_of_compartment):
    """
    Reads out each window's fitted Ks (post-softplus rate constants) from
    the stacked Ks_net parameters -- the batched equivalent of
    Trainer.save_ks, without needing a live Ks_net instance per window.
    Ks_net.Ks = F.softplus(Ks_raw); Ks_raw is stacked as (N, C+1, n_voxels).

    Returns: (N, C+1, n_voxels) tensor.
    """
    import torch.nn.functional as F
    Ks_raw_stacked = ks_stacked["params"]["Ks_raw"]  # (N, C+1, n_voxels)
    return F.softplus(Ks_raw_stacked).detach()
