import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class SineLayer(nn.Module):
    """SIREN sine-activation layer (Sitzmann et al., 2020)."""

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=10):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(in_features)

    def init_weights(self, in_features):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = np.sqrt(6 / in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class f_x(nn.Module):
    """
    SIREN-style continuous-time network: maps a scalar time input to
    concentration curves for every voxel/compartment simultaneously.

    NOTE: `tacs_num` here is the number of independent curves (voxels),
    not the number of time points — the network is evaluated once per
    time point and produces all voxel outputs from that single scalar.
    """

    def __init__(
        self,
        tacs_num,
        num_of_compartment,
        hidden_size=40,
        bottleneck_size=2,
        omega_0=1.0,
        activation="sine",
    ):
        """
        Parameters
        ----------
        tacs_num : int
            Number of independent curves (voxels) predicted per compartment.
        num_of_compartment : int
            Number of kinetic compartments.
        hidden_size : int
            Width of the hidden layers. Previously this argument was
            silently discarded (hardcoded to 10 internally) — now actually
            used.
        bottleneck_size : int or None
            Width of the middle hidden layer. Defaults to `hidden_size`
            (no bottleneck). Previously this was `hidden_size // 10`,
            which — given the hardcoded hidden_size=10 bug — collapsed to
            a single-unit layer; that looked like an unintended side
            effect rather than a deliberate design choice, so the default
            here restores full width. Pass an explicit value if you did
            want a deliberate bottleneck.
        omega_0 : float
            Frequency scale for the hidden SineLayers (see SIREN paper).
            Unused when activation="tanh".
        activation : "sine" (default, SIREN-style, this session's
            hyperparameter sweep found omega_0=1.0 best for this specific
            continuous-time-trunk use) or "tanh" (a standard MLP, no
            special init -- the activation choice most PINN papers in the
            literature actually use, e.g. van Herten et al. 2022; provided
            here so noise/sparsity comparisons can include it directly
            rather than only ever testing this repo's own Sine default).
        """
        super().__init__()
        self.input_dim = 1
        self.output_dim = tacs_num * num_of_compartment + 1
        self.hidden_size = hidden_size
        self.activation = activation
        mid_size = bottleneck_size if bottleneck_size is not None else hidden_size

        if activation == "sine":
            self.mlp = nn.Sequential(
                SineLayer(self.input_dim, self.hidden_size, bias=True, is_first=True, omega_0=omega_0),
                SineLayer(self.hidden_size, hidden_size, bias=True, is_first=False, omega_0=omega_0),
            )
        elif activation == "tanh":
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_size), nn.Tanh(),
                nn.Linear(self.hidden_size, hidden_size), nn.Tanh(),
            )
        else:
            raise ValueError(f"activation must be 'sine' or 'tanh', got {activation!r}")

        # Plain linear readout (no sin/tanh activation): concentration
        # values need arbitrary positive range, which a bounded final
        # activation would clip.
        self.final = nn.Linear(self.hidden_size, self.output_dim)
        self.a=nn.Sigmoid()

    def forward(self, x):
        # x: (T, 1)
        h = self.mlp(x)
        return self.a(self.final(h))  # (T, output_dim)

    def forward_with_dt(self, t):
        """
        Returns (output, d_output/dt), with the derivative computed
        exactly per output column (i.e. per voxel/per-compartment),
        never aggregated across columns.

        Why this is cheap regardless of voxel count: `final` is a single
        linear layer applied to a shared hidden trunk h(t), so by the
        chain rule d(final(h))/dt = final.weight @ dh/dt. Computing dh/dt
        costs `hidden_size` backward passes (independent of how many
        voxels the output has); one matrix multiply then gives the exact
        per-voxel derivative for every output column at once. This is
        the batched-per-voxel derivative you asked for, implemented via
        the trunk's small Jacobian instead of a literal per-voxel loop —
        looping per voxel would redundantly recompute the same trunk
        pass (which doesn't depend on voxel identity) thousands of times.
        """
        h = self.mlp(t)  # (T, hidden_size)
        T, H = h.shape

        dh_dt_cols = []
        for j in range(H):
            grad_j = torch.autograd.grad(
                h[:, j],
                t,
                grad_outputs=torch.ones_like(h[:, j]),
                retain_graph=True,
                create_graph=True,
            )[0]  # (T, 1)
            dh_dt_cols.append(grad_j)
        dh_dt = torch.cat(dh_dt_cols, dim=1)  # (T, hidden_size)

        output = self.final(h)  # (T, output_dim)
        doutput_dt = dh_dt @ self.final.weight.T  # (T, output_dim), exact per-column
        return output, doutput_dt


class f_x_bpinn(f_x):
    """
    "Sine B-PINN": MC-Dropout Bayesian version of the SIREN f_x network
    above. Adds dropout after each SineLayer's activation and keeps it
    ACTIVE at inference time too (not just training) -- this is the
    standard MC-Dropout approximate-Bayesian-inference trick (Gal &
    Ghahramani, 2016), cited as one of the established ways to get
    uncertainty out of a PINN in "VI-PINNs" (arXiv:2211.16753): running
    N stochastic forward passes with dropout on gives a distribution
    over f_x's output (and, downstream, over the physics residuals and
    the fitted Ks) instead of a single point estimate.

    Everything else (SineLayer definitions, forward_with_dt's exact
    per-voxel derivative trick) is inherited unchanged from f_x -- this
    subclass only inserts dropout and forces train()-mode dropout
    behavior regardless of the module's train/eval flag, so callers
    don't need to remember to call .train() before sampling at
    inference time.
    """

    def __init__(self, tacs_num, num_of_compartment, hidden_size=40,
                 bottleneck_size=2, omega_0=1.0, dropout_p=0.1, activation="sine"):
        super().__init__(tacs_num, num_of_compartment, hidden_size, bottleneck_size, omega_0, activation)
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        h = x
        for layer in self.mlp:
            h = self.dropout(layer(h))   # dropout after each SineLayer
        return self.a(self.final(h))

    def forward_with_dt(self, t):
        # Same per-voxel-exact derivative trick as f_x.forward_with_dt,
        # but routed through the dropout-augmented trunk so the sampled
        # dropout mask is shared between the value and its derivative
        # within one stochastic forward pass (as it must be -- they're
        # two views of the same sampled sub-network, not independent
        # draws).
        h = t
        for layer in self.mlp:
            h = self.dropout(layer(h))
        T, H = h.shape

        dh_dt_cols = []
        for j in range(H):
            grad_j = torch.autograd.grad(
                h[:, j], t, grad_outputs=torch.ones_like(h[:, j]),
                retain_graph=True, create_graph=True,
            )[0]
            dh_dt_cols.append(grad_j)
        dh_dt = torch.cat(dh_dt_cols, dim=1)

        output = self.final(h)
        doutput_dt = dh_dt @ self.final.weight.T
        return output, doutput_dt

    def train(self, mode=True):
        # Force dropout to stay active even in eval() mode -- MC-Dropout
        # requires this; without it, .eval() would silently turn the
        # network back into a deterministic point estimate.
        super().train(mode)
        self.dropout.train(True)
        return self


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of F.softplus, for initializing a softplus-reparameterized
    parameter so its *forward* value starts at `y`."""
    return torch.log(torch.expm1(y))


class Ks_net(nn.Module):
    """
    Compartmental-model kinetic parameter network. Solves the ODE system
    in closed form via exponential convolution (trapezoidal rule) and
    returns physics residuals for the PINN loss.

    Kinetic rate constants are stored via a softplus reparameterization
    (`Ks = softplus(Ks_raw)`) to guarantee positivity. This matters
    beyond physical plausibility: the closed-form 2-tissue-compartment
    solution below (`convolve_2cm_for_minimize`) is only mathematically
    valid for k2, k3 >= 0 — an unconstrained parameter going negative
    doesn't just give an unphysical result, it silently invalidates the
    algebraic identity the closed form depends on.
    """

    def __init__(self, tac_num, num_of_compartment, slice_dt, ks_init=None):
        """
        ks_init : torch.Tensor, optional, shape (num_of_compartment+1, tac_num)
            Per-voxel initial values for Ks (post-softplus, i.e. in the
            same units as `Ks`, not `Ks_raw`), e.g. from a warm-start
            (VAE latent, prior estimate, etc.) instead of the flat
            per-compartment constant used by default. Any entry that is
            NaN falls back to the default constant for that
            compartment/voxel.
        """
        super().__init__()
        self.tac_num = tac_num
        self.num_of_compartment = num_of_compartment

        # Register as a buffer (not a plain numpy array / python attribute)
        # so it moves with the module across devices via .to(device).
        self.register_buffer(
            "slice_dt", torch.as_tensor(np.asarray(slice_dt), dtype=torch.float32)
        )

        if num_of_compartment == 1:
            self.minimize = self.convolve_1cm_for_minimize
            self.get_x = self.get_x_1tcm
            init_vals = torch.tensor([[0.1] * tac_num, [0.1] * tac_num], dtype=torch.float32)
        elif num_of_compartment == 2:
            self.minimize = self.convolve_2cm_for_minimize
            self.get_x = self.get_x_2tcm
            init_vals = torch.tensor(
                [[0.08] * tac_num, [0.35] * tac_num, [0.04] * tac_num], dtype=torch.float32
            )
        else:
            raise ValueError("num_of_compartment must be 1 or 2")

        if ks_init is not None:
            ks_init = torch.as_tensor(ks_init, dtype=torch.float32)
            valid = torch.isfinite(ks_init) & (ks_init > 0)
            init_vals = torch.where(valid, ks_init, init_vals)

        # BUGFIX: init_vals was previously computed above and then
        # discarded -- Ks_raw (the actual learnable parameter behind the
        # `Ks` property) was never assigned, so every Ks_net crashed with
        # AttributeError on its first forward() call. Register it here via
        # inverse-softplus so the *forward* (post-softplus) value starts
        # at init_vals, matching what the class docstring already says.
        self.Ks_raw = nn.Parameter(_inverse_softplus(init_vals))

    @property
    def Ks(self) -> torch.Tensor:
        """Positive kinetic rate constants, softplus(Ks_raw)."""
        return F.softplus(self.Ks_raw)

    # ============================================================
    # Core Exponential Convolution (Trapezoidal Rule)
    # ============================================================
    def exp_conv_trap(self, blood, theta):
        """
        Sequential trapezoidal convolution of `blood` with exp(-theta*t).
        This is an O(T) recurrence with per-step decay `alpha_i` that
        depends on the (possibly non-uniform) frame duration `dt_i` — PET
        frame durations vary, so this can't be collapsed into a single
        closed-form matrix power; the loop is doing necessary work, not
        an accidental inefficiency. (Not using torch.compile here: this
        method mixes a Python-level loop with per-step tensor ops, which
        is a poor fit for torch.compile and previously risked excessive
        recompilation rather than a speedup.)
        """
        T = blood.shape[0]
        h_prev = torch.zeros_like(theta)
        H_list = [h_prev]

        for i in range(1, T):
            dt = self.slice_dt[i]
            alpha = torch.exp(-theta * dt)
            h_curr = alpha * h_prev + 0.5 * dt * (blood[i] + alpha * blood[i - 1])
            H_list.append(h_curr)
            h_prev = h_curr

        return torch.stack(H_list, dim=0)

    # ============================================================
    # 1-Tissue Compartment Model
    # ============================================================
    def one_tcm_trap(self, blood, Ktrans, kep):
        H = self.exp_conv_trap(blood, kep)
        return Ktrans * H

    def convolve_1cm_for_minimize(self, blood):
        Ktrans = self.Ks[0]
        Ve = self.Ks[1]
        return self.one_tcm_trap(blood, Ktrans, Ktrans / Ve)

    # ============================================================
    # 2-Tissue Compartment Model (irreversible: k1, k2, k3)
    # ============================================================
    def two_tcm_trap(self, blood, theta1, theta2, phi1, phi2):
        H1 = self.exp_conv_trap(blood, theta1) * phi1
        H2 = self.exp_conv_trap(blood, theta2) * phi2
        return H1 + H2

    def convolve_2cm_for_minimize(self, blood):
        k1 = self.Ks[0]
        k2 = self.Ks[1]
        k3 = self.Ks[2]

        # k2, k3 >= 0 is now guaranteed by the softplus reparameterization
        # above, so delta = |k2+k3| = k2+k3 exactly — using k2+k3 directly
        # (instead of sqrt((k2+k3)**2)) avoids the non-differentiable
        # point at k2+k3=0 that the sqrt-of-square form has.
        delta = k2 + k3

        theta1 = (k2 + k3 + delta) / 2  # == k2 + k3
        theta2 = (k2 + k3 - delta) / 2  # == 0

        phi1 = (k1 * (theta1 - k3)) / delta
        phi2 = (k1 * (theta2 - k3)) / (-delta)

        return self.two_tcm_trap(blood, theta1, theta2, phi1, phi2)

    # ============================================================
    # Physics Residuals (PINN-style)
    # ============================================================
    def get_x_1tcm(self, C0, C1, dC1, dC2, ct):
        # dC1: (T, n_voxels) — true per-voxel derivative (see
        # f_x.forward_with_dt). No cross-voxel summation: each voxel's
        # residual is computed independently.
        r1 = dC1 - (self.Ks[0].view(1, -1) * C0.view(-1, 1) - self.Ks[1].view(1, -1) * C1)
        return [r1, ct]

    def get_x_2tcm(self, C0, C1, dC1, dC2, ct):
        # dC1, dC2: (T, n_voxels) — true per-voxel derivatives.
        k1 = self.Ks[0].view(1, -1)
        k2 = self.Ks[1].view(1, -1)
        k3 = self.Ks[2].view(1, -1)

        r1 = (dC1 + dC2) - (k1 * C0.view(-1, 1) - k2 * C1)
        r2 = dC1 - (k1 * C0.view(-1, 1) - (k2 + k3) * C1)
        r3 = dC2 - (k3 * C1)

        return [r1, 10*r3, 10*r2, ct]

    # ============================================================
    # Forward Pass
    # ============================================================
    def forward(self, C0, C1, dC1, dC2, c_p_tensor):
        """
        C0 : (T,)   arterial input at each time point
        C1 : (T, tac_num)  per-voxel tissue concentration
        dC1, dC2 : (T, tac_num)  true per-voxel time-derivatives, from
            f_x.forward_with_dt (see that method for why this is exact
            and cheap regardless of voxel count).
        c_p_tensor : (T,)  arterial input used for the model-predicted TAC
        """
        ct = self.minimize(C0.detach())  # (T, tac_num)
        return self.get_x(C0, C1, dC1, dC2, ct)