import logging
import os
from os.path import join
from typing import Optional

import numpy as np
import nibabel as nib
import torch

from config import PINN
from core.pinn import PhysicsInformedNN

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trains a Physics-Informed Neural Network (PINN) fitting a
    compartmental kinetic model to PET/DCE time-activity curves (TACs),
    and saves the fitted kinetic parameter maps (Ks) as NIfTI.

    This is the single entry point for both windowed and whole-volume
    training -- windowed=True (the default) splits the image into
    independent spatial windows (van Herten et al., Medical Image
    Analysis 2022 style) and fits a separate sub-Trainer per window (see
    `axis`/`window_size`/`stride`/`slice_window`, and `_train_windowed`'s
    docstring); windowed=False fits ONE PINN jointly across the whole
    volume in a single pass (already inherently 3D -- `train()` flattens
    all X,Y,Z voxels together regardless of this flag). What used to be
    a separate free function (core/windowed.py's train_windowed) is now
    folded into this class as `_train_windowed`/`_fit_window`.

    Tuned defaults (1TCM/DCE, multi-seed confirmed -- see
    simulation/pinn_hyperparam_sweep.py for the sweeps behind these):
      - hidden_size=10, omega_0=1.0 beat hidden_size=40 (better AND more
        consistent K1, ~40% faster).
      - physics_weight=100 flips derived-kep correlation from reliably
        negative to reliably positive, vs the old 0.01 default -- but
        only when annealed in via physics_weight_start=0.01 (holding it
        constant from epoch 1 caused K1 to collapse mid-training on one
        seed; annealing fixed that).
      - causality_eps_final=2000 (up from 100): the clearest single win
        in the sweep -- K1 correlation 0.90->0.94, and k2's correlation
        flips sign from negative to positive.
      - Pure AdamW for the full epoch count beat every two-phase
        Adam+L-BFGS schedule tried -- see use_lbfgs_schedule below,
        which keeps that as the default and makes any other schedule
        opt-in.
      - NOT yet verified for 2TCM/PET -- the Ve/kep bug this recipe was
        built around was 1TCM-only. Override explicitly for 2TCM until
        it gets its own sweep.

    arch/activation trunk options (core/model.py): arch="mlp" +
    activation="sine" (default, SIREN -- this recipe's tuning basis),
    arch="mlp" + activation="tanh" (standard MLP), or arch="kan"
    (Kolmogorov-Arnold, silu-only -- activation is ignored). Only
    sine+mlp was covered by the sweep above; treat the others as
    untuned starting points. kan_grid_size/kan_spline_order/
    kan_grid_range only apply to arch="kan"; grid_range defaults wider
    than KANLinear's own (-1,1) because self.t is z-score-normalized and
    routinely lands outside that range for non-uniform time sampling.

    optimizer/use_lbfgs_schedule: opt-in experiments, not part of the
    tuned recipe -- see their own docstrings below.
    """

    def __init__(
        self,
        c_p,
        num_of_compartment,
        t,
        device=config.DEVICE,
        affine=None,
        save_path=None,
        epochs=PINN["epochs"],
        lr=0.043431928291308805,#PINN["learning_rate"],
        grad_clip: Optional[float] = 0.1,
        causality_eps_final: float = 1000,
        physics_weight_start: Optional[float] = 0.0001,
        hidden_size: int = 100,
        bottleneck_size: Optional[int] = None,
        omega_0: float = 0.1595269002697269,
        activation: str = "sine",
        arch: str = "mlp",

        kan_grid_size: int = 3,
        kan_spline_order: int = 1,
        kan_grid_range=(-3.0, 1.0),
        physics_weight: float = 0.028952212937853314,
        tac_consistency_weight: Optional[float] =0.24095114278078722,
        reg_weight: float = 0.0008221670354318431,
        optimizer: str = "adamw",
        sgd_momentum: float = 0.9,
        use_lbfgs_schedule: bool = True,
        first_order_phase_epochs: int = 0,
        lbfgs_phase_epochs: int = 10,
        lbfgs_max_iter: int =20 ,
        windowed: bool = True,
        axis: str = "xy",
        window_size=64,
        stride=None,
        slice_window: int = 1,

    ):


        """
        optimizer : "adamw" (tuned default) or "sgd" (untested -- lets
            you check whether plain SGD converges at all on this loss;
            not expected to beat AdamW). Only affects the first-order
            phase(s); L-BFGS is unaffected.
        sgd_momentum : only used when optimizer="sgd".
        use_lbfgs_schedule : if True, alternates first_order_phase_epochs
            of the first-order optimizer with lbfgs_phase_epochs of
            L-BFGS, cycling until `epochs` total steps are used. If False
            (default): the first-order optimizer alone for the full
            `epochs` count -- the setting that already won the sweep.
            grad_clip is applied during first-order phases only (L-BFGS's
            line search doesn't compose cleanly with manual clipping).
        first_order_phase_epochs, lbfgs_phase_epochs, lbfgs_max_iter :
            phase lengths (only used when use_lbfgs_schedule=True) and
            L-BFGS's own internal line-search iterations per .step().
        windowed : bool, default True -- see `train()`'s docstring.
        axis : "xy" (default), "z", or "xyz" -- only used when
            windowed=True. See `_train_windowed`'s docstring.
        window_size : int, (int, int), or (int, int, int), default 16 --
            only used when windowed=True. For axis="xy": int or (X,Y)
            tuple. For axis="xyz": REQUIRED, (X,Y,Z) tuple.
        stride : int, optional -- only used when windowed=True and
            axis in ("xy", "xyz"). Defaults to window_size (non-overlapping
            windows); a smaller stride gives overlapping windows (blended
            by averaging in the overlap).
        slice_window : int, default 1 -- only used when windowed=True and
            axis="z". Number of Z slices per window.
        """
        c_p = np.asarray(c_p, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        if c_p.shape[0] != t.shape[0]:
            raise ValueError(
                f"c_p (len={c_p.shape[0]}) and t (len={t.shape[0]}) "
                "must have the same number of time points."
            )
        if num_of_compartment < 1:
            raise ValueError("num_of_compartment must be >= 1")
        if optimizer not in ("adamw", "sgd"):
            raise ValueError(f"optimizer must be 'adamw' or 'sgd', got {optimizer!r}")

        # kept raw (pre-normalization) so _fit_window can build identical
        # sub-Trainers per window without re-deriving these from tensors.
        self._c_p_raw = c_p.copy()
        self._t_raw = t.copy()

        self.pinn = None
        self.tacs_tensor = None
        self.device = device
        self.epochs = epochs
        self.affine = affine
        self.save_path = save_path
        self.num_of_compartment = num_of_compartment
        self.grad_clip = grad_clip
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.omega_0 = omega_0
        self.activation = activation
        self.arch = arch
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.kan_grid_range = kan_grid_range
        self.physics_weight = physics_weight
        self.tac_consistency_weight = tac_consistency_weight
        self.reg_weight = reg_weight
        self.causality_eps_final = causality_eps_final
        self.physics_weight_start = physics_weight_start
        self.optimizer_name = optimizer
        self.sgd_momentum = sgd_momentum
        self.use_lbfgs_schedule = use_lbfgs_schedule
        self.first_order_phase_epochs = first_order_phase_epochs
        self.lbfgs_phase_epochs = lbfgs_phase_epochs
        self.lbfgs_max_iter = lbfgs_max_iter
        self.windowed = windowed
        self.axis = axis
        self.window_size = window_size
        self.stride = stride
        self.slice_window = slice_window
        self.history = {"loss": [], "phase": []}

        self.num = np.max(c_p)
        if not np.isfinite(self.num) or self.num <= 0:
            raise ValueError(
                f"Invalid normalization constant from c_p (max={self.num}); "
                "expected a positive finite value."
            )

        self.dt = np.diff(np.insert(t, 0, 0))
        self.lr = lr

        t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
        t_std = t_tensor.std()
        if t_std == 0:
            raise ValueError("t has zero variance; cannot normalize.")
        self.t = (t_tensor - t_tensor.mean()) / t_std
        self.t.requires_grad_(True)
        self.sigma_t = (1 / t_std).detach()

        self.c_p = c_p / self.num
        self.c_p_tensor = torch.tensor(self.c_p, dtype=torch.float32, device=self.device)

    @staticmethod
    def prepare_tacs(images):
        n_time, size_x, size_y, size_z = images.shape
        tacs = images.reshape(n_time, size_x * size_y * size_z)
        return tacs, (size_x, size_y, size_z, n_time)

    def train_step(self):
        # forward_with_dt gives exact per-voxel/per-compartment derivatives
        # in one pass -- summing across columns (the old approach) let
        # different voxels' physics violations cancel before squaring.
        output1, doutput1_dt = self.pinn.f_x.forward_with_dt(self.t.view(-1, 1))
        doutput1_dt = doutput1_dt * self.sigma_t  # chain rule: d/dt_orig = d/dt_norm * (1/std)

        n_voxels = self.tacs_tensor.shape[1]
        if self.num_of_compartment == 2:
            # output1 width = 2*n_voxels + 1: [0:n_voxels)=compartment 1,
            # [n_voxels:2*n_voxels)=compartment 2, [-1]=shared c_p pred.
            dC1 = doutput1_dt[:, :n_voxels]
            dC2 = doutput1_dt[:, n_voxels : 2 * n_voxels]
        else:
            dC1 = doutput1_dt[:, :-1]
            dC2 = None

        output2 = self.pinn.Ks_net(
            output1[:, -1], output1[:, :n_voxels], dC1, dC2, self.c_p_tensor,
        )
        return self.pinn.loss_fn([output1, output2], [self.tacs_tensor, self.c_p_tensor])

    # ---------------------------
    # Optimizer helpers
    # ---------------------------
    def _build_first_order_optimizer(self, params):
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(params, lr=self.lr, momentum=self.sgd_momentum)
        return torch.optim.AdamW(params, lr=self.lr)

    def _anneal(self, epoch_counter: int):
        """Advances causality_eps / physics_weight using a GLOBAL epoch
        counter (shared across phases), so annealing spans the full
        `epochs` budget regardless of which optimizer is active."""
        progress = epoch_counter / max(self.epochs - 1, 1)
        if self.causality_eps_final > 0 and hasattr(self.pinn.loss_fn, "causality_eps"):
            self.pinn.loss_fn.causality_eps = self.causality_eps_final * progress
        if self.physics_weight_start is not None and hasattr(self.pinn.loss_fn, "physics_weight"):
            self.pinn.loss_fn.physics_weight = self.physics_weight_start + progress * (
                self.physics_weight - self.physics_weight_start
            )

    def _log_step(self, phase: str, epoch_counter: int, loss: torch.Tensor):
        self.history["loss"].append(loss.item())
        self.history["phase"].append(phase)
        if (epoch_counter + 1) % 100:
            return
        components = getattr(self.pinn.loss_fn, "last_components", None)
        comp_str = "  ".join(f"{k}={v.item():.4f}" for k, v in components.items()) if components else ""
        logger.info(f"[{phase}] Epoch {epoch_counter + 1}/{self.epochs} - Loss: {loss.item():.6f}  {comp_str}")

    def _run_first_order_phase(self, params, f_x_params, ks_params, n_steps, epoch_counter, z):
        """Up to n_steps first-order optimizer steps. Returns (epoch_counter, stopped_early)."""
        optimizer = self._build_first_order_optimizer(params)
        for _ in range(n_steps):
            if epoch_counter >= self.epochs:
                break
            self._anneal(epoch_counter)
            optimizer.zero_grad()
            loss = self.train_step()

            if not torch.isfinite(loss):
                logger.error(f"Non-finite loss ({loss.item()}) at epoch {epoch_counter + 1}, z={z}; stopping this slice.")
                return epoch_counter, True

            loss.backward()
            if self.grad_clip is not None:
                # Two separate groups, not one combined norm -- f_x's
                # final layer has far more params than Ks_net, so a
                # single global-norm clip would starve Ks_net's updates.
                torch.nn.utils.clip_grad_norm_(f_x_params, self.grad_clip)
                torch.nn.utils.clip_grad_norm_(ks_params, self.grad_clip)
            optimizer.step()

            self._log_step(self.optimizer_name, epoch_counter, loss)
            epoch_counter += 1
        return epoch_counter, False

    def _run_lbfgs_phase(self, params, n_steps, epoch_counter, z):
        """Up to n_steps L-BFGS .step() calls. grad_clip is NOT applied
        here (doesn't compose with L-BFGS's line search)."""
        optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=self.lbfgs_max_iter, line_search_fn="strong_wolfe")
        for _ in range(n_steps):
            if epoch_counter >= self.epochs:
                break
            self._anneal(epoch_counter)
            stopped = {"value": False}

            def closure():
                optimizer.zero_grad()
                loss = self.train_step()
                if not torch.isfinite(loss):
                    stopped["value"] = True
                    return loss
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            if stopped["value"] or not torch.isfinite(loss):
                logger.error(f"Non-finite loss ({loss.item()}) at epoch {epoch_counter + 1}, z={z}; stopping this slice.")
                return epoch_counter, True

            self._log_step("lbfgs", epoch_counter, loss)
            epoch_counter += 1
        return epoch_counter, False

    def _run_optimization(self, params, f_x_params, ks_params, z):
        """Default: first-order optimizer alone for the full epoch count
        (the setting that won the sweep). If use_lbfgs_schedule=True:
        alternates first_order_phase_epochs / lbfgs_phase_epochs until
        `epochs` is used."""
        if not self.use_lbfgs_schedule:
            return self._run_first_order_phase(params, f_x_params, ks_params, self.epochs, 0, z)

        epoch_counter, stopped = 0, False
        while epoch_counter < self.epochs and not stopped:
            remaining = self.epochs - epoch_counter
            epoch_counter, stopped = self._run_first_order_phase(
                params, f_x_params, ks_params, min(self.first_order_phase_epochs, remaining), epoch_counter, z,
            )
            if stopped or epoch_counter >= self.epochs:
                break
            remaining = self.epochs - epoch_counter
            epoch_counter, stopped = self._run_lbfgs_phase(
                params, min(self.lbfgs_phase_epochs, remaining), epoch_counter, z,
            )
        return epoch_counter, stopped

    # ---------------------------
    # Run training
    # ---------------------------
    def train(self, images, z_slices: Optional[range] = None):
        """
        Train the PINN on a 4D image volume (time, x, y, z).

        images : np.ndarray, shape (n_time, size_x, size_y, size_z).
        z_slices : range, optional -- only used when windowed=False; see
            `_train_whole`'s docstring.

        Dispatches on `self.windowed` (constructor arg, default True):
          - windowed=True: splits `images` into independent spatial
            windows per `self.axis`/`self.window_size`/`self.stride`/
            `self.slice_window` and fits a separate sub-Trainer per
            window (see `_train_windowed`) -- the recommended default
            for real subject volumes.
          - windowed=False: fits ONE PINN jointly across the whole
            volume in a single pass (see `_train_whole`) -- already
            inherently 3D (all X,Y,Z voxels trained together), just
            without the windowing.

        Both paths return (results, history): results is a
        (num_of_compartment+1, X, Y, Z) ndarray of fitted Ks maps;
        history is a dict (populated with "loss"/"phase" for the
        windowed=False path, "n_windows" for windowed=True).
        """
        if images.ndim != 4:
            raise ValueError(f"images must be 4D (time, x, y, z); got shape {images.shape}")
        if images.shape[0] != self.c_p_tensor.shape[0]:
            raise ValueError(
                f"images time dim ({images.shape[0]}) must match c_p length ({self.c_p_tensor.shape[0]})."
            )

        if self.windowed:
            return self._train_windowed(images)
        return self._train_whole(images, z_slices)

    def _train_whole(self, images, z_slices: Optional[range] = None):
        """
        Fit ONE PINN jointly on the whole `images` volume (all X,Y,Z
        voxels together in a single forward pass -- already inherently
        3D).

        z_slices : range, optional -- which z-slices to process; defaults
            to just slice 0. Pass range(images.shape[3]) for the full volume.
        """
        results = np.zeros([self.num_of_compartment + 1] + list(images.shape[1:]))
        z_slices = z_slices if z_slices is not None else [0]

        for z in z_slices:
            tacs, shape = self.prepare_tacs(images)  # [:, :, :, z:z + 1] if slicing per-z
            self.tacs_tensor = torch.tensor(np.asarray(tacs) / self.num, dtype=torch.float32, device=self.device)

            self.pinn = PhysicsInformedNN(
                self.num_of_compartment, self.dt, shape, device=self.device,
                hidden_size=self.hidden_size, bottleneck_size=self.bottleneck_size,
                omega_0=self.omega_0, activation=self.activation, arch=self.arch,
                kan_grid_size=self.kan_grid_size, kan_spline_order=self.kan_spline_order,
                kan_grid_range=self.kan_grid_range, physics_weight=self.physics_weight,
                tac_consistency_weight=self.tac_consistency_weight, reg_weight=self.reg_weight,
            )

            f_x_params = list(self.pinn.f_x.parameters())
            ks_params = list(self.pinn.Ks_net.parameters())
            params = f_x_params + ks_params

            self._run_optimization(params, f_x_params, ks_params, z)

            # save_ks does NOT write to disk -- it just extracts/reshapes
            # the trained Ks map into `results`. Must run unconditionally
            # (a prior version gated this behind save_path, which meant
            # train() silently returned all-zeros whenever save_path was
            # None -- found via _fit_window's per-window sub-training).
            results = self.save_ks(z, self.pinn, shape, results)

        if self.save_path is not None and self.affine is not None:
            nib.save(
                nib.Nifti1Image(results.transpose((1, 2, 3, 0)).astype(np.float64), self.affine),
                join(self.save_path, "K_kan.nii"),
            )
        return results, self.history

    def save_ks(self, z, pinn, shape, results):
        Ks = pinn.Ks_net.Ks.detach().cpu().numpy()
        size_x, size_y, size_z, _ = shape
        return Ks.reshape((self.num_of_compartment + 1), size_x, size_y, size_z)

    # ---------------------------
    # Windowed training
    # ---------------------------
    def _fit_window(self, window_img):
        """Fits a fresh, independent sub-Trainer (same hyperparams, no
        shared state) on one spatial window -- whole-volume mode
        (`_train_whole`) applied to just that window's sub-volume."""
        sub_trainer = Trainer(
            c_p=self._c_p_raw, num_of_compartment=self.num_of_compartment, t=self._t_raw,
            device=self.device, affine=None, save_path=None, epochs=self.epochs, lr=self.lr,
            grad_clip=self.grad_clip, causality_eps_final=self.causality_eps_final,
            physics_weight_start=self.physics_weight_start, hidden_size=self.hidden_size,
            bottleneck_size=self.bottleneck_size, omega_0=self.omega_0, activation=self.activation,
            arch=self.arch, kan_grid_size=self.kan_grid_size, kan_spline_order=self.kan_spline_order,
            kan_grid_range=self.kan_grid_range, physics_weight=self.physics_weight,
            tac_consistency_weight=self.tac_consistency_weight, reg_weight=self.reg_weight,
            optimizer=self.optimizer_name, sgd_momentum=self.sgd_momentum,
            use_lbfgs_schedule=self.use_lbfgs_schedule,
            first_order_phase_epochs=self.first_order_phase_epochs,
            lbfgs_phase_epochs=self.lbfgs_phase_epochs, lbfgs_max_iter=self.lbfgs_max_iter,
            windowed=False,
        )
        mean, _ = sub_trainer._train_whole(window_img, z_slices=[0])
        return mean

    def _train_windowed(self, images):
        """
        Splits `images` into independent spatial windows and trains a
        SEPARATE PINN per window via `_fit_window` (modeled on van Herten
        et al., Medical Image Analysis 2022), then stitches the
        per-window results back into a full-size volume (overlapping
        windows are averaged).

        self.axis : "xy" (default, in-plane patch-tiling): splits X and Y
            into `self.window_size` patches (Z kept whole -- so each
            window is still a full-depth 3D sub-volume). Requires
            `self.window_size` to be set (default 16).
            "z" (slice-wise): keeps the FULL X,Y extent in every window,
            splitting only along Z into groups of `self.slice_window`
            slices -- avoids the small-XY-patch data-starvation problem
            and never risks a patch boundary cutting through the
            anatomy, at the cost of fewer, larger, more expensive
            per-window fits. `self.window_size`/`self.stride` are
            ignored in this mode.
            "xyz": a genuine 3D box, e.g. window_size=(32, 32, 20) --
            splits all three spatial dimensions. Requires
            `self.window_size` as a 3-tuple.
        self.window_size : int, (int, int), or (int, int, int), default
            16 -- spatial window size. For axis="xy": int or (X,Y)
            tuple. For axis="xyz": REQUIRED, (X,Y,Z) tuple.
        self.stride : int, optional -- (axis="xy"/"xyz") defaults to
            window_size (non-overlapping windows). A stride < window_size
            gives overlapping windows (blended by averaging in the overlap).
        self.slice_window : int, default 1 -- (axis="z" only) number of Z
            slices per window.

        Returns (K_mean_full, history): K_mean_full is a
        (num_of_compartment+1, X, Y, Z) ndarray; history is
        {"n_windows": int}. If save_path/affine are set, also saves
        K_mean_windowed.nii.
        """
        T, X, Y, Z = images.shape
        C1 = self.num_of_compartment + 1
        K_mean_full = np.zeros((C1, X, Y, Z))
        weight_full = np.zeros((X, Y, Z))
        n_windows = 0

        if self.axis == "xy":
            if self.window_size is None:
                raise ValueError("window_size is required when axis='xy'")
            if isinstance(self.window_size, int):
                wx, wy = self.window_size, self.window_size
            else:
                wx, wy = self.window_size
            stride = self.stride if self.stride is not None else (wx, wy)
            sx, sy = stride if isinstance(stride, tuple) else (stride, stride)

            for x0 in range(0, X, sx):
                for y0 in range(0, Y, sy):
                    x1, y1 = min(x0 + wx, X), min(y0 + wy, Y)
                    if x1 - x0 < 2 or y1 - y0 < 2:
                        continue  # skip degenerate slivers at the edge
                    window_img = images[:, x0:x1, y0:y1, :]
                    n_windows += 1
                    mean = self._fit_window(window_img)
                    K_mean_full[:, x0:x1, y0:y1, :] += mean
                    weight_full[x0:x1, y0:y1, :] += 1

        elif self.axis == "z":
            for z0 in range(0, Z, self.slice_window):
                z1 = min(z0 + self.slice_window, Z)
                window_img = images[:, :, :, z0:z1]
                n_windows += 1
                mean = self._fit_window(window_img)
                K_mean_full[:, :, :, z0:z1] += mean
                weight_full[:, :, z0:z1] += 1

        elif self.axis == "xyz":
            if (self.window_size is None or not isinstance(self.window_size, (tuple, list))
                    or len(self.window_size) != 3):
                raise ValueError("axis='xyz' requires window_size as a 3-tuple, e.g. (32, 32, 20)")
            wx, wy, wz = self.window_size
            if self.stride is None:
                sx, sy, sz = wx, wy, wz
            else:
                sx, sy, sz = self.stride if isinstance(self.stride, (tuple, list)) else (self.stride,) * 3

            for x0 in range(0, X, sx):
                for y0 in range(0, Y, sy):
                    for z0 in range(0, Z, sz):
                        x1, y1, z1 = min(x0 + wx, X), min(y0 + wy, Y), min(z0 + wz, Z)
                        if x1 - x0 < 2 or y1 - y0 < 2 or z1 - z0 < 1:
                            continue  # skip degenerate slivers at the edge
                        window_img = images[:, x0:x1, y0:y1, z0:z1]
                        n_windows += 1
                        mean = self._fit_window(window_img)
                        K_mean_full[:, x0:x1, y0:y1, z0:z1] += mean
                        weight_full[x0:x1, y0:y1, z0:z1] += 1

        else:
            raise ValueError(f"axis must be 'xy', 'z', or 'xyz', got {self.axis!r}")

        weight_full = np.clip(weight_full, 1, None)
        K_mean_full = K_mean_full / weight_full[None]
        self.history = {"n_windows": n_windows}

        if self.save_path is not None and self.affine is not None:
            os.makedirs(self.save_path, exist_ok=True)
            nib.save(
                nib.Nifti1Image(K_mean_full.transpose((1, 2, 3, 0)).astype(np.float64), self.affine),
                join(self.save_path, "K_mean_windowed.nii"),
            )

        return K_mean_full, self.history