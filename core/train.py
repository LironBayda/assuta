import logging
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
    Full trainer for a Physics-Informed Neural Network (PINN) fitting a
    compartmental kinetic model to PET time-activity curves (TACs).

    Handles TAC preparation, normalization, training, and saving kinetic
    parameter maps (Ks) as NIfTI.
    """

    def __init__(
        self,
        c_p,
        num_of_compartment,
        t,
        device="cpu",
        affine=None,
        save_path=None,
        epochs=PINN["epochs"],
        lr=PINN["learning_rate"],
        grad_clip: Optional[float] = 1.0,
        causality_eps_final: float = 2000.0,
        hidden_size: int = 40,
        bottleneck_size: Optional[int] = None,
        omega_0: float = 1.0,
        activation: str = "sine",
        physics_weight: float = 0.01,
        tac_consistency_weight: Optional[float] = None,
        reg_weight: float = 1e-4,
    ):
        """
        hidden_size, bottleneck_size, omega_0 : f_x/f_x_bpinn structural
            hyperparameters -- see core/model.py's f_x docstring and
            core/pinn.py's PhysicsInformedNN docstring. Swept in
            simulation/pinn_hyperparam_sweep.py.
        physics_weight, tac_consistency_weight, reg_weight : PINNLoss
            term weights -- see core/losses.py. Swept in
            simulation/pinn_hyperparam_sweep.py.
        causality_eps_final : float
            If > 0 and `self.pinn.loss_fn` exposes a `causality_eps`
            attribute, linearly anneal it from 0 to this value across
            training (causal time-weighting of the physics loss; see
            Wang et al. 2022). 0.0 disables annealing and preserves the
            original uniform-weighting behavior exactly.

            Swept 0-10000 (at lr=0.01, on the phantom simulation): this is
            the single clearest, most unambiguous win found in the whole
            PINN hyperparameter search this session. K1 correlation rises
            steadily from ~0.90 (eps=0) to ~0.94 (eps=2000-5000), and
            critically, k2's correlation FLIPS SIGN from negative to
            positive somewhere between eps=100 and eps=1000 -- i.e.
            causal weighting isn't just a magnitude improvement, it
            changes which basin the optimizer converges to. **Default
            changed to 2000** (from the old 100) accordingly. See
            simulation/pinn_hyperparam_sweep.py.

            tac_consistency_weight and grad_clip were ALSO swept at this
            same (lr=0.01, causality_eps_final=2000) setting and were NOT
            changed:
            - tac_consistency_weight: a genuine K1-vs-k2 trade-off, not
              a single winner -- raising it (e.g. to 0.5) pushes K1 up to
              ~0.96 but collapses k2 to near-zero/unstable (0.04, std
              0.18). Left at the existing default (favors a usable k2)
              rather than picking a side; if your priority is K1 only,
              consider raising this explicitly.
            - grad_clip: no clear winner. `None` (no clipping) had the
              best MEAN k2 (0.43) but with ~2x the variance of the
              default across seeds (std 0.18 vs 0.09) -- a real
              risk/reward trade-off, not a clean improvement.
              `grad_clip=0.5` looked good on a single seed (k2=0.35) but
              did NOT hold up across seeds (mean dropped to 0.13) --
              another single-seed false lead, same pattern as the
              physics_weight=0.02 case found earlier. Kept at 1.0.

        A prior two-phase Adam+L-BFGS training schedule (adam_frac,
        lbfgs_max_iter params) was REMOVED after a hyperparameter sweep
        found pure Adam for the full epoch count matched or beat every
        two-phase schedule tried -- see simulation/pinn_hyperparam_sweep.py's
        "Training schedule" section.
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
        self.physics_weight = physics_weight
        self.tac_consistency_weight = tac_consistency_weight
        self.reg_weight = reg_weight
        self.causality_eps_final = causality_eps_final
        self.history = {"loss": [], "phase": []}

        self.num = np.max(c_p)
        if not np.isfinite(self.num) or self.num <= 0:
            raise ValueError(
                f"Invalid normalization constant from c_p (max={self.num}); "
                "expected a positive finite value."
            )

        self.dt = np.diff(np.insert(t, 0, 0))
        self.lr = lr

        # Prepare time tensor
        t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
        t_std = t_tensor.std()
        if t_std == 0:
            raise ValueError("t has zero variance; cannot normalize.")
        self.t = (t_tensor - t_tensor.mean()) / t_std
        self.t.requires_grad_(True)
        self.sigma_t = (1 / t_std).detach()

        # Prepare c_p
        self.c_p = c_p / self.num
        self.c_p_tensor = torch.tensor(self.c_p, dtype=torch.float32, device=self.device)

    # ---------------------------
    # TAC Preparation
    # ---------------------------
    @staticmethod
    def prepare_tacs(images):
        n_time, size_x, size_y, size_z = images.shape
        tacs = images.reshape(n_time, size_x * size_y * size_z)
        return tacs, (size_x, size_y, size_z, n_time)

    # ---------------------------
    # Single training step
    # ---------------------------
    def train_step(self):
        # forward_with_dt gives exact per-voxel/per-compartment derivatives
        # in one pass (see f_x.forward_with_dt) — replaces the old
        # compute_derivatives, which used grad_outputs=torch.ones_like(...)
        # over a column range. That summed derivatives across whatever
        # columns were selected instead of giving one per column, which
        # silently let different voxels' physics violations cancel out
        # before they were ever squared in the loss.
        output1, doutput1_dt = self.pinn.f_x.forward_with_dt(self.t.view(-1, 1))
        doutput1_dt = doutput1_dt * self.sigma_t  # chain rule: d/dt_orig = d/dt_norm * (1/std)

        n_voxels = self.tacs_tensor.shape[1]
        if self.num_of_compartment == 2:
            # output1 has width 2*n_voxels + 1: columns [0:n_voxels) are
            # compartment 1, [n_voxels:2*n_voxels) are compartment 2, and
            # column -1 is the shared c_p prediction (matches the split
            # PINNLoss uses: output1[:, :tacs_num] + output1[:, tacs_num:-1]).
            # The previous split point, (n_voxels - 1) // 2, used n_voxels
            # as the base instead of output1's actual width — off by
            # roughly a factor of 2, so "dC2" was actually being computed
            # from the second half of compartment 1's own columns rather
            # than compartment 2 at all.
            dC1 = doutput1_dt[:, :n_voxels]
            dC2 = doutput1_dt[:, n_voxels : 2 * n_voxels]
        else:
            dC1 = doutput1_dt[:, :-1]
            dC2 = None

        output2 = self.pinn.Ks_net(
            output1[:, -1],
            output1[:, :n_voxels],
            dC1,
            dC2,
            self.c_p_tensor,
        )
        return self.pinn.loss_fn([output1, output2], [self.tacs_tensor, self.c_p_tensor])

    # ---------------------------
    # Run training
    # ---------------------------
    def train(self, images, z_slices: Optional[range] = None,
              bayesian: bool = False, dropout_p: float = 0.1, ks_init=None,
              voxel_weight=None):
        """
        Train the PINN on a 4D image volume (time, x, y, z).

        Parameters
        ----------
        images : np.ndarray
            Shape (n_time, size_x, size_y, size_z).
        z_slices : range, optional
            Which z-slices to process. Defaults to just slice 0 (matching
            prior behavior). Pass `range(images.shape[3])` to process the
            full volume slice-by-slice.
        bayesian : bool
            Use f_x_bpinn (MC-Dropout "Sine B-PINN") instead of the
            deterministic f_x. A single `train()` call still returns one
            point estimate of Ks (dropout affects f_x's TAC
            reconstruction and, through training, the fitted Ks -- but
            Ks_raw itself is a plain nn.Parameter, not sampled at
            inference). For an actual per-voxel uncertainty *map*, use
            `train_ensemble()` below, which trains several bayesian runs
            and reports their mean/std.
        dropout_p : float
            Dropout probability for f_x_bpinn (only used if bayesian=True).
        ks_init : array, optional, shape (num_of_compartment+1, X, Y, Z)
            Per-voxel Ks initial values (e.g. from a VAE latent warm
            start), reshaped internally to match Ks_net's flat (C+1,
            tac_num) convention. See core.model.Ks_net's ks_init
            docstring.
        voxel_weight : array, optional, shape (X, Y, Z)
            Per-voxel loss weight (e.g. inverse-class-frequency, from a
            coarse segmentation/prior voxelwise pass), flattened
            internally to match PINNLoss's (tacs_num,) convention. Fixes
            a minority-class-gets-drowned-out failure mode -- see
            PINNLoss's voxel_weight docstring (core/losses.py) for the
            full explanation. None (default): uniform weighting.
        """
        if images.ndim != 4:
            raise ValueError(f"images must be 4D (time, x, y, z); got shape {images.shape}")
        if images.shape[0] != self.c_p_tensor.shape[0]:
            raise ValueError(
                f"images time dim ({images.shape[0]}) must match c_p length "
                f"({self.c_p_tensor.shape[0]})."
            )

        results = np.zeros([self.num_of_compartment + 1] + list(images.shape[1:]))
        z_slices = z_slices if z_slices is not None else [0]

        for z in z_slices:
            tacs, shape = self.prepare_tacs(images)  # [:, :, :, z:z + 1] if slicing per-z
            self.tacs_tensor = torch.tensor(
                np.asarray(tacs) / self.num, dtype=torch.float32, device=self.device
            )

            ks_init_flat = None
            if ks_init is not None:
                ks_init_arr = np.asarray(ks_init)
                ks_init_flat = ks_init_arr.reshape(ks_init_arr.shape[0], -1)

            voxel_weight_flat = None
            if voxel_weight is not None:
                voxel_weight_flat = np.asarray(voxel_weight).reshape(-1)

            self.pinn = PhysicsInformedNN(
                self.num_of_compartment, self.dt, shape, device=self.device,
                bayesian=bayesian, dropout_p=dropout_p,
                hidden_size=self.hidden_size, bottleneck_size=self.bottleneck_size,
                omega_0=self.omega_0, activation=self.activation, physics_weight=self.physics_weight,
                tac_consistency_weight=self.tac_consistency_weight, reg_weight=self.reg_weight,
                ks_init=ks_init_flat, voxel_weight=voxel_weight_flat,
            )

            f_x_params = list(self.pinn.f_x.parameters())
            ks_params = list(self.pinn.Ks_net.parameters())
            params = f_x_params + ks_params

            # ---- Adam optimization ----
            # Adam handles the noisy, non-convex PINN residual landscape
            # well. A prior L-BFGS second-phase refinement (standard in
            # much PINN literature since Raissi et al.) was tested via
            # hyperparameter sweep this session and REMOVED: Adam-only
            # for the full epoch count matched or beat every two-phase
            # Adam+L-BFGS schedule tried (see
            # simulation/pinn_hyperparam_sweep.py's "Training schedule"
            # section) while being simpler and faster.
            optimizer = torch.optim.AdamW(params, lr=self.lr)
            stopped_early = False

            for epoch in range(self.epochs):
                if self.causality_eps_final > 0 and hasattr(self.pinn.loss_fn, "causality_eps"):
                    self.pinn.loss_fn.causality_eps = self.causality_eps_final * (
                        epoch / max(self.epochs - 1, 1)
                    )

                optimizer.zero_grad()
                loss = self.train_step()

                if not torch.isfinite(loss):
                    logger.error(
                        f"Non-finite loss ({loss.item()}) at epoch {epoch + 1}, "
                        f"z={z}; stopping this slice."
                    )
                    stopped_early = True
                    break

                loss.backward()
                if self.grad_clip is not None:
                    # Clipped as two separate groups, not one combined
                    # norm: f_x's final layer has ~n_voxels * hidden_size
                    # weights, which typically dwarfs Ks_net's parameter
                    # count. A single global-norm clip would scale down
                    # both groups by whatever factor f_x needed, which can
                    # starve Ks_net's updates specifically.
                    torch.nn.utils.clip_grad_norm_(f_x_params, self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(ks_params, self.grad_clip)
                optimizer.step()

                self.history["loss"].append(loss.item())
                self.history["phase"].append("adam")
                if not (epoch + 1) % 100:
                    components = getattr(self.pinn.loss_fn, "last_components", None)
                    if components:
                        comp_str = "  ".join(
                            f"{k}={v.item():.4f}" for k, v in components.items()
                        )
                        logger.info(
                            f"[Adam] Epoch {epoch + 1}/{self.epochs} - Loss: {loss.item():.6f}  "
                            f"({comp_str})"
                        )
                    else:
                        logger.info(
                            f"[Adam] Epoch {epoch + 1}/{self.epochs} - Loss: {loss.item():.6f}"
                        )

            # save_ks despite its name does NOT write to disk -- it just
            # extracts/reshapes the trained Ks map into `results`. It must
            # run unconditionally (previously gated behind the save_path
            # check below, which meant train() silently returned all-
            # zeros whenever save_path was None -- a real bug, found via
            # core/windowed.py calling train() with save_path=None for
            # per-window sub-training).
            results = self.save_ks(z, self.pinn, shape, results)

        if self.save_path is not None and self.affine is not None:
            nib.save(
                nib.Nifti1Image(results.transpose((1, 2, 3, 0)).astype(np.float64), self.affine),
                join(self.save_path, "K.nii"),
            )
        return results, self.history

    # ---------------------------
    # Save Ks as NIfTI
    # ---------------------------
    def save_ks(self, z, pinn, shape, results):
        Ks = pinn.Ks_net.Ks.detach().cpu().numpy()
        size_x, size_y, size_z, _ = shape
        i_img = Ks.reshape((self.num_of_compartment + 1), size_x, size_y, size_z)
        return i_img

    # ---------------------------
    # Deep-ensemble Sine B-PINN: per-voxel Ks mean AND uncertainty
    # ---------------------------
    def train_ensemble(self, images, n_ensemble=5, z_slices: Optional[range] = None,
                        dropout_p=0.1, save=True, tissue_mask=None):
        """
        Trains `n_ensemble` independent Sine B-PINN fits (bayesian=True,
        fresh random init + fresh dropout masks each run -- a Deep
        Ensemble of Bayesian members, combining both established PINN
        uncertainty routes from the literature rather than picking one)
        and reports the per-voxel mean and standard deviation of Ks
        across the ensemble: the mean is the point-estimate K1/k2(/k3)
        map, and the std is a per-voxel epistemic+approximate-aleatoric
        uncertainty map, directly analogous to the deep-ensemble-of-MVE
        approach in the DCE-MRI uncertainty literature (Van Elburg et
        al.), just with each ensemble member being a Sine B-PINN instead
        of an MVE network.

        IMPORTANT CAVEAT (found via direct inspection this session): all
        voxels within one ensemble run share the same f_x trunk, so a
        run's random init/dropout can shift the WHOLE scan's K estimate
        together -- this run-level effect was found to be ~75% the
        magnitude of the reported per-voxel std itself, i.e. a large
        fraction of "std" is really a per-subject/per-fit effect, not
        independent per-voxel noise. `std_demeaned` below subtracts each
        run's own tissue-mean before computing the spread, isolating the
        within-run RELATIVE variation across voxels -- closer to a true
        per-voxel signal. `std` (raw) is kept for comparison/transparency.

        tissue_mask : (X, Y, Z) bool array, optional
            Used only for the run-level-mean subtraction in
            `std_demeaned`. If omitted, all voxels with nonzero mean Ks
            are used.

        Returns
        -------
        mean          : (C+1, X, Y, Z) -- point estimate
        std           : (C+1, X, Y, Z) -- raw ensemble std (per-voxel
                         format, but confounded with the run-level effect)
        std_demeaned  : (C+1, X, Y, Z) -- run-demeaned std (isolates the
                         within-run per-voxel signal)
        run_level_shift : (C+1, n_ensemble) -- each run's own tissue-mean,
                         for inspecting the confound directly
        all_runs      : list of the raw per-run (C+1, X, Y, Z) arrays
        """
        all_runs = []
        for m in range(n_ensemble):
            logger.info(f"[Ensemble] training member {m + 1}/{n_ensemble}")
            ks_out, _ = self.train(images, z_slices=z_slices, bayesian=True, dropout_p=dropout_p)
            all_runs.append(ks_out)

        stacked = np.stack(all_runs, axis=0)   # (M, C+1, X, Y, Z)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)

        # De-meaned version: subtract each run's own tissue-mean (per
        # channel) before computing std, so a whole-scan shift in one run
        # doesn't inflate every voxel's "uncertainty" identically.
        C1 = stacked.shape[1]
        mask = tissue_mask if tissue_mask is not None else (mean[0] != 0)
        run_level_shift = np.zeros((C1, len(all_runs)))
        demeaned = stacked.copy()
        for c in range(C1):
            for m in range(len(all_runs)):
                run_mean = stacked[m, c][mask].mean()
                run_level_shift[c, m] = run_mean
                demeaned[m, c] = stacked[m, c] - run_mean
        std_demeaned = demeaned.std(axis=0)

        if save and self.save_path is not None and self.affine is not None:
            nib.save(
                nib.Nifti1Image(mean.transpose((1, 2, 3, 0)).astype(np.float64), self.affine),
                join(self.save_path, "K_mean.nii"),
            )
            nib.save(
                nib.Nifti1Image(std.transpose((1, 2, 3, 0)).astype(np.float64), self.affine),
                join(self.save_path, "K_uncertainty.nii"),
            )
            nib.save(
                nib.Nifti1Image(std_demeaned.transpose((1, 2, 3, 0)).astype(np.float64), self.affine),
                join(self.save_path, "K_uncertainty_demeaned.nii"),
            )
            np.save(join(self.save_path, "K_all_runs.npy"), stacked)

        return mean, std, std_demeaned, run_level_shift, all_runs