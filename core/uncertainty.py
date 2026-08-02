"""
B-PINN uncertainty: run a Sine B-PINN deep ensemble and get back both the
point-estimate K maps and their per-voxel uncertainty, in one call. Thin
wrapper around Trainer.train_ensemble (see its docstring for the full
mechanism and the run-level-effect caveat) -- this module exists so
"calculate K and its uncertainty" has one obvious entry point in the PINN
package, rather than requiring callers to know about the Trainer class
internals.
"""
import os

import numpy as np

from core.train import Trainer


def estimate_with_uncertainty(
    images, c_p, t, num_of_compartment, save_path=None, affine=None,
    n_ensemble=5, epochs=300, dropout_p=0.1, tissue_mask=None,
    lr=0.01, causality_eps_final=2000.0, device="cpu", activation="sine",
):
    """
    Fit a Sine B-PINN ensemble to `images` and return both the K
    point-estimate maps and their per-voxel uncertainty.

    images : (T, X, Y, Z) array
    c_p : (T,) arterial input function
    t : (T,) timestamps
    num_of_compartment : 1 (DCE-style 1TCM) or 2 (PET-style 2TCM)
    tissue_mask : (X, Y, Z) bool array, optional -- used for the
        run-level-mean subtraction in the de-meaned uncertainty (see
        Trainer.train_ensemble). If omitted, inferred from nonzero mean K.

    Returns a dict:
        K_mean            -- (C+1, X, Y, Z) point estimate
        K_uncertainty      -- (C+1, X, Y, Z) raw ensemble std (per-voxel
                               FORMAT, but partly a per-subject/per-fit
                               effect -- see caveat below)
        K_uncertainty_demeaned -- (C+1, X, Y, Z) run-demeaned std, a
                               cleaner isolation of the true per-voxel
                               signal
        run_level_shift    -- (C+1, n_ensemble) each run's own tissue
                               mean, for inspecting the per-subject/
                               per-fit confound directly
        all_runs           -- (n_ensemble, C+1, X, Y, Z) raw stack

    Caveat (found via direct inspection): all voxels within one ensemble
    run share the same f_x trunk, so a run's random init/dropout can
    shift the WHOLE scan's estimate together. That run-level effect was
    found to be a substantial fraction of the raw `K_uncertainty`
    magnitude -- i.e. a lot of "this voxel's uncertainty" is actually
    "which random seed did the whole scan's fit happen to land on" not
    independent per-voxel noise. Prefer `K_uncertainty_demeaned` when you
    need a genuinely per-voxel signal (e.g. for calibration checks
    against per-voxel error); use `run_level_shift`'s own spread when you
    specifically want the per-subject/per-fit uncertainty.
    """
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    trainer = Trainer(
        c_p=c_p, num_of_compartment=num_of_compartment, t=t, device=device,
        affine=affine if affine is not None else np.eye(4),
        save_path=save_path, epochs=epochs,
        lr=lr, causality_eps_final=causality_eps_final, activation=activation,
    )
    mean, std, std_demeaned, run_level_shift, all_runs = trainer.train_ensemble(
        images, n_ensemble=n_ensemble, z_slices=[0], dropout_p=dropout_p,
        save=save_path is not None, tissue_mask=tissue_mask,
    )
    return {
        "K_mean": mean,
        "K_uncertainty": std,
        "K_uncertainty_demeaned": std_demeaned,
        "run_level_shift": run_level_shift,
        "all_runs": np.stack(all_runs, axis=0),
    }
