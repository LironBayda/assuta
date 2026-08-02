"""
Windowed PINN training -- an alternative to the whole-image joint
training in core/train.py's Trainer, modeled on the windowed/patch-based
processing style used in van Herten et al. (Medical Image Analysis,
2022). Splits the image into independent spatial windows and trains a
SEPARATE PINN (or B-PINN ensemble) per window, with no shared state
across windows, then stitches the per-window results back into a
full-size volume.

This is the architectural mechanism that CAN introduce patch-boundary
artifacts: since each window's Ks_net/f_x is fit completely
independently (different random init, no cross-window information), two
adjacent windows have no reason to agree at their shared boundary even
if the true underlying tissue is smooth there. Compare against
core.train.Trainer's whole-image approach (which has no window
boundaries to begin with, since every voxel's Ks_net parameters are fit
jointly in one optimization) to see whether this specific mechanism
actually produces visible artifacts in this codebase's setup.
"""
import numpy as np

from core.train import Trainer


def train_windowed(images, c_p, t, num_of_compartment, window_size, stride=None,
                    bayesian=False, n_ensemble=5, dropout_p=0.1, epochs=300,
                    device="cpu", affine=None, save_path=None, **trainer_kwargs):
    """
    images : (T, X, Y, Z) array
    window_size : int or (int, int) -- spatial window size in (X, Y).
        Z is not windowed (kept whole) since our phantoms are typically
        thin in Z; window over Z too if needed by passing a 3-tuple and
        extending the logic below.
    stride : int, optional -- defaults to window_size (non-overlapping
        windows, matching a patch-tiling scheme). A stride < window_size
        gives overlapping windows (blended by averaging in the overlap).
    bayesian : bool -- if True, each window is fit with train_ensemble
        (B-PINN deep ensemble) instead of a single train() call, and a
        per-voxel uncertainty map is also stitched together.

    Returns a dict:
        K_mean         -- (C+1, X, Y, Z) stitched point estimate
        K_uncertainty  -- (C+1, X, Y, Z) stitched uncertainty (bayesian only, else None)
        n_windows      -- number of windows processed
    """
    T, X, Y, Z = images.shape
    if isinstance(window_size, int):
        wx, wy = window_size, window_size
    else:
        wx, wy = window_size
    stride = stride if stride is not None else (wx, wy)
    sx, sy = stride if isinstance(stride, tuple) else (stride, stride)

    C1 = num_of_compartment + 1
    K_mean_full = np.zeros((C1, X, Y, Z))
    K_unc_full = np.zeros((C1, X, Y, Z)) if bayesian else None
    weight_full = np.zeros((X, Y, Z))   # for averaging in overlap regions

    n_windows = 0
    for x0 in range(0, X, sx):
        for y0 in range(0, Y, sy):
            x1, y1 = min(x0 + wx, X), min(y0 + wy, Y)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue  # skip degenerate slivers at the edge

            window_img = images[:, x0:x1, y0:y1, :]
            n_windows += 1

            trainer = Trainer(
                c_p=c_p, num_of_compartment=num_of_compartment, t=t, device=device,
                affine=affine, save_path=None, epochs=epochs, **trainer_kwargs,
            )

            if bayesian:
                mean, std, std_demeaned, run_level_shift, all_runs = trainer.train_ensemble(
                    window_img, n_ensemble=n_ensemble, z_slices=[0], dropout_p=dropout_p, save=False,
                )
                K_unc_full[:, x0:x1, y0:y1, :] += std_demeaned
            else:
                mean, _ = trainer.train(window_img, z_slices=[0])

            K_mean_full[:, x0:x1, y0:y1, :] += mean
            weight_full[x0:x1, y0:y1, :] += 1

    weight_full = np.clip(weight_full, 1, None)
    K_mean_full = K_mean_full / weight_full[None]
    if bayesian:
        K_unc_full = K_unc_full / weight_full[None]

    if save_path is not None and affine is not None:
        import os
        import nibabel as nib
        os.makedirs(save_path, exist_ok=True)
        nib.save(nib.Nifti1Image(K_mean_full.transpose((1, 2, 3, 0)).astype(np.float64), affine),
                  os.path.join(save_path, "K_mean_windowed.nii"))
        if bayesian:
            nib.save(nib.Nifti1Image(K_unc_full.transpose((1, 2, 3, 0)).astype(np.float64), affine),
                      os.path.join(save_path, "K_uncertainty_windowed.nii"))

    return {"K_mean": K_mean_full, "K_uncertainty": K_unc_full, "n_windows": n_windows}
