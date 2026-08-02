"""
Uncertainty correlation analysis for the Sine B-PINN ensemble
(core/uncertainty.py). Two questions:

1. Is a voxel's uncertainty correlated between DCE and PET, on the same
   underlying anatomy? (Same phantom geometry, independently simulated
   DCE-1TCM and PET-2TCM data and independently trained ensembles.)
2. Does restricting to LOW-uncertainty voxels change the K1-vs-ground-
   truth correlation -- and if so, is that a genuine accuracy effect
   (check RMSE too) or a statistical range-restriction artifact (Pearson
   r shrinks when you restrict the range of the x-variable, even if
   predictions in that subset are MORE accurate)?

Produces a PNG figure (maps + scatter plots) so results can be inspected
visually, not just as printed numbers.
"""
import os
import sys

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.phantom import build_prostate_phantom
from simulation.kinetics_literature import DCE_1TCM_PARAMS, PET_2TCM_PARAMS, sample_param_maps
from simulation.forward_models import parker_aif, feng_input_function, simulate_1tcm_volume, simulate_2tcm_volume
from core.uncertainty import estimate_with_uncertainty

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "simulation_data", "validation_runs", "uncertainty_correlation",
)
os.makedirs(OUT_DIR, exist_ok=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run(shape=(24, 24, 6), seed=0, n_ensemble=5, epochs=200):
    info = build_prostate_phantom(shape, seed=seed)
    label = info.label
    tissue = label > 0

    # --- DCE (1TCM) ---
    gt_dce = sample_param_maps(label, DCE_1TCM_PARAMS, ["K1", "k2"], seed=seed)
    t_dce = np.cumsum(np.asarray([0.1522] * 35))
    aif_dce = parker_aif(t_dce)
    img_dce, _ = simulate_1tcm_volume(t_dce, aif_dce, gt_dce["K1"], gt_dce["k2"], noise_std=0.03, rng=seed)

    print("[DCE] running B-PINN ensemble...")
    res_dce = estimate_with_uncertainty(
        img_dce, aif_dce, t_dce, num_of_compartment=1,
        save_path=os.path.join(OUT_DIR, "dce"), n_ensemble=n_ensemble, epochs=epochs,
        tissue_mask=tissue,
    )

    # --- PET (2TCM) ---
    gt_pet = sample_param_maps(label, PET_2TCM_PARAMS, ["K1", "k2", "k3"], seed=seed)
    pet_dt = np.asarray([10, 20, 30, 40, 50, 60, 90, 120, 150, 180, 210, 240,
                          270, 300, 330, 360, 410, 460, 510, 560, 610, 660, 960, 1200])
    t_pet = np.cumsum(pet_dt) / 60.0
    aif_pet = feng_input_function(t_pet)
    img_pet, _ = simulate_2tcm_volume(t_pet, aif_pet, gt_pet["K1"], gt_pet["k2"], gt_pet["k3"],
                                       noise_std=0.03, rng=seed)

    print("[PET] running B-PINN ensemble...")
    res_pet = estimate_with_uncertainty(
        img_pet, aif_pet, t_pet, num_of_compartment=2,
        save_path=os.path.join(OUT_DIR, "pet"), n_ensemble=n_ensemble, epochs=epochs,
        tissue_mask=tissue,
    )

    # ---------------- Analysis 1: DCE vs PET uncertainty correlation ----------------
    dce_unc = res_dce["K_uncertainty_demeaned"][0][tissue]   # K1 channel
    pet_unc = res_pet["K_uncertainty_demeaned"][0][tissue]
    r_unc_cross_modal = pearsonr(dce_unc, pet_unc)[0]
    print(f"\ncorr(DCE K1 uncertainty, PET K1 uncertainty), same anatomy: r={r_unc_cross_modal:.3f}")

    # ---------------- Analysis 2: low-uncertainty subset effect ----------------
    results = {}
    for name, res, gt in [("DCE", res_dce, gt_dce), ("PET", res_pet, gt_pet)]:
        K1_mean = res["K_mean"][0]
        K1_unc = res["K_uncertainty_demeaned"][0]
        gtK1 = gt["K1"]

        pred_t, gt_t, unc_t = K1_mean[tissue], gtK1[tissue], K1_unc[tissue]
        r_full = pearsonr(pred_t, gt_t)[0]
        rmse_full = rmse(pred_t, gt_t)
        gt_range_full = gt_t.max() - gt_t.min()

        median_unc = np.median(unc_t)
        low_mask = unc_t <= median_unc
        r_low = pearsonr(pred_t[low_mask], gt_t[low_mask])[0]
        rmse_low = rmse(pred_t[low_mask], gt_t[low_mask])
        gt_range_low = gt_t[low_mask].max() - gt_t[low_mask].min()

        print(f"\n[{name}] full set:  n={len(gt_t)}  corr={r_full:.3f}  RMSE={rmse_full:.4f}  "
              f"GT range={gt_range_full:.3f}")
        print(f"[{name}] low-unc:   n={low_mask.sum()}  corr={r_low:.3f}  RMSE={rmse_low:.4f}  "
              f"GT range={gt_range_low:.3f}")
        print(f"[{name}] -> correlation {'DROPPED' if r_low < r_full else 'rose'} "
              f"({r_full:.3f} -> {r_low:.3f}), RMSE {'improved' if rmse_low < rmse_full else 'worsened'} "
              f"({rmse_full:.4f} -> {rmse_low:.4f}); GT range shrank "
              f"{gt_range_full:.3f} -> {gt_range_low:.3f} "
              f"({'likely range-restriction artifact' if (r_low < r_full and rmse_low <= rmse_full) else 'genuine'})")

        results[name] = dict(pred=pred_t, gt=gt_t, unc=unc_t, low_mask=low_mask,
                              r_full=r_full, r_low=r_low, rmse_full=rmse_full, rmse_low=rmse_low)

    return res_dce, res_pet, gt_dce, gt_pet, label, tissue, results, r_unc_cross_modal


if __name__ == "__main__":
    run()
