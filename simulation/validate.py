"""
End-to-end correctness check for the kinetic-parameter-estimation
methods in this repo (voxelwise NLLS, PINN) on a synthetic prostate
phantom with known ground-truth K1/k2(/k3):

    DCE-MRI -> 1-tissue compartment model (K1 = Ktrans, k2 = kep)
    PET     -> irreversible 2-tissue compartment model (K1, k2, k3)

For each modality: build a 64x64x20 prostate+cancer phantom, sample
literature-informed ground-truth parameter maps, forward-simulate noisy
TACs, then run both estimation methods and score them against the known
ground truth (bias / RMSE / correlation, per tissue class).

NOTE on compute cost: voxelwise NLLS and the PINN are both far too slow
to run on the full 64x64x20 volume in a quick check (voxelwise is a
per-voxel scipy optimization; the PINN's z_slices loop -- see the note in
`run_dce_validation` -- doesn't actually reduce work per slice). This
script therefore validates on a reduced central sub-volume by default
(`Z_SLICES`, `VOXELWISE_SUBSAMPLE`) and prints how to scale up.
"""
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.phantom import build_prostate_phantom
from simulation.kinetics_literature import DCE_1TCM_PARAMS, PET_2TCM_PARAMS, sample_param_maps
from simulation.forward_models import parker_aif, feng_input_function, simulate_1tcm_volume, simulate_2tcm_volume
from simulation.voxelwise_pet import calculate_pet_voxelwise

from dce.analysis import calculate_dce_voxelwise
from core.train import Trainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "simulation_data", "validation_runs",
)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
def voxel_metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    valid = np.isfinite(p) & np.isfinite(g)
    p, g = p[valid], g[valid]
    if len(p) < 2:
        return dict(n=len(p), bias=np.nan, rmse=np.nan, corr=np.nan)
    bias = float(np.mean(p - g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    corr = float(pearsonr(p, g)[0]) if np.std(p) > 0 and np.std(g) > 0 else np.nan
    return dict(n=int(len(p)), bias=bias, rmse=rmse, corr=corr)


def print_metric_table(title, per_param):
    print(f"\n--- {title} ---")
    for param, per_class in per_param.items():
        for cls, m in per_class.items():
            print(f"  {param:>3s} | {cls:>8s} | n={m['n']:5d}  "
                  f"bias={m['bias']:+.4f}  rmse={m['rmse']:.4f}  corr={m['corr']:.3f}")


# =======================================================================
# DCE (1TCM) validation
# =======================================================================
def run_dce_validation(shape=(64, 64, 20), z_slices=None, voxelwise_subsample=1200,
                        pinn_epochs=150, seed=0):
    print("\n================= DCE-MRI (1TCM) validation =================")
    info = build_prostate_phantom(shape, seed=seed)
    label = info.label
    if z_slices is None:
        cz = int(round(info.cancer_center[2]))
        z_slices = tuple(sorted(set(z for z in (cz - 1, cz, cz + 1) if 0 <= z < shape[2])))
        print(f"  auto-selected z_slices={z_slices} around cancer center {info.cancer_center}")
    gt = sample_param_maps(label, DCE_1TCM_PARAMS, ["K1", "k2"], seed=seed)

    t = np.cumsum(np.asarray([0.1522] * 35))          # minutes, matches config.DCE["dt"]
    aif = parker_aif(t)

    noisy, clean = simulate_1tcm_volume(t, aif, gt["K1"], gt["k2"], noise_std=0.03, rng=seed)

    z_idx = list(z_slices)
    img = noisy[:, :, :, z_idx]                        # (T, X, Y, Zsub)
    lab = label[:, :, z_idx]
    gtK1, gtk2 = gt["K1"][:, :, z_idx], gt["k2"][:, :, z_idx]
    tissue = lab > 0
    class_masks = {"normal": lab == 1, "cancer": lab == 2}

    results = {}

    # ---- voxelwise ----
    idx_all = np.argwhere(tissue)
    rng = np.random.default_rng(seed)
    vox_mask = np.zeros_like(tissue)
    if len(idx_all) > voxelwise_subsample:
        sel = rng.choice(len(idx_all), size=voxelwise_subsample, replace=False)
        for x, y, z in idx_all[sel]:
            vox_mask[x, y, z] = True
    else:
        vox_mask = tissue

    t0 = time.time()
    maps_vw = calculate_dce_voxelwise(img, t, aif, mask=vox_mask)
    print(f"[voxelwise] {time.time() - t0:.1f}s on {vox_mask.sum()} voxels")
    results["voxelwise"] = {
        "K1": {c: voxel_metrics(maps_vw["K1"], gtK1, vox_mask & m) for c, m in class_masks.items()},
        "k2": {c: voxel_metrics(maps_vw["k2"], gtk2, vox_mask & m) for c, m in class_masks.items()},
    }
    print_metric_table("DCE voxelwise", results["voxelwise"])

    # ---- PINN ----
    affine = np.eye(4)
    save_path = os.path.join(OUT_DIR, "dce_pinn")
    os.makedirs(save_path, exist_ok=True)
    trainer = Trainer(c_p=aif, num_of_compartment=1, t=t, device=DEVICE,
                       affine=affine, save_path=save_path, epochs=pinn_epochs, windowed=False)
    t0 = time.time()
    ks_out, _ = trainer.train(img, z_slices=[0])
    print(f"[PINN] {time.time() - t0:.1f}s, output shape {ks_out.shape}")
    # ks_out: (num_compartment+1=2, X, Y, Zsub) -> channel 0 = K1 (Ktrans),
    # channel 1 = repo's second 1TCM parameter (labelled "Ve" in
    # convolve_1cm_for_minimize but used as if it were kep/k2 in the ODE
    # residual get_x_1tcm -- see the note printed below).
    pinn_K1 = ks_out[0]
    pinn_second = ks_out[1]
    results["pinn"] = {
        "K1": {c: voxel_metrics(pinn_K1, gtK1, m) for c, m in class_masks.items()},
        "k2 (raw 2nd param)": {c: voxel_metrics(pinn_second, gtk2, m) for c, m in class_masks.items()},
    }
    print_metric_table("DCE PINN", results["pinn"])
    print("  NOTE: core/model.py's 1TCM Ks_net treats its 2nd parameter "
          "inconsistently -- convolve_1cm_for_minimize divides K1 by it as "
          "if it were 've' (Ktrans/ve = kep), but get_x_1tcm's physics "
          "residual uses it directly as if it were kep already. The 'k2' "
          "row above is therefore an ambiguous comparison; see the printed "
          "report for a fixed alternative (K1/ve).")
    ve_alt = pinn_K1 / np.clip(pinn_second, 1e-6, None)
    results["pinn"]["k2 (if 2nd param is ve, kep=K1/ve)"] = {
        c: voxel_metrics(ve_alt, gtk2, m) for c, m in class_masks.items()
    }
    print_metric_table("DCE PINN (alt kep interpretation)",
                        {"k2 (if 2nd param is ve, kep=K1/ve)":
                         results["pinn"]["k2 (if 2nd param is ve, kep=K1/ve)"]})

    return results


# =======================================================================
# PET (irreversible 2TCM) validation
# =======================================================================
def run_pet_validation(shape=(64, 64, 20), z_slices=None, voxelwise_subsample=800,
                        pinn_epochs=150, seed=1):
    print("\n================= Dynamic PET (2TCM) validation =================")
    info = build_prostate_phantom(shape, seed=seed)
    label = info.label
    if z_slices is None:
        cz = int(round(info.cancer_center[2]))
        z_slices = tuple(sorted(set(z for z in (cz - 1, cz, cz + 1) if 0 <= z < shape[2])))
        print(f"  auto-selected z_slices={z_slices} around cancer center {info.cancer_center}")
    gt = sample_param_maps(label, PET_2TCM_PARAMS, ["K1", "k2", "k3"], seed=seed)

    pet_dt = np.asarray([10, 20, 30, 40, 50, 60, 90, 120, 150, 180, 210, 240,
                          270, 300, 330, 360, 410, 460, 510, 560, 610, 660, 960, 1200])
    t = np.cumsum(pet_dt) / 60.0     # minutes; see note below re: pet/analysis.py
    aif = feng_input_function(t)

    noisy, clean = simulate_2tcm_volume(t, aif, gt["K1"], gt["k2"], gt["k3"], noise_std=0.03, rng=seed)

    z_idx = list(z_slices)
    img = noisy[:, :, :, z_idx]
    lab = label[:, :, z_idx]
    gtK1, gtk2, gtk3 = gt["K1"][:, :, z_idx], gt["k2"][:, :, z_idx], gt["k3"][:, :, z_idx]
    tissue = lab > 0
    class_masks = {"normal": lab == 1, "cancer": lab == 2}

    print("  NOTE: pet/analysis.py's pipeline() used to pass PET['dt'] "
          "(frame *durations*) directly as `t` -- fixed (now cumsum'd, "
          "matching dce/analysis.py's convention and what this validation "
          "already used).")

    results = {}

    # ---- voxelwise ----
    idx_all = np.argwhere(tissue)
    rng = np.random.default_rng(seed)
    vox_mask = np.zeros_like(tissue)
    if len(idx_all) > voxelwise_subsample:
        sel = rng.choice(len(idx_all), size=voxelwise_subsample, replace=False)
        for x, y, z in idx_all[sel]:
            vox_mask[x, y, z] = True
    else:
        vox_mask = tissue

    t0 = time.time()
    maps_vw = calculate_pet_voxelwise(img, t, aif, mask=vox_mask, verbose=False)
    print(f"[voxelwise] {time.time() - t0:.1f}s on {vox_mask.sum()} voxels")
    results["voxelwise"] = {
        "K1": {c: voxel_metrics(maps_vw["K1"], gtK1, vox_mask & m) for c, m in class_masks.items()},
        "k2": {c: voxel_metrics(maps_vw["k2"], gtk2, vox_mask & m) for c, m in class_masks.items()},
        "k3": {c: voxel_metrics(maps_vw["k3"], gtk3, vox_mask & m) for c, m in class_masks.items()},
    }
    print_metric_table("PET voxelwise", results["voxelwise"])

    # ---- PINN ----
    affine = np.eye(4)
    save_path = os.path.join(OUT_DIR, "pet_pinn")
    os.makedirs(save_path, exist_ok=True)
    trainer = Trainer(c_p=aif, num_of_compartment=2, t=t, device=DEVICE,
                       affine=affine, save_path=save_path, epochs=pinn_epochs, windowed=False)
    t0 = time.time()
    ks_out, _ = trainer.train(img, z_slices=[0])
    print(f"[PINN] {time.time() - t0:.1f}s, output shape {ks_out.shape}")
    # 2TCM Ks_net IS internally consistent (k1,k2,k3 used the same way in
    # both convolve_2cm_for_minimize and get_x_2tcm), so channels map
    # directly to K1, k2, k3.
    pinn_K1, pinn_k2, pinn_k3 = ks_out[0], ks_out[1], ks_out[2]
    results["pinn"] = {
        "K1": {c: voxel_metrics(pinn_K1, gtK1, m) for c, m in class_masks.items()},
        "k2": {c: voxel_metrics(pinn_k2, gtk2, m) for c, m in class_masks.items()},
        "k3": {c: voxel_metrics(pinn_k3, gtk3, m) for c, m in class_masks.items()},
    }
    print_metric_table("PET PINN", results["pinn"])

    return results


if __name__ == "__main__":
    import json

    dce_results = run_dce_validation()
    pet_results = run_pet_validation()

    with open(os.path.join(OUT_DIR, "dce_results.json"), "w") as f:
        json.dump(dce_results, f, indent=2)
    with open(os.path.join(OUT_DIR, "pet_results.json"), "w") as f:
        json.dump(pet_results, f, indent=2)
    print(f"\nSaved JSON summaries to {OUT_DIR}")
