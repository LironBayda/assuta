import os

import numpy as np

from config import PINN, DEVICE, PET
from core.train import Trainer
from pet.preprocessing import preprocessing
from simulation.voxelwise_pet import calculate_pet_voxelwise, calculate_pet_voxelwise_1tcm
from dce.analysis import save_maps_as_nifti


def pipeline(
        path,
        num_of_compartment=2,
        epochs=PINN["epochs"],
        device=DEVICE,
        method="pinn",
        n_ensemble=5,
        dropout_p=0.1,
):
    """
    Full PET kinetic pipeline -- mirrors dce.analysis.pipeline's
    structure/method dispatch ("pinn" / "voxelwise" / "bayesian") rather
    than being a PINN-only hardcoded function, for consistency across
    modalities.

    method:
        "pinn"       -> PINN kinetic fitting (1TCM if num_of_compartment=1,
                        2TCM/irreversible if num_of_compartment=2)
        "voxelwise" -> classical voxelwise NLLS fitting -- 1TCM
                        (calculate_pet_voxelwise_1tcm) if
                        num_of_compartment=1, else the full irreversible
                        2TCM fit (calculate_pet_voxelwise)
        "bayesian"  -> Sine B-PINN deep ensemble: same K1/k2(/k3) point
                        estimate as "pinn", PLUS a per-voxel uncertainty
                        map (see core/uncertainty.py). Saves K_mean.nii,
                        K_uncertainty.nii, and K_uncertainty_demeaned.nii
                        (prefer the demeaned map for per-voxel calibration
                        checks). Costs ~n_ensemble x the runtime of a
                        single "pinn" fit.

    n_ensemble, dropout_p : only used by method="bayesian" -- see
        core.uncertainty.estimate_with_uncertainty.
    """

    # -----------------------------
    # 1. Preprocessing
    # -----------------------------
    img, aif, affine = preprocessing(path)
    # img: (X, Y, Z, T)

    x, y, z, _ = np.asarray(img.shape) // 7 * 3
    cropped = img[x:-x, y:-y, z:-z, :].transpose(3, 0, 1, 2)
    # (T, X, Y, Z)

    new_affine = affine.copy()
    new_affine[:3, 3] += new_affine[:3, :3] @ np.array([x, y, z])

    # BUGFIX (found via simulation-based validation this session):
    # PET["dt"] holds per-frame *durations*, not cumulative timestamps --
    # passing it straight through (as this pipeline previously did) gives
    # the PINN/voxelwise fitters a non-monotonic-in-general, physically
    # wrong time axis. dce.analysis.pipeline already cumsum()s DCE["dt"]
    # for exactly this reason; do the same here for consistency and
    # correctness.
    t = np.cumsum(PET["dt"])

    # -----------------------------
    # 2. PINN
    # -----------------------------
    if method == "pinn":

        trainer = Trainer(
            c_p=aif,
            num_of_compartment=num_of_compartment,
            t=t,
            device=device,
            affine=new_affine,
            save_path=path,
            epochs=epochs,
        )

        return trainer.train_ensemble(cropped)

    # -----------------------------
    # 2b. Sine B-PINN ensemble (point estimate + per-voxel uncertainty)
    # -----------------------------
    elif method == "bayesian":
        from core.uncertainty import estimate_with_uncertainty

        tissue_mask = cropped.max(axis=0) > (cropped.max() * 0.05)
        return estimate_with_uncertainty(
            cropped, aif, t, num_of_compartment=num_of_compartment,
            save_path=path, affine=new_affine, device=device,
            n_ensemble=n_ensemble, epochs=epochs, dropout_p=dropout_p,
            tissue_mask=tissue_mask,
        )

    # -----------------------------
    # 3. Classical voxelwise fitting
    # -----------------------------
    elif method == "voxelwise":

        if num_of_compartment == 1:
            maps = calculate_pet_voxelwise_1tcm(cropped, t, aif)
        elif num_of_compartment == 2:
            maps = calculate_pet_voxelwise(cropped, t, aif)
        else:
            raise ValueError("voxelwise PET fitting supports num_of_compartment in {1, 2}")

        save_maps_as_nifti(maps, new_affine, out_dir=path)
        return maps

    else:
        raise ValueError(f"Unknown method {method}")


def run_all_pet(root_path, epochs=1000, device="cpu", method="pinn", n_ensemble=5, dropout_p=0.1):
    """
    Batch executor for PET pipelines -- mirrors dce.analysis.run_all_dce.
    Processes all subjects matching sub*/pet within the root directory.
    n_ensemble, dropout_p are only used when method="bayesian".
    """
    import glob

    print(f"[INFO] Searching for subjects in: {root_path}")
    subject_paths = sorted(glob.glob(os.path.join(root_path, "sub*")))

    if len(subject_paths) == 0:
        print("[WARNING] No subject folders found matching sub*/")
        return

    print(f"[INFO] Found {len(subject_paths)} subject(s).")

    for i, sub_path in enumerate(subject_paths, start=1):
        subject_id = os.path.basename(sub_path)
        pet_path = os.path.join(sub_path, "pet")
        if not os.path.isdir(pet_path):
            print(f"[SKIP] {subject_id}: no pet/ folder")
            continue
        print(f"[{i}/{len(subject_paths)}] Processing {subject_id}")
        try:
            pipeline(pet_path, epochs=epochs, device=device, method=method,
                     n_ensemble=n_ensemble, dropout_p=dropout_p)
        except Exception as e:
            print(f"[ERROR] {subject_id}: {e}")
