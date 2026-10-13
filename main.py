import glob
import os

import numpy as np

import config
from core.DCE_preprocessing import preprocessing as dce_preprocessing
from core.PET_preprocessing import preprocessing as pet_preprocessing
from core.pipleine import pipeline, DEFAULT_OPT_METHOD

try:
    from tqdm import tqdm

    _HAVE_TQDM = True
except ImportError:  # tqdm is an optional progress-bar dependency
    _HAVE_TQDM = False


def run_all(root_path, method=DEFAULT_OPT_METHOD, n_jobs=None):
    """
    Batch executor for DCE (1TCM) + PET (2TCM) voxelwise kinetic pipelines.
    Processes all subjects matching sub*/dce (and sub*/pet, if present)
    within the root directory.

    method : optimizer passed to scipy.optimize.minimize for every voxel
        fit -- see core.pipleine.calculate_voxelwise_1tcm/2tcm.
    n_jobs : CPU parallelism across voxels within each subject's fit.
        Default None auto-detects and uses all CPU threads; pass 1 for
        sequential, or an explicit count -- see core.pipleine._fit_voxels.
    """

    dce_t = np.cumsum(config.DCE["dt"])
    pet_t = np.cumsum(config.PET["dt"]) / 60.0

    print(f"[INFO] Searching for subjects in: {root_path}")

    # Find all folders like sub001, sub108, sub9, etc.
    subject_paths = sorted(glob.glob(os.path.join(root_path, "sub108"))) #   108 46 20 32

    if len(subject_paths) == 0:
        print("[WARNING] No subject folders found matching sub/")
        return

    print(f"[INFO] Found {len(subject_paths)} subject(s).")

    # tqdm.write (instead of print) keeps the per-subject status lines from
    # garbling the progress bar's own redraws when a bar is active.
    log = tqdm.write if _HAVE_TQDM else print
    subjects = tqdm(subject_paths, desc="subjects", unit="subject") if _HAVE_TQDM else subject_paths

    for i, sub_path in enumerate(subjects, start=1): #17 301
        subject_id = os.path.basename(sub_path)

        dce_path = os.path.join(sub_path, "dce")
        pet_path = os.path.join(sub_path, "pet")

        if not os.path.isdir(dce_path):
            log(f"[SKIP] {subject_id}: No /dce folder found. Skipping.")
            continue

        log(f"\n[INFO] ({i}/{len(subject_paths)}) Processing {subject_id}")
        log(f"[INFO] DCE path: {dce_path}")

        try:
            pipeline(dce_path, dce_preprocessing, dce_t, num_of_compartment=1, method=method, n_jobs=n_jobs)

           # pet_preprocessing(pet_path)
            pipeline(pet_path, pet_preprocessing, pet_t, num_of_compartment=1, method=method, n_jobs=n_jobs)

            log(f"[SUCCESS] {subject_id} completed.\n")

        except Exception as e:
            log(f"[ERROR] {subject_id} failed with error:")
            log(f"        {e}")
            log("[ACTION] Continuing to next subject.\n")

    print("[INFO] Batch execution completed.")


if __name__ == "__main__":
    path = "/home/liron/Documents/dce_pet/lesion/"
    run_all(path)

