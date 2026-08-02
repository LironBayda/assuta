import glob
import os

import dce.analysis as dce
import pet.analysis as pet


def run_all(root_path, epochs=1000, device="cpu"):
    """
    FDA-style batch executor for DCE pipelines.
    Processes all subjects matching sub*/dce within the root directory.
    """

    print(f"[INFO] Searching for subjects in: {root_path}")

    # Find all folders like sub001, sub108, sub9, etc.
    subject_paths = sorted(glob.glob(os.path.join(root_path, "sub*")))

    if len(subject_paths) == 0:
        print("[WARNING] No subject folders found matching sub*/")
        return

    print(f"[INFO] Found {len(subject_paths)} subject(s).")

    for i, sub_path in enumerate(subject_paths, start=1): # dce 20
        subject_id = os.path.basename(sub_path)

        dce_path = os.path.join(sub_path, "dce")
        pet_path = os.path.join(sub_path, "pet")

        if not os.path.isdir(dce_path):
            print(f"[SKIP] {subject_id}: No /dce folder found. Skipping.")
            continue

        print(f"\n[INFO] ({i}/{len(subject_paths)}) Processing {subject_id}")
        print(f"[INFO] DCE path: {dce_path}")

        try:
            dce.pipeline(dce_path)
            pet.pipeline(pet_path)

            print(f"[SUCCESS] {subject_id} completed.\n")

        except Exception as e:
            print(f"[ERROR] {subject_id} failed with error:")
            print(f"        {e}")
            print("[ACTION] Continuing to next subject.\n")

    print("[INFO] Batch DCE execution completed.")

if "__main__":
    path = "/home/liron/Documents/dce_pet/lesion/"
    run_all(path)

