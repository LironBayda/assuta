"""
Run the Sine B-PINN uncertainty analysis on REAL DCE-MRI and/or dynamic
PET data (not simulation) -- companion to
simulation/uncertainty_correlation.py, which does the same analysis on
simulated data where ground truth is known.

What it does
------------
1. Loads DCE and/or PET data from preprocessed subject directories (the
   same layout dce.preprocessing / pet.preprocessing already expect).
2. Runs a Sine B-PINN deep ensemble (core.uncertainty) on each modality
   given, producing K_mean, K_uncertainty, and K_uncertainty_demeaned
   maps (see core/uncertainty.py's docstring for what "demeaned" means
   and why it matters).
3. If BOTH --dce-path and --pet-path are given (same subject, and
   assumed already co-registered / same voxel grid -- see the warning
   below): 
     - correlates the DCE and PET K1 uncertainty maps directly (no
       ground truth needed for this part)
     - since real data has no ground-truth K1 to check accuracy against,
       uses DCE-K1 vs PET-K1 correlation as the real-data analog of the
       simulation's "K1 vs ground truth" check, on the full voxel set vs.
       a low-joint-uncertainty subset -- and flags a likely
       range-restriction artifact if the K1 value RANGE collapses in the
       low-uncertainty subset (a real risk if you conclude a stronger
       correlation in "confident" voxels is a bigger clinical effect,
       when it may just be a narrower spread inflating/deflating r).

IMPORTANT ASSUMPTION: DCE and PET volumes must already be co-registered
to the same voxel grid (same shape, same physical space) for the
cross-modal comparisons to be meaningful. This script does NOT perform
registration -- if your DCE and PET aren't already aligned, register
them first (e.g. via registration/pipeline.py in this repo) or the
per-voxel correspondence will be wrong.

Usage
-----
DCE only:
    python run_uncertainty_analysis.py --dce-path /data/sub01/dce

PET only:
    python run_uncertainty_analysis.py --pet-path /data/sub01/pet

Both (paired, co-registered -- enables the cross-modal analysis):
    python run_uncertainty_analysis.py --dce-path /data/sub01/dce --pet-path /data/sub01/pet

Restrict to an ROI/lesion mask (strongly recommended for full real
volumes -- a B-PINN ensemble over an entire real scan can be slow; see
"Performance notes" below):
    python run_uncertainty_analysis.py --dce-path /data/sub01/dce --roi-path /data/sub01/lesion_mask.nii.gz

Performance notes
------------------
Real volumes are typically much larger than the phantoms used for
validation in this repo. A B-PINN ensemble trains `--n-ensemble`
independent full PINN fits, so runtime scales with both voxel count and
ensemble size. Recommended for a first real-data run:
  - pass --roi-path to restrict to a lesion/organ mask rather than the
    whole FOV, or
  - reduce --n-ensemble (3 is usually enough to see the effect; 5+ for a
    more stable uncertainty estimate) and/or --epochs.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_modality(path, modality):
    if modality == "dce":
        from dce.preprocessing import preprocessing
    else:
        from pet.preprocessing import preprocessing
    img, aif, affine = preprocessing(path)   # img: (X, Y, Z, T)
    return img.transpose(3, 0, 1, 2), aif, affine   # -> (T, X, Y, Z)


def load_roi(roi_path, shape_xyz):
    import nibabel as nib
    roi_img = nib.load(roi_path)
    roi = roi_img.get_fdata() > 0
    if roi.shape != shape_xyz:
        raise ValueError(
            f"--roi-path mask shape {roi.shape} doesn't match image shape {shape_xyz} "
            f"-- resample the mask to the image's voxel grid first."
        )
    return roi


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_one_modality(path, modality, out_dir, n_ensemble, epochs, dropout_p, num_of_compartment):
    from core.uncertainty import estimate_with_uncertainty
    from config import DCE, PET

    print(f"[{modality}] loading from {path}")
    img, aif, affine = load_modality(path, modality)

    if modality == "dce":
        t = np.cumsum(np.asarray(DCE["dt"]))
    else:
        # matches the PET["dt"]-is-durations-not-timestamps bugfix found
        # earlier this session (pet/analysis.py) -- cumsum, don't use raw
        t = np.cumsum(np.asarray(PET["dt"]))
    if len(t) != img.shape[0]:
        raise ValueError(
            f"config's {'DCE' if modality=='dce' else 'PET'}['dt'] has {len(t)} frames but the "
            f"loaded image has {img.shape[0]} -- check config.py matches your acquisition protocol."
        )

    tissue_mask = img.max(axis=0) > (img.max() * 0.05)   # crude default: voxels with real signal

    save_path = os.path.join(out_dir, modality)
    os.makedirs(save_path, exist_ok=True)
    print(f"[{modality}] running B-PINN ensemble ({n_ensemble} members, {epochs} epochs) -- this can take a while on a full real volume, see the Performance notes in this script's docstring")
    result = estimate_with_uncertainty(
        img, aif, t, num_of_compartment=num_of_compartment, save_path=save_path,
        n_ensemble=n_ensemble, epochs=epochs, dropout_p=dropout_p, tissue_mask=tissue_mask,
    )
    print(f"[{modality}] saved K_mean.nii, K_uncertainty.nii, K_uncertainty_demeaned.nii to {save_path}")
    return result, tissue_mask, affine


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dce-path", default=None, help="Preprocessed DCE subject directory.")
    parser.add_argument("--pet-path", default=None, help="Preprocessed PET subject directory.")
    parser.add_argument("--roi-path", default=None,
                         help="Optional NIfTI mask (same voxel grid as the image) to restrict analysis to, "
                              "e.g. a lesion or organ segmentation. Strongly recommended for full real volumes.")
    parser.add_argument("--out-dir", default="uncertainty_analysis_output")
    parser.add_argument("--n-ensemble", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    args = parser.parse_args()

    if args.dce_path is None and args.pet_path is None:
        raise SystemExit("Provide at least one of --dce-path / --pet-path")

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}

    dce_result = pet_result = None
    dce_mask = pet_mask = None

    if args.dce_path is not None:
        dce_result, dce_mask, dce_affine = run_one_modality(
            args.dce_path, "dce", args.out_dir, args.n_ensemble, args.epochs, args.dropout_p,
            num_of_compartment=1,
        )
        if args.roi_path is not None:
            dce_mask = load_roi(args.roi_path, dce_result["K_mean"].shape[1:])

    if args.pet_path is not None:
        pet_result, pet_mask, pet_affine = run_one_modality(
            args.pet_path, "pet", args.out_dir, args.n_ensemble, args.epochs, args.dropout_p,
            num_of_compartment=2,
        )
        if args.roi_path is not None:
            pet_mask = load_roi(args.roi_path, pet_result["K_mean"].shape[1:])

    # ---------------- Cross-modal analysis (only if both given) ----------------
    if dce_result is not None and pet_result is not None:
        if dce_result["K_mean"].shape[1:] != pet_result["K_mean"].shape[1:]:
            print("\n[WARNING] DCE and PET volumes have different shapes "
                  f"({dce_result['K_mean'].shape[1:]} vs {pet_result['K_mean'].shape[1:]}) -- "
                  "they must be co-registered to the same voxel grid for the cross-modal "
                  "comparisons below to be meaningful. Skipping cross-modal analysis.")
        else:
            shared_mask = dce_mask & pet_mask

            dce_unc = dce_result["K_uncertainty_demeaned"][0][shared_mask]
            pet_unc = pet_result["K_uncertainty_demeaned"][0][shared_mask]
            r_unc = pearsonr(dce_unc, pet_unc)[0]
            print(f"\ncorr(DCE K1 uncertainty, PET K1 uncertainty), same anatomy: r={r_unc:.3f}")
            summary["dce_pet_uncertainty_correlation"] = float(r_unc)

            dce_K1 = dce_result["K_mean"][0][shared_mask]
            pet_K1 = pet_result["K_mean"][0][shared_mask]
            r_full = pearsonr(dce_K1, pet_K1)[0]
            dce_range_full = dce_K1.max() - dce_K1.min()

            joint_unc = dce_unc + pet_unc   # simple combined "how confident are we here, across both modalities"
            median_unc = np.median(joint_unc)
            low_mask = joint_unc <= median_unc
            r_low = pearsonr(dce_K1[low_mask], pet_K1[low_mask])[0]
            dce_range_low = dce_K1[low_mask].max() - dce_K1[low_mask].min()

            range_shrink_frac = 1 - (dce_range_low / dce_range_full) if dce_range_full > 0 else 0
            likely_artifact = (r_low < r_full) and (range_shrink_frac > 0.3)

            print(f"\nDCE-K1 vs PET-K1 correlation (real-data analog of the simulation's "
                  f"K1-vs-ground-truth check, since no ground truth exists here):")
            print(f"  full set:        n={shared_mask.sum()}  corr={r_full:.3f}  DCE-K1 range={dce_range_full:.3f}")
            print(f"  low-joint-unc:   n={low_mask.sum()}  corr={r_low:.3f}  DCE-K1 range={dce_range_low:.3f} "
                  f"({100*range_shrink_frac:.0f}% narrower)")
            if likely_artifact:
                print("  -> correlation dropped AND the value range collapsed substantially: "
                      "this looks like a statistical range-restriction artifact, not necessarily a real "
                      "accuracy difference. Check absolute agreement (e.g. Bland-Altman) in each subset "
                      "before concluding anything from the correlation numbers alone.")
            elif r_low > r_full:
                print("  -> correlation improved in the low-uncertainty subset without the range collapsing "
                      "much -- more likely a genuine effect of confidence tracking agreement.")
            else:
                print("  -> inconclusive from this check alone -- inspect the full distributions.")

            summary["dce_pet_K1_correlation_full"] = float(r_full)
            summary["dce_pet_K1_correlation_low_uncertainty"] = float(r_low)
            summary["dce_K1_range_full"] = float(dce_range_full)
            summary["dce_K1_range_low_uncertainty"] = float(dce_range_low)
            summary["range_restriction_flag"] = bool(likely_artifact)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
