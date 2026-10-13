#!/usr/bin/env python3
"""
register_lesion_masks_pet_to_dce.py
-------------------------------------

One self-contained script, no command-line arguments -- edit the
CONFIG block below and run with `python register_lesion_masks_pet_to_dce.py`.

Walks a folder tree of the shape:

    <ROOT>/
        sub001/
            pet/
                sub001_pet.nii.gz        <- dynamic PET (4D)
                *lesion*.nii.gz          <- lesion mask, in PET space
            dce/
                <..something..dyn..>.nii.gz   <- dynamic DCE (4D)
        sub002/
            pet/  ...
            dce/  ...
        ...

For every sub* subject folder found directly under ROOT:
  1. Finds that subject's PET file: "<sub>_pet.nii.gz" inside sub*/pet.
  2. Finds that subject's lesion mask: a file whose name contains
     "lesion" (case-insensitive) inside sub*/pet -- the mask is
     assumed to be drawn in PET space.
  3. Finds that subject's reference DCE file inside sub*/dce: the file
     whose name contains "dyn" somewhere in the MIDDLE of the filename
     (real characters on both sides -- so "sub001_dyn_dce.nii.gz"
     matches, but "dyn_sub001.nii.gz" or "sub001_dyn.nii.gz" do not).
  4. Extracts the LAST time frame of both the PET and DCE series
     (closest to steady-state uptake / most contrast-enhanced phase,
     usually the most anatomically informative frame of each).
  5. Registers PET-last-frame (moving) onto DCE-last-frame (fixed)
     using multi-resolution Mattes Mutual Information + regular-step
     gradient descent -- the standard SimpleITK recipe for multimodal
     (PET/MRI) rigid or affine registration.
  6. Resamples the lesion mask into DCE space with NEAREST-NEIGHBOR
     interpolation (mask labels are discrete and must not be blurred
     the way image intensities are).
  7. Writes the result to sub*/dce/<sub>_lesion_in_dce.nii.gz>.

One subject failing (missing/ambiguous file, registration error, etc.)
is printed and skipped; the batch continues with the remaining
subjects rather than aborting the whole run.

Requires: SimpleITK (`pip install SimpleITK`).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import List, Optional

import SimpleITK as sitk


# ======================================================================
# CONFIG -- edit these and run the script, no command-line flags needed
# ======================================================================

ROOT = "lesion"                          # folder containing sub* subject folders
TRANSFORM_TYPE = "affine"                 # "rigid" or "affine"
OUT_SUFFIX = "_lesion_in_dce.nii.gz"     # output filename = "<sub>" + OUT_SUFFIX, written into sub*/dce/
DRY_RUN = False                          # True: only print matched files/output paths, don't register/write anything
VERBOSE_REGISTRATION = False             # True: print per-iteration registration progress for every subject


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

def load_image(path: str) -> sitk.Image:
    """Loads a single image file (any ITK-readable format)."""
    return sitk.ReadImage(str(path))


def extract_last_frame(dynamic_image: sitk.Image) -> sitk.Image:
    """
    Extracts the last 3D time frame from a 4D dynamic series.

    Works whether SimpleITK loaded it as a genuine 4D image (size along
    dim 3 > 1) or as a 3D image (already a single frame) -- in the
    latter case it's returned as-is.
    """
    size = dynamic_image.GetSize()
    if dynamic_image.GetDimension() == 3:
        return dynamic_image

    if dynamic_image.GetDimension() != 4:
        raise ValueError(
            f"Expected a 3D or 4D image, got {dynamic_image.GetDimension()}D with size {size}."
        )

    num_frames = size[3]
    last_index = num_frames - 1

    extractor = sitk.ExtractImageFilter()
    # Extract size of 0 along the time axis collapses that dimension,
    # producing a 3D output (standard SimpleITK idiom for 4D->3D slicing).
    extractor.SetSize([size[0], size[1], size[2], 0])
    extractor.SetIndex([0, 0, 0, last_index])
    return extractor.Execute(dynamic_image)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------

def build_initial_transform(fixed: sitk.Image, moving: sitk.Image, transform_type: str) -> sitk.Transform:
    """
    Geometric (centered) alignment of the two images' centers of mass,
    used as the optimizer's starting point -- gets it into a
    reasonable basin instead of searching from a naive identity
    transform, which matters since PET and DCE volumes often differ in
    field of view and patient positioning.
    """
    if transform_type == "rigid":
        transform_template = sitk.Euler3DTransform()
    elif transform_type == "affine":
        transform_template = sitk.AffineTransform(3)
    else:
        raise ValueError(f"transform_type must be 'rigid' or 'affine', got {transform_type!r}")

    return sitk.CenteredTransformInitializer(
        fixed, moving, transform_template,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )


def register_last_frames(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform_type: str = "rigid",
    verbose: bool = True,
) -> sitk.Transform:
    """
    Registers `moving` (PET last frame) onto `fixed` (DCE last frame)
    using multi-resolution Mattes Mutual Information + regular-step
    gradient descent. Mattes MI is the standard similarity metric for
    multimodal registration -- unlike mean-squares or correlation, it
    doesn't assume any fixed relationship between PET uptake and
    DCE-MRI contrast-enhancement intensities.

    The returned transform maps points in fixed (DCE) space to moving
    (PET) space, i.e. it can be plugged directly into
    sitk.Resample(..., transform, ...) with `fixed` as the reference
    grid to resample anything defined in PET space (like the lesion
    mask) into DCE space.
    """
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    initial_transform = build_initial_transform(fixed_f, moving_f, transform_type)

    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.20)

    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=300,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid: coarse-to-fine, standard practice to
    # avoid local optima and speed up convergence.
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(initial_transform, inPlace=False)

    if verbose:
        def _print_progress(reg):
            print(f"    iteration {reg.GetOptimizerIteration():3d}  metric = {reg.GetMetricValue():.6f}")
        registration.AddCommand(sitk.sitkIterationEvent, lambda: _print_progress(registration))

    final_transform = registration.Execute(fixed_f, moving_f)

    if verbose:
        print(
            f"    registration finished: {registration.GetOptimizerStopConditionDescription()} "
            f"(final metric = {registration.GetMetricValue():.6f})"
        )

    return final_transform


def transform_mask(mask: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    """
    Resamples `mask` (PET-space labels) onto `reference`'s grid (DCE
    space) using `transform`. NEAREST-NEIGHBOR interpolation, not
    linear/BSpline -- mask values are discrete labels, and smoothing
    interpolation would create meaningless fractional values at
    boundaries.
    """
    return sitk.Resample(
        mask, reference, transform, sitk.sitkNearestNeighbor, 0, mask.GetPixelID(),
    )


# ----------------------------------------------------------------------
# File-matching helpers
# ----------------------------------------------------------------------

NIFTI_GLOB_PATTERNS = ("*.nii.gz", "*.nii")


def _all_image_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in NIFTI_GLOB_PATTERNS:
        files.extend(sorted(folder.glob(pattern)))
    return files


def _contains_substring_in_middle(filename_stem: str, substring: str) -> bool:
    """
    True if `substring` appears in `filename_stem` with at least one
    character before it and at least one character after it (i.e.
    genuinely in the "middle" of the name, not merely a prefix or a
    suffix). Case-insensitive.

    Examples (substring="dyn"):
        "sub001_dyn_dce"   -> True
        "dyn_sub001"       -> False  (nothing before "dyn")
        "sub001_dyn"       -> False  (nothing after "dyn")
        "dyn"              -> False  (nothing before or after)
    """
    lower = filename_stem.lower()
    sub = substring.lower()
    idx = lower.find(sub)
    if idx <= 0:
        return False
    end = idx + len(sub)
    if end >= len(lower):
        return False
    return True


def _stem_without_all_nifti_suffixes(path: Path) -> str:
    """'sub001_dyn_dce.nii.gz' -> 'sub001_dyn_dce' (strips .nii.gz or .nii)."""
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def find_pet_file(pet_dir: Path, sub_name: str) -> Path:
    """
    Expects a file literally named '<sub_name>_pet.nii.gz' (or .nii) in
    pet_dir. Falls back to any single '*_pet.nii*' match if the exact
    name isn't found.
    """
    exact_candidates = [pet_dir / f"{sub_name}_pet.nii.gz", pet_dir / f"{sub_name}_pet.nii"]
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate

    fallback = sorted(pet_dir.glob("*_pet.nii.gz")) + sorted(pet_dir.glob("*_pet.nii"))
    if len(fallback) == 1:
        print(f"  [warn] expected {sub_name}_pet.nii.gz, using fallback match: {fallback[0].name}")
        return fallback[0]
    if len(fallback) > 1:
        raise FileNotFoundError(
            f"No exact match for {sub_name}_pet.nii(.gz) in {pet_dir}, and multiple "
            f"'*_pet.nii*' fallback candidates found: {[f.name for f in fallback]}. Ambiguous."
        )
    raise FileNotFoundError(f"No PET file found in {pet_dir} (expected {sub_name}_pet.nii.gz).")


def find_lesion_mask_file(pet_dir: Path) -> Path:
    """Any file in pet_dir whose name contains 'lesion' (case-insensitive)."""
    candidates = [f for f in _all_image_files(pet_dir) if "lesion" in f.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"No file containing 'lesion' found in {pet_dir}.")
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Multiple files containing 'lesion' found in {pet_dir}: {[f.name for f in candidates]}. Ambiguous."
        )
    return candidates[0]


def find_dyn_dce_file(dce_dir: Path) -> Path:
    """Any file in dce_dir with 'dyn' in the middle of its filename (see
    _contains_substring_in_middle)."""
    candidates = [
        f for f in _all_image_files(dce_dir)
        if _contains_substring_in_middle(_stem_without_all_nifti_suffixes(f), "dyn")
    ]
    if not candidates:
        raise FileNotFoundError(f"No file with 'dyn' in the middle of its filename found in {dce_dir}.")
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Multiple candidate dynamic-DCE files found in {dce_dir}: {[f.name for f in candidates]}. Ambiguous."
        )
    return candidates[0]


# ----------------------------------------------------------------------
# Per-subject processing
# ----------------------------------------------------------------------

def process_subject(sub_dir: Path) -> Optional[Path]:
    sub_name = sub_dir.name
    pet_dir = sub_dir / "pet"
    dce_dir = sub_dir / "dce"

    if not pet_dir.is_dir():
        raise FileNotFoundError(f"Missing pet/ subfolder: {pet_dir}")
    if not dce_dir.is_dir():
        raise FileNotFoundError(f"Missing dce/ subfolder: {dce_dir}")

    pet_path = find_pet_file(pet_dir, sub_name)
    mask_path = find_lesion_mask_file(pet_dir)
    dce_path = find_dyn_dce_file(dce_dir)

    print(f"  PET:  {pet_path.relative_to(sub_dir)}")
    print(f"  mask: {mask_path.relative_to(sub_dir)}")
    print(f"  DCE:  {dce_path.relative_to(sub_dir)}")

    out_path = dce_dir / f"{sub_name}{OUT_SUFFIX}"

    if DRY_RUN:
        print(f"  [dry-run] would write: {out_path.relative_to(sub_dir)}")
        return None

    dyn_pet = load_image(str(pet_path))
    pet_last_frame = extract_last_frame(dyn_pet)

    dyn_dce = load_image(str(dce_path))
    dce_last_frame = extract_last_frame(dyn_dce)

    transform = register_last_frames(
        fixed=dce_last_frame,
        moving=pet_last_frame,
        transform_type=TRANSFORM_TYPE,
        verbose=VERBOSE_REGISTRATION,
    )

    mask = load_image(str(mask_path))
    mask_in_dce_space = transform_mask(mask, reference=dce_last_frame, transform=transform)

    sitk.WriteImage(mask_in_dce_space, str(out_path))
    print(f"  -> wrote {out_path.relative_to(sub_dir)}")
    return out_path


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    root = "/home/liron/Documents/dce_pet"

    if not root.is_dir():
        print(f"Root folder not found: {root}", file=sys.stderr)
        return 1

    sub_dirs = sorted(p for p in root.glob("sub*") if p.is_dir())
    if not sub_dirs:
        print(f"No sub* subject folders found directly under {root}.", file=sys.stderr)
        return 1

    print(f"Found {len(sub_dirs)} subject folder(s) under {root}.")

    succeeded, failed = [], []
    for sub_dir in sub_dirs:
        print(f"\n=== {sub_dir.name} ===")
        try:
            process_subject(sub_dir)
            succeeded.append(sub_dir.name)
        except Exception as exc:
            print(f"  [ERROR] {sub_dir.name}: {exc}")
            traceback.print_exc()
            failed.append(sub_dir.name)

    print(f"\nDone. {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed subjects: {failed}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())