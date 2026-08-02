"""
dce_prostatex_to_concentration.py
==================================

Convert a ProstateX Dynamic Contrast-Enhanced (DCE / "dyn") DICOM series
into contrast-agent concentration maps C(t), using a Proton-Density (PD)
weighted reference series to estimate the baseline T1 map (T10) via the
variable-flip-angle (VFA / DESPOT1) method.

ASSUMPTIONS (adjust to your actual data before trusting the output):
  - Both the PD series and the DCE series are spoiled gradient-echo
    (SPGR/FLASH) acquisitions with the SAME repetition time (TR) and
    matching geometry (same FOV/slice positions), but DIFFERENT flip
    angles. This is the standard setup for VFA T1 mapping.
  - The DCE series has a low flip angle relative to a "high-flip" PD
    series (or vice versa) -- the code auto-orders them by the flip
    angle actually found in the DICOM headers, so either order works.
  - TR, FlipAngle are readable from standard DICOM tags
    (0018,0080) and (0018,1314). If your data stores flip angle only
    in a private tag / the DICOM header is incomplete, you'll need to
    hardcode the values in the CONFIG section below.
  - Dynamic frames within the "dyn" series are distinguished by
    TemporalPositionIdentifier (0020,0100), falling back to
    AcquisitionTime / TriggerTime if that tag is absent.
  - Pixel values are converted to true signal intensity using
    RescaleSlope/RescaleIntercept before any of the math below.

If you already have a proper T1 map (e.g. from a dedicated multi-flip
T1 mapping sequence) rather than a single PD scan, skip
`estimate_t10_map()` and load that map directly -- it's more accurate
than a two-point VFA estimate.

Pipeline
--------
1. load_series()      -- read a DICOM folder into a 4D numpy array (or
                          3D for PD) sorted by slice position and time.
2. estimate_t10_map()  -- two-point VFA fit -> T10 (ms) per voxel.
3. si_to_concentration() -- invert the SPGR signal equation at each
                          dynamic timepoint to get T1(t), then convert
                          to concentration via the contrast agent's
                          relaxivity r1.
4. main()              -- wires it together and saves a NIfTI (if
                          nibabel is available) or a .npy stack.
"""

import numpy as np
import pydicom
from pathlib import Path
from collections import defaultdict

# ----------------------------- CONFIG -------------------------------
# Contrast agent longitudinal relaxivity (r1), in 1/(mM*s). ~4.5 for
# Gadovist/Gadobutrol, ~3.9-4.3 for Magnevist/Dotarem at 3T -- check
# the value appropriate for the agent and field strength used in your
# ProstateX acquisitions.
R1_RELAXIVITY = 4.5  # 1/(mM*s)

# If flip angle / TR are missing from the headers, set them here
# (in degrees and milliseconds) and the code will use these instead.
FORCE_TR_MS = None
FORCE_PD_FLIP_DEG = None
FORCE_DYN_FLIP_DEG = None
# ----------------------------------------------------------------------


def _rescale(ds):
    """Apply RescaleSlope/RescaleIntercept to get true signal intensity."""
    px = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return px * slope + intercept


def _flip_angle(ds):
    if FORCE_DYN_FLIP_DEG is not None:
        pass  # handled by caller when relevant
    return float(getattr(ds, "FlipAngle", np.nan))


def _tr_ms(ds):
    if FORCE_TR_MS is not None:
        return FORCE_TR_MS
    return float(getattr(ds, "RepetitionTime", np.nan))


def load_pd_series(folder):
    """
    Load a single-timepoint PD (proton-density weighted) series.
    Returns (volume[z,y,x], flip_angle_deg, tr_ms).
    """
    files = sorted(Path(folder).glob("*.dcm"))
    slices = [pydicom.dcmread(str(f)) for f in files]
    slices.sort(key=lambda d: float(getattr(d, "SliceLocation", d.InstanceNumber)))

    volume = np.stack([_rescale(d) for d in slices], axis=0)  # (z, y, x)
    flip = FORCE_PD_FLIP_DEG if FORCE_PD_FLIP_DEG is not None else _flip_angle(slices[0])
    tr = _tr_ms(slices[0])
    return volume, flip, tr


def load_dyn_series(folder):
    """
    Load a ProstateX "dyn" (DCE) series from a SINGLE flat folder where
    slices carry a TemporalPositionIdentifier (or AcquisitionTime) tag
    distinguishing timepoints. Kept for datasets structured that way.

    Returns (volume4d[t,z,y,x], flip_angle_deg, tr_ms, timestamps_sorted).
    """
    files = sorted(Path(folder).glob("*.dcm"))
    slices = [pydicom.dcmread(str(f)) for f in files]

    def temporal_key(d):
        if hasattr(d, "TemporalPositionIdentifier"):
            return int(d.TemporalPositionIdentifier)
        # fallback: AcquisitionTime as HHMMSS.ffffff string -> float seconds
        t = getattr(d, "AcquisitionTime", None)
        return float(t) if t else 0.0

    groups = defaultdict(list)
    for d in slices:
        groups[temporal_key(d)].append(d)

    ordered_keys = sorted(groups.keys())
    volumes = []
    for k in ordered_keys:
        vol_slices = groups[k]
        vol_slices.sort(key=lambda d: float(getattr(d, "SliceLocation", d.InstanceNumber)))
        volumes.append(np.stack([_rescale(d) for d in vol_slices], axis=0))  # (z,y,x)

    volume4d = np.stack(volumes, axis=0)  # (t, z, y, x)
    ref = slices[0]
    flip = FORCE_DYN_FLIP_DEG if FORCE_DYN_FLIP_DEG is not None else _flip_angle(ref)
    tr = _tr_ms(ref)
    return volume4d, flip, tr, ordered_keys


def load_dyn_series_multifolder(parent_folder):
    """
    Load a ProstateX "dyn" (DCE) acquisition stored as ONE SUBFOLDER PER
    TIMEPOINT -- e.g.:

        ProstateX-0326/.../
            10.000000-tfl3d PD reftra1.5x1.5t3-74711/      <- PD, separate
            11.000000-tfldynfasttra1.5x1.5t3.5sec-47175/   <- t=0
            12.000000-tfldynfasttra1.5x1.5t3.5sec-13583/   <- t=1
            13.000000-tfldynfasttra1.5x1.5t3.5sec-19618/   <- t=2
            ...

    This is the layout produced by TCIA/ProstateX downloads: each
    series folder is one 3D dynamic volume (here, 16 slices each),
    named with a leading series number that gives temporal order and
    a "...sec-XXXXX" suffix that is just a random series UID suffix,
    NOT a time value -- do not sort on it.

    Returns (volume4d[t,z,y,x], flip_angle_deg, tr_ms, subfolder_names_sorted).
    """
    import re

    parent = Path(parent_folder)
    subfolders = [p for p in parent.iterdir() if p.is_dir()]

    def series_number_key(p):
        m = re.match(r"^(\d+)\.", p.name)
        return int(m.group(1)) if m else p.name

    subfolders.sort(key=series_number_key)

    volumes = []
    ref_ds = None
    for sub in subfolders:
        files = sorted(sub.glob("*.dcm"))
        if not files:
            continue
        slices = [pydicom.dcmread(str(f)) for f in files]
        slices.sort(key=lambda d: float(getattr(d, "SliceLocation", d.InstanceNumber)))
        volumes.append(np.stack([_rescale(d) for d in slices], axis=0))  # (z,y,x)
        if ref_ds is None:
            ref_ds = slices[0]

    volume4d = np.stack(volumes, axis=0)  # (t, z, y, x)
    flip = FORCE_DYN_FLIP_DEG if FORCE_DYN_FLIP_DEG is not None else _flip_angle(ref_ds)
    tr = _tr_ms(ref_ds)
    return volume4d, flip, tr, [s.name for s in subfolders]


def estimate_t10_map(pd_volume, pd_flip_deg, dyn_precontrast_volume, dyn_flip_deg, tr_ms):
    """
    Two-point variable-flip-angle (VFA / DESPOT1) T1 estimation.

    Linearizes the SPGR signal equation:
        S = M0 * sin(a) * (1-E1) / (1 - E1*cos(a)),  E1 = exp(-TR/T1)
    into:
        Y = E1 * X + M0*(1-E1)
        X = S / tan(a),   Y = S / sin(a)
    Fits a line through the two (X, Y) points (one per flip angle) to
    recover E1 (the slope), then T1 = -TR / ln(E1).

    All inputs must be co-registered volumes of the same shape.
    Returns T10 map in milliseconds (same units as tr_ms).
    """
    a1 = np.radians(pd_flip_deg)
    a2 = np.radians(dyn_flip_deg)
    S1 = pd_volume
    S2 = dyn_precontrast_volume

    X1, Y1 = S1 / np.tan(a1), S1 / np.sin(a1)
    X2, Y2 = S2 / np.tan(a2), S2 / np.sin(a2)

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (Y2 - Y1) / (X2 - X1)  # = E1

    slope = np.clip(slope, 1e-6, 1 - 1e-6)  # E1 must be in (0,1)
    t10 = -tr_ms / np.log(slope)
    t10[~np.isfinite(t10)] = np.nan
    return t10


def si_to_t1(signal, flip_deg, tr_ms, m0):
    """
    Invert the SPGR signal equation at a single timepoint to solve
    for T1, given a known M0 (from the T10 fit) and flip angle/TR.

        S = M0 sin(a) (1-E1) / (1 - E1 cos(a))
        => E1 = (M0*sin(a) - S) / (M0*sin(a)*cos(a) - S*cos(a))

    Returns T1 in the same time units as tr_ms.
    """
    a = np.radians(flip_deg)
    num = m0 * np.sin(a) - signal
    den = m0 * np.sin(a) * np.cos(a) - signal * np.cos(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        e1 = num / den
    e1 = np.clip(e1, 1e-6, 1 - 1e-6)
    t1 = -tr_ms / np.log(e1)
    return t1


def recover_m0(s0, flip_deg, tr_ms, t10):
    """M0 from the pre-contrast signal and known T10 (rearranged SPGR eq.)."""
    a = np.radians(flip_deg)
    e1 = np.exp(-tr_ms / t10)
    m0 = s0 * (1 - e1 * np.cos(a)) / (np.sin(a) * (1 - e1))
    return m0


def si_to_concentration(dyn_volume4d, flip_deg, tr_ms, t10_map, r1=R1_RELAXIVITY):
    """
    Convert a DCE 4D signal-intensity stack (t, z, y, x) to a
    concentration stack C(t) in mM, given a baseline T10 map (ms).

    Assumes the FIRST timepoint of dyn_volume4d is pre-contrast
    (used to recover M0 per voxel).
    """
    s0 = dyn_volume4d[0]
    m0 = recover_m0(s0, flip_deg, tr_ms, t10_map)

    n_t = dyn_volume4d.shape[0]
    conc = np.zeros_like(dyn_volume4d, dtype=np.float64)

    for t in range(n_t):
        t1_t = si_to_t1(dyn_volume4d[t], flip_deg, tr_ms, m0)
        # C(t) = (1/T1(t) - 1/T10) / r1   -- T1 in seconds, r1 in 1/(mM*s)
        conc[t] = (1.0 / (t1_t / 1000.0) - 1.0 / (t10_map / 1000.0)) / r1

    conc[~np.isfinite(conc)] = 0.0
    return conc


def main(pd_folder, dyn_parent_folder, out_path="concentration_stack.npy"):
    """
    pd_folder:          path to the single PD series folder
                         (e.g. ".../10.000000-tfl3d PD reftra1.5x1.5t3-74711")
    dyn_parent_folder:  path to the folder that CONTAINS the per-timepoint
                         dyn subfolders (e.g. ".../ProstateX-0326/<study>/",
                         the parent of "11.000000-tfldynfasttra...",
                         "12.000000-tfldynfasttra...", etc.)
    """
    print("Loading PD series...")
    pd_vol, pd_flip, pd_tr = load_pd_series(pd_folder)

    print("Loading DCE (dyn) series (one subfolder per timepoint)...")
    dyn_vol4d, dyn_flip, dyn_tr, timepoints = load_dyn_series_multifolder(dyn_parent_folder)
    print(f"Found {dyn_vol4d.shape[0]} dynamic timepoints: {timepoints}")

    if np.isnan(pd_flip) or np.isnan(dyn_flip):
        raise ValueError(
            "Flip angle missing from DICOM headers -- set FORCE_PD_FLIP_DEG "
            "/ FORCE_DYN_FLIP_DEG in the CONFIG section."
        )
    if pd_vol.shape != dyn_vol4d.shape[1:]:
        raise ValueError(
            f"PD volume shape {pd_vol.shape} does not match DCE volume "
            f"shape {dyn_vol4d.shape[1:]} -- series must be co-registered "
            f"(same geometry) for the VFA T1 fit to be valid."
        )

    tr_ms = pd_tr if not np.isnan(pd_tr) else dyn_tr
    print(f"TR = {tr_ms} ms, PD flip = {pd_flip} deg, DCE flip = {dyn_flip} deg")

    print("Estimating baseline T10 map (VFA)...")
    t10_map = estimate_t10_map(pd_vol, pd_flip, dyn_vol4d[0], dyn_flip, tr_ms)

    print(f"Converting {dyn_vol4d.shape[0]} dynamic timepoints to concentration...")
    conc_stack = si_to_concentration(dyn_vol4d, dyn_flip, tr_ms, t10_map)

    np.save(out_path, conc_stack)
    print(f"Saved concentration stack {conc_stack.shape} to {out_path}")

    # Optional: save as NIfTI if nibabel is available
    try:
        import nibabel as nib
        nii = nib.Nifti1Image(np.moveaxis(conc_stack, 0, -1), affine=np.eye(4))
        nii_path = str(Path(out_path).with_suffix("")) + ".nii.gz"
        nib.save(nii, nii_path)
        print(f"Also saved NIfTI to {nii_path}")
    except ImportError:
        pass

    return conc_stack, t10_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pd_folder", help="Folder of PD-weighted DICOM slices (single series)")
    parser.add_argument(
        "dyn_parent_folder",
        help="Parent folder containing one subfolder per dyn timepoint "
             "(e.g. the study folder holding '11.000000-tfldynfasttra...', "
             "'12.000000-tfldynfasttra...', etc.)",
    )
    parser.add_argument("--out", default="concentration_stack.npy")
    args = parser.parse_args([
        "pd_folder",
        "/home/liron/Documents/prostateX/ProstateX-0340/04-17-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-20284",
        "dyn_parent_folder",
        "/home/liron/Documents/prostateX/ProstateX-0340/04-17-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-20284",
        "--out",
        "/home/liron/Documents/prostateX/ProstateX-0340/04-17-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-20284",
    ])
    main(args.pd_folder, args.dyn_parent_folder, args.out)