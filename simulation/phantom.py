"""
Synthetic prostate phantom generator.

Label convention:
    0 = background
    1 = normal prostate tissue
    2 = cancer lesion (random size / location, constrained to lie inside
        the prostate)

The phantom is a simple geometric stand-in (ellipsoid prostate, smaller
ellipsoid lesion) -- it is not meant to be anatomically realistic, only to
give three distinct, spatially contiguous tissue classes with known extent
so that voxel-wise parameter-recovery methods (voxelwise NLLS / PINN / VAE)
can be scored against a known ground truth.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PhantomInfo:
    label: np.ndarray            # (X, Y, Z) uint8, values in {0, 1, 2}
    prostate_center: tuple
    cancer_center: tuple
    cancer_radii: tuple          # (rx, ry, rz) of the lesion ellipsoid


def build_prostate_phantom(shape=(64, 64, 20), seed=None) -> PhantomInfo:
    """
    Build a (X, Y, Z) label volume with a centered ellipsoidal "prostate"
    occupying the middle axial slices, and a smaller ellipsoidal "cancer"
    lesion of random size and position fully contained within it.
    """
    rng = np.random.default_rng(seed)
    X, Y, Z = shape

    xx, yy, zz = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")

    # --- Prostate: centered ellipsoid spanning the middle ~60% of slices ---
    cx, cy = (X - 1) / 2.0, (Y - 1) / 2.0
    z_lo, z_hi = int(round(Z * 0.2)), int(round(Z * 0.8))
    cz = (z_lo + z_hi) / 2.0
    rx, ry, rz = X * 0.27, Y * 0.22, max((z_hi - z_lo) / 2.0, 1.0)

    prostate_ellipsoid = (
        ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2
    ) <= 1.0

    label = np.zeros(shape, dtype=np.uint8)
    label[prostate_ellipsoid] = 1

    # --- Cancer: random center inside the prostate, random (smaller) size ---
    prostate_idx = np.argwhere(prostate_ellipsoid)
    center_vox = prostate_idx[rng.integers(len(prostate_idx))]
    ccx, ccy, ccz = center_vox.astype(float)

    # size as a random fraction of the prostate's own radii, biased small
    # so the lesion is a plausible focal finding rather than half the gland
    frac = rng.uniform(0.15, 0.45)
    crx = max(rx * frac * rng.uniform(0.8, 1.2), 1.5)
    cry = max(ry * frac * rng.uniform(0.8, 1.2), 1.5)
    crz = max(rz * frac * rng.uniform(0.8, 1.2), 1.0)

    cancer_ellipsoid = (
        ((xx - ccx) / crx) ** 2 + ((yy - ccy) / cry) ** 2 + ((zz - ccz) / crz) ** 2
    ) <= 1.0
    cancer_ellipsoid &= prostate_ellipsoid  # stay inside the gland

    label[cancer_ellipsoid] = 2

    return PhantomInfo(
        label=label,
        prostate_center=(cx, cy, cz),
        cancer_center=(ccx, ccy, ccz),
        cancer_radii=(crx, cry, crz),
    )
