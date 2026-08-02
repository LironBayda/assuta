"""
Generic circular phantom + 1TCM image simulator.

Unlike simulation/phantom.py (a prostate-shaped ellipsoid with a lesion
constrained inside it, used for the PINN/voxelwise validation), this is
a simpler, more generic phantom: a centered "big circle" (organ/region
of interest) containing a "small circle" (lesion) of random size and
position -- for quick, non-anatomy-specific 1TCM simulation experiments.

Label convention: 0 = background, 1 = big circle, 2 = small circle.
"""
from dataclasses import dataclass

import numpy as np

from simulation.forward_models import simulate_1tcm_volume, parker_aif


@dataclass
class CirclePhantomInfo:
    label: np.ndarray            # (X, Y, Z) uint8, values in {0, 1, 2}
    big_center: tuple
    big_radius: float
    small_center: tuple
    small_radius: float


def build_circle_phantom(shape=(64, 64, 20), big_radius_frac=0.35,
                          small_radius_frac_range=(0.15, 0.45), seed=None):
    """
    Build a (X, Y, Z) label volume with:
      - a centered "big circle" (sphere, radius = big_radius_frac * min(X,Y,Z)/2)
      - a "small circle" (sphere) of random size (a random fraction, in
        small_radius_frac_range, of the big circle's radius) and random
        position, fully contained within the big circle.
    """
    rng = np.random.default_rng(seed)
    X, Y, Z = shape
    xx, yy, zz = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")

    cx, cy, cz = (X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0
    big_r = big_radius_frac * min(X, Y, Z)

    big_sphere = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= big_r ** 2

    label = np.zeros(shape, dtype=np.uint8)
    label[big_sphere] = 1

    # random center for the small circle, fully inside the big one
    big_idx = np.argwhere(big_sphere)
    center_vox = big_idx[rng.integers(len(big_idx))].astype(float)
    frac = rng.uniform(*small_radius_frac_range)
    small_r = max(big_r * frac, 1.0)

    # keep the small circle's own volume inside the big one: cap how far
    # its center can be from the big circle's center
    max_offset = max(big_r - small_r, 0.0)
    direction = rng.normal(size=3)
    direction /= (np.linalg.norm(direction) + 1e-8)
    offset = rng.uniform(0.0, max_offset)
    small_center = np.array([cx, cy, cz]) + direction * offset
    scx, scy, scz = small_center

    small_sphere = (
        (xx - scx) ** 2 + (yy - scy) ** 2 + (zz - scz) ** 2
    ) <= small_r ** 2
    small_sphere &= big_sphere

    label[small_sphere] = 2

    return CirclePhantomInfo(
        label=label,
        big_center=(cx, cy, cz), big_radius=big_r,
        small_center=(scx, scy, scz), small_radius=small_r,
    )


def simulate_1tcm_image(shape=(64, 64, 20), k1_big=0.15, k2_big=0.35,
                         k1_small=0.45, k2_small=0.9, k_std_frac=0.15,
                         n_points=35, dt_sec=9.5, noise_std=0.03, seed=None):
    """
    Convenience wrapper: build a circle phantom, sample per-voxel K1/k2
    (Gaussian around the given big/small circle means, std =
    k_std_frac * mean), forward-simulate a 1TCM (X,Y,Z,T) image with a
    Parker AIF, and return everything needed to validate an estimation
    method against ground truth.

    Returns dict with keys: label, K1_map, k2_map, image (T,X,Y,Z),
    clean (T,X,Y,Z, noise-free), t, aif, phantom_info.
    """
    rng = np.random.default_rng(seed)
    info = build_circle_phantom(shape, seed=seed)
    label = info.label

    K1_map = np.zeros(shape, dtype=np.float32)
    k2_map = np.zeros(shape, dtype=np.float32)
    for cls_val, (k1_mean, k2_mean) in [(1, (k1_big, k2_big)), (2, (k1_small, k2_small))]:
        mask = label == cls_val
        n = int(mask.sum())
        if n == 0:
            continue
        K1_map[mask] = np.clip(rng.normal(k1_mean, k1_mean * k_std_frac, size=n), 1e-4, None)
        k2_map[mask] = np.clip(rng.normal(k2_mean, k2_mean * k_std_frac, size=n), 1e-4, None)

    dt_min = dt_sec / 60.0
    t = np.arange(n_points) * dt_min
    aif = parker_aif(t)

    noisy, clean = simulate_1tcm_volume(t, aif, K1_map, k2_map, noise_std=noise_std, rng=seed)

    return {
        "label": label, "K1_map": K1_map, "k2_map": k2_map,
        "image": noisy, "clean": clean, "t": t, "aif": aif,
        "phantom_info": info,
    }
