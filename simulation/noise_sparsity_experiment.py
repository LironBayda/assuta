"""
Noise and temporal-sparsity sweep: voxelwise NLLS vs tanh-PINN vs
Sine-PINN vs Sine-B-PINN (deep ensemble with uncertainty), for both DCE
(1TCM, Rician+Gaussian noise) and PET (2TCM, Poisson+Gaussian noise).

This is meant to be edited and re-run, not just executed once -- the
grids (NOISE_LEVELS, SPARSITY_LEVELS), phantom size, epoch count, and
ensemble size are all constants at the top for exactly that. Full runs
(1000 epochs, n_ensemble=5, all four methods x both axes x both
modalities) are compute-heavy -- see "Performance notes" below before
scaling up.

Usage
-----
    python simulation/noise_sparsity_experiment.py                # full grid, both axes, both modalities
    python simulation/noise_sparsity_experiment.py --axis noise --modality dce
    python simulation/noise_sparsity_experiment.py --methods voxelwise pinn_sine --epochs 300 --n-ensemble 2

Output: simulation_data/validation_runs/noise_sparsity/results.json
(one row per modality x axis x level x method), plus a printed table.

Performance notes
------------------
- Bayesian (Sine-B-PINN) is ~n_ensemble x the cost of a single PINN fit.
  At 1000 epochs / n_ensemble=5 / a 32x32x8 phantom, ONE bayesian data
  point took roughly 3-4 minutes on CPU in earlier testing this session
  -- a full 4-method x 2-axis x N-levels x 2-modality grid at that scale
  is realistically a multi-hour run, not something to fire off and wait
  on interactively. Start with --epochs 300 --n-ensemble 2 --shape 24 24 6
  to sanity check the whole pipeline runs end-to-end in a few minutes,
  then scale up.
- voxelwise and plain PINN methods are much cheaper (seconds to ~1 min
  each at the same phantom size) -- fine to run at full settings even
  while you're still tuning the bayesian settings.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.phantom import build_prostate_phantom
from simulation.kinetics_literature import DCE_1TCM_PARAMS, PET_2TCM_PARAMS, sample_param_maps
from simulation.forward_models import (
    parker_aif, feng_input_function, simulate_1tcm_volume, simulate_2tcm_volume,
)
from dce.analysis import calculate_dce_voxelwise
from simulation.voxelwise_pet import calculate_pet_voxelwise
from core.train import Trainer
from core.uncertainty import estimate_with_uncertainty

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "simulation_data", "validation_runs", "noise_sparsity",
)
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- grids (edit these) --------------------
NOISE_LEVELS = {"low": 30.0, "moderate": 15.0, "high": 5.0}     # SNR values, lower = noisier
SPARSITY_LEVELS = {
    "fine": 0.05,       # dt (minutes) -- fine temporal sampling
    "moderate": 0.1522,  # this repo's default DCE frame duration
    "coarse": 0.5,       # coarse -- likely misses the AIF peak
}
METHODS = ["voxelwise", "tanh_pinn", "sine_pinn", "sine_bpinn"]
DEFAULT_SHAPE = (32, 32, 8)


def build_dce_data(shape, seed, snr, dt):
    info = build_prostate_phantom(shape, seed=seed)
    label = info.label
    gt = sample_param_maps(label, DCE_1TCM_PARAMS, ["K1", "k2"], seed=seed)
    t = np.arange(0, 5.3, dt)
    aif = parker_aif(t)
    img, _ = simulate_1tcm_volume(t, aif, gt["K1"], gt["k2"], noise_model="dce", snr=snr, rng=seed)
    return img, t, aif, label, gt, 1


def build_pet_data(shape, seed, snr, dt_sec):
    info = build_prostate_phantom(shape, seed=seed)
    label = info.label
    gt = sample_param_maps(label, PET_2TCM_PARAMS, ["K1", "k2", "k3"], seed=seed)
    n_frames = max(int(1200 / dt_sec), 5)   # ~20-minute scan, varying frame count
    t = np.arange(n_frames) * (dt_sec / 60.0)
    aif = feng_input_function(t)
    img, _ = simulate_2tcm_volume(t, aif, gt["K1"], gt["k2"], gt["k3"], noise_model="pet", snr=snr, rng=seed)
    return img, t, aif, label, gt, 2


def run_method(method, img, t, aif, num_of_compartment, epochs, n_ensemble, dropout_p, mask):
    """Returns (K1_map, extra_info_dict)."""
    if method == "voxelwise":
        if num_of_compartment == 1:
            maps = calculate_dce_voxelwise(img, t, aif, mask=mask)
        else:
            maps = calculate_pet_voxelwise(img, t, aif, mask=mask, verbose=False)
        return maps["K1"], {}

    elif method == "tanh_pinn":
        trainer = Trainer(c_p=aif, num_of_compartment=num_of_compartment, t=t, device="cpu",
                           affine=np.eye(4), save_path=None, epochs=epochs, activation="tanh")
        ks_out, hist = trainer.train(img, z_slices=[0])
        return ks_out[0], {"final_loss": hist["loss"][-1]}

    elif method == "sine_pinn":
        trainer = Trainer(c_p=aif, num_of_compartment=num_of_compartment, t=t, device="cpu",
                           affine=np.eye(4), save_path=None, epochs=epochs, activation="sine")
        ks_out, hist = trainer.train(img, z_slices=[0])
        return ks_out[0], {"final_loss": hist["loss"][-1]}

    elif method == "sine_bpinn":
        res = estimate_with_uncertainty(
            img, aif, t, num_of_compartment=num_of_compartment, save_path=None,
            n_ensemble=n_ensemble, epochs=epochs, dropout_p=dropout_p, tissue_mask=mask,
            activation="sine",
        )
        K1_unc = res["K_uncertainty_demeaned"][0]
        return res["K_mean"][0], {"mean_uncertainty": float(K1_unc[mask].mean())}

    else:
        raise ValueError(f"unknown method {method!r}")


def run_sweep(axis, modality, methods, epochs, n_ensemble, dropout_p, shape, seed):
    """axis: 'noise' or 'sparsity'. modality: 'dce' or 'pet'."""
    levels = NOISE_LEVELS if axis == "noise" else SPARSITY_LEVELS
    results = []

    for level_name, level_val in levels.items():
        if modality == "dce":
            snr = level_val if axis == "noise" else NOISE_LEVELS["moderate"]
            dt = SPARSITY_LEVELS["moderate"] if axis == "noise" else level_val
            img, t, aif, label, gt, num_c = build_dce_data(shape, seed, snr, dt)
        else:
            snr = level_val if axis == "noise" else NOISE_LEVELS["moderate"]
            dt_sec = 30.0 if axis == "noise" else level_val * 60  # rough PET analog of the DCE dt grid
            img, t, aif, label, gt, num_c = build_pet_data(shape, seed, snr, dt_sec)

        mask = label > 0
        gtK1 = gt["K1"]

        for method in methods:
            t0 = time.time()
            try:
                K1_pred, extra = run_method(method, img, t, aif, num_c, epochs, n_ensemble, dropout_p, mask)
                r = float(pearsonr(K1_pred[mask], gtK1[mask])[0])
            except Exception as e:
                r = None
                extra = {"error": str(e)}
            elapsed = time.time() - t0

            row = dict(modality=modality, axis=axis, level=level_name, level_value=level_val,
                       method=method, K1_corr=r, time_s=round(elapsed, 1), n_frames=len(t), **extra)
            results.append(row)
            print(f"[{modality}/{axis}={level_name}] {method:12s} K1_corr={r} ({elapsed:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis", choices=["noise", "sparsity", "both"], default="both")
    parser.add_argument("--modality", choices=["dce", "pet", "both"], default="both")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n-ensemble", type=int, default=5)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--shape", type=int, nargs=3, default=list(DEFAULT_SHAPE))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    axes = ["noise", "sparsity"] if args.axis == "both" else [args.axis]
    modalities = ["dce", "pet"] if args.modality == "both" else [args.modality]

    all_results = []
    for modality, axis in itertools.product(modalities, axes):
        all_results += run_sweep(axis, modality, args.methods, args.epochs, args.n_ensemble,
                                  args.dropout_p, tuple(args.shape), args.seed)

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} rows to {out_path}")


if __name__ == "__main__":
    main()
