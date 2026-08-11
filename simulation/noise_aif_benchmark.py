"""
Benchmark voxelwise NLLS vs PINN (tanh / sine / KAN trunks) kinetic-
parameter recovery on a synthetic 64x64x20 prostate phantom (ellipsoid
gland with a smaller ellipsoid lesion inside it, see
simulation/phantom.py), across:

  - a noise-level sweep (SNR grid, high->low i.e. noise getting larger):
    DCE uses 1TCM + Rician/Gaussian "dce" noise, PET uses 2TCM +
    Poisson/Gaussian "pet" noise (simulation/forward_models.py's
    dce_noise/pet_noise, matching each modality's actual noise physics).
  - an AIF-quality axis: "correct" (the same AIF the data was simulated
    with) vs "sparse" (that AIF undersampled in time then linearly
    interpolated back -- simulating a coarse/incomplete arterial
    sampling protocol, a real acquisition failure mode) fed to the
    fitter, with the tissue data itself unchanged -- isolates AIF
    measurement error from everything else.

For each PINN method, a small hyperparameter search (random, or
Optuna/TPE Bayesian if optuna is installed) picks the best-scoring
config per (modality, aif, noise_level) cell, scored against the known
ground truth K1 -- valid here because this is a controlled simulation
with ground truth available, not real data.

This is a SLOW, exhaustive benchmark by design (grid x hyperparameter
search x whole-volume PINN fits) -- meant to be kicked off and left
running (results are written incrementally after every cell so a long
run's progress survives an interruption). Use --smoke for a fast sanity
check that the whole pipeline (data gen -> every method -> search -> CSV)
actually runs end to end -- NOT a real result, just wiring verification.

CAVEAT (inherited from simulation/validate.py): core/model.py's 1TCM
Ks_net treats its 2nd output channel inconsistently between the two
places that use it (kep directly vs K1/ve) -- see that script's note.
To sidestep this, the hyperparameter-search objective here uses ONLY K1
correlation (recoverable unambiguously by every method/modality), while
the CSV still reports k2 (DCE)/k2,k3 (PET) bias/RMSE/corr for reference.

Usage
-----
    python simulation/noise_aif_benchmark.py --smoke
    python simulation/noise_aif_benchmark.py                      # full run (default grid, slow)
    python simulation/noise_aif_benchmark.py --modality dce --methods voxelwise pinn_sine
    python simulation/noise_aif_benchmark.py --search bayesian --search-iters 10

Output (in <out-dir>, default simulation_data/validation_runs/noise_aif_benchmark):
    results_best.csv   -- one row per (modality, aif, snr, method): best
                           config found + full bias/RMSE/corr breakdown
                           (all voxels, normal-only, cancer-only).
    results_trials.csv -- every hyperparameter-search trial (long format).
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAVE_OPTUNA = True
except ImportError:
    _HAVE_OPTUNA = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_SHAPE = (64, 64, 20)
DEFAULT_SNR_LEVELS = [30.0, 20.0, 12.0, 7.0, 4.0]   # high -> low SNR = noise getting larger

PINN_METHODS = ("pinn_tanh", "pinn_sine", "pinn_kan")
ALL_METHODS = ("voxelwise",) + PINN_METHODS

PINN_KWARGS = {
    "pinn_tanh": dict(activation="tanh", arch="mlp"),
    "pinn_sine": dict(activation="sine", arch="mlp"),
    "pinn_kan":  dict(activation="silu", arch="kan"),
}

# Per-method PINN hyperparameter search spaces -- kept small and centered
# near this repo's already-tuned recipe (core/train.py Trainer's
# docstring: hidden_size=10, omega_0=1.0, physics_weight=100). omega_0 is
# only meaningful for activation="sine" (unused for "tanh"/kan's "silu"
# -- see core/model.py); kan_grid_size is KAN-only.
SEARCH_SPACES = {
    "pinn_tanh": {
        "hidden_size": [10, 20, 40],
        "lr": [5e-4, 1e-3, 2e-3],
        "physics_weight": [10.0, 100.0, 300.0],
    },
    "pinn_sine": {
        "hidden_size": [10, 20, 40],
        "omega_0": [0.5, 1.0, 2.0],
        "lr": [5e-4, 1e-3, 2e-3],
        "physics_weight": [10.0, 100.0, 300.0],
    },
    "pinn_kan": {
        "hidden_size": [10, 20, 40],
        "kan_grid_size": [3, 5, 8],
        "lr": [5e-4, 1e-3, 2e-3],
        "physics_weight": [10.0, 100.0, 300.0],
    },
}

OUT_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "simulation_data", "validation_runs", "noise_aif_benchmark",
)


# ---------------------------------------------------------------------
# AIF degradation
# ---------------------------------------------------------------------
def degrade_aif(t, aif, keep_every=3):
    """Simulate an undersampled/incomplete AIF measurement: keep only
    every `keep_every`-th sample (endpoints always kept), linearly
    interpolate the rest back onto the full time grid -- the standard
    way a coarsely-sampled AIF is actually used for fitting in practice.
    """
    idx = list(range(0, len(t), keep_every))
    if idx[-1] != len(t) - 1:
        idx.append(len(t) - 1)
    return np.interp(t, t[idx], aif[idx])


# ---------------------------------------------------------------------
# Metrics
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


def k1_score(maps, gt, mask):
    """Hyperparameter-search objective: K1 correlation over `mask`
    (recoverable unambiguously by every method/modality -- see module
    docstring). NaN (e.g. a degenerate fit) scores -1 so the search never
    silently prefers a broken fit over a mediocre one."""
    m = voxel_metrics(maps["K1"], gt["K1"], mask)
    return m["corr"] if np.isfinite(m["corr"]) else -1.0


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------
def build_dce_case(shape, seed, snr):
    info = build_prostate_phantom(shape, seed=seed)
    label = info.label
    gt = sample_param_maps(label, DCE_1TCM_PARAMS, ["K1", "k2"], seed=seed)
    t = np.arange(0, 5.3, 0.1522)                 # minutes, matches config.DCE's frame convention
    aif = parker_aif(t)
    img, _ = simulate_1tcm_volume(t, aif, gt["K1"], gt["k2"], noise_model="dce", snr=snr, rng=seed)
    return img, t, aif, label, gt


def build_pet_case(shape, seed, snr):
    info = build_prostate_phantom(shape, seed=seed)
    label = info.label
    gt = sample_param_maps(label, PET_2TCM_PARAMS, ["K1", "k2", "k3"], seed=seed)
    # ~20-minute scan, fine sampling around the bolus then coarsening --
    # matches this repo's other PET simulation conventions.
    dt_sec = np.asarray([10] * 6 + [30] * 10 + [50] * 6 + [300] + [240])
    t = np.cumsum(dt_sec) / 60.0
    aif = feng_input_function(t)
    img, _ = simulate_2tcm_volume(t, aif, gt["K1"], gt["k2"], gt["k3"], noise_model="pet", snr=snr, rng=seed)
    return img, t, aif, label, gt


# ---------------------------------------------------------------------
# Fitters
# ---------------------------------------------------------------------
def fit_voxelwise(modality, img, t, aif, mask, subsample, seed):
    idx_all = np.argwhere(mask)
    vox_mask = mask
    if len(idx_all) > subsample:
        rng = np.random.default_rng(seed)
        vox_mask = np.zeros_like(mask)
        sel = rng.choice(len(idx_all), size=subsample, replace=False)
        for x, y, z in idx_all[sel]:
            vox_mask[x, y, z] = True
    if modality == "dce":
        maps = calculate_dce_voxelwise(img, t, aif, mask=vox_mask)
    else:
        maps = calculate_pet_voxelwise(img, t, aif, mask=vox_mask, verbose=False)
    return maps, vox_mask


def fit_pinn(method, num_of_compartment, img, t, aif, epochs, params):
    kwargs = dict(PINN_KWARGS[method])
    kwargs.update(params)
    trainer = Trainer(c_p=aif, num_of_compartment=num_of_compartment, t=t, device=DEVICE,
                       affine=np.eye(4), save_path=None, epochs=epochs, windowed=False, **kwargs)
    ks_out, _ = trainer.train(img, z_slices=[0])
    if num_of_compartment == 1:
        return {"K1": ks_out[0], "k2": ks_out[1]}
    return {"K1": ks_out[0], "k2": ks_out[1], "k3": ks_out[2]}


def sample_params(space, rng):
    return {k: values[int(rng.integers(len(values)))] for k, values in space.items()}


def search_pinn(method, num_of_compartment, img, t, aif, gt, mask, epochs,
                 search_mode, search_iters, seed):
    """Runs `search_iters` PINN fits (or exactly 1 if search_mode='none'),
    scoring each against K1 ground truth (k1_score). Returns
    (best_maps, best_trial, all_trials)."""
    space = SEARCH_SPACES[method]
    trials = []

    def evaluate(params):
        t0 = time.time()
        maps = fit_pinn(method, num_of_compartment, img, t, aif, epochs, params)
        score = k1_score(maps, gt, mask)
        trials.append(dict(params=params, score=score, time_s=round(time.time() - t0, 1)))
        return score

    if search_mode == "none":
        evaluate({})
    elif search_mode == "bayesian" and _HAVE_OPTUNA:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: evaluate({k: trial.suggest_categorical(k, v) for k, v in space.items()}),
            n_trials=search_iters, show_progress_bar=False,
        )
    else:
        if search_mode == "bayesian" and not _HAVE_OPTUNA:
            print("  [WARN] optuna not installed -- falling back to random search")
        rng = np.random.default_rng(seed)
        for _ in range(search_iters):
            evaluate(sample_params(space, rng))

    best = max(trials, key=lambda r: r["score"])
    best_maps = fit_pinn(method, num_of_compartment, img, t, aif, epochs, best["params"])
    return best_maps, best, trials


# ---------------------------------------------------------------------
# One (modality, aif_mode, snr) grid cell -- all methods
# ---------------------------------------------------------------------
def run_cell(modality, aif_mode, snr, methods, epochs, search_mode, search_iters,
             voxelwise_subsample, shape, seed):
    if modality == "dce":
        img, t, aif_correct, label, gt = build_dce_case(shape, seed, snr)
        num_c, param_names = 1, ["K1", "k2"]
    else:
        img, t, aif_correct, label, gt = build_pet_case(shape, seed, snr)
        num_c, param_names = 2, ["K1", "k2", "k3"]

    aif = aif_correct if aif_mode == "correct" else degrade_aif(t, aif_correct)
    mask = label > 0
    class_masks = {"normal": label == 1, "cancer": label == 2}

    rows_best, rows_trials = [], []

    for method in methods:
        t0 = time.time()
        try:
            if method == "voxelwise":
                maps, eval_mask = fit_voxelwise(modality, img, t, aif, mask, voxelwise_subsample, seed)
                best = dict(params={}, score=k1_score(maps, gt, eval_mask), time_s=round(time.time() - t0, 1))
                trials = [best]
            else:
                maps, best, trials = search_pinn(method, num_c, img, t, aif, gt, mask,
                                                  epochs, search_mode, search_iters, seed)
                eval_mask = mask
        except Exception as e:
            print(f"  [ERROR] {modality}/aif={aif_mode}/snr={snr}/{method}: {e}")
            continue

        elapsed = time.time() - t0
        for i, tr in enumerate(trials):
            rows_trials.append(dict(
                modality=modality, aif=aif_mode, snr=snr, method=method, trial=i,
                score=tr["score"], time_s=tr["time_s"], params=json.dumps(tr["params"]),
            ))

        row = dict(
            modality=modality, aif=aif_mode, snr=snr, method=method,
            best_params=json.dumps(best["params"]), best_k1_score=best["score"],
            n_trials=len(trials), total_time_s=round(elapsed, 1), n_eval_voxels=int(eval_mask.sum()),
        )
        cls_masks = {"all": eval_mask, "normal": eval_mask & class_masks["normal"],
                     "cancer": eval_mask & class_masks["cancer"]}
        for p in param_names:
            for cls_name, cls_mask in cls_masks.items():
                m = voxel_metrics(maps[p], gt[p], cls_mask)
                row[f"{p}_{cls_name}_n"] = m["n"]
                row[f"{p}_{cls_name}_bias"] = m["bias"]
                row[f"{p}_{cls_name}_rmse"] = m["rmse"]
                row[f"{p}_{cls_name}_corr"] = m["corr"]
        rows_best.append(row)

        print(f"[{modality}/aif={aif_mode}/snr={snr}] {method:10s} "
              f"K1_score={best['score']:+.3f} ({len(trials)} trial(s), {elapsed:.1f}s)")

    return rows_best, rows_trials


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true",
                         help="tiny fast run (10x10x4 phantom, ~12 epochs, 2 SNR levels, "
                              "2 search trials) to verify the pipeline runs end to end -- "
                              "NOT a real result. Overrides --shape/--epochs/--snr-levels/--search-iters.")
    parser.add_argument("--modality", choices=["dce", "pet", "both"], default="both")
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
    parser.add_argument("--aif", choices=["correct", "sparse", "both"], default="both")
    parser.add_argument("--snr-levels", type=float, nargs="+", default=None,
                         help=f"SNR grid, high->low SNR = noise getting larger. Default: {DEFAULT_SNR_LEVELS}")
    parser.add_argument("--shape", type=int, nargs=3, default=list(DEFAULT_SHAPE))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--search", choices=["none", "random", "bayesian"], default="random",
                         help="'bayesian' uses Optuna/TPE if installed, else falls back to random.")
    parser.add_argument("--search-iters", type=int, default=4)
    parser.add_argument("--voxelwise-subsample", type=int, default=1500,
                         help="cap on voxels fit per NLLS call (it's a per-voxel scipy optimization "
                              "-- too slow on the full volume otherwise).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.smoke:
        shape = (10, 10, 4)
        epochs = 12
        snr_levels = [20.0, 5.0]
        search_iters = 2
        voxelwise_subsample = 200
        print("[SMOKE MODE] tiny shape/epochs/grid -- verifying the pipeline runs, not a real result.\n")
    else:
        shape = tuple(args.shape)
        epochs = args.epochs
        snr_levels = args.snr_levels or DEFAULT_SNR_LEVELS
        search_iters = args.search_iters
        voxelwise_subsample = args.voxelwise_subsample

    modalities = ["dce", "pet"] if args.modality == "both" else [args.modality]
    aif_modes = ["correct", "sparse"] if args.aif == "both" else [args.aif]
    out_dir = args.out_dir or OUT_DIR_DEFAULT
    os.makedirs(out_dir, exist_ok=True)

    print(f"device={DEVICE}  modalities={modalities}  aif={aif_modes}  snr_levels={snr_levels}\n"
          f"methods={args.methods}  search={args.search}(x{search_iters})  shape={shape}  epochs={epochs}\n"
          f"out_dir={out_dir}\n")

    all_best, all_trials = [], []
    t_start = time.time()
    for modality, aif_mode, snr in itertools.product(modalities, aif_modes, snr_levels):
        rows_best, rows_trials = run_cell(modality, aif_mode, snr, args.methods, epochs,
                                           args.search, search_iters, voxelwise_subsample,
                                           shape, args.seed)
        all_best += rows_best
        all_trials += rows_trials
        # write incrementally so a long run's progress survives an interruption
        write_csv(os.path.join(out_dir, "results_best.csv"), all_best)
        write_csv(os.path.join(out_dir, "results_trials.csv"), all_trials)

    print(f"\nDone in {time.time() - t_start:.1f}s. "
          f"{len(all_best)} best-config rows, {len(all_trials)} trial rows -> {out_dir}")


if __name__ == "__main__":
    main()
