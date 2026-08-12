"""
Optuna hyperparameter search for the prostate-phantom PINN.

The search maximizes Pearson correlation between recovered K1 and the
known phantom K1 map.

Workflow
--------
1. Run many inexpensive Optuna/TPE trials using SEEDS_PER_TRIAL seeds.
2. Take the best N_CONFIRM trials according to the search-time score.
3. Re-run those candidates on all N_SEEDS_CONFIRM seeds.
4. Select the final winner using the multi-seed mean.

IMPORTANT
---------
Do NOT treat study.best_value as the final result when
SEEDS_PER_TRIAL < N_SEEDS_CONFIRM. A single seed can produce a false lead.

The objective remains K1-only, matching the original run_config behavior.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import optuna
from optuna.samplers import TPESampler
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.phantom import build_prostate_phantom
from simulation.kinetics_literature import DCE_1TCM_PARAMS, sample_param_maps
from simulation.forward_models import parker_aif, simulate_1tcm_volume
from core.train import Trainer


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SHAPE = (16, 16, 8)
Z_IDX = [3, 4]

FRAME_DURATION = 0.1522
N_FRAMES = 35
NOISE_STD = 0.1

EPOCHS = 100

# Cheap score during Bayesian search.
SEEDS_PER_TRIAL = 1
N_TRIALS = 160

# More reliable final confirmation.
N_SEEDS_CONFIRM = 3
N_CONFIRM = 3

SCRATCH_DIR = PROJECT_ROOT / "simulation_data" / "validation_runs" / "_sweep_scratch"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson r, or NaN when correlation is undefined."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size < 2 or y.size < 2:
        return float("nan")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

    if x.size < 2:
        return float("nan")

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(pearsonr(x, y)[0])


def parse_grad_clip(value) -> float | None:
    """Convert Optuna's categorical grad-clip value to float/None."""
    if value is None or value == "none":
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Phantom generation
# ---------------------------------------------------------------------------

def build_case(seed: int):
    """Build one noisy prostate phantom and return the PINN inputs/targets."""
    info = build_prostate_phantom(SHAPE, seed=seed)
    label = info.label

    gt = sample_param_maps(
        label,
        DCE_1TCM_PARAMS,
        ["K1", "k2"],
        seed=seed,
    )

    t = np.cumsum(np.full(N_FRAMES, FRAME_DURATION, dtype=float))
    aif = parker_aif(t)

    noisy, _ = simulate_1tcm_volume(
        t,
        aif,
        gt["K1"],
        gt["k2"],
        noise_std=NOISE_STD,
        rng=seed,
    )

    # Only use the requested phantom slices.
    img = noisy[:, :, :, Z_IDX]
    lab = label[:, :, Z_IDX]
    gt_k1 = gt["K1"][:, :, Z_IDX]

    return t, aif, img, lab, gt_k1


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def run_config(
    tag: str,
    seeds: Iterable[int],
    **kwargs,
) -> float:
    """
    Train/evaluate one hyperparameter configuration across multiple seeds.

    Returns
    -------
    float
        Mean K1 Pearson correlation across valid seeds.
    """
    correlations = []

    for seed in seeds:
        t, aif, img, lab, gt_k1 = build_case(seed)

        save_path = SCRATCH_DIR / f"{tag}_{seed}"
        save_path.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(
            c_p=aif,
            num_of_compartment=1,
            t=t,
            device="cpu",
            affine=np.eye(4),
            save_path=str(save_path),
            epochs=EPOCHS,
            windowed=True,
            **kwargs,
        )

        ks_out, _ = trainer.train(img, z_slices=[0])
        recovered_k1 = ks_out[0]

        mask = lab > 0

        r = safe_pearsonr(
            recovered_k1[mask],
            gt_k1[mask],
        )

        if np.isfinite(r):
            correlations.append(r)

    if not correlations:
        return float("nan")

    mean_r = float(np.mean(correlations))
    std_r = float(np.std(correlations))

    print(
        f"{tag:24s} "
        f"K1 r = {mean_r:.3f} ± {std_r:.3f} "
        f"(n={len(correlations)})"
    )

    return mean_r


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """Sample one configuration and return its search-time K1 score."""

    hidden_size = trial.suggest_categorical(
        "hidden_size",
        [10, 20, 40, 60, 80, 100],
    )

    # The previous grid search showed that very large omega_0 values can
    # become unstable. Keep the broad historical range, but sample
    # logarithmically.
    omega_0 = trial.suggest_float(
        "omega_0",
        0.1,
        50.0,
        log=True,
    )

    physics_weight = trial.suggest_float(
        "physics_weight",
        1e-3,
        100.0,
        log=True,
    )

    tac_consistency_weight = trial.suggest_float(
        "tac_consistency_weight",
        1e-3,
        50.0,
        log=True,
    )

    reg_weight = trial.suggest_float(
        "reg_weight",
        1e-5,
        1e-2,
        log=True,
    )

    causality_eps_final = trial.suggest_categorical(
        "causality_eps_final",
        [0, 100, 1000, 2000, 5000, 10000],
    )

    grad_clip = trial.suggest_categorical(
        "grad_clip",
        ["none", "0.1", "0.5", "1.0"],
    )
    grad_clip = parse_grad_clip(grad_clip)

    lr = trial.suggest_float(
        "lr",
        1e-4,
        1,
        log=True,
    )

    # These parameters were already present in the supplied script.
    kan_grid_size = trial.suggest_categorical(
        "kan_grid_size",
        [3, 5],
    )

    kan_spline_order = trial.suggest_categorical(
        "kan_spline_order",
        [1, 2, 3],
    )

    kan_grid_range = trial.suggest_categorical(
        "kan_grid_range",
        [3, 5]
    )

    score = run_config(
        f"trial_{trial.number}",
        seeds=range(SEEDS_PER_TRIAL),
        hidden_size=hidden_size,
        omega_0=omega_0,
        physics_weight=physics_weight,
        tac_consistency_weight=tac_consistency_weight,
        reg_weight=reg_weight,
        causality_eps_final=float(causality_eps_final),
        grad_clip=grad_clip,
        lr=lr,
        kan_grid_size=kan_grid_size,
        kan_spline_order=kan_spline_order,
        kan_grid_range=(-kan_grid_range, kan_grid_range),
    )

    if not np.isfinite(score):
        raise optuna.TrialPruned(
            "K1 correlation was undefined for this trial."
        )

    return score


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------

def confirm_top_trials(study: optuna.Study):
    """Re-score the top search trials on all confirmation seeds."""

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and np.isfinite(trial.value)
    ]

    top_trials = sorted(
        completed,
        key=lambda trial: trial.value,
        reverse=True,
    )[:N_CONFIRM]

    print(
        f"\n=== Confirming top {len(top_trials)} trials "
        f"on {N_SEEDS_CONFIRM} seeds ==="
    )

    confirmed = []

    for trial in top_trials:
        params = dict(trial.params)

        # Optuna stores grad_clip as a string category.
        grad_clip = parse_grad_clip(params.pop("grad_clip"))

        params["causality_eps_final"] = float(
            params["causality_eps_final"]
        )

        confirmed_mean = run_config(
            f"confirm_trial_{trial.number}",
            seeds=range(N_SEEDS_CONFIRM),
            grad_clip=grad_clip,
            **params,
        )

        confirmed.append(
            {
                "trial_number": trial.number,
                "search_score": float(trial.value),
                "confirmed_score": confirmed_mean,
                "params": trial.params,
            }
        )

    confirmed.sort(
        key=lambda row: (
            -np.inf
            if not np.isfinite(row["confirmed_score"])
            else -row["confirmed_score"]
        )
    )

    return confirmed


def print_confirmed_results(confirmed):
    """Print the final multi-seed ranking."""

    print(
        "\n=== Confirmed ranking "
        "(use this ranking, NOT study.best_value) ==="
    )

    for row in confirmed:
        search_score = row["search_score"]
        confirmed_score = row["confirmed_score"]

        if (
            np.isfinite(confirmed_score)
            and confirmed_score < search_score - 0.05
        ):
            flag = "  <-- large drop during confirmation"
        else:
            flag = ""

        print(
            f"trial_{row['trial_number']}: "
            f"search={search_score:.3f}, "
            f"confirmed={confirmed_score:.3f}"
            f"{flag}"
        )
        print(f"  params={row['params']}")

    if confirmed:
        winner = confirmed[0]

        print("\n=== FINAL WINNER ===")
        print(f"trial_{winner['trial_number']}")
        print(f"confirmed K1 r = {winner['confirmed_score']:.4f}")
        print(f"parameters = {winner['params']}")

        return winner

    print("\nNo valid confirmed trials were produced.")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()

    print("Starting Optuna PINN hyperparameter search")
    print(f"Trials: {N_TRIALS}")
    print(f"Seeds per trial: {SEEDS_PER_TRIAL}")
    print(f"Confirmation seeds: {N_SEEDS_CONFIRM}")
    print(f"Final candidates: {N_CONFIRM}")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=0),
    )

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
        )
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]

    print(
        f"\n=== Optuna search finished: "
        f"{len(study.trials)} trials ==="
    )

    if not completed:
        print("No completed trials.")
        return

    best = study.best_trial

    print(
        f"Best search-time trial: trial_{best.number}, "
        f"value={best.value:.3f}"
    )
    print(
        "WARNING: this is only the search-time score and is not "
        "the final result when using fewer seeds per trial."
    )
    print(f"Parameters: {best.params}")

    confirmed = confirm_top_trials(study)
    winner = print_confirmed_results(confirmed)

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f} s")

    return study, confirmed, winner


if __name__ == "__main__":
    main()