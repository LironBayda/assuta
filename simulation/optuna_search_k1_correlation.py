"""
Optuna hyperparameter search for the prostate-phantom PINN.

CHANGED OBJECTIVE
------------------
The search no longer maximizes per-modality K1 accuracy. Instead, for each
trial it reconstructs K1 from BOTH modalities on the same phantom:

    - DCE via the 1TCM model  (num_of_compartment=1)
    - PET via the 2TCM model  (num_of_compartment=2)

and scores the trial by how well the DCE/PET K1 *cross-modality*
relationship is preserved:

    gt_corr    = Pearson r( K1_dce_ground_truth,  K1_pet_ground_truth  )
    recon_corr = Pearson r( K1_dce_reconstructed,  K1_pet_reconstructed )
    score      = |gt_corr - recon_corr|      <-- MINIMIZE this

A trial that reconstructs both modalities' K1 accurately in isolation but
distorts their inter-modality relationship is penalized just as much as one
that reconstructs the relationship "by accident" while being inaccurate
per-voxel -- per-modality Pearson r vs. ground truth is still computed and
logged for diagnostics, but it is NOT the optimized quantity anymore.

ASSUMPTIONS ABOUT THE REPO (please check/adjust the two marked below)
-----------------------------------------------------------------------
1. `simulation.kinetics_literature` exposes a `PET_2TCM_PARAMS` dict with
   the same shape/contract as `DCE_1TCM_PARAMS`, usable via
   `sample_param_maps(label, PET_2TCM_PARAMS, ["K1", "k2", "k3"], seed=...)`.
   -> If the real name differs, fix the import below.
2. `simulation.forward_models` exposes `simulate_2tcm_volume(t, aif, K1,
   k2, k3, noise_std=..., rng=...)` mirroring `simulate_1tcm_volume`.
   -> If the real signature differs, fix `build_case()` below.

GROUND-TRUTH K1 CORRELATION
----------------------------
DCE-K1 and PET-K1 are sampled independently from their own literature
marginals (they're different tracers/models, so their marginal
distributions differ), then their voxel ranks are recoupled to a target
Pearson correlation (TARGET_K1_CORR) using an Iman-Conover style rank
match. This keeps each modality's own marginal distribution intact while
giving you a known, controllable ground-truth cross-modality correlation
to try to preserve through reconstruction.

Workflow
--------
1. Run many inexpensive Optuna/TPE trials using SEEDS_PER_TRIAL seeds.
2. Take the best N_CONFIRM trials according to the search-time score.
3. Re-run those candidates on all N_SEEDS_CONFIRM seeds.
4. Select the final winner using the multi-seed mean |gt_corr - recon_corr|.

IMPORTANT
---------
Do NOT treat study.best_value as the final result when
SEEDS_PER_TRIAL < N_SEEDS_CONFIRM. A single seed can produce a false lead.

SPEED NOTES (CPU)
------------------
Three changes make this materially faster on CPU without changing what's
being optimized:

1. `build_case(seed)` is now cached. The phantom + simulated DCE/PET
   volumes depend ONLY on `seed`, never on the trial's hyperparameters --
   the original script silently rebuilt and re-simulated them from scratch
   on every single trial (160x redundant work for SEEDS_PER_TRIAL=1). Now
   each seed's data is built once and reused.
2. Search-time epochs (EPOCHS_SEARCH) are decoupled from confirmation-time
   epochs (EPOCHS_CONFIRM). TPE only needs a directionally useful signal
   to rank configurations, not a fully converged model; full epochs are
   still used for the final confirmation re-scoring.
3. `torch.set_num_threads(TORCH_NUM_THREADS)` avoids CPU oversubscription.
   If you set N_JOBS > 1 to run trials in parallel (via joblib/loky
   processes), keep TORCH_NUM_THREADS low (e.g. 1-2) so the per-trial
   thread pools don't fight each other for cores.
"""

from __future__ import annotations

import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler
from scipy.stats import pearsonr
import config
# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.phantom import build_prostate_phantom
from simulation.kinetics_literature import (
    DCE_1TCM_PARAMS,
    PET_2TCM_PARAMS,  # ASSUMPTION 1 -- fix name/import if different in repo
    sample_param_maps,
)
from simulation.forward_models import (
    parker_aif,
    simulate_1tcm_volume,
    simulate_2tcm_volume,  # ASSUMPTION 2 -- fix signature if different
)
from core.train import Trainer


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SHAPE = (8, 8, 8)
Z_IDX = [3, 4]

FRAME_DURATION = 0.1522
N_FRAMES = 35
NOISE_STD = 0.1

# Fewer epochs while ranking configurations; full epochs only for the final
# multi-seed confirmation pass. This is the second-biggest speed lever after
# caching build_case() -- tune EPOCHS_SEARCH down further if TPE still
# ranks configs sensibly with less training.
EPOCHS_SEARCH = 250
EPOCHS_CONFIRM = 250

# Target ground-truth Pearson correlation between DCE-K1 and PET-K1 maps.
TARGET_K1_CORR = 0.9

# Cheap score during Bayesian search.
SEEDS_PER_TRIAL = 1
N_TRIALS = 160

# More reliable final confirmation.
N_SEEDS_CONFIRM = 3
N_CONFIRM = 3

# Number of Optuna trials to run concurrently. >1 uses joblib/loky
# processes. Keep TORCH_NUM_THREADS low when this is >1 to avoid every
# process fighting for all cores at once.
# NOTE: the build_case() cache is per-process, so N_JOBS > 1 means each
# worker process rebuilds/re-simulates each seed's phantom the first time
# it sees it (still far cheaper than the original script, which rebuilt
# on literally every trial).
N_JOBS = 1
TORCH_NUM_THREADS = 4

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


def induce_rank_correlation(
    x1: np.ndarray,
    x2: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reorder x1 and x2 (independently sampled, flattened) so that their
    *ranks* follow a pair of correlated Gaussians with correlation `rho`,
    while each array keeps its own original marginal distribution
    (Iman-Conover rank matching).
    """
    x1 = np.asarray(x1).ravel()
    x2 = np.asarray(x2).ravel()
    n = x1.size

    z = rng.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]], size=n)
    order1 = np.argsort(np.argsort(z[:, 0]))
    order2 = np.argsort(np.argsort(z[:, 1]))

    x1_sorted = np.sort(x1)
    x2_sorted = np.sort(x2)

    return x1_sorted[order1], x2_sorted[order2]


# ---------------------------------------------------------------------------
# Phantom generation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _build_case_cached(seed: int):
    """
    Cached core of build_case(). The phantom + simulated DCE/PET volumes
    depend only on `seed`, never on trial hyperparameters, so every trial
    that reuses a seed gets this for free instead of re-simulating from
    scratch.
    """
    info = build_prostate_phantom(SHAPE, seed=seed)
    label = info.label
    mask = label > 0

    gt_dce = sample_param_maps(label, DCE_1TCM_PARAMS, ["K1", "k2"], seed=seed)
    gt_pet = sample_param_maps(
        label, PET_2TCM_PARAMS, ["K1", "k2", "k3"], seed=seed + 1_000_000
    )

    # Recouple DCE-K1 and PET-K1 ranks within the tissue mask so their
    # ground-truth correlation matches TARGET_K1_CORR, without touching
    # either modality's own marginal distribution.
    rng = np.random.default_rng(seed)
    k1_dce_masked, k1_pet_masked = induce_rank_correlation(
        gt_dce["K1"][mask], gt_pet["K1"][mask], TARGET_K1_CORR, rng
    )
    gt_dce["K1"][mask] = k1_dce_masked
    gt_pet["K1"][mask] = k1_pet_masked

    t = np.cumsum(np.full(N_FRAMES, FRAME_DURATION, dtype=float))
    aif = parker_aif(t)

    noisy_dce, _ = simulate_1tcm_volume(
        t, aif, gt_dce["K1"], gt_dce["k2"], noise_std=NOISE_STD, rng=seed,
    )
    noisy_pet, _ = simulate_2tcm_volume(
        t,
        aif,
        gt_pet["K1"],
        gt_pet["k2"],
        gt_pet["k3"],
        noise_std=NOISE_STD,
        rng=seed + 1_000_000,
    )

    img_dce = noisy_dce[:, :, :, Z_IDX]
    img_pet = noisy_pet[:, :, :, Z_IDX]
    lab = label[:, :, Z_IDX]
    gt_k1_dce = gt_dce["K1"][:, :, Z_IDX]
    gt_k1_pet = gt_pet["K1"][:, :, Z_IDX]

    return t, aif, img_dce, img_pet, lab, gt_k1_dce, gt_k1_pet


def build_case(seed: int):
    """
    Public entry point: returns a fresh COPY of the cached arrays for this
    seed, so nothing downstream (e.g. in-place ops inside Trainer.train)
    can corrupt the cached original for the next trial that reuses it.
    """
    cached = _build_case_cached(seed)
    t, aif, img_dce, img_pet, lab, gt_k1_dce, gt_k1_pet = cached
    return (
        t.copy(),
        aif.copy(),
        img_dce.copy(),
        img_pet.copy(),
        lab.copy(),
        gt_k1_dce.copy(),
        gt_k1_pet.copy(),
    )


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def run_config(
    tag: str,
    seeds: Iterable[int],
    epochs: int,
    **kwargs,
) -> dict:
    """
    Train/evaluate one hyperparameter configuration across multiple seeds,
    for both DCE (1TCM) and PET (2TCM).

    Returns
    -------
    dict with:
        corr_diff_mean / corr_diff_std   -- |gt_corr - recon_corr| stats
                                              (this is the value to MINIMIZE)
        r_dce_mean, r_pet_mean           -- per-modality diagnostics only
        n_valid                          -- number of seeds that produced
                                              a finite score
    """
    corr_diffs = []
    r_dce_list = []
    r_pet_list = []

    for seed in seeds:
        t, aif, img_dce, img_pet, lab, gt_k1_dce, gt_k1_pet = build_case(seed)
        mask = lab > 0

        save_path_dce = SCRATCH_DIR / f"{tag}_{seed}_dce"
        save_path_dce.mkdir(parents=True, exist_ok=True)
        trainer_dce = Trainer(
            c_p=aif,
            num_of_compartment=1,
            t=t,
            device=config.DEVICE,
            affine=np.eye(4),
            save_path=str(save_path_dce),
            epochs=epochs,
            windowed=True,
            **kwargs,
        )
        ks_dce, _ = trainer_dce.train(img_dce, z_slices=[0])
        recovered_k1_dce = ks_dce[0]

        save_path_pet = SCRATCH_DIR / f"{tag}_{seed}_pet"
        save_path_pet.mkdir(parents=True, exist_ok=True)
        trainer_pet = Trainer(
            c_p=aif,
            num_of_compartment=2,
            t=t,
            device=config.DEVICE,
            affine=np.eye(4),
            save_path=str(save_path_pet),
            epochs=epochs,
            windowed=True,
            **kwargs,
        )
        ks_pet, _ = trainer_pet.train(img_pet, z_slices=[0])
        recovered_k1_pet = ks_pet[0]

        gt_corr = safe_pearsonr(gt_k1_dce[mask], gt_k1_pet[mask])
        recon_corr = safe_pearsonr(recovered_k1_dce[mask], recovered_k1_pet[mask])

        if np.isfinite(gt_corr) and np.isfinite(recon_corr):
            corr_diffs.append(abs(gt_corr - recon_corr))

        r_dce_list.append(safe_pearsonr(recovered_k1_dce[mask], gt_k1_dce[mask]))
        r_pet_list.append(safe_pearsonr(recovered_k1_pet[mask], gt_k1_pet[mask]))

    result = {
        "corr_diff_mean": float(np.mean(corr_diffs)) if corr_diffs else float("nan"),
        "corr_diff_std": float(np.std(corr_diffs)) if corr_diffs else float("nan"),
        "r_dce_mean": float(np.nanmean(r_dce_list)) if r_dce_list else float("nan"),
        "r_pet_mean": float(np.nanmean(r_pet_list)) if r_pet_list else float("nan"),
        "n_valid": len(corr_diffs),
    }

    print(
        f"{tag:28s} "
        f"|gt_corr-recon_corr| = {result['corr_diff_mean']:.3f} "
        f"± {result['corr_diff_std']:.3f} "
        f"(r_dce={result['r_dce_mean']:.3f}, r_pet={result['r_pet_mean']:.3f}, "
        f"n={result['n_valid']})"
    )

    return result


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """Sample one configuration and return its search-time |gt_corr - recon_corr|."""

    hidden_size = trial.suggest_categorical(
        "hidden_size",
        [10, 20, 40, 60, 80, 100],
    )

    omega_0 = trial.suggest_float("omega_0", 0.1, 50.0, log=True)
    physics_weight = trial.suggest_float("physics_weight", 1e-3, 100.0, log=True)
    tac_consistency_weight = trial.suggest_float(
        "tac_consistency_weight", 1e-3, 50.0, log=True
    )
    reg_weight = trial.suggest_float("reg_weight", 1e-5, 1e-2, log=True)

    causality_eps_final = trial.suggest_categorical(
        "causality_eps_final",
        [0, 100, 1000, 2000, 5000, 10000],
    )

    grad_clip = trial.suggest_categorical("grad_clip", ["none", "0.1", "0.5", "1.0"])
    grad_clip = parse_grad_clip(grad_clip)

    lr = trial.suggest_float("lr", 1e-4, 1, log=True)

    kan_grid_size = trial.suggest_categorical(
        "kan_grid_size", [3, 5,6,7,8,9]
    )
    kan_spline_order = trial.suggest_categorical("kan_spline_order", [1, 2, 3])
    kan_grid_range = trial.suggest_categorical(
        "kan_grid_range", [3, 5,6,7,8,9]
    )

    result = run_config(
        f"trial_{trial.number}",
        seeds=range(SEEDS_PER_TRIAL),
        epochs=EPOCHS_SEARCH,
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

    score = result["corr_diff_mean"]

    if not np.isfinite(score):
        raise optuna.TrialPruned(
            "gt/recon K1 correlation difference was undefined for this trial."
        )

    # Stash diagnostics on the trial so confirm_top_trials() can report them
    # without re-deriving them from trial.value alone.
    trial.set_user_attr("r_dce_mean", result["r_dce_mean"])
    trial.set_user_attr("r_pet_mean", result["r_pet_mean"])

    return score


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------

def confirm_top_trials(study: optuna.Study):
    """Re-score the top search trials (lowest corr_diff) on all confirmation seeds."""

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and np.isfinite(trial.value)
    ]

    # Lower is better now (minimizing |gt_corr - recon_corr|).
    top_trials = sorted(completed, key=lambda trial: trial.value)[:N_CONFIRM]

    print(
        f"\n=== Confirming top {len(top_trials)} trials "
        f"on {N_SEEDS_CONFIRM} seeds ==="
    )

    confirmed = []

    for trial in top_trials:
        params = dict(trial.params)
        grad_clip = parse_grad_clip(params.pop("grad_clip"))
        params["causality_eps_final"] = float(params["causality_eps_final"])

        result = run_config(
            f"confirm_trial_{trial.number}",
            seeds=range(N_SEEDS_CONFIRM),
            epochs=EPOCHS_CONFIRM,
            grad_clip=grad_clip,
            **params,
        )

        confirmed.append(
            {
                "trial_number": trial.number,
                "search_score": float(trial.value),
                "confirmed_score": result["corr_diff_mean"],
                "r_dce_mean": result["r_dce_mean"],
                "r_pet_mean": result["r_pet_mean"],
                "params": trial.params,
            }
        )

    # Lower confirmed_score is better; NaN sorts last.
    confirmed.sort(
        key=lambda row: (
            np.inf if not np.isfinite(row["confirmed_score"]) else row["confirmed_score"]
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

        # Worse now means the diff GREW during confirmation (higher = worse).
        if (
            np.isfinite(confirmed_score)
            and confirmed_score > search_score + 0.05
        ):
            flag = "  <-- large increase during confirmation"
        else:
            flag = ""

        print(
            f"trial_{row['trial_number']}: "
            f"search |diff|={search_score:.3f}, "
            f"confirmed |diff|={confirmed_score:.3f} "
            f"(r_dce={row['r_dce_mean']:.3f}, r_pet={row['r_pet_mean']:.3f})"
            f"{flag}"
        )
        print(f"  params={row['params']}")

    if confirmed:
        winner = confirmed[0]

        print("\n=== FINAL WINNER ===")
        print(f"trial_{winner['trial_number']}")
        print(f"confirmed |gt_corr - recon_corr| = {winner['confirmed_score']:.4f}")
        print(
            f"(diagnostics: r_dce={winner['r_dce_mean']:.3f}, "
            f"r_pet={winner['r_pet_mean']:.3f})"
        )
        print(f"parameters = {winner['params']}")

        return winner

    print("\nNo valid confirmed trials were produced.")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()

    torch.set_num_threads(TORCH_NUM_THREADS)

    print("Starting Optuna PINN hyperparameter search")
    print("Objective: minimize |gt K1 corr(DCE,PET) - reconstructed K1 corr(DCE,PET)|")
    print(f"Target ground-truth K1 correlation: {TARGET_K1_CORR}")
    print(f"Trials: {N_TRIALS}  (n_jobs={N_JOBS})")
    print(f"Epochs -- search: {EPOCHS_SEARCH}, confirm: {EPOCHS_CONFIRM}")
    print(f"Seeds per trial: {SEEDS_PER_TRIAL}")
    print(f"Confirmation seeds: {N_SEEDS_CONFIRM}")
    print(f"Final candidates: {N_CONFIRM}")
    print(f"torch.set_num_threads({TORCH_NUM_THREADS})")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=0),
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]

    print(f"\n=== Optuna search finished: {len(study.trials)} trials ===")

    if not completed:
        print("No completed trials.")
        return

    best = study.best_trial

    print(f"Best search-time trial: trial_{best.number}, value={best.value:.3f}")
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