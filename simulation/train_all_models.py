"""
Train the MVE (Mean-Variance Estimation) ensemble -- the supervised,
calibrated-uncertainty kinetic-parameter estimator in this repo -- on a
simulated population, score it against known ground-truth K1/k2, and
save weights + a results summary to simulation_data/trained_models/.

(This used to also train SineBetaVAE and DynamicBetaVAE; both were
removed from the repo. MVE remains because it's a distinct, non-VAE
supervised model that was found this session to be by far the strongest
and best-calibrated estimator of the methods tried -- see
simulation_validation_report.md.)

Usage: python simulation/train_all_models.py
Output: simulation_data/trained_models/mve_member_{i}.pt (weights) and
        simulation_data/trained_models/results.json (config +
        correlations + uncertainty calibration).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VAE_initi.dataset import SimulatedTACDataset, digit_scale_normalize
from VAE_initi.mve import MVEEnsemble

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "simulation_data", "trained_models",
)
os.makedirs(OUT_DIR, exist_ok=True)

# -------- config (best-found settings from this session) --------
LENGTH = 35
DT_SEC = 9.5
DEVICE = "cpu"
# Sample-count sweep this session found MVE benefits substantially from
# more training samples at the SAME k1/k2 range (K1 r: 0.45 at n=1500 ->
# 0.65 at n=6000-15000, then plateaus) -- unlike widening the range,
# which hurt (K1 r dropped to ~0.19 at 3-5x range).
MVE_N_SAMPLES = 8000


def main():
    print("=== MVEEnsemble ===")
    t0 = time.time()
    ds_mve = SimulatedTACDataset(n_samples=MVE_N_SAMPLES, length_range=(LENGTH, LENGTH),
                                  dt_range_sec=(DT_SEC, DT_SEC), seed=1)
    raw_mve = np.stack([tac.numpy() for tac in ds_mve.tacs])
    X_mve = digit_scale_normalize(raw_mve)
    Y_np = np.stack([ds_mve.k1, ds_mve.k2], axis=1)

    ens = MVEEnsemble(n_members=3, input_dim=LENGTH, n_params=2, hidden_dim=64)
    ens.fit(X_mve, Y_np, epochs=300, lr=5e-4, batch_size=256, device=DEVICE)
    for i, m in enumerate(ens.members):
        torch.save(m.state_dict(), os.path.join(OUT_DIR, f"mve_member_{i}.pt"))

    mean, aleatoric, epistemic, total = ens.predict(X_mve)
    r_k1 = float(pearsonr(mean[:, 0], ds_mve.k1)[0])
    r_k2 = float(pearsonr(mean[:, 1], ds_mve.k2)[0])
    err1 = np.abs(mean[:, 0] - ds_mve.k1)
    calib_k1 = float(pearsonr(total[:, 0], err1)[0])

    results = {
        "config": dict(n_samples=MVE_N_SAMPLES, length=LENGTH, dt_sec=DT_SEC),
        "MVEEnsemble": dict(
            time_s=round(time.time() - t0, 1),
            n_samples=MVE_N_SAMPLES,
            K1_corr=r_k1, k2_corr=r_k2,
            uncertainty_calibration_K1=calib_k1,
            n_members=3,
            weights=[f"mve_member_{i}.pt" for i in range(3)],
        ),
    }
    print(f"  K1 r={r_k1:.3f}  k2 r={r_k2:.3f}  uncertainty-calibration(K1)={calib_k1:.3f}  "
          f"[{time.time()-t0:.1f}s]")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWeights + results.json saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
