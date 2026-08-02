"""
Hyperparameter sweep for the PINN (core/pinn.py PhysicsInformedNN /
core/train.py Trainer), scored against the known ground truth in the
prostate phantom simulation. Covers three axes:

  1. Training schedule (Adam-only vs. Adam+L-BFGS, learning rate)
  2. f_x structure (hidden_size, with/without a bottleneck) and omega_0
     (SIREN frequency scale)
  3. Loss term weights (physics_weight, tac_consistency_weight, reg_weight)

Findings baked into the repo defaults as of this sweep:
  - Pure Adam for the full epoch count (no L-BFGS phase) matched or
    beat every two-phase Adam+L-BFGS schedule tried. The L-BFGS phase
    has since been removed from core/train.py's Trainer entirely.
  - omega_0: swept 0.1-50. Correlation peaks around omega_0=1
    (mean K1 r=0.81-0.83 across seeded phantoms) and DEGRADES SHARPLY
    above ~10 -- by omega_0=30-50 the loss itself diverges (loss=5.7 and
    88.1 respectively, vs ~1 elsewhere), i.e. genuine training
    instability, not just a worse optimum. **omega_0=1.0 is now the
    default** (up from the old 0.1) in f_x/f_x_bpinn (core/model.py),
    PhysicsInformedNN (core/pinn.py), and Trainer (core/train.py). This
    is a markedly different value than the omega_0=10 found best for
    the unrelated MVE fixed-length curve encoder --
    f_x is a continuous-time trunk evaluated at arbitrary t, a different
    regime, and the two sweeps land on different optima as expected.
  - hidden_size: swept 20-100, with and without a bottleneck
    (bottleneck_size = hidden_size/4). Best: hidden_size=40, NO
    bottleneck (mean K1 r=0.827 vs 0.810 at hidden_size=20, vs 0.808 for
    the best bottleneck variant tried, h60/b15) -- bottlenecking did not
    help at any width tested. Beyond hidden_size=40 performance drifts
    back down (h60/h80/h100 all sit around 0.86-0.87 on the single-seed
    check, below h40's 0.90). **hidden_size=40 is now the default.**
  - Loss weights: swept physics_weight in {0.001, 0.005, 0.01(default),
    0.02, 0.1}, tac_consistency_weight in {0.001, 0.1, 1(->default via
    None), 5, 20}, reg_weight in {1e-5, 1e-4(default), 1e-3, 1e-2}, at
    the new best structure (omega_0=1.0, hidden_size=40). On a SINGLE
    seed, physics_weight=0.02 (r=0.894) and reg_weight=0.01 (r=0.889)
    both looked better than the defaults (r=0.877) -- but re-run across
    3 seeds, physics_weight=0.02 turned out WORSE on average (mean
    0.774 vs 0.826 for the defaults) -- a single-seed false lead caught
    only by the multi-seed check. reg_weight=0.01 held up marginally
    (0.829 vs 0.826) but the difference is within noise (std ~0.05).
    Raising tac_consistency_weight to 1 or 5 was clearly worse at every
    seed tried (r=0.55-0.76). **No loss-weight defaults were changed**
    -- nothing tested robustly beat physics_weight=0.01,
    tac_consistency_weight=None (defaults to physics_weight),
    reg_weight=1e-4 once checked across seeds.

Run this file directly to reproduce the sweep (takes several minutes on
CPU for the full ranges below); adjust N_SEEDS / EPOCHS to trade off
thoroughness vs. runtime. Note the loss-weight section specifically
demonstrates why the multi-seed check matters -- if you only run one
seed, physics_weight=0.02 will look like a real win.
"""
import os
import sys
import time

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.phantom import build_prostate_phantom
from simulation.kinetics_literature import DCE_1TCM_PARAMS, sample_param_maps
from simulation.forward_models import parker_aif, simulate_1tcm_volume
from core.train import Trainer

SHAPE = (32, 32, 8)
Z_IDX = [3, 4]
EPOCHS = 300
N_SEEDS = 3


def build_case(seed):
    info = build_prostate_phantom(SHAPE, seed=seed)
    label = info.label
    gt = sample_param_maps(label, DCE_1TCM_PARAMS, ["K1", "k2"], seed=seed)
    t = np.cumsum(np.asarray([0.1522] * 35))
    aif = parker_aif(t)
    noisy, _ = simulate_1tcm_volume(t, aif, gt["K1"], gt["k2"], noise_std=0.03, rng=seed)
    img = noisy[:, :, :, Z_IDX]
    lab = label[:, :, Z_IDX]
    gtK1 = gt["K1"][:, :, Z_IDX]
    return t, aif, img, lab, gtK1


def run_config(tag, seeds=range(N_SEEDS), **kwargs):
    rs = []
    for seed in seeds:
        t, aif, img, lab, gtK1 = build_case(seed)
        affine = np.eye(4)
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "simulation_data", "validation_runs", "_sweep_scratch", f"{tag}_{seed}",
        )
        os.makedirs(save_path, exist_ok=True)
        trainer = Trainer(c_p=aif, num_of_compartment=1, t=t, device="cpu",
                           affine=affine, save_path=save_path, epochs=EPOCHS, **kwargs)
        ks_out, hist = trainer.train(img, z_slices=[0])
        K1 = ks_out[0]
        m = lab > 0
        rs.append(pearsonr(K1[m], gtK1[m])[0])
    print(f"{tag:30s} {kwargs}\n"
          f"{'':30s} K1 corr mean={np.mean(rs):.3f} std={np.std(rs):.3f}")
    return float(np.mean(rs))


if __name__ == "__main__":
    t0 = time.time()
    print("=== Training schedule ===")
    run_config("adam_only_default", seeds=[0])
    # (two-phase Adam+L-BFGS comparison removed along with the L-BFGS
    # phase itself, which the result below made obsolete)

    print("\n=== omega_0 (wide sweep, up to 50) ===")
    for om in [1, 2, 5, 10, 20, 30, 50]:
        run_config(f"omega{om}", seeds=[0], omega_0=float(om), hidden_size=20)

    print("\n=== hidden_size (up to 100), no bottleneck ===")
    for hs in [20, 40, 60, 80, 100]:
        run_config(f"nb_h{hs}", seeds=[0], omega_0=1.0, hidden_size=hs)

    print("\n=== hidden_size with bottleneck (size = hidden/4) ===")
    for hs, bn in [(40, 10), (60, 15), (80, 20), (100, 25), (100, 10)]:
        run_config(f"bn_h{hs}_b{bn}", seeds=[0], omega_0=1.0, hidden_size=hs, bottleneck_size=bn)

    print("\n=== Best structure, multi-seed confirmation ===")
    run_config("h20_confirm", seeds=range(N_SEEDS), omega_0=1.0, hidden_size=20)
    run_config("h40_confirm", seeds=range(N_SEEDS), omega_0=1.0, hidden_size=40)
    run_config("h60_b15_confirm", seeds=range(N_SEEDS), omega_0=1.0, hidden_size=60, bottleneck_size=15)

    print("\n=== Loss weights (at best structure: omega_0=1.0, hidden_size=40) ===")
    run_config("loss_default", seeds=[0], omega_0=1.0, hidden_size=40)
    for pw in [0.001, 0.005, 0.02, 0.1]:
        run_config(f"physw_{pw}", seeds=[0], omega_0=1.0, hidden_size=40, physics_weight=pw)
    for tcw in [0.001, 0.1, 1, 5]:
        run_config(f"tacw_{tcw}", seeds=[0], omega_0=1.0, hidden_size=40, tac_consistency_weight=tcw)
    for rw in [1e-5, 1e-3, 1e-2]:
        run_config(f"regw_{rw}", seeds=[0], omega_0=1.0, hidden_size=40, reg_weight=rw)

    print("\n=== Loss weights, multi-seed confirmation (this is where physics_weight=0.02 falls apart) ===")
    run_config("loss_base_confirm", seeds=range(N_SEEDS), omega_0=1.0, hidden_size=40)
    run_config("physw002_confirm", seeds=range(N_SEEDS), omega_0=1.0, hidden_size=40, physics_weight=0.02)
    run_config("regw001_confirm", seeds=range(N_SEEDS), omega_0=1.0, hidden_size=40, reg_weight=0.01)

    print("\n=== causality_eps_final / tac_consistency_weight / grad_clip (lr=0.01) ===")
    common = dict(lr=0.01)
    for ce in [0, 1, 10, 100, 1000, 2000, 5000, 10000]:
        run_config(f"ce{ce}", seeds=[0], causality_eps_final=float(ce), **common)
    print("\n--- tac_consistency_weight at causality_eps_final=2000, multi-seed ---")
    run_config("tcw_default", seeds=range(N_SEEDS), causality_eps_final=2000.0, **common)
    run_config("tcw0.05", seeds=range(N_SEEDS), causality_eps_final=2000.0,
               tac_consistency_weight=0.05, **common)
    run_config("tcw0.5", seeds=range(N_SEEDS), causality_eps_final=2000.0,
               tac_consistency_weight=0.5, **common)
    print("\n--- grad_clip at causality_eps_final=2000, multi-seed ---")
    run_config("gc_default_1.0", seeds=range(N_SEEDS), causality_eps_final=2000.0,
               grad_clip=1.0, **common)
    run_config("gc_0.5", seeds=range(N_SEEDS), causality_eps_final=2000.0,
               grad_clip=0.5, **common)
    run_config("gc_none", seeds=range(N_SEEDS), causality_eps_final=2000.0,
               grad_clip=None, **common)

    print(f"\nTotal sweep time: {time.time()-t0:.1f}s")

