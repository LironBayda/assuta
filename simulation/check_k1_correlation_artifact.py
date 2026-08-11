"""
Diagnostic: is the PET-K1 <-> DCE-K1 correlation you're seeing real
physiology, or an artifact of shared estimation (epistemic) error?

Logic:
  1. Build a phantom where PET-K1 and DCE-K1 are sampled INDEPENDENTLY
     from their literature distributions (true correlation ~ 0 by
     construction -- if your kinetics_literature sampler already
     correlates them, this whole test is meaningless; check that first,
     see ASSUMPTION note below).
  2. Fit both modalities' PINNs on the same phantom to get recovered
     K1 maps.
  3. Compare:
       corr(GT_PET_K1,        GT_DCE_K1)        -- should be ~0
       corr(recovered_PET_K1, recovered_DCE_K1) -- the number you
                                                    actually observed
     A gap between these two is direct evidence of shared estimation
     bias (epistemic), since the ground truth had no coupling to begin
     with.
  4. Stratify the recovered-vs-recovered correlation by per-voxel fit
     uncertainty (here: MC-dropout-free proxy = residual TAC fit error;
     swap in an ensemble/MC-dropout std if you have one -- that's a
     better epistemic-uncertainty proxy than residual error, which
     conflates epistemic and aleatoric). If the correlation is
     concentrated in the high-uncertainty voxels, that's a strong
     signature of a shared-estimation-error artifact rather than real
     coupled physiology.

ASSUMPTIONS TO FIX BEFORE RUNNING (I don't have your PET-side module
names / signatures, only DCE's from your earlier sweep script):
  - `simulate_2tcm_volume` and `PET_2TCM_PARAMS` are guessed names,
    mirroring `simulate_1tcm_volume` / `DCE_1TCM_PARAMS`. Update the
    import + call below to match whatever your actual PET forward
    model / param-sampling functions are called.
  - Whatever function samples GT K1/k2/k3 per phantom for PET and DCE
    must sample them INDEPENDENTLY (no shared RNG stream reused
    across modalities, no shared spatial correlation structure
    injected) for step 3's "GT correlation ~ 0" baseline to be
    meaningful. If they're NOT independent by design, this test can't
    distinguish "artifact" from "correlation you deliberately built
    in" -- check simulation/kinetics_literature.py before trusting the
    result.
  - `Trainer(..., num_of_compartment=...)` args mirrored from your
    sweep script; adjust per-modality kwargs (e.g. PET likely wants
    num_of_compartment=2 for k1/k2/k3, given your DCE/PET = 1TCM/2TCM
    convention) to whatever your actual best-known / tuned config is
    -- this script is about the correlation diagnostic, not about
    re-doing your hyperparameter search.
"""
import os
import sys

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.phantom import build_prostate_phantom
from simulation.kinetics_literature import DCE_1TCM_PARAMS, PET_2TCM_PARAMS, sample_param_maps
from simulation.forward_models import parker_aif, simulate_1tcm_volume, simulate_2tcm_volume
from core.train import Trainer

SHAPE = (64, 64, 8)
Z_IDX = [3, 4]
EPOCHS = 100
N_SEEDS = 10  # more seeds = more voxels = more stable correlation estimate

# fill in with your actual best/tuned hyperparameters per modality
DCE_TRAIN_KWARGS = dict(hidden_size=40, omega_0=1.0, num_of_compartment=1)
PET_TRAIN_KWARGS = dict(hidden_size=40, omega_0=1.0, num_of_compartment=2)


def build_dce_case(seed):
    info = build_prostate_phantom(SHAPE, seed=seed)
    label = info.label
    gt = sample_param_maps(label, DCE_1TCM_PARAMS, ["K1", "k2"], seed=seed)
    t = np.cumsum(np.asarray([0.1522] * 35))
    aif = parker_aif(t)
    noisy, _ = simulate_1tcm_volume(t, aif, gt["K1"], gt["k2"], noise_std=0.1, rng=seed)
    img = noisy[:, :, :, Z_IDX]
    return t, aif, img, label[:, :, Z_IDX], gt["K1"][:, :, Z_IDX]


def build_pet_case(seed):
    # Uses a DIFFERENT seed offset so PET's independent literature draw
    # doesn't share the RNG stream with DCE's draw above -- if your
    # sampler derives noise/params from `seed` deterministically, reusing
    # the same seed for both modalities could itself inject spurious
    # correlation. Adjust if your sampler already guarantees independence.
    pet_seed = seed + 100_000
    info = build_prostate_phantom(SHAPE, seed=seed)  # SAME anatomy as DCE case
    label = info.label
    gt = sample_param_maps(label, PET_2TCM_PARAMS, ["K1", "k2", "k3"], seed=pet_seed)
    t = np.cumsum(np.asarray([5.0] * 35))  # placeholder PET frame timing -- replace with yours
    aif = parker_aif(t)  # placeholder -- replace with your PET input function if different
    noisy, _ = simulate_2tcm_volume(t, aif, gt["K1"], gt["k2"], gt["k3"], noise_std=0.1, rng=pet_seed)
    img = noisy[:, :, :, Z_IDX]
    return t, aif, img, label[:, :, Z_IDX], gt["K1"][:, :, Z_IDX]


def fit_k1(t, aif, img, label, num_of_compartment, train_kwargs, tag):
    save_path = os.path.join("simulation_data", "validation_runs", "_corr_check", tag)
    os.makedirs(save_path, exist_ok=True)
    affine = np.eye(4)
    trainer = Trainer(c_p=aif, t=t, device="cpu", affine=affine, save_path=save_path,
                       epochs=EPOCHS, windowed=True, **train_kwargs)
    ks_out, hist = trainer.train(img, z_slices=[0])
    K1_recovered = ks_out[0]
    # residual TAC-fit error as a crude per-voxel uncertainty proxy --
    # replace with MC-dropout / ensemble std over K1 if you have one,
    # that separates epistemic from aleatoric far better than residual
    # error alone (residual error mixes in real measurement noise too)
    residual = hist.get("final_residual_per_voxel") if isinstance(hist, dict) else None
    return K1_recovered, residual


def main():
    gt_pet_k1, gt_dce_k1 = [], []
    rec_pet_k1, rec_dce_k1 = [], []
    uncertainty = []
    mask_all = []

    for seed in range(N_SEEDS):
        t_d, aif_d, img_d, lab_d, gtK1_d = build_dce_case(seed)
        t_p, aif_p, img_p, lab_p, gtK1_p = build_pet_case(seed)

        recK1_d, resid_d = fit_k1(t_d, aif_d, img_d, lab_d, 1, DCE_TRAIN_KWARGS, f"dce_{seed}")
        recK1_p, resid_p = fit_k1(t_p, aif_p, img_p, lab_p, 2, PET_TRAIN_KWARGS, f"pet_{seed}")

        m = (lab_d > 0) & (lab_p > 0)
        gt_dce_k1.append(gtK1_d[m])
        gt_pet_k1.append(gtK1_p[m])
        rec_dce_k1.append(recK1_d[m])
        rec_pet_k1.append(recK1_p[m])
        if resid_d is not None and resid_p is not None:
            uncertainty.append(resid_d[m] + resid_p[m])
        mask_all.append(m)

    gt_dce_k1 = np.concatenate(gt_dce_k1)
    gt_pet_k1 = np.concatenate(gt_pet_k1)
    rec_dce_k1 = np.concatenate(rec_dce_k1)
    rec_pet_k1 = np.concatenate(rec_pet_k1)

    r_gt, p_gt = pearsonr(gt_pet_k1, gt_dce_k1)
    r_rec, p_rec = pearsonr(rec_pet_k1, rec_dce_k1)

    print("=== GT vs recovered correlation ===")
    print(f"GT   PET-K1 vs DCE-K1:        r={r_gt:.3f}  p={p_gt:.3g}  (n={len(gt_dce_k1)})")
    print(f"Recovered PET-K1 vs DCE-K1:   r={r_rec:.3f}  p={p_rec:.3g}  (n={len(rec_dce_k1)})")
    gap = r_rec - r_gt
    print(f"Gap (recovered - GT): {gap:+.3f}", end="  ")
    if abs(gap) > 0.15 and abs(r_gt) < 0.1:
        print("-- sizeable correlation appeared that ISN'T in the ground truth: "
              "consistent with a shared-estimation (epistemic) artifact.")
    elif abs(gap) < 0.05:
        print("-- recovered correlation tracks GT closely: no evidence of an artifact here.")
    else:
        print("-- ambiguous; check the uncertainty-stratified breakdown below.")

    if uncertainty:
        uncertainty = np.concatenate(uncertainty)
        # split into uncertainty terciles
        q1, q2 = np.percentile(uncertainty, [33, 66])
        low = uncertainty <= q1
        high = uncertainty >= q2
        r_low, _ = pearsonr(rec_pet_k1[low], rec_dce_k1[low])
        r_high, _ = pearsonr(rec_pet_k1[high], rec_dce_k1[high])
        print("\n=== Stratified by fit uncertainty (residual TAC error proxy) ===")
        print(f"Low-uncertainty voxels  (best fits): r={r_low:.3f}  (n={low.sum()})")
        print(f"High-uncertainty voxels (worst fits): r={r_high:.3f}  (n={high.sum()})")
        if r_high - r_low > 0.15:
            print("-- correlation concentrated in poorly-fit voxels: "
                  "consistent with shared estimation error, not physiology.")
    else:
        print("\n(no per-voxel residual/uncertainty returned by Trainer.train() -- "
              "add that to hist, or swap in an MC-dropout/ensemble K1 std, "
              "to run the stratified check.)")


if __name__ == "__main__":
    main()
