# assuta — kinetic parameter estimation (PINN / voxelwise / MVE)

Estimates DCE-MRI (1-tissue compartment) and dynamic PET (irreversible
2-tissue compartment) kinetic parameters (K1, k2, k3) via PINN, classical
voxelwise NLLS fitting, and MVE (a supervised, calibrated-uncertainty
network), validated against a literature-grounded simulation.

## Install

```
pip install -e .
```

## Quick start

```
assuta --source simulation --modality dce --method voxelwise   # no real data needed
assuta --source real --path /data/sub01/dce --modality dce --method pinn
```

That's the whole interface for the common case -- everything below is
either background/rationale or advanced options you likely don't need
to touch. `assuta --help` for the full option list.

Run on a real (preprocessed) subject directory:
```
assuta --source real --path /data/sub01/pet --modality pet --method pinn
assuta --source real --path /data/sub01/dce --modality dce --method voxelwise
```

Run on a whole folder of subjects at once (expects `root/sub*/dce`
and/or `root/sub*/pet`; any subject missing the relevant modality
subfolder is skipped, not an error):
```
assuta --source real --batch-root /data --modality dce --method pinn
assuta --source real --batch-root /data --modality both --method voxelwise
```

`--help` for the full option list (`assuta_cli.py` if not installed).

## Background

**Irreversible 2TCM (`num_of_compartment=2`, k4=0) is the default and
literature-validated model for PSMA PET specifically** -- not an
arbitrary choice. Directly tested against two alternatives (a reversible
1TCM and a reversible 2TCM) on real ⁶⁸Ga-PSMA-11 dynamic PET data,
irreversible 2TCM won by goodness-of-fit and information-loss criteria,
"consistently appropriate across prostatic zones" (Smith et al., EJNMMI
Research 2024, PMC10781928). A second, independent cohort gives the
mechanistic reason: for prostate cancer lesions, k4≈0 because PSMA
binding/internalization is predominantly irreversible (PMC10105814,
EJNMMI Research 2023). See `simulation/kinetics_literature.py` for the
full citation list behind every parameter range used in the simulation.

This repo used to also include a VAE model family (SineBetaVAE,
DynamicBetaVAE), a PCA baseline, and a Bayesian/deep-ensemble PINN
("B-PINN") uncertainty-estimation pathway. All were removed -- the VAE
and PCA baselines underperformed MVE and voxelwise/PINN throughout
validation, and the B-PINN ensemble was found too computationally
expensive for the calibration quality it delivered relative to MVE; see
`simulation_validation_report.md` for the numbers behind both calls.

## Package layout

```
core/           PINN (f_x, Ks_net, PINNLoss, Trainer), windowed.py
                 (independent per-window/per-slice PINN training)
dce/            DCE-MRI pipeline (preprocessing, analysis: pinn/voxelwise)
pet/            PET pipeline (preprocessing, analysis: pinn/voxelwise)
VAE_initi/      Shared Sine/SIREN infra (model.py) + TAC simulation
                 (dataset.py) + MVE (mve.py) -- name kept for history,
                 no longer holds VAE models (see note above)
simulation/     Everything for validating the above against known ground truth
                (code only -- see simulation_data/ for its outputs):
  phantom.py               prostate-shaped ellipsoid + random cancer lesion
  circle_phantom.py        generic background/big-circle/small-circle-inside phantom
  kinetics_literature.py   literature-derived K1/k2/k3 distributions, plus
                            sample_correlated_adc() for a diffusion-MRI
                            (ADC) companion map correlated with PET uptake
                            (rho~-0.73, from Domachevsky et al. 2018)
  forward_models.py        AIF models, 1TCM/2TCM forward simulation, modality noise
  voxelwise_pet.py         PET voxelwise NLLS fitting (1TCM and 2TCM)
  validate.py               end-to-end voxelwise/PINN validation against ground truth
  pinn_hyperparam_sweep.py PINN hyperparameter search (see below)
  train_all_models.py      trains the MVE ensemble, saves to simulation_data/
  noise_sparsity_experiment.py  voxelwise vs tanh/Sine-PINN sweep across
                            noise level and temporal sampling density
simulation_data/  All generated data / run outputs (NOT source code) --
                  phantoms/, trained_models/, validation_runs/. See its own README.md.
assuta_cli.py                The `assuta` console command's entry point.
```

Every existing top-level package (`core`, `dce`, `pet`, `VAE_initi`,
`simulation`, `registration`, `io_utils`) is kept as its own top-level
import path rather than nested under a single `assuta/` directory --
this avoids rewriting every `from core.X import Y`-style import built up
across the project's history, while still making the whole thing
pip-installable and giving it one CLI entry point.

## Key defaults (set from hyperparameter sweeps -- see
`simulation/pinn_hyperparam_sweep.py` and the docstrings in
`core/model.py` / `core/train.py` / `core/windowed.py` / `VAE_initi/mve.py`
for the full search and multi-seed confirmation behind each):

| component | parameter | value | note |
|---|---|---|---|
| 1TCM `Ks_net` | parameterization | (Ktrans, Ve) everywhere | see "1TCM parameterization fix" below -- was internally inconsistent |
| `Trainer` | `hidden_size` | 10 (was 40) | multi-seed: better AND more consistent K1, ~40% faster |
| `Trainer` | `physics_weight` | 100.0 (was 0.01) | flips derived-kep correlation from reliably negative to reliably positive, combined with the parameterization fix above |
| `Trainer` | `physics_weight_start` | 0.01 (anneals up to `physics_weight`) | holding physics_weight=100 constant from epoch 1 caused real K1 instability (one seed collapsed 0.65->0.20); annealing fixed it |
| PINN `f_x` | `omega_0` | 1.0 | 0.1 tested as an alternative under the new recipe and was consistently WORSE (kep flipped negative every seed) |
| `core/windowed.py` | `axis`, `window_size` | `"xy"`, 16 | windowed (16x16 patches) is now the CLI/pipeline default for `method="pinn"`; pass `--no-windowed` for the old whole-image behavior |
| `config.PINN["epochs"]` | epochs | 1000 (was 200) | `physics_weight`'s anneal ramps across the full schedule -- cutting epochs short cuts the anneal short too; still improving at 2000 in testing |
| PINN | `causality_eps_final` | 2000 | the clearest single win found early on -- also flips k2's correlation sign, not just magnitude |
| `MVE` encoder | `omega_0` | 10.0 | different regime from the PINN's `f_x` — don't conflate the two |
| `digit_scale_normalize` | `offset` | 3 | divide TAC by `10^(digits+offset)`; per-curve min-max erases K1's amplitude signal, this doesn't |
| `Trainer` | `grad_clip` | 1.0 | **critical**: `grad_clip=0.0` silently zeros every gradient |

**CAVEAT**: the `hidden_size`/`physics_weight`/`physics_weight_start`/
windowed-default recipe above was found and multi-seed-tested on
**1TCM/DCE data specifically**. It's still the `Trainer` default for
2TCM/PET too (shared architecture), but hasn't been independently
verified there -- the 1TCM parameterization fix doesn't even apply to
2TCM (which was already internally consistent). If PET results look
worse than before, override `hidden_size=40`, `physics_weight=0.01`,
`physics_weight_start=None`, `windowed=False` explicitly (the old
defaults) until 2TCM gets its own sweep.

## 1TCM parameterization fix

`core/model.py`'s `Ks_net` (1TCM case) used to treat its second fitted
parameter as `kep` directly in the physics/ODE residual
(`get_x_1tcm`), but as `Ve` (with `kep = Ktrans/Ve`) in the closed-form
solution used for the tac-consistency loss target
(`convolve_1cm_for_minimize`) -- since BOTH are computed every single
training step, the parameter was being pulled toward two different
physical quantities simultaneously. Fixed by unifying on `(Ktrans, Ve)`
everywhere; `kep` (if you need it) is `Ktrans / Ve`, computed downstream.
Directly tested: unifying alone (without also raising `physics_weight`)
did not change K1/k2 recovery -- the fix is correctness-motivated, not
itself the source of the accuracy improvement (that came from combining
it with the stronger, annealed `physics_weight`).

## Known issues fixed / flagged (see
`simulation_validation_report.md` for the full write-up with numbers)

- **Fixed**: `Ks_net.Ks_raw` was never registered as an `nn.Parameter` — the PINN crashed on its first forward pass.
- **Fixed**: `Trainer`'s old `grad_clip=0.0` default silently zeroed all gradients.
- **Fixed**: `pet/analysis.py` was passing frame *durations* as timestamps (now `np.cumsum`'d, matching `dce/analysis.py`) -- **and** a second bug in the same line found later: the cumsum'd result was missing a `/60.0` (config.PET["dt"] is in seconds, everything else in this codebase assumes minutes), feeding a ~60x-too-large time axis to every PET fit through this pipeline. Both are now fixed together.
- **Fixed**: `simulation/voxelwise_pet.py`'s `fit_1tcm_voxel`/`fit_2tcm_voxel` used `method="Powell"` explicitly -- same catastrophic-blowup risk on low-signal noisy voxels as the DCE case above (validated for DCE; applied here by the same reasoning, not independently re-tested for PET specifically).
- **Fixed**: a `ve = K1/k2` ratio computed in `dce/analysis.py` could blow up to absurd values when `k2` landed near its lower fitting bound (root cause of a real-data table showing `ve` values in the thousands mislabeled as "kep") -- now masked to NaN outside the physically valid `[0, 1]` range.
- **Fixed**: the 1TCM `Ks_net` parameter inconsistency (`kep` vs `Ve`) described above.
- **Removed**: the two-phase Adam+L-BFGS training schedule -- pure Adam matched or beat it in every hyperparameter sweep comparison.
- **Removed**: the VAE model family (SineBetaVAE, DynamicBetaVAE) and the PCA baseline -- both underperformed MVE and voxelwise/PINN throughout validation.
- **Removed**: the Bayesian/deep-ensemble PINN ("B-PINN") uncertainty pathway -- too computationally expensive (an ensemble of full PINN retrains) for calibration quality that consistently underperformed MVE; see the report for the comparison numbers.
- **Flagged**: `Trainer.train()`'s `z_slices` loop doesn't actually slice per z — every iteration retrains on the whole passed-in volume.

## Windowed PINN training

`core/windowed.py`'s `train_windowed()` trains independent per-window
PINNs (no shared state across windows) instead of one whole-image fit,
modeled on van Herten et al. (Medical Image Analysis, 2022), and is now
**the default for `method="pinn"`** across the CLI and both pipelines.
Two modes: `axis="xy"` (default -- small in-plane patches, `window_size`
x `window_size`, e.g. 16x16, Z kept whole) or `axis="z"` (full X,Y
resolution per window, split only along Z, e.g. a 64x64x1-style window
-- avoids the small-patch data-starvation problem at the cost of fewer,
larger, more expensive per-window fits; not independently re-verified
against `axis="xy"` under the current recipe). Available via
`--windowed`/`--no-windowed` (`--axis`, `--slice-window`,
`--window-size`) on the CLI for `--method pinn`, or by calling
`train_windowed()` directly.

## Adaptive per-voxel 1TCM-vs-2TCM model selection (PET)

`core/model_selection.py` provides a cheap, per-voxel diagnostic for
whether a PET voxel's kinetics need the full irreversible 2TCM model
(K1, k2, k3) or are adequately described by simpler 1TCM (K1, k2) --
useful specifically for short scans, where a low-k3 voxel's trapping
signature may be too weak to reliably estimate directly (see
`simulation_validation_report.md`'s Fisher-information identifiability
discussion), but where blanket-fitting 2TCM everywhere still wastes a
free parameter and its associated instability risk on voxels that don't
need it.

**Mechanism**: a small network is trained to predict k2 from a TAC,
using ONLY simulated true-1TCM (k3=0) curves. Applied to a true-2TCM
voxel, its prediction is systematically biased downward (the network
misinterprets the extra accumulation as unusually slow washout) -- that
bias, not the raw prediction or any latent representation, is the
diagnostic signal. Two alternative approaches (a VAE's latent space, and
VAE reconstruction error as a novelty/anomaly score) were tried first
and found NOT to work for this (chance-level separation, or inverted in
one case) -- see the chat history for why; the misspecification-bias
approach works because it exploits a real, well-understood statistical
phenomenon (model misspecification produces systematic bias, not just
noise) rather than hoping an unsupervised representation happens to
capture the right feature.

**Validated**: circle phantom (full realistic PSMA k3 range 0.03-0.11,
multiple seeds, varying K1/k2): highly significant separation at every
k3 level. Real prostate phantom geometry (realistic class imbalance,
65 true-2TCM vs 6007 true-1TCM voxels): ROC-AUC=0.885, sensitivity=0.83/
specificity=0.79 at the data-driven optimal cutoff. Works specifically
well in the low-k3 regime where direct nonlinear 2TCM fitting (voxelwise
and PINN, both tested extensively elsewhere in this repo) struggled.

**Usage**:
```python
# train once (schedule/noise-specific -- see the module docstring):
python simulation/train_all_models.py
# saves simulation_data/trained_models/k2_regressor_1tcm_only.pt

# use via the PET pipeline -- engine="voxelwise" (default) or "pinn":
from pet.analysis import pipeline
pipeline(subject_path, method="voxelwise", num_of_compartment=2,
         adaptive_model_selection=True, adaptive_fitting_engine="pinn",
         k2_regressor_path="simulation_data/trained_models/k2_regressor_1tcm_only.pt")
```
`adaptive_fitting_engine` controls what actually fits each model-order
class of voxels AFTER the 1TCM-vs-2TCM decision: classical NLLS
(`"voxelwise"`) or a PINN (`"pinn"`, via `core.train.Trainer` -- one fit
per class, since voxel spatial position doesn't matter to `Ks_net`
anyway). **If K1 is your main interest, "pinn" is the better choice** --
tested at 0.64 vs. 0.20 correlation in one comparison, consistent with
PINN's K1 advantage found elsewhere in this repo; voxelwise remains the
validated choice specifically for k3.

Output maps include `model_selected` (1=1TCM chosen, 2=2TCM chosen per
voxel) and `predicted_k2_diagnostic` (the raw regressor output, for
inspection/re-thresholding) alongside the usual K1/k2/k3.

**CAVEAT, worth repeating from the module docstring**: the default
cutoff (0.259) and the trained weights are specific to the exact frame
schedule and noise level they were derived on. Retrain
(`core.model_selection.train_k2_regressor`) and re-derive the cutoff via
ROC/Youden analysis on your own simulated validation data before
trusting this on a different protocol or on real data -- it has not yet
been tested on anything but simulation.

## Reproducing the validation

```
python simulation/validate.py                 # voxelwise/PINN vs ground truth, DCE + PET
python simulation/pinn_hyperparam_sweep.py     # PINN structure/omega_0/loss-weight sweep
python simulation/train_all_models.py          # train the MVE ensemble, save to trained_models/
python simulation/noise_sparsity_experiment.py # voxelwise vs tanh/Sine-PINN across noise + sparsity
```

Everything is CPU-runnable at reduced scale (small phantoms / few
z-slices / modest epoch counts) for quick iteration; see comments in
each script for how to scale up to the full 64x64x20 volume and larger
training budgets.
