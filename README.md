# assuta — kinetic parameter estimation (PINN / voxelwise / MVE)

Estimates DCE-MRI (1-tissue compartment) and dynamic PET (irreversible
2-tissue compartment) kinetic parameters (K1, k2, k3) via PINN, classical
voxelwise NLLS fitting, and , validated against a literature-grounded simulation.

## Install

```
pip install -e .
```

Installs the `assuta` console command and makes every package below
importable from anywhere, not just from inside the repo directory.

## Quick start: the `assuta` CLI

Run on a built-in simulated phantom (no real data needed):
```
assuta --source simulation --modality dce --method voxelwise
assuta --source simulation --modality pet --method pinn --shape 32 32 8
assuta --source simulation --modality dce --method bayesian --n-ensemble 5
```

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

## Package layout

```
core/           PINN (f_x / f_x_bpinn, Ks_net, PINNLoss, Trainer,
                 uncertainty.py for the B-PINN deep-ensemble uncertainty)
dce/            DCE-MRI pipeline (preprocessing, analysis: pinn/voxelwise)
pet/            PET pipeline (preprocessing, analysis: pinn/voxelwise)
simulation/     Everything for validating the above against known ground truth
                (code only -- see simulation_data/ for its outputs):
  phantom.py               prostate-shaped ellipsoid + random cancer lesion
  circle_phantom.py        generic background/big-circle/small-circle-inside phantom
  kinetics_literature.py   literature-derived K1/k2/k3 distributions
  forward_models.py        AIF models, 1TCM/2TCM forward simulation, composite noise
  voxelwise_pet.py         PET voxelwise NLLS fitting (1TCM and 2TCM)
  validate.py               end-to-end voxelwise/PINN validation against ground truth
  pinn_hyperparam_sweep.py PINN hyperparameter search (see Results below)
  train_all_models.py      trains the MVE ensemble, saves to simulation_data/
  uncertainty_correlation.py  DCE-vs-PET B-PINN uncertainty correlation,
                            low-uncertainty-subset / range-restriction check
simulation_data/  All generated data / run outputs (NOT source code) --
                  phantoms/, trained_models/, validation_runs/. See its own README.md.
assuta_cli.py                The `assuta` console command's entry point.
run_uncertainty_analysis.py  Standalone script: run the B-PINN uncertainty
                              analysis on your OWN real DCE/PET data.
```

Every existing top-level package (`core`, `dce`, `pet`, `VAE_initi`,
`simulation`, `registration`, `io_utils`) is kept as its own top-level
import path rather than nested under a single `assuta/` directory --
this avoids rewriting every `from core.X import Y`-style import built up
across the project's history, while still making the whole thing
pip-installable and giving it one CLI entry point.

## Key defaults (set from the hyperparameter sweeps this session — see
`simulation/pinn_hyperparam_sweep.py` and the docstrings in
`core/model.py` / `core/train.py` / `VAE_initi/mve.py` for the full
search and multi-seed confirmation behind each):

| component | parameter | value | note |
|---|---|---|---|
| PINN `f_x` | `omega_0` | 1.0 | collapses above ~10; loss diverges by 30-50 |
| PINN `f_x` | `hidden_size` | 40 | no bottleneck; bottleneck didn't help at any width tried |
| PINN training | schedule | pure Adam, full epoch count | a two-phase Adam+L-BFGS schedule was tested and REMOVED -- Adam-only matched or beat it every time |
| PINN | `causality_eps_final` | 2000 | the single clearest win found this session -- also flips k2's correlation sign, not just magnitude |
| PINN loss weights | `physics_weight`, `tac_consistency_weight`, `reg_weight` | 0.01, None(→0.01), 1e-4 | unchanged — a single-seed win for `physics_weight=0.02` did NOT hold up across seeds |
| `MVE` encoder | `omega_0` | 10.0 | different regime from the PINN's `f_x` — don't conflate the two |
| `digit_scale_normalize` | `offset` | 3 | divide TAC by `10^(digits+offset)`; per-curve min-max erases K1's amplitude signal, this doesn't |
| `Trainer` | `grad_clip` | 1.0 | **critical**: `grad_clip=0.0` silently zeros every gradient (found this session) |

## Known issues fixed / flagged this session (see
`simulation_validation_report.md` for the full write-up with numbers)

- **Fixed**: `Ks_net.Ks_raw` was never registered as an `nn.Parameter` — the PINN crashed on its first forward pass.
- **Fixed**: `Trainer`'s old `grad_clip=0.0` default silently zeroed all gradients.
- **Fixed**: `pet/analysis.py` was passing frame *durations* as timestamps (now `np.cumsum`'d, matching `dce/analysis.py`).
- **Removed**: the two-phase Adam+L-BFGS training schedule -- pure Adam matched or beat it in every hyperparameter sweep comparison.
- **Removed**: the VAE model family (SineBetaVAE, DynamicBetaVAE) and the PCA baseline -- both underperformed MVE and voxelwise/PINN throughout validation.
- **Flagged, not fixed**: the 1TCM `Ks_net`'s second parameter is used inconsistently as `ve` in one function and `kep` directly in another — affects the DCE PINN's "k2" output specifically. 2TCM does not have this issue.
- **Flagged**: `Trainer.train()`'s `z_slices` loop doesn't actually slice per z — every iteration retrains on the whole passed-in volume.

## B-PINN uncertainty

Any pipeline (`dce.analysis.pipeline`, `pet.analysis.pipeline`, the
`assuta` CLI, and the batch executors) now supports `method="bayesian"`
alongside `"pinn"`/`"voxelwise"` -- it computes the same K1/k2(/k3)
point estimate as `"pinn"`, plus a per-voxel uncertainty map, and saves
`K_mean.nii`, `K_uncertainty.nii`, and `K_uncertainty_demeaned.nii`.
Tunable via `n_ensemble` (default 5) and `dropout_p` (default 0.1);
runtime scales ~linearly with `n_ensemble`.

Under the hood, `core/uncertainty.py`'s `estimate_with_uncertainty()`
trains a deep ensemble of Sine B-PINNs (MC-Dropout + independent
re-inits). Important caveat found via direct inspection: voxels within
one ensemble run share the same `f_x` trunk, so a run's random seed can
shift the WHOLE scan's estimate together -- `K_uncertainty_demeaned`
(not raw `K_uncertainty`) isolates the true per-voxel signal from that
per-subject/per-fit confound. See the function's docstring for the full
explanation.

- `simulation/uncertainty_correlation.py` runs this on simulated DCE +
  PET data (known ground truth) and checks whether uncertainty is
  correlated across modalities, and whether restricting to
  low-uncertainty voxels is a genuine accuracy gain or a statistical
  range-restriction artifact (checks RMSE alongside correlation to tell
  them apart).
- `run_uncertainty_analysis.py` (repo root) is the real-data version of
  the same analysis -- run it on your own preprocessed DCE/PET subject
  directories; see its docstring for usage, co-registration
  requirements, and performance notes.

## Reproducing the validation

```
python simulation/validate.py              # voxelwise/PINN vs ground truth, DCE + PET
python simulation/pinn_hyperparam_sweep.py  # PINN structure/omega_0/loss-weight sweep
python simulation/train_all_models.py       # train the MVE ensemble, save to trained_models/
python simulation/uncertainty_correlation.py  # B-PINN uncertainty analysis on simulated data
```

Everything is CPU-runnable at reduced scale (small phantoms / few
z-slices / modest epoch counts) for quick iteration; see comments in
each script for how to scale up to the full 64x64x20 volume and larger
training budgets.
