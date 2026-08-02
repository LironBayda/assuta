# simulation_data/

Output data for everything under `simulation/` (the code) writes here,
not `/tmp` or scattered ad-hoc paths -- keeps generated data separate
from source and out of the package itself.

- `phantoms/` -- saved phantom volumes / ground-truth parameter maps
- `trained_models/` -- VAE/MVE weights + results.json from
  `simulation/train_all_models.py`
- `validation_runs/` -- output of `simulation/validate.py` and
  `simulation/pinn_hyperparam_sweep.py` (K.nii, K_mean.nii,
  K_uncertainty.nii, results JSON, etc.)

Nothing in here is meant to be committed except this README and
.gitkeep -- add the rest to .gitignore if this becomes a real repo.
