
import os
import warnings

import numpy as np
import nibabel as nib
from scipy.optimize import minimize, dual_annealing
from scipy.interpolate import interp1d

try:
    from joblib import Parallel, delayed

    _HAVE_JOBLIB = True
except ImportError:  # joblib is an optional speed-up, not a hard dependency
    _HAVE_JOBLIB = False

try:
    from tqdm import tqdm

    _HAVE_TQDM = True
except ImportError:  # tqdm is an optional progress-bar dependency
    _HAVE_TQDM = False

# Global optimizer (simulated annealing over the bounded parameter space).
# More robust to local minima than derivative-based/Powell searches on
# low-signal/noisy voxels, at the cost of more function evaluations per
# voxel. Kept as the default but always overridable per-call via the
# `method` parameter on every fitting function below -- any other string
# is forwarded to scipy.optimize.minimize as its `method` argument.
DEFAULT_OPT_METHOD = "L-BFGS-B"


def exp_conv_trap(t, blood, theta):
    """
    Trapezoidal-rule convolution of `blood` with exp(-theta*t) -- the
    exact solution of dCt/dt = blood(t) - theta*Ct(t), evaluated at `t`.
    """
    t = np.asarray(t, float)
    blood = np.asarray(blood, float)
    dt = np.diff(t, prepend=t[0])
    H = np.zeros_like(blood)
    for i in range(1, len(t)):
        alpha = np.exp(-theta * dt[i])
        H[i] = alpha * H[i - 1] + 0.5 * dt[i] * (blood[i] + alpha * blood[i - 1])
    return H


def one_tcm_solution(t, K1, k2, aif_interp):
    """Forward 1-tissue-compartment solution: Ct(t) = K1 * conv(Cp, exp(-k2*t))."""
    cp = aif_interp(t)
    return K1 * exp_conv_trap(t, cp, k2)


def two_tcm_solution(t, K1, k2, k3, aif_interp):
    """Forward irreversible 2-tissue-compartment solution (k4 = 0)."""
    cp = aif_interp(t)
    delta = max(k2 + k3, 1e-8)
    phi_fast = K1 * (delta - k3) / delta
    phi_trap = K1 * k3 / delta
    H_fast = exp_conv_trap(t, cp, delta)
    H_trap = exp_conv_trap(t, cp, 0.0)
    return phi_fast * H_fast + phi_trap * H_trap


def _fit_voxel(tac, model_fn, n_params, p0, bounds, method):
    """
    Fit one voxel's TAC by minimizing sum((model(params) - tac)**2).

    method == "dual_annealing" routes to scipy.optimize.dual_annealing,
    a global optimizer that searches the full bounded box (bounds are
    required for it, unlike scipy.optimize.minimize where they're
    optional) -- more robust to local minima than local optimizers on
    noisy voxels, at the cost of more function evaluations. `p0` is
    passed through as its `x0` seed. Any other `method` string is
    forwarded as-is to scipy.optimize.minimize (None lets scipy
    auto-select from `bounds`, which resolves to L-BFGS-B).
    """
    tac = np.asarray(tac, float)
    if tac.max() < 1e-6:
        return (np.nan,) * n_params

    def objective(params):
        return np.sum((model_fn(*params) - tac) ** 2)

    try:
        if method == "dual_annealing":
            result = dual_annealing(objective, bounds=bounds, x0=p0)
        else:
            result = minimize(objective, x0=p0, method=method, bounds=bounds)
        return tuple(result.x)
    except Exception:
        return (np.nan,) * n_params


def _fit_voxel_chunk(tacs, model_fn, n_params, p0, bounds, method):
    """Sequentially fit every row of `tacs` (N, T) -- the unit of work
    handed to each parallel worker by _fit_voxels."""
    return np.array([_fit_voxel(tac, model_fn, n_params, p0, bounds, method) for tac in tacs],
                     dtype=float).reshape(-1, n_params)


def _resolve_n_jobs(n_jobs):
    """n_jobs=None -> auto-detect the number of CPU threads available on
    this machine (os.cpu_count()) and use all of them, printing what was
    detected. Any other value (1 = sequential, -1 = all cores, or an
    explicit count) is passed through unchanged."""
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
        print(f"[INFO] Detected {n_jobs} CPU thread(s) -- using n_jobs={n_jobs}")
    return n_jobs


def _fit_voxels(tacs, model_fn, n_params, p0, bounds, method, n_jobs, verbose, label):
    """
    Fit every row of `tacs` (N, T) independently -- each voxel's fit has
    no dependency on any other, so this is embarrassingly parallel across
    CPU processes (both scipy.optimize.minimize and dual_annealing are
    CPU-only, no GPU path).

    n_jobs : None (default) auto-detects and uses all CPU threads (see
        _resolve_n_jobs). 1 runs sequentially in this process. Any other
        value splits `tacs` into n_jobs contiguous chunks (n_jobs=-1 uses
        os.cpu_count() chunks) and fits each chunk in a separate worker
        process via joblib -- chunked rather than one task per voxel,
        since these fits are cheap enough that per-task dispatch overhead
        would otherwise dominate. Falls back to sequential (with a
        warning) if joblib isn't installed.
    """
    n_jobs = _resolve_n_jobs(n_jobs)
    n = len(tacs)
    if n == 0:
        return np.zeros((0, n_params))

    if n_jobs == 1:
        results = np.empty((n, n_params))
        indices = range(n)
        if verbose and _HAVE_TQDM:
            indices = tqdm(indices, desc=f"voxelwise-{label}", unit="voxel")
        for i in indices:
            if verbose and not _HAVE_TQDM and i % 500 == 0:
                print(f"[voxelwise-{label}] {i}/{n}")
            results[i] = _fit_voxel(tacs[i], model_fn, n_params, p0, bounds, method)
        return results

    if not _HAVE_JOBLIB:
        warnings.warn(f"n_jobs={n_jobs} requested but joblib is not installed; running sequentially.")
        return _fit_voxel_chunk(tacs, model_fn, n_params, p0, bounds, method)

    n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    # More, smaller chunks than one-per-worker, purely so the progress
    # bar (and load balancing across workers) has finer granularity than
    # "n_workers ticks total" -- dispatch overhead stays low since each
    # chunk still batches many voxels.
    n_chunks = max(1, min(n, n_workers * 8))
    chunks = np.array_split(tacs, n_chunks)
    tasks = (delayed(_fit_voxel_chunk)(chunk, model_fn, n_params, p0, bounds, method) for chunk in chunks)

    if verbose and _HAVE_TQDM:
        try:
            chunk_results = list(tqdm(
                Parallel(n_jobs=n_jobs, return_as="generator")(tasks),
                total=n_chunks, desc=f"voxelwise-{label}", unit="chunk",
            ))
        except TypeError:  # joblib < 1.3 has no return_as="generator"
            print(f"[voxelwise-{label}] fitting {n} voxels across {n_chunks} parallel chunk(s)")
            chunk_results = Parallel(n_jobs=n_jobs)(tasks)
    else:
        if verbose:
            print(f"[voxelwise-{label}] fitting {n} voxels across {n_chunks} parallel chunk(s)")
        chunk_results = Parallel(n_jobs=n_jobs)(tasks)

    return np.concatenate(chunk_results, axis=0)


def calculate_voxelwise_1tcm(
        img, t, aif,
        mask=None,
        method=DEFAULT_OPT_METHOD,
        p0=(0.3, 0.2),
        bounds=((1e-6, 1.0), (1e-6, 1.0)),
        verbose=True,
        n_jobs=None,
):
    """
    Voxelwise 1-tissue-compartment (K1, k2) NLLS fit.

    img : ndarray (X, Y, Z, T)
    t : ndarray (T,) frame timestamps
    aif : ndarray (T,) arterial input function sampled at `t`
    mask : optional (X, Y, Z) bool array restricting the fit; defaults to
        voxels whose signal ever exceeds 1e-6.
    method : "dual_annealing" (default -- global optimizer, bounds
        required) or any scipy.optimize.minimize method string (e.g.
        "L-BFGS-B", "Powell", "Nelder-Mead"); see _fit_voxel's docstring,
        or use benchmark_optimizer_methods() to pick this empirically
        against a simulated phantom with known ground truth.
    n_jobs : int or None. Default None auto-detects and uses all CPU
        threads (see _resolve_n_jobs); pass 1 for sequential, or an
        explicit count. See _fit_voxels's docstring.

    Returns dict of (X, Y, Z) maps: K1, k2.
    """
    img = np.asarray(img, float)
    X, Y, Z, T = img.shape
    flat = img.reshape(-1, T)

    flat_mask = mask.reshape(-1) if mask is not None else flat.max(axis=1) > 1e-6
    valid_idx = np.where(flat_mask)[0]
    aif_interp = interp1d(t, aif, kind="linear", fill_value="extrapolate")

    def model_fn(K1, Ve):
        return one_tcm_solution(t, K1, K1/Ve, aif_interp)

    fitted = _fit_voxels(flat[valid_idx], model_fn, 2, p0, bounds, method, n_jobs, verbose, "1TCM")

    K1_flat = np.full(X * Y * Z, np.nan)
    k2_flat = np.full(X * Y * Z, np.nan)
    K1_flat[valid_idx] = fitted[:, 0]
    k2_flat[valid_idx] = fitted[:, 1]

    return {"K1": K1_flat.reshape(X, Y, Z), "k2": k2_flat.reshape(X, Y, Z)}


def calculate_voxelwise_2tcm(
        img, t, aif,
        mask=None,
        method=DEFAULT_OPT_METHOD,
        p0=(0.2, 0.3, 0.05),
        bounds=((1e-6, 2.0), (1e-6, 2.0), (1e-6, 2.0)),
        verbose=True,
        n_jobs=None,
):
    """
    Voxelwise irreversible 2-tissue-compartment (K1, k2, k3; k4 = 0) NLLS
    fit. Same conventions as calculate_voxelwise_1tcm; see its docstring
    for `mask` / `method` / `n_jobs`.

    Returns dict of (X, Y, Z) maps: K1, k2, k3.
    """
    img = np.asarray(img, float)
    X, Y, Z, T = img.shape
    flat = img.reshape(-1, T)

    flat_mask = mask.reshape(-1) if mask is not None else flat.max(axis=1) > 1e-6
    valid_idx = np.where(flat_mask)[0]
    aif_interp = interp1d(t, aif, kind="linear", fill_value="extrapolate")

    def model_fn(K1, k2, k3):
        return two_tcm_solution(t, K1, k2, k3, aif_interp)

    fitted = _fit_voxels(flat[valid_idx], model_fn, 3, p0, bounds, method, n_jobs, verbose, "2TCM")

    K1_flat = np.full(X * Y * Z, np.nan)
    k2_flat = np.full(X * Y * Z, np.nan)
    k3_flat = np.full(X * Y * Z, np.nan)
    K1_flat[valid_idx] = fitted[:, 0]
    k2_flat[valid_idx] = fitted[:, 1]
    k3_flat[valid_idx] = fitted[:, 2]

    return {
        "K1": K1_flat.reshape(X, Y, Z),
        "k2": k2_flat.reshape(X, Y, Z),
        "k3": k3_flat.reshape(X, Y, Z),
    }


def save_maps_as_nifti(maps, affine, out_dir):
    """Save each parametric map (K1, k2, k3, ...) as a .nii.gz file."""
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in maps.items():
        arr_clean = np.nan_to_num(np.asarray(arr, float), nan=0.0)
        img_nii = nib.Nifti1Image(arr_clean.astype(np.float32), affine)
        out_path = os.path.join(out_dir, f"{name}_map.nii.gz")
        nib.save(img_nii, out_path)
        print(f"[SAVED] {out_path}")


def benchmark_optimizer_methods(
        true_params,
        t,
        aif,
        model="1tcm",
        methods=("dual_annealing", "L-BFGS-B", "Powell", "Nelder-Mead", "TNC"),
        noise_std=0.02,
        n_trials=5,
        seed=0,
        p0=None,
        bounds=None,
):
    """
    Compare optimizer methods for voxelwise kinetic fitting by fitting
    simulated, noisy TACs generated from known ground-truth parameters,
    and scoring each method's parameter-recovery error. "dual_annealing"
    routes to scipy.optimize.dual_annealing (global optimizer); every
    other entry in `methods` is forwarded to scipy.optimize.minimize.

    Why this (and not a grid search over K1/k2/k3 themselves): the local
    optimizers already do continuous optimization over the kinetic
    parameters, and dual_annealing already does a global bounded search
    -- hand-rolling a grid search over that continuous space would just
    be a slower, coarser version of what they're already doing. What
    genuinely isn't continuous is the *choice of optimizer*, and its
    real-data accuracy can't be checked directly (real voxels have no
    known ground truth), so comparing a handful of candidate methods
    against a simulated phantom -- where ground truth IS known -- is
    exactly the useful comparison, and a small discrete list is all
    there is to search.

    true_params : dict, e.g. {"K1": 0.2, "k2": 0.3} (model="1tcm") or
        {"K1": 0.2, "k2": 0.3, "k3": 0.05} (model="2tcm").
    t, aif : ndarray (T,) -- frame timestamps and AIF sampled at `t`.
    methods : candidate method names to compare (see _fit_voxel).
    noise_std : Gaussian noise std added to the simulated TAC before
        fitting (0 = noise-free sanity check).
    n_trials : independent noise draws averaged per method.
    seed : base RNG seed; trial i uses seed + i.

    Returns dict: {method_name: {"rmse": float, "mean_params": {...}}, ...,
                   "best_method": name with the lowest rmse}.
    """
    aif_interp = interp1d(t, aif, kind="linear", fill_value="extrapolate")

    if model == "1tcm":
        names = ["K1", "k2"]
        model_fn = lambda K1, k2: one_tcm_solution(t, K1, k2/K1, aif_interp)
        default_p0, default_bounds = (0.13, 0.3), ((1e-6, 1.0), (1e-6, 1.0))
    elif model == "2tcm":
        names = ["K1", "k2", "k3"]
        model_fn = lambda K1, k2, k3: two_tcm_solution(t, K1, k2, k3, aif_interp)
        default_p0, default_bounds = (0.2, 0.3, 0.05), ((1e-6, 2.0), (1e-6, 2.0), (1e-6, 2.0))
    else:
        raise ValueError("model must be '1tcm' or '2tcm'")

    p0 = default_p0 if p0 is None else p0
    bounds = default_bounds if bounds is None else bounds
    true_vec = np.array([true_params[n] for n in names], float)
    clean_tac = model_fn(*true_vec)

    results = {}
    for method in methods:
        errs = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + trial)
            noisy_tac = clean_tac + rng.normal(0, noise_std, size=clean_tac.shape)
            fitted = np.array(_fit_voxel(noisy_tac, model_fn, len(names), p0, bounds, method))
            errs.append(fitted - true_vec)
        errs = np.asarray(errs)
        rmse = float(np.sqrt(np.nanmean(errs ** 2)))
        results[method] = {
            "rmse": rmse,
            "mean_params": dict(zip(names, np.nanmean(errs, axis=0) + true_vec)),
        }

    results["best_method"] = min(methods, key=lambda m: results[m]["rmse"])
    return results




def pipeline(
        path,
        preprocessing,  # need to be preproccesing function for pet or dce
        t,
        num_of_compartment=2,
        mask=None,
        method=DEFAULT_OPT_METHOD,
        n_jobs=None,
):
    """
    method : optimizer used for every voxel fit -- "dual_annealing"
        (global optimizer, default) or a scipy.optimize.minimize method
        string; see calculate_voxelwise_1tcm/2tcm, _fit_voxel, and
        benchmark_optimizer_methods() for picking this empirically.
    n_jobs : CPU parallelism across voxels, forwarded to
        calculate_voxelwise_1tcm/2tcm -- see _fit_voxels's docstring.
    """

    # -----------------------------
    # 1. Preprocessing
    # -----------------------------
    img, aif, affine = preprocessing(path)
    # img: (X, Y, Z, T)

    if num_of_compartment == 1:
        maps = calculate_voxelwise_1tcm(img, t, aif, mask=mask, method=method, n_jobs=n_jobs)
    elif num_of_compartment == 2:
        maps = calculate_voxelwise_2tcm(img, t, aif, mask=mask, method=method, n_jobs=n_jobs)
    else:
        raise ValueError("voxelwise PET fitting supports num_of_compartment in {1, 2}")

    save_maps_as_nifti(maps, affine, out_dir=path)
    return maps

