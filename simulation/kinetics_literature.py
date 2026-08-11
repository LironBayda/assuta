"""
Literature-informed kinetic parameter distributions for a synthetic
prostate phantom, used to sample per-voxel ground-truth K1/k2(/k3) for the
DCE (1TCM / Tofts) and PET (2TCM, irreversible) simulations.

These are *illustrative* ranges assembled from published group-level
summaries -- not a substitute for fitting your own data -- intended to
give the simulation physiologically plausible contrast between normal
prostate and cancer, in the right ballpark and with the right *direction*
of effect (which tissue is higher/lower), which is what matters for
validating whether voxelwise / PINN recover the correct parameters.

DCE-MRI (1-tissue-compartment / standard Tofts model: K1 == Ktrans, k2 == kep)
--------------------------------------------------------------------------
- Meta-analysis across 14 studies / 484 patients: Ktrans and kep are both
  significantly higher in prostate cancer (PCa) than in non-cancerous
  peripheral-zone (PZ) tissue (Ktrans SMD ~1.57, kep SMD ~1.41).
  Rosenkrantz/Vos-type meta-analysis, PMC5207570.
- A 150-patient study reports cutoff values separating cancer from normal
  tissue of Ktrans ~ 0.205 /min and kep ~ 0.63-0.665 /min (extended Tofts
  model); PubMed 29662622.
- Hectors et al. (J Magn Reson Imaging 2017;46:837-849) report Tofts-model
  Ktrans/kep/ve for prostate cancer vs. peripheral zone.

We use these to center "normal prostate" below the reported cutoffs and
"cancer" above them, with spread wide enough to overlap somewhat (as real
tissue does) but with a clear group-level separation.

Dynamic PET (irreversible 2-tissue-compartment model: K1, k2, k3, k4=0)
--------------------------------------------------------------------------
- 68Ga-PSMA-11 kinetic modeling in intermediate/high-risk PCa (23 lesions,
  18 patients): irreversible 2TCM fit best; lesions show significantly
  *elevated* K1 and k3 and significantly *decreased* k2 relative to
  reference (non-cancerous) prostate tissue. PMC10781928 / EJNMMI Research
  2024. Model selection was tested directly against two alternatives (a
  reversible 1TCM and a reversible 2TCM) using goodness-of-fit and
  information-loss criteria -- irreversible 2TCM won, "consistently
  appropriate across prostatic zones."
- Mechanistic confirmation, independent cohort: dynamic whole-body
  [68Ga]Ga-PSMA-11 / [18F]PSMA-1007 PET, 20 patients -- "for PCa lesions,
  k4 ~ 0, as the binding is predominantly irreversible" (PSMA
  internalization after receptor binding). PMC10105814 / EJNMMI Research
  2023.
- [18F]DCFPyL vs [18F]fluorocholine dynamic PET (flow-modified 2TCM):
  K1 ~0.30 vs 0.24 mL/min/g for dominant intraprostatic lesion vs. benign
  tissue (modest but significant increase). PMC7782622.
- [18F]PSMA-1007 whole-body kinetics: tumor lesions show lower K1/k2 than
  normal high-uptake organs (liver/spleen/parotid) but this is an
  inter-organ, not intra-prostate, comparison; used here only as a sanity
  check on plausible K1/k2 magnitude ranges (mL/ccm/min and 1/min).
  PMC11139746.

We therefore model PET cancer voxels as having *higher* K1 and k3, and
*lower* k2, than normal prostate voxels -- matching the PSMA-11 finding
above, which is the most directly relevant intra-prostate comparison.

All rate constants are in 1/min (PET K1 nominally mL/ccm/min, treated as
numerically equivalent to 1/min here since tissue density ~1). Values are
truncated to be positive when sampled.
"""

# ---------------------------------------------------------------------
# DCE-MRI, 1TCM / Tofts (K1 == Ktrans, k2 == kep), units: 1/min
# ---------------------------------------------------------------------
DCE_1TCM_PARAMS = {
    "background": {
        "K1": (0.0, 0.0),
        "k2": (0.0, 0.0),
    },
    "prostate_normal": {
        # below the ~0.205 / ~0.65 cancer cutoffs
        "K1_mean": 0.12, "K1_std": 0.035,
        "k2_mean": 0.35, "k2_std": 0.08,
    },
    "prostate_cancer": {
        # above the cutoffs, matching reported SMD ~1.4-1.6 elevation
        "K1_mean": 0.40, "K1_std": 0.10,
        "k2_mean": 0.90, "k2_std": 0.18,
    },
}

# ---------------------------------------------------------------------
# Dynamic PET, irreversible 2TCM (K1, k2, k3; k4 = 0), units: 1/min
# ---------------------------------------------------------------------
PET_2TCM_PARAMS = {
    "background": {
        "K1": (0.0, 0.0),
        "k2": (0.3, 0.3),   # avoid div-by-zero in closed-form solution; K1=0 -> Ct=0 regardless
        "k3": (0.0, 0.0),
    },
    "prostate_normal": {
        "K1_mean": 0.18, "K1_std": 0.04,
        "k2_mean": 0.42, "k2_std": 0.08,
        "k3_mean": 0.03, "k3_std": 0.010,
    },
    "prostate_cancer": {
        # PSMA-11 finding: K1 up, k3 up, k2 down vs. normal reference tissue
        "K1_mean": 0.38, "K1_std": 0.08,
        "k2_mean": 0.22, "k2_std": 0.06,
        "k3_mean": 0.11, "k3_std": 0.03,
    },
}

TISSUE_CLASSES = {0: "background", 1: "prostate_normal", 2: "prostate_cancer"}

# ---------------------------------------------------------------------
# ADC (diffusion-MRI cellularity marker), correlated with PET/PSMA uptake
# ---------------------------------------------------------------------
# Domachevsky et al. (Eur Radiol 2018;28:5275-5283, PMC/DOI
# 10.1007/s00330-018-5484-1): 22 patients, 44 prostate regions (22
# intra-prostatic cancer [IPC], 22 normal prostatic tissue [NPT]).
# IPC had significantly HIGHER PSMA SUVmax and significantly LOWER
# ADCmin/ADCmean than NPT (both p<0.0001). Measured correlation strength
# between PSMA SUVmax and ADC: rho=-0.717 to -0.740 (SUVmax vs ADCmin,
# both PET/MR-MRAC and PET/CT-CTAC) and rho=-0.737 (SUVmax vs ADCmean) --
# i.e. higher PSMA uptake goes with lower ADC (denser tissue) fairly
# strongly, not just directionally.
#
# The paper's abstract does not give exact ADCmin/ADCmean median values
# (only that IPC vs NPT differ, p<0.0001) -- ADC_TARGET_RANGES below
# uses commonly-reported prostate ADC ranges from the broader DWI
# literature for the baseline scale (illustrative, same caveat as the
# rest of this module), while the CORRELATION STRENGTH (rho~-0.73) is
# the actual number taken from this paper specifically.
ADC_TARGET_RANGES = {
    # (mean, std) in units of 10^-3 mm^2/s -- typical reported prostate
    # peripheral-zone ranges; illustrative baseline scale only.
    "prostate_normal": (1.6, 0.3),
    "prostate_cancer": (0.85, 0.20),
}


def sample_correlated_adc(K1_map, label, rho=-0.73, seed=None):
    """
    Sample a per-voxel ADC map (diffusion-MRI cellularity marker,
    10^-3 mm^2/s) correlated with an already-sampled K1_map (PSMA
    uptake), at approximately the target Spearman rho -- see the
    citation above (Domachevsky et al. 2018) for where rho=-0.73 comes
    from. Uses a Gaussian-copula construction: within each tissue class,
    convert K1 to a normal rank-score, mix in independent noise to hit
    the target correlation, then map back through that class's target
    ADC distribution -- so the MARGINAL ADC distribution per class comes
    from ADC_TARGET_RANGES while the K1-ADC correlation comes from rho.

    K1_map : (X, Y, Z) float array, already sampled (e.g. via
        sample_param_maps(..., PET_2TCM_PARAMS, ["K1", ...])).
    label : (X, Y, Z) uint8 array, same convention as sample_param_maps.
    rho : target Spearman-style correlation between K1 and ADC within
        each tissue class (default -0.73, this paper's measured value).

    Returns (X, Y, Z) float32 ADC map. Background voxels are set to 0.
    """
    import numpy as np
    from scipy.stats import rankdata, norm

    rng = np.random.default_rng(seed)
    adc_map = np.zeros(label.shape, dtype=np.float32)

    for cls_val, cls_name in TISSUE_CLASSES.items():
        if cls_name not in ADC_TARGET_RANGES:
            continue  # background: leave at 0
        mask = label == cls_val
        n = int(mask.sum())
        if n == 0:
            continue

        k1_vals = K1_map[mask]
        # rank -> uniform -> standard normal (avoiding exact 0/1 which
        # would map to +-inf)
        ranks = rankdata(k1_vals) / (n + 1)
        z_k1 = norm.ppf(ranks)

        z_indep = rng.normal(size=n)
        z_adc = rho * z_k1 + np.sqrt(max(1 - rho ** 2, 0.0)) * z_indep

        mean, std = ADC_TARGET_RANGES[cls_name]
        adc_vals = mean + std * z_adc
        adc_map[mask] = np.clip(adc_vals, 0.05, None)

    return adc_map


def sample_param_maps(label, params_dict, param_names, seed=None):
    """
    Sample per-voxel ground-truth kinetic parameters for a label volume.

    label : (X, Y, Z) uint8 array with values in {0, 1, 2}
    params_dict : DCE_1TCM_PARAMS or PET_2TCM_PARAMS
    param_names : e.g. ["K1", "k2"] or ["K1", "k2", "k3"]

    Returns dict[param_name] -> (X, Y, Z) float32 array.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    maps = {p: np.zeros(label.shape, dtype=np.float32) for p in param_names}

    for cls_val, cls_name in TISSUE_CLASSES.items():
        mask = label == cls_val
        n = int(mask.sum())
        if n == 0:
            continue
        cfg = params_dict[cls_name]
        for p in param_names:
            if f"{p}_mean" in cfg:
                vals = rng.normal(cfg[f"{p}_mean"], cfg[f"{p}_std"], size=n)
                vals = np.clip(vals, 1e-4, None)
            else:
                lo, hi = cfg[p]
                vals = rng.uniform(lo, hi, size=n) if hi > lo else np.full(n, lo)
            maps[p][mask] = vals

    return maps
