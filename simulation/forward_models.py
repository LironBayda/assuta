"""
Forward compartmental models used to *simulate* ground-truth TACs.

The exponential-convolution math here is deliberately kept identical to
`core.model.Ks_net` (trapezoidal exp-convolution, and the same closed-form
irreversible-2TCM decomposition used in `convolve_2cm_for_minimize`) so
that the PINN's own forward model and the data it's asked to fit are
generated the same way -- any recovery error is then attributable to the
estimation method, not to a mismatch between simulation and model.
"""
import numpy as np


# ---------------------------------------------------------------------
# Arterial input functions
# ---------------------------------------------------------------------
def parker_aif(t_min):
    """
    Parker et al. (2006) population AIF for DCE-MRI, gadolinium
    concentration in mmol/L, t in minutes.
    """
    t = np.asarray(t_min, dtype=np.float64)
    A1, A2 = 0.809, 0.330
    T1, T2 = 0.17046, 0.365
    sigma1, sigma2 = 0.0563, 0.132
    alpha, beta = 1.050, 0.1685
    s, tau = 38.078, 0.483

    gauss1 = (A1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-((t - T1) ** 2) / (2 * sigma1 ** 2))
    gauss2 = (A2 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-((t - T2) ** 2) / (2 * sigma2 ** 2))
    sigmoid = alpha * np.exp(-beta * t) / (1 + np.exp(-s * (t - tau)))
    cp = gauss1 + gauss2 + sigmoid
    return np.clip(cp, 0.0, None)


def feng_input_function(t_min, A1=851.1, A2=21.9, A3=20.8,
                         lam1=-4.134, lam2=-0.1191, lam3=-0.0104, t0=0.5):
    """
    Simplified three-exponential (Feng-type) population arterial input
    function for PET, t in minutes. Same functional form already used in
    VAE_initi/dataset.py, reproduced here so this module has no import-
    time dependency on the VAE package.
    """
    t = np.asarray(t_min, dtype=np.float64)
    tt = t - t0
    cp = np.where(
        tt > 0,
        (A1 * tt - A2 - A3) * np.exp(lam1 * tt) + A2 * np.exp(lam2 * tt) + A3 * np.exp(lam3 * tt),
        0.0,
    )
    return np.clip(cp, 0.0, None)


# ---------------------------------------------------------------------
# Trapezoidal exponential convolution, vectorized over many (voxel) thetas
# ---------------------------------------------------------------------
def exp_conv_trap_vec(t, blood, theta):
    """
    blood: (T,) plasma/AIF curve sampled at t
    theta: (N,) per-voxel decay rate
    returns H: (T, N), H(t) = int_0^t blood(s) * exp(-theta*(t-s)) ds
    (trapezoidal recursion, identical math to core.model.Ks_net.exp_conv_trap)
    """
    t = np.asarray(t, dtype=np.float64)
    blood = np.asarray(blood, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    T = len(t)
    N = len(theta)
    dt = np.diff(t, prepend=t[0])

    H = np.zeros((T, N), dtype=np.float64)
    h_prev = np.zeros(N, dtype=np.float64)
    for i in range(1, T):
        alpha = np.exp(-theta * dt[i])
        h_curr = alpha * h_prev + 0.5 * dt[i] * (blood[i] + alpha * blood[i - 1])
        H[i] = h_curr
        h_prev = h_curr
    return H


def dce_noise(signal, snr=15.0, rng=None):
    """
    DCE-MRI noise: Rician + Gaussian. MRI magnitude images are
    sqrt(I^2+Q^2) of two independently Gaussian-noised (real/imaginary)
    channels -- Rician, not Gaussian, by construction. No Poisson
    component: DCE-MRI is not a counting/photon modality, so a
    count-statistics term doesn't apply here (unlike PET).

    signal : (..., T) clean TAC array.
    snr : peak-signal-to-noise ratio (SNR 10-30 is the literature-typical
        range for DCE-MRI simulation studies).
    """
    rng = np.random.default_rng(rng)
    signal = np.asarray(signal, dtype=np.float64)
    peak = np.max(np.abs(signal)) + 1e-8
    sigma = peak / snr
    n_real = rng.normal(0.0, sigma, size=signal.shape)
    n_imag = rng.normal(0.0, sigma, size=signal.shape)
    noisy = np.sqrt((signal + n_real) ** 2 + n_imag ** 2)
    return np.clip(noisy, 0.0, None)


def pet_noise(signal, snr=15.0, poisson_scale=1.0, rng=None):
    """
    Dynamic PET noise: Poisson + Gaussian. PET is a photon-counting
    modality (Poisson count statistics, signal-proportional variance)
    with additional Gaussian measurement/reconstruction noise layered on
    top. No Rician component: that's specific to MRI's complex-channel
    magnitude reconstruction, not how PET images are formed.

    signal : (..., T) clean TAC array.
    snr : controls the Gaussian noise floor.
    poisson_scale : float, higher = more effective counts = less
        relative Poisson noise (1.0 is a deliberately low-count/noisy
        regime; increase for a higher-count acquisition).
    """
    rng = np.random.default_rng(rng)
    signal = np.asarray(signal, dtype=np.float64)
    peak = np.max(np.abs(signal)) + 1e-8

    counts_scale = 1000.0 * poisson_scale / peak
    counts = np.clip(signal * counts_scale, 0.0, None)
    noisy_counts = rng.poisson(counts).astype(np.float64)
    signal_poisson = noisy_counts / counts_scale

    sigma = peak / snr
    gaussian_term = rng.normal(0.0, sigma, size=signal.shape)
    return np.clip(signal_poisson + gaussian_term, 0.0, None)


def composite_noise(signal, snr=15.0, poisson_scale=1.0, rng=None):
    """
    Composite Poisson + Gaussian + Rician noise model, combining the
    noise sources actually reported in the literature for PET/DCE-MRI
    simulation studies rather than a single idealized noise type:

    - Poisson: PET is fundamentally a photon-counting modality: raw
      reconstructed activity has count-dependent (signal-proportional
      variance) Poisson-like noise before any other noise is added.
      `poisson_scale` controls the effective count level (higher ->
      more counts -> less relative Poisson noise); this is applied to a
      rescaled-to-counts copy of the signal and rescaled back.
    - Gaussian: thermal/electronic noise in the underlying (complex,
      pre-magnitude) MR signal, or additional Gaussian measurement noise
      layered on top of the Poisson-noised PET signal.
    - Rician: MRI magnitude images are computed as sqrt(I^2+Q^2) of two
      independently Gaussian-noised (real/imaginary) channels, which is
      Rician-distributed, not Gaussian -- this is the standard way
      Rician noise is simulated in the DCE-MRI literature (e.g. "Robust
      estimation of hemo-dynamic parameters in traditional DCE-MRI
      models", PMC6317807, and typical SNR sweep studies for
      NLLS/DL pharmacokinetic fitting).

    signal : (..., T) clean TAC array.
    snr : peak-signal-to-noise ratio controlling the Gaussian/Rician
        noise floor (SNR=10-30 is the literature-typical range used in
        DCE-MRI simulation studies).
    poisson_scale : float
        Higher = more effective counts = less Poisson noise. 1.0 is a
        deliberately noisy/low-count regime; increase for a higher-count
        (e.g. static or well-sampled) acquisition.

    Returns the noisy signal, same shape as input.
    """
    rng = np.random.default_rng(rng)
    signal = np.asarray(signal, dtype=np.float64)
    peak = np.max(np.abs(signal)) + 1e-8

    # 1) Poisson (count-statistics) noise on a rescaled-to-counts copy
    counts_scale = 1000.0 * poisson_scale / peak
    counts = np.clip(signal * counts_scale, 0.0, None)
    noisy_counts = rng.poisson(counts).astype(np.float64)
    signal_poisson = noisy_counts / counts_scale

    # 2) Rician magnitude noise: two independent Gaussian (real/imag)
    # channels, then magnitude -- this is where the "Gaussian" and
    # "Rician" pieces are actually the same physical noise source, not
    # two separate additive terms (adding both directly would double-
    # count the same underlying thermal noise).
    sigma = peak / snr
    n_real = rng.normal(0.0, sigma, size=signal.shape)
    n_imag = rng.normal(0.0, sigma, size=signal.shape)
    signal_rician = np.sqrt((signal_poisson + n_real) ** 2 + n_imag ** 2)

    return np.clip(signal_rician, 0.0, None)


def simulate_1tcm_volume(t, cp, K1_map, k2_map, noise_std=0.02, rng=None, noise_model="gaussian", snr=15.0):
    """
    1-tissue-compartment forward model: Ct(t) = K1 * conv(Cp, exp(-k2 t)).
    K1_map, k2_map: (X, Y, Z). Returns (T, X, Y, Z) noisy TAC volume and
    the clean (noise-free) volume.

    noise_model : "gaussian" (default, backward compatible, uses noise_std)
        or "dce" (Rician+Gaussian, uses `snr` instead -- see
        simulation.forward_models.dce_noise).
    """
    rng = np.random.default_rng(rng)
    shape = K1_map.shape
    K1 = K1_map.reshape(-1)
    k2 = np.clip(k2_map.reshape(-1), 1e-4, None)

    H = exp_conv_trap_vec(t, cp, k2)          # (T, N)
    Ct = K1[None, :] * H                      # (T, N)
    clean = Ct.reshape(len(t), *shape)

    if noise_model == "dce":
        noisy = dce_noise(clean, snr=snr, rng=rng)
    elif noise_std > 0:
        scale = noise_std * (clean.max() + 1e-8)
        noisy = clean + rng.normal(0.0, scale, size=clean.shape)
        noisy = np.clip(noisy, 0.0, None)
    else:
        noisy = clean.copy()

    return noisy.astype(np.float32), clean.astype(np.float32)


def simulate_2tcm_volume(t, cp, K1_map, k2_map, k3_map, noise_std=0.02, rng=None, noise_model="gaussian", snr=15.0):
    """
    Irreversible 2-tissue-compartment forward model (k4 = 0), same closed
    form as core.model.Ks_net.convolve_2cm_for_minimize:

        delta  = k2 + k3
        theta1 = delta,  theta2 = 0
        phi1   = K1*(theta1 - k3)/delta
        phi2   = K1*(theta2 - k3)/(-delta)
        Ct     = phi1 * conv(Cp, exp(-theta1 t)) + phi2 * conv(Cp, exp(-theta2 t))

    K1_map, k2_map, k3_map: (X, Y, Z). Returns (T, X, Y, Z) noisy and clean
    TAC volumes.

    noise_model : "gaussian" (default, backward compatible, uses noise_std)
        or "pet" (Poisson+Gaussian, uses `snr` instead -- see
        simulation.forward_models.pet_noise).
    """
    rng = np.random.default_rng(rng)
    shape = K1_map.shape
    K1 = K1_map.reshape(-1)
    k2 = k2_map.reshape(-1)
    k3 = k3_map.reshape(-1)

    delta = k2 + k3
    delta = np.where(delta <= 1e-8, 1e-8, delta)
    theta1 = delta
    theta2 = np.zeros_like(delta)
    phi1 = K1 * (theta1 - k3) / delta
    phi2 = K1 * (theta2 - k3) / (-delta)

    H1 = exp_conv_trap_vec(t, cp, theta1)
    H2 = exp_conv_trap_vec(t, cp, theta2)
    Ct = phi1[None, :] * H1 + phi2[None, :] * H2
    clean = Ct.reshape(len(t), *shape)

    if noise_model == "pet":
        noisy = pet_noise(clean, snr=snr, rng=rng)
    elif noise_std > 0:
        scale = noise_std * (clean.max() + 1e-8)
        noisy = clean + rng.normal(0.0, scale, size=clean.shape)
        noisy = np.clip(noisy, 0.0, None)
    else:
        noisy = clean.copy()

    return noisy.astype(np.float32), clean.astype(np.float32)
