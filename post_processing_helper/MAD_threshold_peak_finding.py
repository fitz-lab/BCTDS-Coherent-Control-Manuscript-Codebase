import numpy as np
import math

def noise_threshold_mad(x, k=4.0, refine_iters=0, clip_k=4.0):
    """
    Estimate a robust noise-floor threshold for 1D data (peaks + noise).

    Threshold returned is:  mu + k * sigma,
    where mu and sigma are estimated robustly (MAD + optional sigma-clipping).

    Args:
        x (array-like): 1D data.
        k (float): number of sigmas for the threshold (e.g., 3, 4, 5).
        refine_iters (int): how many sigma-clipping refinements after initial MAD.
        clip_k (float): clipping factor used during refinement (e.g., 3σ).

    Returns:
        float: threshold (height) you can pass to scipy.signal.find_peaks(..., height=threshold)
    """
    x = np.asarray(x).ravel()

    # Initial robust center and spread from MAD
    mu = np.median(x)
    sigma = 1.4826 * np.median(np.abs(x - mu))  # Gaussian-consistent MAD

    # Optional: refine via a few rounds of sigma-clipping to reduce peak bias
    for _ in range(max(0, refine_iters)):
        if sigma == 0:
            break
        mask = np.abs(x - mu) <= clip_k * sigma
        if not np.any(mask):
            break
        mu = np.median(x[mask])
        sigma = 1.4826 * np.median(np.abs(x[mask] - mu))

    # Fallback if data are flat
    if sigma == 0:
        return float(mu)

    return float(mu + k * sigma)

def peak_count_uncertainty_neffN(N_obs, N, k=4.0, two_sided=False):
    """
    N_obs: observed peaks above threshold
    N:     total number of samples (Neff = N)
    k:     sigma threshold (e.g., 3, 4, 5)
    """
    p_tail = 0.5 * math.erfc(k / math.sqrt(2.0))
    if two_sided:
        p_tail *= 2.0
    E_FP = N * p_tail
    N_true = max(0.0, N_obs - E_FP)
    err = math.sqrt(max(N_obs, 0.0) + max(E_FP, 0.0))  # 1σ
    ci95 = (max(0.0, N_true - 1.96 * err), N_true + 1.96 * err)
    return {"N_true": N_true, "err": err, "E_FP": E_FP, "p_tail": p_tail, "ci95": ci95}