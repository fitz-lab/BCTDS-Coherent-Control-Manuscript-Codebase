import traceback
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def piecewise_model_continuous(t, x_c, k, b):
    return np.where(t < x_c, k*t + b, k*x_c + b)

def lifetime_fit_one_trace(t, trace, n_sigma=2.0, slice=None):
    """Fit a single trace; return (tau, tau_lo, tau_hi, xc, xc_lo, xc_hi) and (popt, fit_y)."""
    if not isinstance(trace, np.ndarray):
        return (np.nan,)*6, (None, None)

    # prepare p0/bounds up front so we can display them even on failure
    # x_c0 = t[len(t)//2]
    x_c0 = t[len(t)//20] if len(t) >= 20 else t[len(t)//2]
    k0   = -1000.0
    b0   = trace[0]
    p0   = [x_c0, k0, b0]
    bounds = ([t[0], -np.inf, -np.inf], [t[-1], np.inf, np.inf])

    try:
        popt, pcov = curve_fit(
            piecewise_model_continuous, t, trace,
            p0=p0, bounds=bounds, maxfev=20000
        )
        x_c_hat, k_hat, _ = popt
        factor = -np.log10(np.e)               # τ (same units as t) = factor / k for log10
        tau = factor / k_hat if k_hat != 0 else np.nan
        se_k = np.sqrt(pcov[1, 1])
        k_lo, k_hi = k_hat - n_sigma*se_k, k_hat + n_sigma*se_k
        t_lo, t_hi = factor/k_lo, factor/k_hi
        tau_lo, tau_hi = (min(t_lo, t_hi), max(t_lo, t_hi))
        if tau_hi - tau_lo > 1e-3:
            pass
            # bounds = ([t[0], -np.inf, -np.inf], [t[-1], -5, np.inf])
            # popt, pcov = curve_fit(
            #     piecewise_model_continuous, t, trace,
            #     p0=p0, bounds=bounds, maxfev=20000
            # )
            # x_c_hat, k_hat, _ = popt
            # factor = -np.log10(np.e)               # τ (same units as t) = factor / k for log10
            # tau = factor / k_hat if k_hat != 0 else np.nan

        # x_c ± nσ
        xc_lo = xc_hi = np.nan
        if pcov is not None and np.isfinite(pcov[0, 0]):
            se_xc = np.sqrt(pcov[0, 0])
            xc_lo = x_c_hat - n_sigma*se_xc
            xc_hi = x_c_hat + n_sigma*se_xc

        # τ bounds from k ± nσ (asymmetric)
        tau_lo = tau_hi = np.nan
        if pcov is not None and np.isfinite(pcov[1, 1]):
            se_k = np.sqrt(pcov[1, 1])
            k_lo, k_hi = k_hat - n_sigma*se_k, k_hat + n_sigma*se_k
            if k_lo * k_hi > 0:
                t_lo, t_hi = factor/k_lo, factor/k_hi
                tau_lo, tau_hi = (min(t_lo, t_hi), max(t_lo, t_hi))

        fit_y = piecewise_model_continuous(t, *popt)

        # if slice==0:
        #     fig2, ax2 = plt.subplots(num="debug fit", figsize=(7, 3.5))

        #     ax2.plot(t, trace, '.', ms=3, label='data')
        #     ax2.plot(t, fit_y, '-', lw=1.6, label=rf'fit: $\tau={tau:.3g}$, $x_c={x_c_hat:.3g}$')
        #     ax2.axvline(x=x_c_hat, ls='--', alpha=0.5, label=r'$x_c$')
        #     ax2.set_xlabel("Time [µs]"); ax2.set_ylabel("Amplitude (log)")
        #     ax2.grid(True, alpha=0.3); ax2.legend()
        #     fig2.tight_layout()
        #     plt.show()

        return (tau, tau_lo, tau_hi, x_c_hat, xc_lo, xc_hi), (popt, fit_y)

    except Exception as e:
        return (np.nan,)*6, (None, None)


def lifetime_fit_matrix(trace_list, time_spacing_us, n_sigma=2.0):
    """Wrapper: apply the single-trace fit over all traces and return arrays."""
    N = len(trace_list)
    t = np.arange(len(trace_list[0])) * float(time_spacing_us)
    tau_array      = np.full(N, np.nan); CI_tau_lower = np.full(N, np.nan); CI_tau_upper = np.full(N, np.nan)
    xc_array       = np.full(N, np.nan); CI_xc_lower  = np.full(N, np.nan); CI_xc_upper  = np.full(N, np.nan)
    fit_curves = []

    for i, trace in enumerate(trace_list):
        (tau, tlo, thi, xc, xlo, xhi), fit = lifetime_fit_one_trace(t, trace, n_sigma=n_sigma, slice=i)
        tau_array[i], CI_tau_lower[i], CI_tau_upper[i] = tau, tlo, thi
        xc_array[i],  CI_xc_lower[i],  CI_xc_upper[i]  = xc,  xlo, xhi

        # large error indicate no decaying signal to fit (purely noise floor)
        if np.abs(thi - tlo) >= 0.05 or np.abs(xhi - xlo) >= 0.05:
            xc_bound = 0.5
            for _ in range(2):
                xc_bound = np.minimum(xc_bound, 3 * xc)
                xc_bound_idx = int(np.maximum(50, xc_bound * 552.96))
                (tau, tlo, thi, xc, xlo, xhi), fit = lifetime_fit_one_trace(t[0:xc_bound_idx], trace[0:xc_bound_idx], n_sigma=n_sigma, slice=i)
                tau_array[i], CI_tau_lower[i], CI_tau_upper[i] = tau, tlo, thi
                xc_array[i],  CI_xc_lower[i],  CI_xc_upper[i]  = xc,  xlo, xhi

        fit_curves.append(fit)

    return tau_array, CI_tau_lower, CI_tau_upper, xc_array, CI_xc_lower, CI_xc_upper
