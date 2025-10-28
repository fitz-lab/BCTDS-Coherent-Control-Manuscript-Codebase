import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# ─── global style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 18,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})


# Parameters
f1 = 3.0e9  # Hz
f2 = 4.0e9  # Hz
omega_1 = 2 * np.pi * f1
omega_2 = 2 * np.pi * f2
J = 5.0e3 * 1 * np.pi
Omega = 2 * np.pi * 1e2
# Time array
t_max = 1e-6
dt = 1.0e-9
t = np.arange(0, t_max, dt)
N = len(t)
# Drive frequency sweep
num_fr = 900
delta_f = 0.9e9
fr = np.linspace((f1 + f2) / 2 - delta_f, (f1 + f2) / 2 + delta_f, num_fr)
# fr = np.linspace(2, 5, 900)
omega_d = 2 * np.pi * fr
# FFT setup
fft_freqs = fftfreq(N, dt)
positive_freqs = fft_freqs[fft_freqs >= 0]
fft_plot_freqs_MHz = positive_freqs / 1e6
window = np.hanning(N)
eps = 1e-15
# Storage for FFT result
phase_eg_fft = np.zeros((len(positive_freqs), num_fr))
print("Starting computation...")
for i, wd in enumerate(omega_d):
    Delta1 = omega_1 - wd
    Delta2 = omega_2 - wd
    delta = Delta1 - Delta2
    # Zeroth order terms
    c_eg_0 = (Omega / 2) * (1 - np.exp(-1j * Delta1 * t)) / (Delta1 + eps)
    # First order terms
    term1 = (1 / (Delta2 + eps)) * (
        (1 - np.exp(-1j * Delta1 * t)) / (Delta1 + eps)
        + (1 - np.exp(1j * delta * t)) / (delta + eps)
    )
    c_eg_1 = (Omega / 2) * J * term1
    # Second order term (triple integral)
    exp_w = np.exp(1j * (omega_1 - wd) * t)
    exp_delta_pos = np.exp(1j * delta * t)
    exp_delta_neg = np.exp(-1j * delta * t)
    cum_tau3 = np.cumsum(exp_w) * dt
    I_tau2 = np.copy(cum_tau3)
    I_tau1 = np.zeros_like(t, dtype=complex)
    for idx1 in range(N):
        integrand_tau2 = exp_delta_pos[:idx1 + 1] * I_tau2[:idx1 + 1]
        I_tau1[idx1] = np.sum(integrand_tau2) * dt
    integrand_tau1 = exp_delta_neg * I_tau1
    I = np.cumsum(integrand_tau1) * dt
    c_eg_2 = 1j * (Omega / 2) * J**2 * I
    # Total amplitude
    c_eg = c_eg_0 + c_eg_1 + c_eg_2
    phi_eg = np.angle(c_eg)
    # FFT of windowed phase
    fft_phi_eg = np.abs(fft(phi_eg * window))[fft_freqs >= 0]
    max_eg = np.max(fft_phi_eg)
    # max_eg = 1.0
    if max_eg > 1e-15:
        phase_eg_fft[:, i] = fft_phi_eg / max_eg
    if i % 10 == 0:
        print(f"Processed {i + 1} / {num_fr} frequencies")

print("Computation finished.")
# Gamma correction
gamma = 0.4
vmax = np.max(phase_eg_fft)
vmin = 0.1 * vmax
phase_eg_fft_gamma = (phase_eg_fft / vmax) ** gamma * vmax

# Plotting setup (single column)
fig = plt.figure(figsize=(7, 7), constrained_layout=True)
# fig = plt.figure(figsize=(7, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.0], hspace=0.05)
# Heatmap
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(
    phase_eg_fft_gamma,
    aspect='auto',
    extent=[fr[0] / 1e9, fr[-1] / 1e9, fft_plot_freqs_MHz[0], fft_plot_freqs_MHz[-1]],
    origin='lower',
    cmap='inferno',
    vmin=vmin,
    vmax=vmax
)

ax1.set_ylabel(r'FFT Freq. [MHz]')
ax1.set_xlabel('Drive Frequency [GHz]')
# ax1.tick_params(labelbottom=False)
ax1.set_ylim(0, 300)
ax1.axhline(0, color='cyan', ls='--', lw=4.0)

# Linecut
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
idx = np.argmin(np.abs(fft_plot_freqs_MHz - 0.0))
ax2.plot(fr / 1e9, phase_eg_fft[idx], color='crimson', lw=3.0)
ax2.set_xlabel('Drive Frequency [GHz]')
ax2.set_ylabel(r'Norm. FFT($\phi$) [arb.]', labelpad=25)

# Colorbar
fig.colorbar(im1, ax=ax1, location='right', label=r'Norm. FFT($\phi$) [arb.]')
# Dashed vertical lines
for f in [3.0, 4.0]:
    ax1.axvline(f, color='white', linestyle='--', alpha=0.4)
    ax2.axvline(f, color='black', linestyle='--', alpha=0.4)

ax1.text(-0.08,1.15,r"\textbf{a}", transform=ax1.transAxes,
             fontsize=25, fontweight='bold')
ax2.text(-0.08,1.15,r"\textbf{b}", transform=ax2.transAxes,
             fontsize=25, fontweight='bold')

OUT_FIG = 'phase_Vs_analytical.png'
os.makedirs("../plots", exist_ok=True)
fig.savefig("../plots/"+OUT_FIG, dpi=300)
plt.close()