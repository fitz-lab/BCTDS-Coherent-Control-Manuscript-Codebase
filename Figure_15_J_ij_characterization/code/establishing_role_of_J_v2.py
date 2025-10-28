#!/usr/bin/env python3
"""
exploring_interactions.py
─────────────────────────
4 × 3 figure (rows × columns):

  row-1 : ⟨σ⁺σ⁻⟩
  row-2 : log₁₀|FFT⟨σ⁺σ⁻⟩|
  row-3 : phase ϕ = atan2(Im⟨σ⁺⟩, Re⟨σ⁺⟩)   (wrapped → (−π,π])
  row-4 : log₁₀|FFT ϕ|

x-axis   → drive-frequency  [GHz]   (all panels)  
y-axis   → time [µs] (rows 1 & 3) / FFT freq [MHz] (rows 2 & 4)

Set AFTER_PULSE_ONLY = False to include the pulse window in the FFTs.
"""
import os, datetime, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from hamiltonian_generator import (
    run_simulation_single_pulse_full,
    build_spin_spin_interactions_random_distribution,
)

# ────────── GLOBAL STYLE ──────────
plt.rcParams.update({
    "font.size": 22,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ────────── SIMULATION KNOBS ──────────
PULSE_NS            = 400
AFTER_PULSE_ONLY    = False      # set False to FFT the full trace
T_MAX_NS, N_T       = 1600, 1000
tlist               = np.linspace(0, T_MAX_NS, N_T)
μs                  = 1e3       # ns → µs

f_min, f_max        = 3.0, 5.0
FREQ_AXIS           = np.linspace(f_min, f_max, 300)

N_TLS          = 4
np.random.seed(42)
base_freqs     = np.random.uniform(f_min, f_max, N_TLS)

GAMMA, GAMMA_PHI    = 0.001, 0.0
AMP_DRIVE           = 0.1
T_RAMP_NS           = 1.0

J_levels = {
    "Low $J$":  0.005,
    "Mid $J$":  0.05,
    "High $J$": 0.5,
}

fft_view_MHz        = 300
MAX_WORKERS         = 70
root                = "../plots_v4/interaction_scan_operator"
os.makedirs(root, exist_ok=True)

# ────────── FIGURE CANVAS ──────────
fig = plt.figure(figsize=(18, 16))
gs  = GridSpec(4, 3, hspace=0.35, wspace=0.30)
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(4)])

def add_cbar(im, ax, label):
    cax = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.012,
                        ax.get_position().height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel(label)

dash_lw = 1.3

# ────────── MAIN LOOP OVER J ──────────
for col, (title, Jval) in enumerate(J_levels.items()):
    print(f"▸ {title} all-to-all coupled with the same (J = {Jval})")

    ## Or J to J...
    H_int = build_spin_spin_interactions_random_distribution(
        N_TLS, Jval, Jval, alpha_x=1.0, alpha_y=0.0, alpha_z=0.0
    )

    pop_mat = np.empty((N_T, len(FREQ_AXIS)))
    phi_mat = np.empty_like(pop_mat)

    # —— frequency sweep ——
    with ProcessPoolExecutor(min(MAX_WORKERS, len(FREQ_AXIS))) as pool:
        futures = {pool.submit(
            run_simulation_single_pulse_full,
            f_drv, AMP_DRIVE, tlist, base_freqs, H_int,
            GAMMA, GAMMA_PHI, PULSE_NS, T_RAMP_NS
        ): j for j, f_drv in enumerate(FREQ_AXIS)}

        for fut in tqdm(as_completed(futures), total=len(futures)):
            j                 = futures[fut]
            pop, Sp, Sm        = fut.result()
            pop_mat[:, j]     = pop
            # homodyne-like phase (wrapped to −π…π)
            phi_mat[:, j]     = np.angle(Sp) - np.angle(Sm)

    # eliminate NaNs (very weak signal)
    phi_mat = np.nan_to_num(phi_mat)

    # —— choose window for FFTs ——
    mask_tail  = (tlist > PULSE_NS) if AFTER_PULSE_ONLY else np.ones_like(tlist, bool)
    tail_t     = tlist[mask_tail]
    fft_freq_MHz = np.fft.rfftfreq(len(tail_t),
                                   d=(tlist[1]-tlist[0]))*1e3
    keep_idx   = fft_freq_MHz <= fft_view_MHz
    fft_freq   = fft_freq_MHz[keep_idx]

    fft_pop = np.log10(
        np.abs(np.fft.rfft(pop_mat[mask_tail, :], axis=0)[keep_idx, :]) + 1e-12
    )
    fft_phi = np.log10(
        np.abs(np.fft.rfft(phi_mat[mask_tail, :], axis=0)[keep_idx, :]) + 1e-12
    )

    # ── row-1 : ⟨σ⁺σ⁻⟩ ──
    ax1 = axes[0, col]
    im1 = ax1.imshow(pop_mat,
                     origin="lower", aspect="auto",
                     extent=[FREQ_AXIS[0], FREQ_AXIS[-1],
                             tlist[0]/μs, tlist[-1]/μs],
                     cmap="inferno")
    if col == 0:
        ax1.set_ylabel("Time [$\\mu$s]", labelpad=25)
    ax1.set_title(title)
    if col == 2:
        add_cbar(im1, ax1, r"$\langle\sigma^{+}\sigma^{-}\rangle$")
    ax1.axhline(PULSE_NS/μs, ls="--", color="white", lw=dash_lw, alpha=0.6)

    # ── row-2 : FFT(pop) ──
    ax2 = axes[1, col]
    im2 = ax2.imshow(fft_pop,
                     origin="lower", aspect="auto",
                     extent=[FREQ_AXIS[0], FREQ_AXIS[-1],
                             fft_freq[0], fft_freq[-1]],
                     cmap="inferno")
    if col == 0:
        ax2.set_ylabel("FFT Freq. [MHz]")
    if col == 2:
        add_cbar(im2, ax2,
                 r"$\log_{10}|\mathrm{FFT}(\langle\sigma^+\sigma^-\rangle)|$")

    # ── row-3 : phase ϕ ──
    ax3 = axes[2, col]
    im3 = ax3.imshow(phi_mat,
                     origin="lower", aspect="auto",
                     extent=[FREQ_AXIS[0], FREQ_AXIS[-1],
                             tlist[0]/μs, tlist[-1]/μs],
                     cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)
    if col == 0:
        ax3.set_ylabel("Time [$\\mu$s]", labelpad=25)
    if col == 2:
        add_cbar(im3, ax3, r"$\phi$ [rad]")
    ax3.axhline(PULSE_NS/μs, ls="--", color="white", lw=dash_lw, alpha=0.6)

    # ── row-4 : FFT(ϕ) ──
    ax4 = axes[3, col]
    im4 = ax4.imshow(fft_phi,
                     origin="lower", aspect="auto",
                     extent=[FREQ_AXIS[0], FREQ_AXIS[-1],
                             fft_freq[0], fft_freq[-1]],
                     cmap="inferno")
    ax4.set_xlabel("Drive Freq. [GHz]")
    if col == 0:
        ax4.set_ylabel("FFT Freq. [MHz]")
    if col == 2:
        add_cbar(im4, ax4, r"$\log_{10}|\mathrm{FFT}(\phi)|$")

    for f0 in base_freqs:
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvline(f0, ls="--", color="white", lw=dash_lw)

# ── (1) Panel row tags: a / b / c / d  ─────────────────────────────
row_tags = ['a', 'b', 'c', 'd']
for r, tag in enumerate(row_tags):
    axes[r, 0].text(
        -0.15, 1.02, rf'\textbf{{{tag}}}',
        transform=axes[r, 0].transAxes,
        ha='left', va='bottom',
        fontsize=28, fontweight='bold'
    )

# ── (2) Column tags: i / ii / iii on EVERY row ─────────────────────
col_tags = ['i', 'ii', 'iii']                 # tags for the three columns
for r in range(axes.shape[0]):                # loop over all 4 rows
    for c, tag in enumerate(col_tags):        # loop over the 3 columns
        axes[r, c].text(
            0.98, 0.97, rf'\textbf{{{tag}}}',
            transform=axes[r, c].transAxes,
            ha='right', va='top',
            fontsize=24, fontweight='bold',
            color='white'
        )

# ── (3) Give every subplot a little extra space under the x-tick labels ──
for ax in axes.flatten():
    ax.tick_params(axis='x', which='major', pad=12)   # adjust pad value as needed

# ─────────── SAVE ───────────
out = os.path.join(root,
                   f"interaction_scan_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"✓ Figure saved → {out}")
