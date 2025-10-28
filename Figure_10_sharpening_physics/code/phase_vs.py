#!/usr/bin/env python3
"""
figure_postpulse_layout_with_phaseV.py

Layout:
  • Row 1 (panel a):  ⟨σ⁺σ⁻⟩ vs time for τ = [min, median, max] (full trajectories; 3 panels)
  • Row 2 (panel b):  Phase V's = normalized | FFT[ φ(t) ] | with φ = arg(⟨σ⁺⟩) − arg(⟨σ⁻⟩),
                       computed for the same three pulse durations (3 panels), FULL trace FFT.
  • Row 3 (panel c):  Post-pulse map (spans 3 columns): ⟨σ⁺σ⁻⟩ sampled at first t > τ

Solver/physics unchanged; only phase-V computation and figure layout.
"""

# ───────────────────────── imports ───────────────────────────────────────
import os, datetime, numpy as np, matplotlib
matplotlib.use("Agg")                     # comment-out for interactive use
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.transforms import Bbox
import matplotlib.colors as mcolors
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.fft import fft, fftfreq

# –––– Hamiltonian helper functions ––––––
from hamiltonian_generator import run_simulation_for_frequency, build_spin_spin_interactions_random_distribution

# ─────────────────────── matplotlib style ───────────────────────────────
plt.rcParams.update({
    "font.size": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ─────────────────────── user knobs / constants ─────────────────────────
# Dense pulse grid for smooth bottom panel
PULSE_MIN_NS   = 20.0
PULSE_MAX_NS   = 200.0
N_PULSES       = 50
PULSES_NS      = np.linspace(PULSE_MIN_NS, PULSE_MAX_NS, N_PULSES)

# Top-row pulses
pulse_short = PULSES_NS[0]
pulse_mid   = PULSES_NS[len(PULSES_NS)//2]
pulse_long  = PULSES_NS[-1]
TOP_PULSES  = [pulse_short, pulse_mid, pulse_long]

# Frequency sweep
f_min, f_max   = 3.0, 5.0                 # GHz
N_FREQS        = 400
FREQ_AXIS      = np.linspace(f_min, f_max, N_FREQS)

# Time grid
T_MAX_NS, N_T  = 1000, 1000
tlist          = np.linspace(0.0, T_MAX_NS, N_T)
dt_ns          = tlist[1] - tlist[0]
μs             = 1e3

# TLS / drive (use your current values)
N_TLS          = 2
init_freqs     = [4.0, 4.1]               # GHz

# interactions: random dipolar with given J (same J_min=J_max)
J = 0.05
H_int = build_spin_spin_interactions_random_distribution(
    N_TLS, J, J, alpha_x=1.0, alpha_y=0.0, alpha_z=0.0
)

GAMMA, GAMMA_PHI, DRIVE_AMPL = 0.002, 0.0, 0.1

# Phase-V FFT settings
FFT_VIEW_MHZ   = 150.0                    # plot up to this frequency (MHz)

# Parallel
MAX_WORKERS    = 80

# Output dir
root = "../plots/ringdowns_final_layout_v5"
os.makedirs(root, exist_ok=True)

# ─────────────────────── helpers ─────────────────────────────────────────
def first_index_strictly_after(time_array: np.ndarray, tau_ns: float) -> int:
    """Return the first index j such that time_array[j] > tau_ns."""
    j = int(np.searchsorted(time_array, tau_ns, side="right"))
    if j >= len(time_array):
        raise ValueError(
            f"Pulse {tau_ns} ns exceeds simulation window (T_MAX_NS={T_MAX_NS} ns)."
        )
    return j

# ─────────────────────── storage for results ────────────────────────────
# Bottom panel data: one value per (pulse, frequency)
postpulse_map = np.zeros((len(PULSES_NS), len(FREQ_AXIS)), dtype=np.float32)

# Row 1 data (population dynamics): full time traces across frequency for selected pulses
top_pop_maps = {p: np.zeros((len(FREQ_AXIS), len(tlist)), dtype=np.float32) for p in TOP_PULSES}

# Row 2 data (phase V's): normalized |FFT(φ)| for the same pulses
phaseV_maps = {}  # dict: pulse_ns -> (phaseV_T, fft_freq_plot_MHz)

# ───────────────────────── main loop (by pulse) ─────────────────────────
for row_idx, pulse_ns in enumerate(PULSES_NS):
    j_post = first_index_strictly_after(tlist, float(pulse_ns))

    want_full = any(np.isclose(pulse_ns, p, rtol=0, atol=1e-12) for p in TOP_PULSES)
    if want_full:
        pop_buffer = np.zeros((len(FREQ_AXIS), len(tlist)), dtype=np.float32)  # (freq, time)
        phi_buffer = np.zeros_like(pop_buffer)                                  # phase(t) per freq

    # Parallel sweep over drive frequencies
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(FREQ_AXIS))) as pool:
        jobs = {
            pool.submit(
                run_simulation_for_frequency, f_drv, tlist,
                init_freqs, H_int,
                GAMMA, GAMMA_PHI,
                DRIVE_AMPL, float(pulse_ns)
            ): (i, f_drv)
            for i, f_drv in enumerate(FREQ_AXIS)
        }

        for fut in tqdm(as_completed(jobs), total=len(jobs),
                        desc=f"τ={pulse_ns:6.1f} ns  (dt≈{dt_ns:.3f} ns)"):
            i, _ = jobs[fut]
            # Generator may return (Sp, Sm, pop) OR (pop, Sp, Sm); detect the two complex arrays
            ret = fut.result()
            arrs = list(ret)
            complex_idx = [k for k, a in enumerate(arrs) if np.iscomplexobj(a)]

            if len(complex_idx) >= 2:
                sp1_tr = arrs[complex_idx[0]]
                sp2_tr = arrs[complex_idx[1]]
                real_idx = [k for k in range(3) if k not in complex_idx][0]
                pop_tr  = np.asarray(arrs[real_idx], dtype=np.float64).real
            else:
                # fallback to (Sp_tr, Sm_tr, pop_tr)
                sp1_tr, sp2_tr, pop_tr = ret
                pop_tr = np.asarray(pop_tr, dtype=np.float64).real

            # Bottom panel metric: first post-pulse sample of population
            postpulse_map[row_idx, i] = pop_tr[j_post].astype(np.float32)

            if want_full:
                # Row 1: population time trace
                pop_buffer[i, :] = pop_tr.astype(np.float32)

                # Row 2: phase(t) = arg(⟨σ⁺⟩) − arg(⟨σ⁻⟩), wrapped to (−π, π]
                phi_tr = np.angle(sp1_tr) - np.angle(sp2_tr)
                phi_tr = ((phi_tr + np.pi) % (2*np.pi) - np.pi).astype(np.float32)
                phi_buffer[i, :] = phi_tr

        if want_full:

            WIN_START_NS = 0
            WIN_STOP_NS = 400

            # Save row-1 data
            top_pop_maps[pulse_ns] = pop_buffer

            # ---- Phase-V map over a SELECTED TIME WINDOW, row-wise normalization ----
            # Build a boolean mask for t in [WIN_START_NS, WIN_STOP_NS]
            tmask = (tlist >= WIN_START_NS) & (tlist <= WIN_STOP_NS)
            if tmask.sum() < 2:
                raise ValueError(
                    f"Phase-V window too small: {tmask.sum()} points in [{WIN_START_NS}, {WIN_STOP_NS}] ns."
                )

            # Frequency axis from the *windowed* trace; keep positive freqs up to FFT_VIEW_MHZ
            freqs_GHz     = np.fft.fftfreq(tmask.sum(), dt_ns)   # 1/ns ≡ GHz
            pos_mask      = (freqs_GHz >= 0.0) & (freqs_GHz <= FFT_VIEW_MHZ/1e3)
            fft_freq_plot = (freqs_GHz[pos_mask] * 1e3).astype(np.float32)  # → MHz

            # FFT along time for each drive frequency (row-wise), using only the windowed data
            phi_win = phi_buffer[:, tmask]                        # (N_FREQS, Nt_win)
            PhiFFT  = np.fft.fft(phi_win, axis=1)[:, pos_mask]   # (N_FREQS, Nf)
            mag     = np.abs(PhiFFT).astype(np.float32)          # (N_FREQS, Nf)

            # Row-wise normalization (same as your reference)
            den = mag.max(axis=1, keepdims=True)                 # (N_FREQS, 1)
            mag_norm = np.divide(mag, den, out=np.zeros_like(mag), where=den > 1e-14)

            # Store as (Nf, N_FREQS) for imshow with origin='lower'
            phaseV_maps[pulse_ns] = (mag_norm.T, fft_freq_plot)
# ─────────────────────── figure layout (3 rows × 3 cols) ─────────────────
fig = plt.figure(figsize=(20, 13))
gs  = gridspec.GridSpec(
    3, 3, height_ratios=[1.0, 1.0, 1.15], width_ratios=[1, 1, 1],
    hspace=0.34, wspace=0.22
)

# Row-1 axes (population dynamics)
ax_top = [fig.add_subplot(gs[0, c]) for c in range(3)]
# Row-2 axes (phase V's)
ax_phase = [fig.add_subplot(gs[1, c]) for c in range(3)]
# Row-3 axis spans all columns (post-pulse map)
ax_bottom = fig.add_subplot(gs[2, :])

cmap = "inferno"

# ─────────────────────── Row 1: population dynamics (panel a) ───────────
all_top_vals = np.concatenate([top_pop_maps[p].ravel() for p in TOP_PULSES])
norm_top = mcolors.Normalize(vmin=np.nanmin(all_top_vals), vmax=np.nanmax(all_top_vals))
labels = [rf"{pulse_short:.0f} ns", rf"{pulse_mid:.0f} ns", rf"{pulse_long:.0f} ns"]

for ax, pulse_ns, lab in zip(ax_top, TOP_PULSES, labels):
    data = top_pop_maps[pulse_ns]  # (freq, time)
    im = ax.imshow(
        data.T, origin="lower", aspect="auto",
        extent=[FREQ_AXIS[0], FREQ_AXIS[-1], tlist[0]/μs, tlist[-1]/μs],
        cmap=cmap, norm=norm_top
    )
    # Pulse end line
    ax.axhline(pulse_ns/μs, ls="--", color="white", lw=1.5, alpha=0.9)
    # TLS guides
    for f0 in init_freqs:
        ax.axvline(f0, ls="--", color="white", alpha=0.5, lw=1.6)

    ax.set_xlabel("Frequency [GHz]", fontsize=30)
    ax.tick_params(axis="both", which="major", labelsize=26)
    ax.text(0.04, 0.08, lab, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=24,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

ax_top[0].set_ylabel("Time [$\\mu$s]", fontsize=30)

# One slim colorbar for the top row
bbox_top = Bbox.union([a.get_position() for a in ax_top])
cax_top = fig.add_axes([bbox_top.x1 + 0.015, bbox_top.y0, 0.012, bbox_top.y1 - bbox_top.y0])
cb_top  = fig.colorbar(plt.cm.ScalarMappable(norm=norm_top, cmap=cmap), cax=cax_top)
cb_top.ax.set_ylabel(r"$\langle \sigma^{+}\sigma^{-} \rangle$", fontsize=28, labelpad=12)
cb_top.ax.tick_params(labelsize=24)

# ─────────────────────── Row 2: phase V's (panel b) ─────────────────────
# Common normalization across the three phase-V panels: values already in [0,1]
norm_phase = mcolors.Normalize(vmin=0.0, vmax=1.0)

for ax, pulse_ns in zip(ax_phase, TOP_PULSES):
    phaseV_T, fft_freq_plot = phaseV_maps[pulse_ns]  # (Nf, N_FREQS), (Nf,)
    im = ax.imshow(
        phaseV_T, origin="lower", aspect="auto",
        extent=[FREQ_AXIS[0], FREQ_AXIS[-1], fft_freq_plot[0], fft_freq_plot[-1]],
        cmap=cmap, norm=norm_phase
    )
    # TLS guides
    for f0 in init_freqs:
        ax.axvline(f0, ls="--", color="white", alpha=0.5, lw=1.6)

    ax.set_xlabel("Frequency [GHz]", fontsize=30)
    ax.tick_params(axis="both", which="major", labelsize=26)

ax_phase[0].set_ylabel("FFT Freq. [MHz]", fontsize=30)

# Slim colorbar for phase-V row
bbox_phase = Bbox.union([a.get_position() for a in ax_phase])
cax_phase = fig.add_axes([bbox_phase.x1 + 0.015, bbox_phase.y0, 0.012, bbox_phase.y1 - bbox_phase.y0])
cb_phase  = fig.colorbar(plt.cm.ScalarMappable(norm=norm_phase, cmap=cmap), cax=cax_phase)
cb_phase.ax.set_ylabel(r"Norm. FFT($\phi$) [arb.]", fontsize=28, labelpad=12)
cb_phase.ax.tick_params(labelsize=24)

# ─────────────────────── Row 3: post-pulse map (panel c) ────────────────
im_bot = ax_bottom.imshow(
    postpulse_map,
    origin="lower",
    aspect="auto",
    extent=[FREQ_AXIS[0], FREQ_AXIS[-1], PULSES_NS[0]/μs, PULSES_NS[-1]/μs],
    cmap=cmap
)
for f0 in init_freqs:
    ax_bottom.axvline(f0, ls="--", color="white", alpha=0.5, lw=3.0)

ax_bottom.set_xlabel("Frequency [GHz]", fontsize=32)
ax_bottom.set_ylabel("Pulse Duration [$\\mu$s]", fontsize=32)
ax_bottom.tick_params(axis='both', which='major', labelsize=28)

# Colorbar for bottom panel
cax_bot = fig.add_axes([ax_bottom.get_position().x1 + 0.015,
                        ax_bottom.get_position().y0,
                        0.012,
                        ax_bottom.get_position().height])
cb_bot  = fig.colorbar(im_bot, cax=cax_bot)
cb_bot.ax.set_ylabel(r"$\langle \sigma^{+}\sigma^{-} \rangle$", fontsize=26, labelpad=12)
cb_bot.ax.tick_params(labelsize=24)

# ─────────────────────── tags & panel letters ───────────────────────────
# Roman i/ii/iii on Row 1 and Row 2
col_tags = ['i', 'ii', 'iii']
for ax, tag in zip(ax_top, col_tags):
    ax.text(0.98, 0.98, rf'\textbf{{{tag}}}', transform=ax.transAxes,
            ha='right', va='top', fontsize=26, fontweight='bold', color='white')
for ax, tag in zip(ax_phase, col_tags):
    ax.text(0.98, 0.98, rf'\textbf{{{tag}}}', transform=ax.transAxes,
            ha='right', va='top', fontsize=26, fontweight='bold', color='white')

# Panel letters a (row 1), b (row 2), c (row 3)
ax_top[0].text(-0.18, 1.02, r'\textbf{a}', transform=ax_top[0].transAxes,
               ha='left', va='bottom', fontsize=42, fontweight='bold', clip_on=False)
ax_phase[0].text(-0.18, 1.02, r'\textbf{b}', transform=ax_phase[0].transAxes,
                 ha='left', va='bottom', fontsize=42, fontweight='bold', clip_on=False)
ax_bottom.text(-0.06, 1.02, r'\textbf{c}', transform=ax_bottom.transAxes,
               ha='left', va='bottom', fontsize=42, fontweight='bold', clip_on=False)

# ───────────────────────── save & exit ───────────────────────────────────
fig.subplots_adjust(left=0.08)  # leave room for the three slim colorbars
outfile_png = os.path.join(
    root, f"NTLS_{N_TLS}_layout_postpulse_phaseV_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
)
fig.savefig(outfile_png, dpi=220)
plt.close(fig)

# Save raw arrays for reuse
np.save(os.path.join(root, "postpulse_map.npy"), postpulse_map)
np.save(os.path.join(root, "postpulse_freq_axis.npy"), FREQ_AXIS)
np.save(os.path.join(root, "postpulse_pulses_ns.npy"), PULSES_NS)
for p in TOP_PULSES:
    np.save(os.path.join(root, f"row1_pop_tau_{int(round(p))}ns.npy"), top_pop_maps[p])
    phaseV_T, fft_freq_plot = phaseV_maps[p]
    np.save(os.path.join(root, f"row2_phaseV_tau_{int(round(p))}ns.npy"), phaseV_T)
    np.save(os.path.join(root, f"row2_phaseV_freqs_MHz_tau_{int(round(p))}ns.npy"), fft_freq_plot)

print("✓ Figure saved →", outfile_png)
print("✓ Data saved   →", os.path.join(root, "postpulse_map.npy"))
