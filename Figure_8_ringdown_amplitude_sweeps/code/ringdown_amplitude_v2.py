#!/usr/bin/env python3
"""
ringdown_amplitude_sweeps.py
────────────────────────────
Three aligned panels with consistent annotations, guides, and panel tags.
"""

import os, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from hamiltonian_generator import (
    run_simulation_single_pulse,
    run_simulation_double_pulse,
    build_spin_spin_interactions_random_distribution,
)

# ────────────────── global style ──────────────────
plt.rcParams.update({
    "font.size": 22,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ────────────────── simulation knobs ring-down experiment ──────────────
PULSE_RING = 200              # pulse used in ring-down map

# ────────────────── simulation knobs dual pulse ──────────────
PULSE1_NS  = PULSE_RING       # duration of pulse-1 (sweeps)
GAP_NS     = 100              # gap after pulse-1
PULSE2_NS  = 200              # duration of pulse-2

f_min, f_max   = 3.0, 5.0
FREQ_AXIS      = np.linspace(f_min, f_max, 400)

T_MAX_NS, N_T  = 1600, 1000
tlist          = np.linspace(0, T_MAX_NS, N_T)
μs             = 1e3          # ns → μs

N_TLS          = 4

np.random.seed(2072025)
init_freqs     = np.random.uniform(f_min, f_max, N_TLS)

print(init_freqs)

H_int = build_spin_spin_interactions_random_distribution(
    N_TLS, -0.05, 0.05, alpha_x=1.0, alpha_y=0.0, alpha_z=0.0
)

print(H_int)
## This was kinda interesting tho. I think we're in the regime of very small interactions.
# H_int = build_spin_spin_interactions_random_distribution(
#     N_TLS, -0.005, 0.005, alpha_x=1.0, alpha_y=0.0, alpha_z=0.0
# )

GAMMA, GAMMA_PHI = 0.002, 0.0
AMP_BASE   = 0.10
AMP1_FIXED = 0.10

AMP_SWEEP  = np.linspace(0.0, 0.10, 60)
AMP2_SWEEP = np.linspace(0.0, 0.10, 60)

MAX_WORKERS = 70
root = "../plots/ringdowns_dual_pulse_column_v2"
os.makedirs(root, exist_ok=True)

# # ────────────────── figure canvas ─────────────────
# fig = plt.figure(figsize=(15, 10))
# gs  = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.30, wspace=0.45)
# ax_ring   = fig.add_subplot(gs[0, :])
# ax_amp    = fig.add_subplot(gs[1, 0])
# ax_double = fig.add_subplot(gs[1, 1])

def add_cbar(im, ax, label):
    """Inset colour-bar that keeps panel widths identical."""
    cax = inset_axes(ax, width="2%", height="100%", loc="right",
                     bbox_to_anchor=(0.05, 0, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel(label)
    return cb

# ────────────────── figure canvas ─────────────────
fig = plt.figure(figsize=(8, 14))      # taller, single-column
gs  = GridSpec(
    3, 1,                # ← 3 rows, 1 column
    height_ratios=[1.0, 1.0, 1.0],      # adjust if you want different heights
    hspace=0.35          # vertical spacing between panels
)

ax_ring   = fig.add_subplot(gs[0])      # top panel
ax_amp    = fig.add_subplot(gs[1])      # middle panel
ax_double = fig.add_subplot(gs[2])      # bottom panel

dash_lw = 1.5   # ← unified line-width for all guides

# ───────────── 1 ▪ ring-down map ─────────────
print("▸ Ring-down map …")
pop_tr = np.zeros((len(FREQ_AXIS), len(tlist)))
with ProcessPoolExecutor(min(MAX_WORKERS, len(FREQ_AXIS))) as pool:
    futures = {pool.submit(
        run_simulation_single_pulse, f_drv, AMP_BASE,
        tlist, init_freqs, H_int, GAMMA, GAMMA_PHI, PULSE_RING
    ): i for i, f_drv in enumerate(FREQ_AXIS)}
    for fut in tqdm(as_completed(futures), total=len(futures)):
        pop_tr[futures[fut]] = fut.result()

mask_tail   = tlist > PULSE1_NS
idx_longest = np.argmax(pop_tr[:, mask_tail].sum(axis=1))
FREQ_STAR   = FREQ_AXIS[idx_longest]

im_ring = ax_ring.imshow(
    pop_tr, origin="lower", aspect="auto",
    extent=[tlist[0]/μs, tlist[-1]/μs, FREQ_AXIS[0], FREQ_AXIS[-1]],
    cmap="inferno"
)
ax_ring.axhline(FREQ_STAR, ls="--", color="cyan", lw=dash_lw, alpha=0.8)
for f0 in init_freqs:
    ax_ring.axhline(f0, ls="--", color="white", lw=dash_lw, alpha=0.3)
ax_ring.axvline(0,           ls="--", color="white", lw=dash_lw)
ax_ring.axvline(PULSE_RING/μs, ls="--", color="white", lw=dash_lw)

ax_ring.set_xlabel("Time [$\\mu$s]")
ax_ring.set_ylabel(r"$\omega_d/2\pi$ [GHz]")
add_cbar(im_ring, ax_ring, r"$\langle \sigma^{+}\sigma^{-} \rangle$")

# annotation box (lower right)
ax_ring.text(
    0.98, 0.05,
    fr"$\tau = {PULSE_RING}\,\mathrm{{ns}}$"
    "\n"
    fr"$A/2\pi = {AMP_BASE:.2f}$",
    transform=ax_ring.transAxes,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8)
)

# panel tag
ax_ring.text(-0.1, 1.12, r"\textbf{a}", transform=ax_ring.transAxes,
             ha="left", va="bottom", fontsize=30, fontweight="bold")

# ───────────── 2 ▪ single-pulse sweep ────────────
print("▸ Single-pulse amplitude sweep …")
pop_amp = np.zeros((len(AMP_SWEEP), len(tlist)))
with ProcessPoolExecutor(min(MAX_WORKERS, len(AMP_SWEEP))) as pool:
    futures = {pool.submit(
        run_simulation_single_pulse,
        FREQ_STAR, amp, tlist, init_freqs, H_int,
        GAMMA, GAMMA_PHI, PULSE1_NS
    ): i for i, amp in enumerate(AMP_SWEEP)}
    for fut in tqdm(as_completed(futures), total=len(futures)):
        pop_amp[futures[fut]] = fut.result()

im_amp = ax_amp.imshow(
    pop_amp, origin="lower", aspect="auto",
    extent=[tlist[0]/μs, tlist[-1]/μs, AMP_SWEEP[0], AMP_SWEEP[-1]],
    cmap="inferno"
)
ax_amp.axvline(0,           ls="--", color="white", lw=dash_lw)
ax_amp.axvline(PULSE1_NS/μs, ls="--", color="white", lw=dash_lw)

ax_amp.set_xlabel("Time [$\\mu$s]")
ax_amp.set_ylabel("$A_1/2\pi$")
add_cbar(im_amp, ax_amp, r"$\langle \sigma^{+}\sigma^{-} \rangle$")

ax_amp.text(
    0.98, 0.05, fr"$\omega_d/2\pi = {FREQ_STAR:.2f}\,\mathrm{{GHz}}$",
    transform=ax_amp.transAxes,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8)
)
ax_amp.text(-0.1, 1.12, r"\textbf{b}", transform=ax_amp.transAxes,
            ha="left", va="bottom", fontsize=30, fontweight="bold")

# ───────────── 3 ▪ two-pulse sweep  (sweep A1, keep A2 fixed) ─────────────
print("▸ First-pulse amplitude sweep …")

AMP1_SWEEP = np.linspace(0.0, 0.10, 60)   # ← sweep A1
AMP2_FIXED = 0.10                         # ← keep A2 constant

pop_double = np.zeros((len(AMP1_SWEEP), len(tlist)))

with ProcessPoolExecutor(min(MAX_WORKERS, len(AMP1_SWEEP))) as pool:
    futures = {pool.submit(
        run_simulation_double_pulse,
        FREQ_STAR,
        amp1,               # A1 swept
        AMP2_FIXED,         # A2 fixed
        PULSE1_NS, GAP_NS, PULSE2_NS,
        tlist, init_freqs, H_int,
        GAMMA, GAMMA_PHI
    ): i for i, amp1 in enumerate(AMP1_SWEEP)}
    for fut in tqdm(as_completed(futures), total=len(futures)):
        pop_double[futures[fut]] = fut.result()

im_d = ax_double.imshow(
    pop_double, origin="lower", aspect="auto",
    extent=[tlist[0]/μs, tlist[-1]/μs, AMP1_SWEEP[0], AMP1_SWEEP[-1]],
    cmap="inferno"
)

# pulse guides (unchanged)
ax_double.axvline(0,                       ls="--", color="white", lw=dash_lw)
ax_double.axvline(PULSE1_NS/μs,            ls="--", color="white", lw=dash_lw)
ax_double.axvline((PULSE1_NS+GAP_NS)/μs,   ls="--", color="white", lw=dash_lw)
ax_double.axvline((PULSE1_NS+GAP_NS+PULSE2_NS)/μs,
                  ls="--", color="white", lw=dash_lw)

ax_double.set_xlabel("Time [$\\mu$s]")
ax_double.set_ylabel("$A_1/2\pi$")                       # ← y-axis now A1
add_cbar(im_d, ax_double, r"$\langle \sigma^{+}\sigma^{-} \rangle$")

# annotation & tag
ax_double.text(
    0.95, 0.05,
    fr"$\omega_d/2\pi = {FREQ_STAR:.2f}\,\mathrm{{GHz}}$"
    "\n"
    fr"$A_2/2\pi = {AMP2_FIXED:.2f}$",                  # ← show fixed A2
    transform=ax_double.transAxes,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8)
)
ax_double.text(-0.1, 1.12, r"\textbf{c}", transform=ax_double.transAxes,
               ha="left", va="bottom", fontsize=30, fontweight="bold")

# ───────────── save ─────────────
outfile = os.path.join(
    root,
    f"ringdown_amp_sweeps_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
)
fig.savefig(outfile, dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ Figure saved →", outfile)
