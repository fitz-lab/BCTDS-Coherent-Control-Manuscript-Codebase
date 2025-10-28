#!/usr/bin/env python3
"""
separation_phase_manipulation.py
────────────────────────────────
Three-panel figure:

 a) ring-down map (single pulse)
 b) gap sweep  (phase2 = 0)
 c) phase sweep (fixed gap)

Colour scale: ⟨σ⁺σ⁻⟩ in all panels.
"""
import os, datetime, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from hamiltonian_generator import (
    run_simulation_single_pulse,
    run_simulation_double_pulse_phase,
    build_spin_spin_interactions_random_distribution,
)

# ─────────── style ───────────
plt.rcParams.update({
    "font.size": 22,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ─────────── simulation knobs ───────────
PULSE_NS     = 200
AMP_DRIVE    = 0.12

T_MAX_NS, N_T = 1600, 1000
tlist          = np.linspace(0, T_MAX_NS, N_T)
μs             = 1e3

f_min, f_max   = 2.0, 5.0
FREQ_AXIS      = np.linspace(f_min, f_max, 400)

GAP_SWEEP_NS   = np.linspace(0, 800, 120)           # panel-b
PHASE_SWEEP    = np.linspace(0, 2*np.pi, 120)       # panel-c
FIXED_GAP_NS   = 150                               # panel-c

N_TLS          = 2
# np.random.seed(2072025)
np.random.seed(42)
init_freqs     = np.random.uniform(f_min, f_max, N_TLS)
init_freqs     = [3.0, 4.0]
H_int          = build_spin_spin_interactions_random_distribution(
                   N_TLS, -0.05, 0.05)

GAMMA, GAMMA_PHI = 0.002, 0.0
MAX_WORKERS      = 80

root = "../plots/ringdowns_sep_phase_manipulation"
os.makedirs(root, exist_ok=True)

# ───────── figure canvas ─────────
fig = plt.figure(figsize=(10, 12))
gs  = GridSpec(3, 1, hspace=0.35)
ax_ring, ax_gap, ax_phase = [fig.add_subplot(gs[i]) for i in range(3)]

def add_cbar(im, ax, label):
    cax = inset_axes(ax, width="1.8%", height="100%", loc="right",
                     bbox_to_anchor=(0.03, 0, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel(label)

dash_lw = 1.5

# ───────── panel-a : ring-down map ─────────
print("▸ Ring-down sweep …")
pop_map = np.zeros((len(FREQ_AXIS), len(tlist)))
with ProcessPoolExecutor(min(MAX_WORKERS, len(FREQ_AXIS))) as pool:
    futs = {pool.submit(run_simulation_single_pulse,
                        f_drv, AMP_DRIVE, tlist, init_freqs, H_int,
                        GAMMA, GAMMA_PHI, PULSE_NS): i
            for i, f_drv in enumerate(FREQ_AXIS)}
    for fut in tqdm(as_completed(futs), total=len(futs)):
        pop_map[futs[fut]] = fut.result()

mask_tail = tlist > PULSE_NS
idx_star  = np.argmax(pop_map[:, mask_tail].sum(axis=1))
FREQ_STAR = FREQ_AXIS[idx_star]

im_r = ax_ring.imshow(pop_map, origin="lower", aspect="auto",
                      extent=[tlist[0]/μs, tlist[-1]/μs,
                              FREQ_AXIS[0], FREQ_AXIS[-1]],
                      cmap="inferno")

# guides
ax_ring.axhline(FREQ_STAR, ls="--", color="cyan", lw=dash_lw)
for f0 in init_freqs:
    ax_ring.axhline(f0, ls="--", color="white", lw=1, alpha=0.4)
ax_ring.axvline(0,           ls="--", color="white", lw=dash_lw)
ax_ring.axvline(PULSE_NS/μs, ls="--", color="white", lw=dash_lw)

# labels
ax_ring.set_xlabel("Time [$\\mu$s]")
ax_ring.set_ylabel("$\omega_d$ [GHz]")
add_cbar(im_r, ax_ring, r"$\langle\sigma^{+}\sigma^{-}\rangle$")
ax_ring.text(-0.1, 1.04, r"\textbf{a}", transform=ax_ring.transAxes,
             ha="left", va="bottom", fontsize=28, fontweight="bold")

# common parameter annotation  (use \mathrm not \text)
ax_ring.text(
    0.98, 0.04,
    rf"$A = {AMP_DRIVE:.2f}$"
    "\n"
    rf"$\tau = {PULSE_NS}\,\mathrm{{ns}}$",
    transform=ax_ring.transAxes,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25",
              fc="white", ec="none", alpha=0.8)
)

# ───────── panel-b : gap sweep ─────────
print("▸ Gap sweep …")
pop_gap = np.zeros((len(GAP_SWEEP_NS), len(tlist)))
with ProcessPoolExecutor(min(MAX_WORKERS, len(GAP_SWEEP_NS))) as pool:
    futs = {pool.submit(run_simulation_double_pulse_phase,
                        FREQ_STAR, AMP_DRIVE, AMP_DRIVE,
                        PULSE_NS, gap, PULSE_NS, 0.0,
                        tlist, init_freqs, H_int,
                        GAMMA, GAMMA_PHI): i
            for i, gap in enumerate(GAP_SWEEP_NS)}
    for fut in tqdm(as_completed(futs), total=len(futs)):
        pop_gap[futs[fut]] = fut.result()

im_g = ax_gap.imshow(pop_gap, origin="lower", aspect="auto",
                     extent=[tlist[0]/μs, tlist[-1]/μs,
                             GAP_SWEEP_NS[0], GAP_SWEEP_NS[-1]],
                     cmap="inferno")

# pulse-1 edges
ax_gap.axvline(0,           ls="--", color="white", lw=dash_lw)
ax_gap.axvline(PULSE_NS/μs, ls="--", color="white", lw=dash_lw)
# pulse-2 start & end (diagonals)
ax_gap.plot(GAP_SWEEP_NS/μs + PULSE_NS/μs, GAP_SWEEP_NS,
            ls="--", color="cyan", lw=dash_lw)
ax_gap.plot((GAP_SWEEP_NS+PULSE_NS)/μs + PULSE_NS/μs, GAP_SWEEP_NS,
            ls="--", color="cyan", lw=dash_lw, alpha=0.6)

ax_gap.set_xlabel("Time [$\\mu$s]")
ax_gap.set_ylabel(r"$\tau_g$ [ns]")
add_cbar(im_g, ax_gap, r"$\langle\sigma^{+}\sigma^{-}\rangle$")
ax_gap.text(-0.1, 1.04, r"\textbf{b}", transform=ax_gap.transAxes,
            ha="left", va="bottom", fontsize=28, fontweight="bold")
ax_gap.text(
    0.98, 0.04,
    rf"$\omega_d = {FREQ_STAR:.2f}\,\mathrm{{GHz}}$",
    transform=ax_gap.transAxes,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25",
              fc="white", ec="none", alpha=0.8)
)

# ───────── panel-c : phase sweep ─────────
print("▸ Phase sweep …")
pop_phase = np.zeros((len(PHASE_SWEEP), len(tlist)))
with ProcessPoolExecutor(min(MAX_WORKERS, len(PHASE_SWEEP))) as pool:
    futs = {pool.submit(run_simulation_double_pulse_phase,
                        FREQ_STAR, AMP_DRIVE, AMP_DRIVE,
                        PULSE_NS, FIXED_GAP_NS, PULSE_NS, phase,
                        tlist, init_freqs, H_int,
                        GAMMA, GAMMA_PHI): i
            for i, phase in enumerate(PHASE_SWEEP)}
    for fut in tqdm(as_completed(futs), total=len(futs)):
        pop_phase[futs[fut]] = fut.result()

im_p = ax_phase.imshow(pop_phase, origin="lower", aspect="auto",
                       extent=[tlist[0]/μs, tlist[-1]/μs,
                               PHASE_SWEEP[0], PHASE_SWEEP[-1]],
                       cmap="inferno")

# pulse-edges
ax_phase.axvline(0,           ls="--", color="white", lw=dash_lw)
ax_phase.axvline(PULSE_NS/μs, ls="--", color="white", lw=dash_lw)
ax_phase.axvline((PULSE_NS+FIXED_GAP_NS)/μs,
                 ls="--", color="cyan", lw=dash_lw)
ax_phase.axvline((PULSE_NS+FIXED_GAP_NS+PULSE_NS)/μs,
                 ls="--", color="cyan", lw=dash_lw, alpha=0.6)

# y-ticks: 0, π, 2π
ax_phase.set_yticks([0, np.pi, 2*np.pi])
ax_phase.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])

ax_phase.set_xlabel("Time [$\\mu$s]")
ax_phase.set_ylabel("$\Delta \phi$")
add_cbar(im_p, ax_phase, r"$\langle\sigma^{+}\sigma^{-}\rangle$")
ax_phase.text(-0.1, 1.04, r"\textbf{c}", transform=ax_phase.transAxes,
              ha="left", va="bottom", fontsize=28, fontweight="bold")
ax_phase.text(
    0.98, 0.04,
    rf"$\tau_g = {FIXED_GAP_NS}\,\mathrm{{ns}}$",
    transform=ax_phase.transAxes,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25",
              fc="white", ec="none", alpha=0.8)
)

# ───────── save ─────────
outfile = os.path.join(
    root, f"sep_phase_layout_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
fig.savefig(outfile, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"✓ Figure saved → {outfile}")
