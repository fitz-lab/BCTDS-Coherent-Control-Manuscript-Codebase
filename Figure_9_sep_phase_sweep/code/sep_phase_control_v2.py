#!/usr/bin/env python3
"""
separation_phase_manipulation.py
────────────────────────────────
Three-panel figure:

 a) ring-down map (single pulse)
 b) gap sweep  (phase2 = 0, second pulse fixed in time)
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

T_MAX_NS, N_T = 2600, 1000
tlist          = np.linspace(0, T_MAX_NS, N_T)
dt_ns          = tlist[1] - tlist[0]          # <── step size (needed later)
μs             = 1e3

f_min, f_max   = 2.0, 5.0
FREQ_AXIS      = np.linspace(f_min, f_max, 400)

GAP_SWEEP_NS   = np.linspace(0, 800, 120)     # panel-b
PHASE_SWEEP    = np.linspace(0, 2*np.pi, 120) # panel-c
FIXED_GAP_NS   = 150                          # panel-c

N_TLS          = 2
np.random.seed(42)
init_freqs     = [3.0, 4.0]                   # deterministic pair
H_int          = build_spin_spin_interactions_random_distribution(
                   N_TLS, -0.05, 0.05)

GAMMA, GAMMA_PHI = 0.002, 0.0
MAX_WORKERS      = 80

root = "../plots/ringdowns_sep_phase_v2"
os.makedirs(root, exist_ok=True)

# ───────── figure canvas ─────────
fig = plt.figure(figsize=(8, 14))
gs  = GridSpec(3, 1, hspace=0.35)
ax_ring, ax_gap, ax_phase = [fig.add_subplot(gs[i]) for i in range(3)]

def add_cbar(im, ax, label):
    cax = inset_axes(ax, width="1.8%", height="100%", loc="right",
                     bbox_to_anchor=(0.03, 0, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
    fig.colorbar(im, cax=cax).ax.set_ylabel(label)

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
ax_ring.axhline(FREQ_STAR, ls="--", color="cyan", lw=dash_lw)
for f0 in init_freqs:
    ax_ring.axhline(f0, ls="--", color="white", lw=1, alpha=0.4)
ax_ring.axvline(0,           ls="--", color="white", lw=dash_lw)
ax_ring.axvline(PULSE_NS/μs, ls="--", color="white", lw=dash_lw)
ax_ring.set_xlabel("Time [$\\mu$s]")
ax_ring.set_ylabel("$\\omega_d/2\pi$ [GHz]")
add_cbar(im_r, ax_ring, r"$\langle\sigma^{+}\sigma^{-}\rangle$")
ax_ring.text(-0.1, 1.04, r"\textbf{a}", transform=ax_ring.transAxes,
             fontsize=30, fontweight="bold")
ax_ring.text(
    0.98, 0.04,
    rf"$A/2\pi = {AMP_DRIVE:.2f}$" + "\n" + rf"$\tau = {PULSE_NS}\,\mathrm{{ns}}$",
    transform=ax_ring.transAxes,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8)
)
ax_ring.set_xlim(0.0, 1.6)

# ───────── panel-b : gap sweep (second pulse fixed) ─────────
# ───────── panel-b : gap-sweep (pulse-2 pinned at t = 0 µs) ─────────
print("▸ Gap sweep …")

# ------------------------------------------------------------------ #
# 1) run traces                                                     #
# ------------------------------------------------------------------ #
pop_gap = np.zeros((len(GAP_SWEEP_NS), len(tlist)))          # raw solver data

with ProcessPoolExecutor(min(MAX_WORKERS, len(GAP_SWEEP_NS))) as pool:
    futures = {
        pool.submit(
            run_simulation_double_pulse_phase,
            FREQ_STAR,
            AMP_DRIVE, AMP_DRIVE,                       # A1, A2
            PULSE_NS, gap, PULSE_NS, 0.0,               # τ1, τgap, τ2, Δφ
            tlist, init_freqs, H_int,
            GAMMA, GAMMA_PHI
        ): i
        for i, gap in enumerate(GAP_SWEEP_NS)
    }
    for fut in tqdm(as_completed(futures), total=len(futures)):
        pop_gap[futures[fut]] = fut.result()

# ------------------------------------------------------------------ #
# 2) build aligned matrix: second pulse always starts at t = 0       #
# ------------------------------------------------------------------ #
dt_ns          = tlist[1] - tlist[0]                          # sample step
shift_samples  = ((PULSE_NS + GAP_SWEEP_NS) / dt_ns).astype(int)
max_shift      = shift_samples.max()                          # left padding
new_len        = len(tlist) + max_shift                       # full width

# pre-fill with zeros  → population is strictly zero where no drive
pop_shift = np.zeros((len(GAP_SWEEP_NS), new_len), dtype=np.float32)

for row, s in enumerate(shift_samples):
    start = max_shift - s
    pop_shift[row, start:start + len(tlist)] = pop_gap[row]

# relative time axis (µs); pulse-2 starts at t = 0
t_rel_ns = (np.arange(new_len) - max_shift) * dt_ns
t_rel_us = t_rel_ns / μs

# ------------------------------------------------------------------ #
# 3) plotting                                                        #
# ------------------------------------------------------------------ #
im_g = ax_gap.imshow(
    pop_shift,
    origin="lower", aspect="auto",
    extent=[t_rel_us[0], t_rel_us[-1],
            GAP_SWEEP_NS[0], GAP_SWEEP_NS[-1]],
    cmap="inferno"
)

# --- visual guides ------------------------------------------------- #
ax_gap.axvline(0,          ls="--", color="cyan",  lw=dash_lw)          # τ2 start
ax_gap.axvline(PULSE_NS/μs, ls="--", color="cyan",  lw=dash_lw, alpha=0.6)  # τ2 end

# pulse-1 start/end after re-centering (see text)
x_start_1 = -(PULSE_NS + GAP_SWEEP_NS) / μs   # τ1 start  (earlier in time)
x_end_1   = -GAP_SWEEP_NS            / μs     # τ1 end

ax_gap.plot(x_start_1, GAP_SWEEP_NS, ls="--", color="white", lw=dash_lw)
ax_gap.plot(x_end_1,   GAP_SWEEP_NS, ls="--", color="white", lw=dash_lw, alpha=0.6)

# --- axes, colour-bar, annotations -------------------------------- #
ax_gap.set_xlabel(r"Time relative to 2$^{\mathrm{nd}}$ pulse  [$\mu$s]")
ax_gap.set_ylabel(r"$\tau_g$ [ns]")
add_cbar(im_g, ax_gap, r"$\langle\sigma^{+}\sigma^{-}\rangle$")

ax_gap.text(-0.08, 1.05, r"\textbf{b}", transform=ax_gap.transAxes,
            fontsize=30, fontweight="bold", va="bottom")
ax_gap.text(
    0.98, 0.04, rf"$\omega_d/2\pi = {FREQ_STAR:.2f}\,\mathrm{{GHz}}$",
    transform=ax_gap.transAxes,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8)
)
# --- crop the view so only physical data are shown -----------------
left_lim  = -(PULSE_NS + GAP_SWEEP_NS.max()) / μs      # earliest τ1 start
right_lim = (T_MAX_NS - PULSE_NS) / μs                 # latest time after τ2
ax_gap.set_xlim(-1.0, 1.6)

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

ax_phase.axvline(PULSE_NS/μs,                 ls="--", color="white", lw=dash_lw)
ax_phase.axvline((PULSE_NS+FIXED_GAP_NS)/μs,  ls="--", color="cyan",  lw=dash_lw)
ax_phase.axvline((PULSE_NS+FIXED_GAP_NS+PULSE_NS)/μs,
                 ls="--", color="cyan", lw=dash_lw, alpha=0.6)

ax_phase.set_yticks([0, np.pi, 2*np.pi])
ax_phase.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
ax_phase.set_xlabel(r"Time [$\mu$s]")
ax_phase.set_ylabel(r"$\Delta \phi$")
add_cbar(im_p, ax_phase, r"$\langle\sigma^{+}\sigma^{-}\rangle$")
ax_phase.text(-0.1, 1.04, r"\textbf{c}", transform=ax_phase.transAxes,
              fontsize=30, fontweight="bold")
ax_phase.text(
    0.98, 0.04,
    fr"$\tau_g = {FIXED_GAP_NS}\,\mathrm{{ns}}$",
    transform=ax_phase.transAxes,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8)
)
ax_phase.set_xlim(0.0, 1.6)
# ───────── save ─────────
outfile = os.path.join(
    root, f"sep_phase_layout_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
fig.savefig(outfile, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"✓ Figure saved → {outfile}")
