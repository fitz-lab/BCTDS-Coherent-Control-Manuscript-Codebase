#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from concurrent.futures import ProcessPoolExecutor, as_completed
from qutip import (tensor, qeye, sigmax, sigmaz, sigmap, sigmam,
                   mesolve, Options)
from scipy.fft import fft, fftfreq

# ── Global style (scaled for a single-column figure)
plt.rcParams.update({
    "font.size": 18,                 # slightly smaller than 2-col
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True
})

# ── Simulation knobs
omega_tls_1 = 4.0      # GHz
omega_tls_2 = 4.12     # GHz   # panels a use (4.0, 4.12)
J            = 0.01    # coupling
Omega_amp    = 0.10    # drive amplitude
T_drive      = 100.0   # ns
T_total      = 800.0   # ns
dt           = 0.01    # ns
omega_d_vals = np.linspace(3.0, 5.0, 400)   # drive sweep (GHz)

# Dissipation
gamma_collective = 0.002   # ns^-1
gamma_local_1    = 0.0001  # ns^-1
gamma_local_2    = 0.0005  # ns^-1

# FFT-of-phase settings
WIN_START_NS  = 0.0
WIN_STOP_NS   = 300.0
FFT_VIEW_MHZ  = 150.0

# Parallelism
MAX_WORKERS = 80

# Output
out_dir = "../plots"
os.makedirs(out_dir, exist_ok=True)
outfile = os.path.join(out_dir, f"phaseV_pop_numerics_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")

# ── Time & freq grids
tlist = np.arange(0.0, T_total, dt)
num_t = tlist.size
num_d = omega_d_vals.size

tmask = (tlist >= WIN_START_NS) & (tlist <= WIN_STOP_NS)
if tmask.sum() < 2:
    raise ValueError("FFT window is too small. Increase WIN_STOP_NS or reduce WIN_START_NS.")

freqs_GHz = fftfreq(tmask.sum(), dt)       # 1/ns = GHz
mask_pos  = (freqs_GHz >= 0.0) & (freqs_GHz <= FFT_VIEW_MHZ/1e3)
fft_freq_plot_MHz = (freqs_GHz[mask_pos] * 1e3).astype(np.float32)  # x-axis for FFT panels
Nf = mask_pos.sum()

# ── Drive coefficient (accepts optional args)
def drive_coeff_factory(omega_d):
    def dcoeff(t, args=None):
        return Omega_amp * np.cos(omega_d * t) if (t <= T_drive) else 0.0
    return dcoeff

# ── Workers
def simulate_two_tls(idx_omega, w1, w2):
    """Two spins with given frequencies (w1, w2)."""
    idx, omega_d = idx_omega

    sx1 = tensor(sigmax(), qeye(2)); sx2 = tensor(qeye(2), sigmax())
    sz1 = tensor(sigmaz(), qeye(2)); sz2 = tensor(qeye(2), sigmaz())
    sp1 = tensor(sigmap(), qeye(2)); sp2 = tensor(qeye(2), sigmap())
    sm1 = tensor(sigmam(), qeye(2)); sm2 = tensor(qeye(2), sigmam())
    sp_tot = sp1 + sp2
    sm_tot = sm1 + sm2
    pop_op = sp_tot * sm_tot

    H0    = 0.5*w1*sz1 + 0.5*w2*sz2
    Hint  = J*sx1*sx2
    Hstat = H0 + Hint
    Hdrv  = [[sx1 + sx2, drive_coeff_factory(omega_d)]]
    Hfull = [Hstat] + Hdrv

    c_ops = [
        np.sqrt(gamma_collective)*sm_tot,
        np.sqrt(gamma_local_1)*sm1,
        np.sqrt(gamma_local_2)*sm2
    ]

    _, evecs = Hstat.eigenstates()
    psi0 = evecs[0]

    opts = Options(nsteps=5000, progress_bar=None)
    res = mesolve(Hfull, psi0, tlist, c_ops=c_ops,
                  e_ops=[pop_op, sp1, sp2],
                  options=opts)
    pop_t = np.real(res.expect[0]).astype(np.float32)
    sp1_t = res.expect[1]; sp2_t = res.expect[2]

    # φ(t) = arg⟨σ1+⟩ − arg⟨σ2+⟩, wrapped to (−π,π]
    phase = (np.angle(sp1_t) - np.angle(sp2_t)).astype(np.float32)
    phase = ((phase + np.pi) % (2*np.pi) - np.pi).astype(np.float32)

    phase_win = phase[tmask]
    fft_row   = np.abs(fft(phase_win))[mask_pos].astype(np.float32)
    if fft_row.max() > 1e-14:
        fft_row /= fft_row.max()
    return idx, pop_t, fft_row

# ── Parallel sweeps
pairs = [(i, w) for i, w in enumerate(omega_d_vals)]

# Panels a: non-degenerate (4.0, 4.12)
print("▸ Running two-TLS sweep for panel a (ω1=4.0, ω2=4.12)…")
pop_a = np.empty((num_d, num_t), dtype=np.float32)
fft_a = np.empty((num_d, Nf),    dtype=np.float32)
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = [pool.submit(simulate_two_tls, p, 4.0, 4.12) for p in pairs]
    for fut in as_completed(futures):
        i, pop_t, fft_row = fut.result()
        pop_a[i, :], fft_a[i, :] = pop_t, fft_row

# Panels c: degenerate (4.0, 4.0)
print("▸ Running two-TLS sweep for panel c (ω1=ω2=4.0)…")
pop_c = np.empty((num_d, num_t), dtype=np.float32)
fft_c = np.empty((num_d, Nf),    dtype=np.float32)
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = [pool.submit(simulate_two_tls, p, 4.0, 4.0) for p in pairs]
    for fut in as_completed(futures):
        i, pop_t, fft_row = fut.result()
        pop_c[i, :], fft_c[i, :] = pop_t, fft_row

# ── Figure: single column with two pairs (a on top, c on bottom)
# Keep pair aspect; shrink width about half vs 2-column figure
fig = plt.figure(figsize=(10.5, 10.0))

outer = GridSpec(
    2, 1,                     # two rows, one column
    left=0.10, right=0.89, bottom=0.08, top=0.92,
    hspace=0.42
)

pair_kwargs = dict(width_ratios=[1.0, 4.2], wspace=0.05)

# Top pair (panel a)
sg_a = outer[0, 0].subgridspec(1, 2, **pair_kwargs)
ax_a_phase = fig.add_subplot(sg_a[0, 0])
ax_a_pop   = fig.add_subplot(sg_a[0, 1], sharey=ax_a_phase)
ax_a_pop.set_ylabel("")
ax_a_pop.tick_params(axis="y", left=False, labelleft=False)

# Bottom pair (panel c)
sg_c = outer[1, 0].subgridspec(1, 2, **pair_kwargs)
ax_c_phase = fig.add_subplot(sg_c[0, 0])
ax_c_pop   = fig.add_subplot(sg_c[0, 1], sharey=ax_c_phase)
ax_c_pop.set_ylabel("")
ax_c_pop.tick_params(axis="y", left=False, labelleft=False)

# Extents
extent_fft = [fft_freq_plot_MHz[0], fft_freq_plot_MHz[-1],
              omega_d_vals[0],     omega_d_vals[-1]]
extent_pop = [tlist[0]/1e3, tlist[-1]/1e3, omega_d_vals[0], omega_d_vals[-1]]

cmap_fft = "inferno"
cmap_pop = "inferno"

# Thin, top-mounted horizontal colorbar for FFT panels (label on top)
def add_phase_cbar_thin_top(ax, im, label=r"Norm.\ FFT($\phi$)"):
    cax = ax.inset_axes([0.05, 1.03, 0.90, 0.035], transform=ax.transAxes)
    cb  = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_label_position('top')
    cb.set_label(label, labelpad=6)
    cb.ax.tick_params(axis='x', pad=10, labelsize=11)
    cb.ax.tick_params(bottom=False, top=False, labelbottom=False, labeltop=False)
    return cb

# Slim vertical colorbar for population panels
def add_cbar_vertical(ax, im, label):
    cax = ax.inset_axes([1.02, 0.00, 0.028, 1.00], transform=ax.transAxes)
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label)
    return cb

# Make labels a bit tighter for single column
for a in [ax_a_phase, ax_a_pop, ax_c_phase, ax_c_pop]:
    a.tick_params(axis="both", which="major", pad=6)

# ── Panel a
im_a_fft = ax_a_phase.imshow(fft_a, origin="lower", aspect="auto",
                             extent=extent_fft, cmap=cmap_fft)
ax_a_phase.set_xlabel("FFT Freq. [MHz]")
ax_a_phase.set_ylabel("Drive Freq. [GHz]")
add_phase_cbar_thin_top(ax_a_phase, im_a_fft)

im_a_pop = ax_a_pop.imshow(pop_a, origin="lower", aspect="auto",
                           extent=extent_pop, cmap=cmap_pop)
ax_a_pop.set_xlabel(r"Time [$\mu$s]")
add_cbar_vertical(ax_a_pop, im_a_pop, r"$\langle\sigma^{+}\sigma^{-}\rangle$")
# im_a_pop.axvline(T_drive/1e3, ls='--', lw=3.0, alpha=0.5, color='white')

# ── Panel c
im_c_fft = ax_c_phase.imshow(fft_c, origin="lower", aspect="auto",
                             extent=extent_fft, cmap=cmap_fft)
ax_c_phase.set_xlabel("FFT Freq. [MHz]")
ax_c_phase.set_ylabel("Drive Freq. [GHz]")
add_phase_cbar_thin_top(ax_c_phase, im_c_fft)

im_c_pop = ax_c_pop.imshow(pop_c, origin="lower", aspect="auto",
                           extent=extent_pop, cmap=cmap_pop)
# im_c_pop.axvline(T_drive/1e3, ls='--', lw=3.0, alpha=0.5, color='white')

ax_c_pop.set_xlabel(r"Time [$\mu$s]")
add_cbar_vertical(ax_c_pop, im_c_pop, r"$\langle\sigma^{+}\sigma^{-}\rangle$")

# ── Panel letters (“a” for top pair, “b” for bottom pair) + i/ii tags
ax_a_phase.text(-0.60, 1.00, r"\textbf{a}",
                transform=ax_a_phase.transAxes, ha="left", va="bottom",
                fontsize=28, fontweight="bold", clip_on=False)
ax_c_phase.text(-0.60, 1.00, r"\textbf{b}",
                transform=ax_c_phase.transAxes, ha="left", va="bottom",
                fontsize=28, fontweight="bold", clip_on=False)

ax_a_pop.axvline(T_drive/1e3, ls='--', lw=3.0, alpha=0.5, color='white')
ax_c_pop.axvline(T_drive/1e3, ls='--', lw=3.0, alpha=0.5, color='white')

for ax in [ax_a_phase, ax_c_phase]:
    ax.text(0.985, 0.985, r"\textbf{i}", transform=ax.transAxes,
            ha="right", va="top", fontsize=24, fontweight="bold", color="white")
for ax in [ax_a_pop, ax_c_pop]:
    ax.text(0.985, 0.985, r"\textbf{ii}", transform=ax.transAxes,
            ha="right", va="top", fontsize=24, fontweight="bold", color="white")

# Save
fig.savefig(outfile, dpi=220)
plt.close(fig)
print(f"✓ Figure saved → {outfile}")
