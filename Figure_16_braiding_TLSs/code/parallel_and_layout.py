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

# ── Global style (LaTeX-safe)
plt.rcParams.update({
    "font.size": 20,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True
})

# ── Simulation knobs
omega_tls_1 = 4.0      # GHz
omega_tls_2 = 4.12     # GHz   ← panels a/b use (4.0, 4.12)
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
outfile = os.path.join(out_dir, f"exp_theory_layout_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")

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
def simulate_two_tls(idx_omega):
    """Two spins with possibly different frequencies (for panels a/b)."""
    idx, omega_d = idx_omega

    sx1 = tensor(sigmax(), qeye(2)); sx2 = tensor(qeye(2), sigmax())
    sz1 = tensor(sigmaz(), qeye(2)); sz2 = tensor(qeye(2), sigmaz())
    sp1 = tensor(sigmap(), qeye(2)); sp2 = tensor(qeye(2), sigmap())
    sm1 = tensor(sigmam(), qeye(2)); sm2 = tensor(qeye(2), sigmam())
    sp_tot = sp1 + sp2
    sm_tot = sm1 + sm2
    pop_op = sp_tot * sm_tot

    H0    = 0.5*omega_tls_1*sz1 + 0.5*omega_tls_2*sz2
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

def simulate_two_tls_degenerate(idx_omega):
    """Two spins with **degenerate** frequencies (for panels c/d)."""
    idx, omega_d = idx_omega

    # Degenerate frequencies 4.0 GHz and 4.0 GHz
    w1 = 4.0
    w2 = 4.0

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

    # Same phase definition
    phase = (np.angle(sp1_t) - np.angle(sp2_t)).astype(np.float32)
    phase = ((phase + np.pi) % (2*np.pi) - np.pi).astype(np.float32)

    phase_win = phase[tmask]
    fft_row   = np.abs(fft(phase_win))[mask_pos].astype(np.float32)
    if fft_row.max() > 1e-14:
        fft_row /= fft_row.max()
    return idx, pop_t, fft_row

# ── Parallel sweeps
pairs = [(i, w) for i, w in enumerate(omega_d_vals)]

print("▸ Running TWO-TLS sweep (panels a/b: ω1=4.0, ω2=4.12)…")
pop_two = np.empty((num_d, num_t), dtype=np.float32)
fft_two = np.empty((num_d, Nf),    dtype=np.float32)
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = [pool.submit(simulate_two_tls, p) for p in pairs]
    for fut in as_completed(futures):
        i, pop_t, fft_row = fut.result()
        pop_two[i, :], fft_two[i, :] = pop_t, fft_row

print("▸ Running TWO-TLS **degenerate** sweep (panels c/d: ω1=ω2=4.0)…")
# Keep variable names 'pop_one/fft_one' so plotting below stays identical
pop_one = np.empty((num_d, num_t), dtype=np.float32)
fft_one = np.empty((num_d, Nf),    dtype=np.float32)
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = [pool.submit(simulate_two_tls_degenerate, p) for p in pairs]
    for fut in as_completed(futures):
        i, pop_t, fft_row = fut.result()
        pop_one[i, :], fft_one[i, :] = pop_t, fft_row

# “Experimental” placeholders (copy theory for layout)
fft_two_exp = fft_two.copy()
pop_two_exp = pop_two.copy()
fft_one_exp = fft_one.copy()
pop_one_exp = pop_one.copy()

# ── Figure using nested GridSpecs so (i, ii) are close & share y-axis ──
fig = plt.figure(figsize=(20.0, 10.0))

outer = GridSpec(
    2, 2,
    height_ratios=[1, 1],
    width_ratios=[1, 1],
    left=0.07, right=0.92, bottom=0.10, top=0.90,
    wspace=0.35, hspace=0.40
)

pair_kwargs = dict(width_ratios=[1.0, 4.0], wspace=0.05)

# Top-left pair (panel a)
sg_a = outer[0, 0].subgridspec(1, 2, **pair_kwargs)
ax_r1_c0 = fig.add_subplot(sg_a[0, 0])
ax_r1_c1 = fig.add_subplot(sg_a[0, 1], sharey=ax_r1_c0)
ax_r1_c1.set_ylabel("")
ax_r1_c1.tick_params(axis="y", left=False, labelleft=False)

# Top-right pair (panel b)
sg_b = outer[0, 1].subgridspec(1, 2, **pair_kwargs)
ax_r1_c2 = fig.add_subplot(sg_b[0, 0])
ax_r1_c3 = fig.add_subplot(sg_b[0, 1], sharey=ax_r1_c2)
ax_r1_c3.set_ylabel("")
ax_r1_c3.tick_params(axis="y", left=False, labelleft=False)

# Bottom-left pair (panel c)
sg_c = outer[1, 0].subgridspec(1, 2, **pair_kwargs)
ax_r2_c0 = fig.add_subplot(sg_c[0, 0])
ax_r2_c1 = fig.add_subplot(sg_c[0, 1], sharey=ax_r2_c0)
ax_r2_c1.set_ylabel("")
ax_r2_c1.tick_params(axis="y", left=False, labelleft=False)

# Bottom-right pair (panel d)
sg_d = outer[1, 1].subgridspec(1, 2, **pair_kwargs)
ax_r2_c2 = fig.add_subplot(sg_d[0, 0])
ax_r2_c3 = fig.add_subplot(sg_d[0, 1], sharey=ax_r2_c2)
ax_r2_c3.set_ylabel("")
ax_r2_c3.tick_params(axis="y", left=False, labelleft=False)

# Extents
extent_fft = [fft_freq_plot_MHz[0], fft_freq_plot_MHz[-1],
              omega_d_vals[0],     omega_d_vals[-1]]
extent_pop = [tlist[0], tlist[-1], omega_d_vals[0], omega_d_vals[-1]]

cmap_fft = "inferno"
cmap_pop = "inferno"

# Thin, top-mounted horizontal colorbar for FFT panels (label on top)
def add_phase_cbar_thin_top(ax, im, label=r"Norm.\ FFT($\phi$)"):
    cax = ax.inset_axes([0.05, 1.03, 0.90, 0.035], transform=ax.transAxes)
    cb  = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_label_position('top')
    cb.set_label(label, labelpad=8)
    cb.ax.tick_params(axis='x', pad=20, labelsize=12)
    return cb

# Slim vertical colorbar for population panels
def add_cbar_vertical(ax, im, label):
    cax = ax.inset_axes([1.02, 0.00, 0.028, 1.00], transform=ax.transAxes)
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=18)
    cb.set_label(label)
    return cb

# Make labels less cramped
for a in [ax_r1_c0, ax_r1_c1, ax_r1_c2, ax_r1_c3, ax_r2_c0, ax_r2_c1, ax_r2_c2, ax_r2_c3]:
    a.tick_params(axis="both", which="major", pad=8)

# ── Row 1 (two TLS, different freqs → panels a/b)
im_r1_fft_exp = ax_r1_c0.imshow(fft_two_exp, origin="lower", aspect="auto",
                                extent=extent_fft, cmap=cmap_fft)
ax_r1_c0.set_xlabel("FFT Freq. [MHz]")
ax_r1_c0.set_ylabel("Drive Freq. [GHz]")
add_phase_cbar_thin_top(ax_r1_c0, im_r1_fft_exp)

im_r1_pop_exp = ax_r1_c1.imshow(pop_two_exp, origin="lower", aspect="auto",
                                extent=extent_pop, cmap=cmap_pop)
ax_r1_c1.set_xlabel("Time [ns]")
add_cbar_vertical(ax_r1_c1, im_r1_pop_exp, r"$\langle\sigma^{+}\sigma^{-}\rangle$")
ax_r1_c1.text(0.03, 0.08, "experiment",
              transform=ax_r1_c1.transAxes, fontsize=24,
              ha="left", va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", fc="0.85", ec="0.7"))

im_r1_fft_theo = ax_r1_c2.imshow(fft_two, origin="lower", aspect="auto",
                                 extent=extent_fft, cmap=cmap_fft)
ax_r1_c2.set_xlabel("FFT Freq. [MHz]")
ax_r1_c2.set_ylabel("Drive Freq. [GHz]")
add_phase_cbar_thin_top(ax_r1_c2, im_r1_fft_theo)

im_r1_pop_theo = ax_r1_c3.imshow(pop_two, origin="lower", aspect="auto",
                                 extent=extent_pop, cmap=cmap_pop)
ax_r1_c3.set_xlabel("Time [ns]")
add_cbar_vertical(ax_r1_c3, im_r1_pop_theo, r"$\langle\sigma^{+}\sigma^{-}\rangle$")
ax_r1_c3.text(0.03, 0.08, "numerics",
              transform=ax_r1_c3.transAxes, fontsize=24,
              ha="left", va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", fc="0.85", ec="0.7"))

# ── Row 2 (two TLS, **degenerate** → panels c/d)
im_r2_fft_exp = ax_r2_c0.imshow(fft_one_exp, origin="lower", aspect="auto",
                                extent=extent_fft, cmap=cmap_fft)
ax_r2_c0.set_xlabel("FFT Freq. [MHz]")
ax_r2_c0.set_ylabel("Drive Freq. [GHz]")
add_phase_cbar_thin_top(ax_r2_c0, im_r2_fft_exp)

im_r2_pop_exp = ax_r2_c1.imshow(pop_one_exp, origin="lower", aspect="auto",
                                extent=extent_pop, cmap=cmap_pop)
ax_r2_c1.set_xlabel("Time [ns]")
add_cbar_vertical(ax_r2_c1, im_r2_pop_exp, r"$\langle\sigma^{+}\sigma^{-}\rangle$")
ax_r2_c1.text(0.03, 0.08, "experiment",
              transform=ax_r2_c1.transAxes, fontsize=24,
              ha="left", va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", fc="0.85", ec="0.7"))

im_r2_fft_theo = ax_r2_c2.imshow(fft_one, origin="lower", aspect="auto",
                                 extent=extent_fft, cmap=cmap_fft)
ax_r2_c2.set_xlabel("FFT Freq. [MHz]")
ax_r2_c2.set_ylabel("Drive Freq. [GHz]")
add_phase_cbar_thin_top(ax_r2_c2, im_r2_fft_theo)

im_r2_pop_theo = ax_r2_c3.imshow(pop_one, origin="lower", aspect="auto",
                                 extent=extent_pop, cmap=cmap_pop)
ax_r2_c3.set_xlabel("Time [ns]")
add_cbar_vertical(ax_r2_c3, im_r2_pop_theo, r"$\langle\sigma^{+}\sigma^{-}\rangle$")
ax_r2_c3.text(0.03, 0.08, "numerics",
              transform=ax_r2_c3.transAxes, fontsize=24,
              ha="left", va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", fc="0.85", ec="0.7"))

# ── Pair letters (a–d) and subindices (i on phase, ii on ring-down)
pair_phase_axes = [ax_r1_c0, ax_r1_c2, ax_r2_c0, ax_r2_c2]
pair_pop_axes   = [ax_r1_c1, ax_r1_c3, ax_r2_c1, ax_r2_c3]
panel_letters   = ['a', 'b', 'c', 'd']

for ax, lab in zip(pair_phase_axes, panel_letters):
    ax.text(-0.6, 1.15, rf"\textbf{{{lab}}}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=35, fontweight="bold", clip_on=False)

for ax in pair_phase_axes:
    ax.text(0.985, 0.985, r"\textbf{i}", transform=ax.transAxes,
            ha="right", va="top", fontsize=22, fontweight="bold", color="white")
for ax in pair_pop_axes:
    ax.text(0.985, 0.985, r"\textbf{ii}", transform=ax.transAxes,
            ha="right", va="top", fontsize=22, fontweight="bold", color="white")

# Save
fig.savefig(outfile, dpi=220)
plt.close(fig)
print(f"✓ Figure saved → {outfile}")
