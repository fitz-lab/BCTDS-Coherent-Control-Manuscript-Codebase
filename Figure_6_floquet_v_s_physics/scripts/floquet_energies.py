#!/usr/bin/env python3
"""
ringdown_phase_floquet.py
─────────────────────────
(a) ⟨σ⁺σ⁻⟩ ring-down (b) phase-FFT (≤ 100 MHz) (c) Floquet branches

Physics & numerics unchanged – only the colour-bar placement is reordered so
they always line up with their panels.
"""

import os, datetime, multiprocessing as mp
import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qutip import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.ticker import FormatStrFormatter
from scipy.fft import fft, fftfreq
from tqdm import tqdm

# ─── global style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 22,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ─── model / simulation knobs (unchanged) ────────────────────────────
OMEGA_1,  OMEGA_2  = 3.0, 4.4
J_COUP,   DRIVE_AMP= 0.05, 0.10
T_PULSE,  T_TOTAL  = 100.0, 400.0         # ns
DT,       GAMMA_COL= 0.05,  2e-3
FREQ_GRID = np.linspace(3.0, 5.0, 400)    # GHz

# output file
OUT_FIG = f"try_ringdown_phase_floquet_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"

# ─── grids ───────────────────────────────────────────────────────────
T_NS  = np.arange(0.0, T_TOTAL, DT)
FFT_F = fftfreq(T_NS.size, DT)                     # GHz
MASK  = (FFT_F >= 0) & (FFT_F <= 0.15)             # 0-100 MHz
FFT_POS = FFT_F[MASK]

# ───────────── worker (must be picklable) ─────────────
def worker(pair):
    idx, f_drive = pair                       # GHz (already linear)

    # ───────────────────────── operators / static H ───────────────────────
    sx1 = tensor(sigmax(), qeye(2));  sx2 = tensor(qeye(2), sigmax())
    sz1 = tensor(sigmaz(), qeye(2));  sz2 = tensor(qeye(2), sigmaz())
    sp1 = tensor(sigmap(),  qeye(2)); sp2 = tensor(qeye(2), sigmap())
    sm1 = tensor(sigmam(),  qeye(2)); sm2 = tensor(qeye(2), sigmam())

    sp_tot, sm_tot = sp1 + sp2, sm1 + sm2
    pop_op = sp_tot * sm_tot                                    # ⟨σ⁺σ⁻⟩

    h_static = 0.5*OMEGA_1*sz1 + 0.5*OMEGA_2*sz2 + J_COUP*sx1*sx2
    psi0     = h_static.eigenstates()[1][0]
    cops = [np.sqrt(GAMMA_COL) * sm_tot]

    ### This is what we've been using this whole time.. If we use this, the pop map looks like the photographic negative of the one we are used to..
    # T_ramp_ns = 1.0
    # def drive_coeff(t, _=None):
    #     if t < T_ramp_ns:
    #         env = 0.5*(1-np.cos(np.pi*t/T_ramp_ns))
    #         return DRIVE_AMP*env*np.cos(f_drive*t)
    #     elif t < T_PULSE:
    #         return DRIVE_AMP*np.cos(f_drive*t)
    #     return 0.0
    
    def drive_coeff(t, args):
        """time-dependent drive: simple square pulse with cosine"""
        if t <= T_PULSE:
            return DRIVE_AMP * np.cos(f_drive * t)
        return 0.0
    
    h_full = [h_static, [sx1 + sx2, drive_coeff]]

    res = mesolve(h_full, psi0, T_NS, cops,
                [pop_op, sp1, sp2],
                options=Options(progress_bar=None, nsteps=5000))

    pop_t, sp1_t, sp2_t = res.expect

    phase  = np.angle(sp1_t) - np.angle(sp2_t)
    phase  = ((phase + np.pi) % (2*np.pi) - np.pi).astype(np.float32)

    fft_row = np.abs(fft(phase))[MASK].astype(np.float32)
    if fft_row.max() > 1e-14:
        fft_row /= fft_row.max()

    # Floquet quasi-energies (1 period = T_PULSE)
    try:
        U = propagator(h_full, T_PULSE, c_ops=[], options=Options(nsteps=5000))
        phases = np.angle(np.linalg.eigvals(U.full()))
        quasi  = np.sort(((phases+np.pi)%(2*np.pi)-np.pi) / T_PULSE).astype(np.float32)
    except Exception:
        quasi  = np.full(4, np.nan, dtype=np.float32)

    return idx, pop_t.real, fft_row, quasi

# ─── parallel sweep ─────────────────────────────────────────────────
POP   = np.empty((FREQ_GRID.size, T_NS.size),          dtype=np.float32)
FFT_H = np.empty((FREQ_GRID.size, FFT_POS.size),       dtype=np.float32)
QUASI = np.empty((FREQ_GRID.size, 4),                  dtype=np.float32)

with ProcessPoolExecutor(mp.cpu_count()) as pool:
    futures = [pool.submit(worker, (i,f)) for i,f in enumerate(FREQ_GRID)]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="drive sweep"):
        i, p, f, q = fut.result()
        POP[i], FFT_H[i], QUASI[i] = p, f, q

# ─── plotting ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(8,12))
gs  = GridSpec(3,1, height_ratios=[1.3,1.05,0.95], hspace=0.35)

ax_pop  = fig.add_subplot(gs[0])
ax_fft  = fig.add_subplot(gs[1])
ax_qfan = fig.add_subplot(gs[2])

im_pop = ax_pop.imshow(POP.T, origin='lower', aspect='auto',
                       extent=[FREQ_GRID[0], FREQ_GRID[-1], T_NS[0], T_NS[-1]],
                       cmap='inferno')
ax_pop.axhline(T_PULSE, ls='--', color='white', lw=1.4)
ax_pop.set_xlabel("Drive frequency [GHz]")
ax_pop.set_ylabel("Time [ns]")
ax_pop.text(-0.1,1.1,r"\textbf{a}", transform=ax_pop.transAxes,
            fontsize=30, fontweight='bold')

ax_pop.tick_params(axis='both', which='major', pad=15)   # 15 pt gap

im_fft = ax_fft.imshow(FFT_H.T, origin='lower', aspect='auto',
                       extent=[FREQ_GRID[0], FREQ_GRID[-1],
                               FFT_POS[0]*1e3, FFT_POS[-1]*1e3],
                       cmap='inferno')
ax_fft.set_xlabel("Drive frequency [GHz]")
ax_fft.set_ylabel("FFT Freq. [MHz]")
ax_fft.text(-0.1,1.1,r"\textbf{b}", transform=ax_fft.transAxes,
            fontsize=30, fontweight='bold')
ax_fft.tick_params(axis='both', which='major', pad=15)   # 15 pt gap

# for k in range(QUASI.shape[1]):
#     ax_qfan.plot(FREQ_GRID, QUASI[:,k]*1e3, lw=3.0, label=f"Q{k}")
# ax_qfan.set_xlabel("Drive frequency [GHz]")
# ax_qfan.set_ylabel("Quasi-energy [MHz]")
# ax_qfan.legend(fontsize=20, ncol=2, loc='upper left', frameon=False)

# --- plot the four curves ---
lines = []
for k in range(QUASI.shape[1]):
    ln, = ax_qfan.plot(FREQ_GRID, QUASI[:, k]*1e3,
                       lw=3.0, label=f"$Q_{k}$")
    lines.append(ln)

# --- build two separate legends: (Q0,Q1) top-left ; (Q2,Q3) bottom-left ---
top_handles   = lines[:2]
bottom_handles= lines[2:]

leg1 = ax_qfan.legend(top_handles, [h.get_label() for h in top_handles],
                      fontsize=17, loc='upper left', frameon=False)

leg2 = ax_qfan.legend(bottom_handles, [h.get_label() for h in bottom_handles],
                      fontsize=17, loc='lower left', frameon=False)

# keep both legends
ax_qfan.add_artist(leg1)
ax_qfan.set_xlabel("Drive frequency [GHz]")
ax_qfan.set_ylabel("Quasi-energy [MHz]")

ax_qfan.text(-0.1,1.1,r"\textbf{c}", transform=ax_qfan.transAxes,
             fontsize=30, fontweight='bold')
ax_qfan.set_xlim(FREQ_GRID.min(), FREQ_GRID.max())   # note the () 
ax_qfan.tick_params(axis='both', which='major', pad=15)   # 15 pt gap

# ---- global margins BEFORE colour-bars ----------------------------
fig.subplots_adjust(left=0.16 , right=0.83, bottom=0.08, top=0.94, hspace=0.35)

# ---- colour-bars (added *after* subplots_adjust) -------------------
def add_cbar(ax, im, label):
    cax = ax.inset_axes([1.02, 0, 0.03, 1], transform=ax.transAxes)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(label)

add_cbar(ax_pop, im_pop, r'$\langle\sigma^{+}\sigma^{-}\rangle$ [arb.]')
add_cbar(ax_fft, im_fft, r'Norm. FFT($\phi)$ [arb.]')

# -------------------------------------------------------------------
os.makedirs("../plots", exist_ok=True)
fig.savefig("../plots/"+OUT_FIG, dpi=260)
plt.close(fig)
print("✓ Figure saved →", OUT_FIG)

# entry-point
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
