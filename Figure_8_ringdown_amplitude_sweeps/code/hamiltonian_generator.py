"""
hamiltonian_generator.py
────────────────────────
QuTiP helpers for driven N-TLS simulations (Python ≥ 3.9, QuTiP ≥ 5).

Changes in this revision
────────────────────────
• `run_simulation_double_pulse` takes
      pulse1_ns, gap_ns, pulse2_ns            (instead of pulse1_ns, delay_ns, pulse2_ns)
  where `gap_ns` = time waited *after pulse 1 finishes*.
• Drive functions use the same envelope as the single-pulse definition.
"""

import numpy as np
from qutip import qeye, destroy, tensor, mesolve


# ────────────────────────── TLS operators ────────────────────────────
def _tls_ops(N):
    I2 = qeye(2)
    sm1 = destroy(2)
    sx1 = sm1 + sm1.dag()
    sy1 = -1j * (sm1 - sm1.dag())
    sz1 = 2 * sm1.dag() * sm1 - I2

    def T(op, k):
        return tensor([I2] * k + [op] + [I2] * (N - k - 1))

    sm = [T(sm1, k) for k in range(N)]
    sx = [T(sx1, k) for k in range(N)]
    sy = [T(sy1, k) for k in range(N)]
    sz = [T(sz1, k) for k in range(N)]
    return sm, sx, sy, sz


# ────────────────────────── single pulse ─────────────────────────────
def run_simulation_single_pulse(freq, amp, tlist, init_freqs, interactions,
                                gamma, gamma_phi, pulse_ns, T_ramp_ns=1.0):
    """
    freq : ns⁻¹ (treat GHz numerically as ns⁻¹, so no 2π factor here)
    amp  : drive amplitude
    """
    N = len(init_freqs)
    sm, sx, _, sz = _tls_ops(N)
    Sx, Sm = sum(sx), sum(sm)

    H_tls = sum(init_freqs[j] * sz[j] / 2 for j in range(N))

    T_ramp = T_ramp_ns
    def drive_coeff(t, _=None):
        if t < T_ramp:
            env = 0.5 * (1 - np.cos(np.pi * t / T_ramp))
            return max(0.0, amp * env * np.cos(freq * t))
        elif t < pulse_ns:
            return max(0.0, amp * np.cos(freq * t))
        return 0.0

    H = [H_tls + interactions, [Sx, drive_coeff]]

    c_ops = [np.sqrt(gamma) * Sm]
    if gamma_phi:
        _, _, _, szs = _tls_ops(N)
        c_ops += [np.sqrt(gamma_phi) * szs[j] for j in range(N)]

    psi0 = (H_tls + interactions).groundstate()[1]
    e_ops = [Sm.dag() * Sm]

    return np.real_if_close(
        mesolve(H, psi0, tlist, c_ops, e_ops, progress_bar=None).expect[0]
    )


# ───────────────────────── two pulses ────────────────────────────────
def run_simulation_double_pulse(freq, amp1, amp2,
                                pulse1_ns, gap_ns, pulse2_ns,
                                tlist, init_freqs, interactions,
                                gamma, gamma_phi, T_ramp_ns=1.0):
    """
    Two pulses at the same frequency.
      • Pulse 1: starts at t=0, duration = pulse1_ns
      • Gap    : gap_ns  (idle time **after** pulse 1 ends)
      • Pulse 2: duration = pulse2_ns, amplitude = amp2
    """
    N = len(init_freqs)
    sm, sx, _, sz = _tls_ops(N)
    Sx, Sm = sum(sx), sum(sm)

    H_tls = sum(init_freqs[j] * sz[j] / 2 for j in range(N))
    t_start_p2 = pulse1_ns + gap_ns       # absolute start of pulse 2

    T_ramp = T_ramp_ns
    def drive_coeff(t, _=None):
        # ------- pulse 1 -------
        if t < T_ramp:
            env = 0.5 * (1 - np.cos(np.pi * t / T_ramp))
            return max(0.0, amp1 * env * np.cos(freq * t))
        elif t < pulse1_ns:
            return max(0.0, amp1 * np.cos(freq * t))

        # ------- gap -------
        if t < t_start_p2:
            return 0.0

        # ------- pulse 2 -------
        t2 = t - t_start_p2
        if t2 < T_ramp:
            env = 0.5 * (1 - np.cos(np.pi * t2 / T_ramp))
            return max(0.0, amp2 * env * np.cos(freq * t))
        elif t2 < pulse2_ns:
            return max(0.0, amp2 * np.cos(freq * t))
        return 0.0

    H = [H_tls + interactions, [Sx, drive_coeff]]

    c_ops = [np.sqrt(gamma) * Sm]
    if gamma_phi:
        _, _, _, szs = _tls_ops(N)
        c_ops += [np.sqrt(gamma_phi) * szs[j] for j in range(N)]

    psi0 = (H_tls + interactions).groundstate()[1]
    e_ops = [Sm.dag() * Sm]

    return np.real_if_close(
        mesolve(H, psi0, tlist, c_ops, e_ops, progress_bar=None).expect[0]
    )


# ─────────────────── random XX couplings (αy,z optional) ─────────────
def build_spin_spin_interactions_random_distribution(N, J_min, J_max,
                                                     alpha_x=1.0, alpha_y=0.0, alpha_z=0.0):
    _, sx, sy, sz = _tls_ops(N)
    J = np.random.uniform(J_min, J_max, size=(N, N))
    J = np.tril(J, -1) + np.tril(J, -1).T

    H_int = 0
    for i in range(N):
        for j in range(i + 1, N):
            Jij = J[i, j]
            H_int += Jij * (
                alpha_x * sx[i] * sx[j] +
                alpha_y * sy[i] * sy[j] +
                alpha_z * sz[i] * sz[j]
            )
    return H_int


# ───────── legacy wrapper (unchanged signature) ─────────
def run_simulation_for_frequency(freq, tlist, init_freqs, interactions,
                                 gamma, gamma_phi, drive_ampl, pulse_duration):
    return run_simulation_single_pulse(freq, drive_ampl, tlist, init_freqs,
                                       interactions, gamma, gamma_phi, pulse_duration)
