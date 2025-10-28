"""
hamiltonian_generator.py
────────────────────────
Driven N-TLS helpers (QuTiP ≥ 5).

New in this version
───────────────────
• `run_simulation_single_pulse_full(...)`  returns
      pop(t)  = ⟨S⁺S⁻⟩
      Sp(t)   = ⟨S⁺⟩
      Sm(t)   = ⟨S⁻⟩
  while keeping the original API intact.
"""

import numpy as np
from qutip import qeye, destroy, tensor, mesolve

# ──────────────────── TLS operators ────────────────────
def _tls_ops(N):
    I2 = qeye(2)
    sm1 = destroy(2)
    sx1 = sm1 + sm1.dag()
    sy1 = -1j * (sm1 - sm1.dag())
    sz1 = 2 * sm1.dag() * sm1 - I2

    def T(op, k):            # tensor helper
        return tensor([I2]*k + [op] + [I2]*(N-k-1))

    sm = [T(sm1, k) for k in range(N)]
    sx = [T(sx1, k) for k in range(N)]
    sy = [T(sy1, k) for k in range(N)]
    sz = [T(sz1, k) for k in range(N)]
    return sm, sx, sy, sz


# ────────────────── single-pulse FULL ──────────────────
def run_simulation_single_pulse_full(freq, amp, tlist, init_freqs, interactions,
                                     gamma, gamma_phi, pulse_ns, T_ramp_ns=1.0):
    """
    Returns   pop(t), Sp(t), Sm(t)   for one drive frequency & amplitude.
      freq, amp – as before (freq in “GHz” ≡ ns⁻¹ numerically).
    """
    N = len(init_freqs)
    sm, sx, _, sz = _tls_ops(N)
    Sp = sum(s.dag() for s in sm)
    Sm = sum(sm)
    Sx = sum(sx)

    H_tls = sum(init_freqs[j]*sz[j]/2 for j in range(N))

    def drive_coeff(t, _=None):
        if t < T_ramp_ns:
            env = 0.5*(1-np.cos(np.pi*t/T_ramp_ns))
            return amp*env*np.cos(freq*t)
        elif t < pulse_ns:
            return amp*np.cos(freq*t)
        return 0.0

    H   = [H_tls + interactions, [Sx, drive_coeff]]
    psi = (H_tls + interactions).groundstate()[1]

    c_ops = [np.sqrt(gamma)*Sm]
    if gamma_phi:
        _, _, _, szs = _tls_ops(N)
        c_ops += [np.sqrt(gamma_phi)*z for z in szs]

    e_ops = [Sm.dag()*Sm, Sp, Sm]         # pop, Sp, Sm
    result = mesolve(H, psi, tlist, c_ops, e_ops, progress_bar=None)

    pop, sp, sm_exp = result.expect
    return np.real_if_close(pop), sp, sm_exp


# ────────────────── original thin wrapper ──────────────
def run_simulation_single_pulse(freq, amp, tlist, init_freqs, interactions,
                                gamma, gamma_phi, pulse_ns, T_ramp_ns=1.0):
    pop, _, _ = run_simulation_single_pulse_full(
        freq, amp, tlist, init_freqs, interactions,
        gamma, gamma_phi, pulse_ns, T_ramp_ns
    )
    return pop


# ────────────────── two-pulse (unchanged) ──────────────
def run_simulation_double_pulse(freq, amp1, amp2,
                                pulse1_ns, gap_ns, pulse2_ns,
                                tlist, init_freqs, interactions,
                                gamma, gamma_phi, T_ramp_ns=1.0):
    sm, sx, _, sz = _tls_ops(len(init_freqs))
    Sx, Sm = sum(sx), sum(sm)

    H_tls      = sum(init_freqs[j]*sz[j]/2 for j in range(len(init_freqs)))
    t_start_p2 = pulse1_ns + gap_ns
    def drive_coeff(t, _=None):
        if t < T_ramp_ns:
            env = 0.5*(1-np.cos(np.pi*t/T_ramp_ns))
            return amp1*env*np.cos(freq*t)
        elif t < pulse1_ns:
            return amp1*np.cos(freq*t)
        if t < t_start_p2:
            return 0.0
        t2 = t - t_start_p2
        if t2 < T_ramp_ns:
            env = 0.5*(1-np.cos(np.pi*t2/T_ramp_ns))
            return amp2*env*np.cos(freq*t)
        elif t2 < pulse2_ns:
            return amp2*np.cos(freq*t)
        return 0.0

    H = [H_tls + interactions, [Sx, drive_coeff]]
    psi0 = (H_tls + interactions).groundstate()[1]
    c_ops = [np.sqrt(gamma)*Sm]
    if gamma_phi:
        _, _, _, szs = _tls_ops(len(init_freqs))
        c_ops += [np.sqrt(gamma_phi)*z for z in szs]

    return np.real_if_close(
        mesolve(H, psi0, tlist, c_ops, [Sm.dag()*Sm], progress_bar=None).expect[0]
    )


# ───────── random XX (+optional YY/ZZ) couplings ───────
def build_spin_spin_interactions_random_distribution(N, J_min, J_max,
                                                     alpha_x=1.0, alpha_y=0.0, alpha_z=0.0):
    _, sx, sy, sz = _tls_ops(N)
    J = np.random.uniform(J_min, J_max, size=(N, N))
    J = np.tril(J, -1) + np.tril(J, -1).T
    H_int = 0
    for i in range(N):
        for j in range(i+1, N):
            Jij = J[i, j]
            H_int += Jij * (
                alpha_x*sx[i]*sx[j] + alpha_y*sy[i]*sy[j] + alpha_z*sz[i]*sz[j]
            )
    return H_int


# ───────── legacy wrapper (unchanged) ─────────
def run_simulation_for_frequency(freq, tlist, init_freqs, interactions,
                                 gamma, gamma_phi, drive_ampl, pulse_duration):
    return run_simulation_single_pulse(
        freq, drive_ampl, tlist, init_freqs, interactions,
        gamma, gamma_phi, pulse_duration
    )
