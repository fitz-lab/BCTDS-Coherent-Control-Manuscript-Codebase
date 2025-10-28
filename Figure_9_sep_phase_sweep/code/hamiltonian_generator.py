"""
hamiltonian_generator.py
────────────────────────
Driven N-TLS helpers (QuTiP ≥ 5).

Adds:
• run_simulation_double_pulse_phase(...)  – second pulse with phase offset φ₂.
• run_simulation_single_pulse_full(...)   – returns ⟨S⁺S⁻⟩, ⟨S⁺⟩, ⟨S⁻⟩.

All earlier APIs continue to work.
"""
import numpy as np
from qutip import qeye, destroy, tensor, mesolve, Options

_opts_no_bar = Options(progress_bar=None)   # silence progress-bar warnings

# ───────── TLS operator factory ─────────
def _tls_ops(N):
    I2 = qeye(2)
    sm1 = destroy(2)
    sx1 = sm1 + sm1.dag()
    sy1 = -1j * (sm1 - sm1.dag())
    sz1 = 2 * sm1.dag() * sm1 - I2

    def T(op, k):
        return tensor([I2]*k + [op] + [I2]*(N-k-1))

    sm = [T(sm1, k) for k in range(N)]
    sx = [T(sx1, k) for k in range(N)]
    sy = [T(sy1, k) for k in range(N)]
    sz = [T(sz1, k) for k in range(N)]
    return sm, sx, sy, sz


# ───────── single-pulse (pop, Sp, Sm) ─────────
def run_simulation_single_pulse_full(freq, amp, tlist, init_freqs, interactions,
                                     gamma, gamma_phi, pulse_ns, T_ramp_ns=1.0):
    N = len(init_freqs)
    sm, sx, _, sz = _tls_ops(N)
    Sp = sum(s.dag() for s in sm)
    Sm = sum(sm)
    Sx = sum(sx)

    H_tls = sum(init_freqs[j]*sz[j]/2 for j in range(N))

    def d_coeff(t, _=None):
        if t < T_ramp_ns:
            env = 0.5*(1-np.cos(np.pi*t/T_ramp_ns))
            return amp*env*np.cos(freq*t)
        elif t < pulse_ns:
            return amp*np.cos(freq*t)
        return 0.0

    H   = [H_tls + interactions, [Sx, d_coeff]]
    ψ0  = (H_tls + interactions).groundstate()[1]

    c_ops  = [np.sqrt(gamma)*Sm]
    if gamma_phi:
        _, _, _, szs = _tls_ops(N)
        c_ops += [np.sqrt(gamma_phi)*z for z in szs]

    pop, sp, sm_exp = mesolve(
        H, ψ0, tlist, c_ops, [Sm.dag()*Sm, Sp, Sm], options=_opts_no_bar
    ).expect
    return np.real_if_close(pop), sp, sm_exp


# convenience → only ⟨S⁺S⁻⟩
def run_simulation_single_pulse(*args, **kw):
    return run_simulation_single_pulse_full(*args, **kw)[0]


# ───────── two-pulse with phase shift φ₂ ─────────
def run_simulation_double_pulse_phase(freq, amp1, amp2,
                                      pulse1_ns, gap_ns, pulse2_ns,
                                      phase2_rad,
                                      tlist, init_freqs, interactions,
                                      gamma, gamma_phi, T_ramp_ns=1.0):
    N = len(init_freqs)
    sm, sx, _, sz = _tls_ops(N)
    Sx, Sm = sum(sx), sum(sm)

    H_tls      = sum(init_freqs[j]*sz[j]/2 for j in range(N))
    t_start_p2 = pulse1_ns + gap_ns

    def d_coeff(t, _=None):
        # pulse-1
        if t < T_ramp_ns:
            env = 0.5*(1-np.cos(np.pi*t/T_ramp_ns))
            return amp1*env*np.cos(freq*t)
        elif t < pulse1_ns:
            return amp1*np.cos(freq*t)

        # gap
        if t < t_start_p2:
            return 0.0

        # pulse-2 (phase-shifted)
        t2 = t - t_start_p2
        if t2 < T_ramp_ns:
            env = 0.5*(1-np.cos(np.pi*t2/T_ramp_ns))
            return amp2*env*np.cos(freq*t + phase2_rad)
        elif t2 < pulse2_ns:
            return amp2*np.cos(freq*t + phase2_rad)
        return 0.0

    H   = [H_tls + interactions, [Sx, d_coeff]]
    ψ0  = (H_tls + interactions).groundstate()[1]
    c_ops = [np.sqrt(gamma)*Sm]
    if gamma_phi:
        _, _, _, szs = _tls_ops(N)
        c_ops += [np.sqrt(gamma_phi)*z for z in szs]

    pop = mesolve(
        H, ψ0, tlist, c_ops, [Sm.dag()*Sm], options=_opts_no_bar
    ).expect[0]
    return np.real_if_close(pop)


# backwards-compat API (phase=0)
def run_simulation_double_pulse(freq, amp1, amp2,
                                pulse1_ns, gap_ns, pulse2_ns,
                                tlist, init_freqs, interactions,
                                gamma, gamma_phi, T_ramp_ns=1.0):
    return run_simulation_double_pulse_phase(
        freq, amp1, amp2, pulse1_ns, gap_ns, pulse2_ns, 0.0,
        tlist, init_freqs, interactions, gamma, gamma_phi, T_ramp_ns
    )


# ───────── random couplings helper ─────────
def build_spin_spin_interactions_random_distribution(N, J_min, J_max,
                                                     alpha_x=1.0, alpha_y=0.0, alpha_z=0.0):
    _, sx, sy, sz = _tls_ops(N)
    J = np.random.uniform(J_min, J_max, size=(N, N))
    J = np.tril(J, -1) + np.tril(J, -1).T
    H_int = 0
    for i in range(N):
        for j in range(i+1, N):
            Jij = J[i, j]
            H_int += Jij*(alpha_x*sx[i]*sx[j] +
                          alpha_y*sy[i]*sy[j] +
                          alpha_z*sz[i]*sz[j])
    return H_int
