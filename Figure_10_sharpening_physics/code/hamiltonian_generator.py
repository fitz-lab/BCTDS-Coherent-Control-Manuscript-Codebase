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
from qutip import *

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

    sx1 = tensor(sigmax(), qeye(2));  sx2 = tensor(qeye(2), sigmax())
    sz1 = tensor(sigmaz(), qeye(2));  sz2 = tensor(qeye(2), sigmaz())
    sp1 = tensor(sigmap(),  qeye(2)); sp2 = tensor(qeye(2), sigmap())
    sm1 = tensor(sigmam(),  qeye(2)); sm2 = tensor(qeye(2), sigmam())


    sm, sx, _, sz = _tls_ops(N)
    Sp = sum(s.dag() for s in sm)
    Sm = sum(sm)
    Sx = sum(sx)

    ## This was the original thing...
    # T_ramp = 1.0
    # def drive_coeff(t, args):
    #     if t < T_ramp:
    #         return max(0, amp * (1 - np.cos(np.pi * t / T_ramp)) / 2 * np.cos(freq * t))
    #     elif t < pulse_ns:
    #         return max(0, amp * np.cos(freq * t))
    #     else:
    #         return 0

    # We can try this next...
    def drive_coeff(t, args):
        """time-dependent drive: simple square pulse with cosine"""
        if t <= pulse_ns:
            return amp * np.cos(freq * t)
        return 0.0
    
    H_tls = sum(init_freqs[j]* sz[j] / 2 for j in range(N))

    # H_int = sum(interactions[i, j] * (sx[i] * sx[j]) for i in range(N_tls) for j in range(i))  # Dipole-Dipole Interaction
    H_int = sum(interactions[i, j] for i in range(N) for j in range(i))  # Dipole-Dipole Interaction
    
    H_drive = [[sum(sx), drive_coeff]]

    H = [H_tls + H_int] + H_drive
    c_ops = [np.sqrt(gamma) * Sm]  # Collective decay

    # c_ops = [np.sqrt(gamma) * Sm]  # collective decay

    if gamma_phi and gamma_phi > 0.0:
        # per-TLS pure dephasing (L = sqrt(gamma_phi/2) * sz_k)
        for k in range(N):
            c_ops.append(np.sqrt(0.5*gamma_phi) * sz[k])

    # ---------------------- Initial State ----------------------
    H_static = H_tls + H_int
    evals, evecs = H_static.eigenstates()
    psi0 = evecs[0]  # Ground state

    # -------- Solve for <Sm^dagger Sm> and <Sm> --------
    e_ops = [sp1, sp2, Sm.dag()*Sm]

    result = mesolve(H, psi0, tlist, c_ops, e_ops, progress_bar=None)

    Sp1_tot = result.expect[0]
    Sp2_tot = result.expect[1]
    expec_pop = result.expect[2]

    return Sp1_tot, Sp2_tot, np.real(expec_pop)

# ────────────────── original thin wrapper ──────────────
def run_simulation_single_pulse(freq, amp, tlist, init_freqs, interactions,
                                gamma, gamma_phi, pulse_ns, T_ramp_ns=1.0):
    sp1, sp2, pop = run_simulation_single_pulse_full(
        freq, amp, tlist, init_freqs, interactions,
        gamma, gamma_phi, pulse_ns, T_ramp_ns
    )
    return sp1, sp2, pop


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


# # ───────── random XX (+optional YY/ZZ) couplings ───────
# def build_spin_spin_interactions_random_distribution(N, J_min, J_max,
#                                                      alpha_x=1.0, alpha_y=0.0, alpha_z=0.0):
#     _, sx, sy, sz = _tls_ops(N)
#     J = np.random.uniform(J_min, J_max, size=(N, N))
#     J = np.tril(J, -1) + np.tril(J, -1).T
#     H_int = 0
#     for i in range(N):
#         for j in range(i+1, N):
#             Jij = J[i, j]
#             H_int += Jij * (
#                 alpha_x*sx[i]*sx[j] + alpha_y*sy[i]*sy[j] + alpha_z*sz[i]*sz[j]
#             )
#     return H_int

# --------------------------------------------------
def build_spin_spin_interactions_random_distribution(N_tls, J_min, J_max, 
                                 alpha_x=1.0, alpha_y=1.0, alpha_z=0.5):
    
    I2 = qeye(2)
    sm_single = destroy(2)
    sx_single = sm_single + sm_single.dag()
    sy_single = -1j*(sm_single - sm_single.dag())
    sz_single = 2*sm_single.dag()*sm_single - I2
    
    def tensor_op(op, k, N):
        return tensor([I2]*k + [op] + [I2]*(N - k - 1))
    
    sx_ops = [tensor_op(sx_single, i, N_tls) for i in range(N_tls)]
    sy_ops = [tensor_op(sy_single, i, N_tls) for i in range(N_tls)]
    sz_ops = [tensor_op(sz_single, i, N_tls) for i in range(N_tls)]
    
    J_mat = np.zeros((N_tls, N_tls), dtype=np.complex128)
    for i in range(N_tls):
        for j in range(i+1, N_tls):
            np.random.seed(42)
            J_rand = np.random.uniform(J_min, J_max)
            J_mat[i,j] = J_rand
            J_mat[j,i] = J_rand
    
    H_int = 0
    for i in range(N_tls):
        for j in range(i+1, N_tls):
            Jij = J_mat[i,j]
            H_int += alpha_x*Jij*(sx_ops[i]*sx_ops[j])
            H_int += alpha_y*Jij*(sy_ops[i]*sy_ops[j])
            H_int += alpha_z*Jij*(sz_ops[i]*sz_ops[j])
    return H_int


# ───────── legacy wrapper (unchanged) ─────────
def run_simulation_for_frequency(freq, tlist, init_freqs, interactions,
                                 gamma, gamma_phi, drive_ampl, pulse_duration):
    return run_simulation_single_pulse(
        freq, drive_ampl, tlist, init_freqs, interactions,
        gamma, gamma_phi, pulse_duration
    )
