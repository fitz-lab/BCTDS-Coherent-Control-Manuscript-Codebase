import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
# ---------------------- System Parameters ----------------------
omega_tls_1 = 4.0         # GHz
omega_tls_2 = 4.12         # GHz
J = 0.01                # Coupling strength
Omega_amp = 0.10         # Drive amplitude
T_drive = 100             # ns (duration of drive)
T_total =500           # ns (total evolution time)
dt = 0.01                 # Time step (ns)
omega_d_vals = np.linspace(2.5, 5.5, 300)
# ---------------------- Operators ----------------------
sx1 = tensor(sigmax(), qeye(2))
sx2 = tensor(qeye(2), sigmax())
sz1 = tensor(sigmaz(), qeye(2))
sz2 = tensor(qeye(2), sigmaz())
sp1 = tensor(sigmap(), qeye(2))
sp2 = tensor(qeye(2), sigmap())
sm1 = tensor(sigmam(), qeye(2))
sm2 = tensor(qeye(2), sigmam())
sp_total = sp1 + sp2
sm_total = sm1 + sm2
collective_excitation = sp_total * sm_total
# ---------------------- Collapse Operators ----------------------
gamma_collective = 0.001   # ns⁻¹
gamma_local_1 = 0.0001      # ns⁻¹
gamma_local_2 = 0.0005      # ns⁻¹
c_ops = [
    np.sqrt(gamma_collective) * sm_total,  # Collective decay
    np.sqrt(gamma_local_1) * sm1,          # Local decay TLS 1
    np.sqrt(gamma_local_2) * sm2           # Local decay TLS 2
]
# ---------------------- Hamiltonian ----------------------
H0 = 0.5 * omega_tls_1 * sz1 + 0.5 * omega_tls_2 * sz2
Hint = J * sx1 * sx2
H_static = H0 + Hint
# ---------------------- Initial State ----------------------
evals, evecs = H_static.eigenstates()
psi0 = evecs[0]  # Ground state
# ---------------------- Time and Storage ----------------------
tlist = np.arange(0, T_total, dt)
num_t = len(tlist)
num_d = len(omega_d_vals)
sigmap_sigmam_data = np.zeros((num_d, num_t))
quasienergies = []
opts = Options(nsteps=5000)
# ---------------------- Main Loop ----------------------
print("Running time evolution and computing Floquet spectrum via propagator...")
for idx, omega_d in enumerate(tqdm(omega_d_vals)):
    args = {'omega_d': omega_d}
    def drive_coeff(t, args):
        return Omega_amp * np.cos(args['omega_d'] * t) if t <= T_drive else 0.0
    H_drive = [[sx1 + sx2, drive_coeff]]
    H_full = [H_static] + H_drive
    # Dissipative evolution with both local and collective decay
    result = mesolve(H_full, psi0, tlist, c_ops=c_ops,
                     e_ops=[collective_excitation], args=args, options=opts)
    signal = np.real(result.expect[0])
    sigmap_sigmam_data[idx, :] = signal
    # Floquet spectrum during drive (unitary part only)
    try:
        U = propagator(H_full, T_drive, c_ops=[], args=args, options=opts)
        phases = np.angle(np.linalg.eigvals(U.full()))
        folded = (phases + np.pi) % (2 * np.pi) - np.pi
        quasienergies.append(np.sort(folded / T_drive))
    except Exception as e:
        print(f"Floquet failed at ω_d = {omega_d:.2f}: {e}")
        quasienergies.append([np.nan] * 4)
# Convert list to array
quasienergies = np.array(quasienergies)
# ---------------------- Plotting ----------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
# 1. Heatmap: ⟨(σ₁⁺+σ₂⁺)(σ₁⁻+σ₂⁻)⟩ vs time and drive frequency
im1 = axs[0].imshow(sigmap_sigmam_data, extent=[tlist[0], tlist[-1], omega_d_vals[0], omega_d_vals[-1]],
                    aspect='auto', origin='lower', cmap='inferno')
axs[0].set_xlabel("Time (ns)")
axs[0].set_ylabel("Drive Frequency $\omega_d$ (GHz)")
axs[0].set_title(r"Time evolution of $\langle (\sigma_1^+ + \sigma_2^+)(\sigma_1^- + \sigma_2^-) \rangle$")
fig.colorbar(im1, ax=axs[0], label=r"$\langle \sigma^+ \sigma^- \rangle$")
# 2. Floquet quasi-energies vs drive frequency
for i in range(quasienergies.shape[1]):
    axs[1].plot(quasienergies[:, i], omega_d_vals, lw=1, label=f'Q{i}')
axs[1].set_ylabel("Drive Frequency $\omega_d$ (GHz)")
axs[1].set_xlabel("Quasi-Energy (GHz)")
axs[1].set_title("Floquet Quasi-Energies from U($T_\mathrm{drive}$)")
axs[1].legend(fontsize=6)
axs[1].grid(True)
plt.tight_layout()
plt.savefig('cool_braiding.png')
plt.close()