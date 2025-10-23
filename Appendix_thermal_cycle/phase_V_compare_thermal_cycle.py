import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import time
import matplotlib
# matplotlib.use('Agg')  # Use the 'Agg' backend, which does not require a display environment.
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit


# #### Preamble
# # Set the font globally to Helvetica
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

##############################################################################
# 1. General matplotlib settings
##############################################################################
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 20
matplotlib.rcParams['figure.titlesize'] = 20

def wrap_phase(phases, offset):
    phases = np.array(phases)  # Ensure input is an array
    return (phases + offset + np.pi) % (2 * np.pi) - np.pi

def fft_custom(trace, time_spacing_us):
    N = len(trace)   # Number of points in FFT
    t = np.arange(N) * time_spacing_us  # Time vector
    fft_result = np.fft.fft(trace, N)
    frequencies = np.fft.fftfreq(N, time_spacing_us)  # Frequency axis
    magnitude = np.abs(fft_result)[:N // 2]  # One-sided spectrum
    frequencies_MHz = frequencies[:N // 2]  # Positive frequencies only
    return frequencies_MHz, magnitude

def g2_correlation(intensity, time_spacing_us, max_tau_us=None):
    intensity = np.array(intensity)
    n = len(intensity)
    max_lag = n if max_tau_us is None else min(n, int(max_tau_us / time_spacing_us))
    corr = np.correlate(intensity, intensity, mode='full')
    corr = corr[len(corr)//2:]
    normalization = np.mean(intensity)**2 * np.arange(n, n - max_lag, -1)
    g2 = corr[:max_lag] / normalization
    tau_values = np.arange(max_lag) * time_spacing_us
    return tau_values, g2

def compute_chi2_matrix(g2_arr, mag_data, time_spacing_us):
    num_freqs, num_tau = g2_arr.shape
    omega = np.fft.fftshift(np.fft.fftfreq(num_tau, d=time_spacing_us))
    pos_idx = omega >= 0
    omega = omega[pos_idx]
    chi2_matrix = np.zeros((num_freqs, len(omega)))
    for i in range(num_freqs):
        S_omega = np.fft.fftshift(np.fft.fft(g2_arr[i]))
        S_omega = S_omega[pos_idx]
        chi2_matrix[i] = -1*np.imag(S_omega)
    return omega, chi2_matrix

script_dir = os.path.dirname(os.path.abspath(__file__))
fig_save_dir = os.path.join(script_dir, 'phase_FFT_analysis')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)

matrix_data_dir = os.path.join(script_dir, 'matrix_npy_save_folder')

save_fig = True

pulse_start_idx = 17
# pulse_end_idx = 133
pulse_end_idx = 0 #75
line_plot_line_width = 1.2
slicing_line_width = 1.5
slicing_line_color = 'white'
slicing_line_style = '--'
slicing_line_alpha = 0.5
slicing_line_dashes = (6,10)
slicing_V_line_style = '--'
slicing_V_line_width = 1.5
slicing_V_line_alpha = 1
slicing_V_line_dashes = (6,10)
phase_V_fft_display_range = [0, 100]
# display_freq_range = [3, 5]
display_freq_range = [3.5, 4.0]
display_tau_range = [0,0.5]
display_phase_fft_range = [0, 400]
display_phase_fft_range_2 = [0, 200]
# frequency_ticks = [3.9, 4.0, 4.1, 4.2, 4.3, 4.4]
# frequency_ticks = [4.2, 4.6]
tau_color = plt.cm.inferno(215)
left_plot_color = 'cornflowerblue'
text_box_x = 0.05
text_box_y = 0.91
text_box_font_size = 16
max_tau = 0.5
i_x = 0.9
i_y = 0.95
ii_x = 0.97
ii_y = 0.95
i_ii_size = 27
abcd_x = -0.15
abcd_y = 1.15
abcd_size = 30

V_base_freq = 4352 # 4352 4487
V_base_idx = V_base_freq - 4100

# pulse_width_list = np.arange(50, 2101, 50)
# pulse_amp_list = np.arange(0, 7600, 200).tolist()
pulse_amp_list = np.concatenate([
    np.arange(0, 1000, 50),
    np.arange(1000, 5000, 200), 
    np.arange(5000, 30001, 1000) 
]).tolist()
# pulse_width_list = [50]
pulse_width = 1950

fig = plt.figure(figsize=(14, 9.5))

gs = gridspec.GridSpec(
    nrows=3,
    ncols=7,
    wspace=0,
    hspace=0.5,
    width_ratios=[6.5, 0.3, .2, 3, 6.5, 0.3, .2]
)

ax_long = fig.add_subplot(gs[2, 0:5])

for subplot_idx, npy_prefix in enumerate(["shipley", "shipley_2nd"]):
    # ---------------- Process Data ----------------
    display_offset = 65 #25
    pulse_frequency_list = np.arange(2000, 5000, 1).tolist()

    IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, rf"{npy_prefix}_IQ_avg_matrix.npy"))
    I_avg_matrix = IQ_avg_matrix[0]
    Q_avg_matrix = IQ_avg_matrix[1]
    mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
    phase_avg_matrix = np.angle(I_avg_matrix + 1j *Q_avg_matrix)
    mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
    mag_avg_log_matrix = mag_avg_log_matrix[0:, display_offset:]
    phase_avg_matrix = phase_avg_matrix[0:, display_offset:]
    print(np.shape(mag_avg_log_matrix))

    mag_avg_log_matrix_pulse = mag_avg_log_matrix[:, pulse_start_idx:pulse_end_idx]
    # mag_avg_log_matrix_pulse_avg = np.mean(mag_avg_log_matrix_pulse, axis=1)

    ### these are the mag and log mag matrixes without pulse 
    mag_avg_log_matrix_transient_region = mag_avg_log_matrix[:,pulse_end_idx:]
    mag_avg_matrix_transient_region = 10**(mag_avg_log_matrix_transient_region) - 0.01

    starting_phase_value = phase_avg_matrix[:,0]
    for idx, pulse_frequency in enumerate(pulse_frequency_list):
        phase_avg_matrix[idx] = wrap_phase(phase_avg_matrix[idx], -1*starting_phase_value[idx])
    phase_transient_region = phase_avg_matrix[:,pulse_end_idx:]

    # initialize the fft and g2 matricies
    fft_avg_list = []
    phase_fft_list = []
    g2_list = []

    # process fft and g2
    for idx, pulse_frequency in enumerate(pulse_frequency_list):

        mag_avg = mag_avg_matrix_transient_region[idx]


        phase_avg = phase_transient_region[idx]
        # phase_avg = wrap_phase(phase_avg, -1*starting_phase_value[idx])
        intensity = mag_avg**2

        tau_values, g2 = g2_correlation(intensity, time_spacing_us=1/552.96, max_tau_us=max_tau)
        g2_list.append(g2)

        mag_avg_log = np.log10(mag_avg + 0.01)
        fft_freq_MHz, fft_of_truncated_trace = fft_custom(mag_avg_log, 1/552.96)
        fft_avg_list.append(fft_of_truncated_trace)
        phase_fft_freq_MHz, phase_fft_of_truncated_trace = fft_custom(phase_avg, 1/552.96)
        phase_fft_list.append(phase_fft_of_truncated_trace)

    g2_matrix = np.array(g2_list)
    g2_matrix_log = np.log10(g2_matrix + 1e-2)
    tau_values = np.array(tau_values)
    fft_matrix = np.array(fft_avg_list)
    phase_fft_matrix = np.array(phase_fft_list)
    fft_matrix_log = np.log10(fft_matrix + 1e-2)
    fft_freq_MHz = np.array(fft_freq_MHz)
    phase_fft_freq_MHz = np.array(phase_fft_freq_MHz)

    time_axis = np.arange(mag_avg_log_matrix_transient_region.shape[1]) / 552.96  # Data points index
    freq_axis = np.array(pulse_frequency_list)/1e3  # Ensure correct length


    # ---------------- top subplot: log mag ----------------
    ax_top = fig.add_subplot(gs[subplot_idx, 0])

    im = ax_top.imshow(
        mag_avg_log_matrix.T,
        aspect='auto',
        extent=[freq_axis[0], freq_axis[-1], time_axis[0], time_axis[-1]],
        origin='lower',
        cmap='inferno',
        interpolation='none',
        vmin=-2,
        vmax=3,
    )

    
    ax_top.set_xlim(display_freq_range)
    ax_top.set_xlabel("Frequency [GHz]")
    ax_top.set_ylim([0, 0.8])
    ax_top.set_ylabel(rf'Time [$\mu$s]')
    ax_top.text(ii_x, ii_y, r'\textbf{i}', transform=ax_top.transAxes,
                fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')
    # cbar = fig.colorbar(im, ax=ax_top, label=r"Log$_{10}$(A) [arb.]")
    cax = fig.add_subplot(gs[subplot_idx, 2])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label=r"Log$_{10}$(A) [arb.]")


    ax_top.text(
        text_box_x, text_box_y, f"Cooldown {subplot_idx+1}",  # position in axes coordinates
        transform=ax_top.transAxes,
        fontsize=text_box_font_size,
        ha='left', va='top',
        color='black',
        bbox=dict(
            facecolor='white',
            alpha=0.5,
            boxstyle='round,pad=0.3',
            edgecolor='none'
        )
    )

    ax_top.text(abcd_x, abcd_y, r'\textbf{a}' if subplot_idx==0 else r'\textbf{b}', transform=ax_top.transAxes,
            fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

    # ---------------- bot subplot: phase V ----------------

    ax_bot = fig.add_subplot(gs[subplot_idx, 4])
    # print(f"phasefft{np.shape(phase_fft_matrix)}")
    im = ax_bot.imshow(
        phase_fft_matrix.T,
        aspect='auto',
        extent=[freq_axis[0], freq_axis[-1], fft_freq_MHz[0], fft_freq_MHz[-1]],
        origin='lower',
        cmap='inferno',
        interpolation='none',
        vmin=0,
        vmax=300,
    )
    ax_bot.set_xlabel("Frequency [GHz]")
    ax_bot.set_xlim(display_freq_range)
    ax_bot.set_ylabel('FFT Freq. [MHz]')
    ax_bot.set_ylim([0, 100])

    ax_bot.text(
        text_box_x, text_box_y, f"Cooldown {subplot_idx+1}",  # position in axes coordinates
        transform=ax_bot.transAxes,
        fontsize=text_box_font_size,
        ha='left', va='top',
        color='black',
        bbox=dict(
            facecolor='white',
            alpha=0.5,
            boxstyle='round,pad=0.3',
            edgecolor='none'
        )
    )

    ax_bot.axhline(y=0, 
        color='dodgerblue' if subplot_idx == 0 else 'C1', 
        linestyle='--', 
        linewidth=4, 
        dashes=(3.8,3.8), 
        alpha = 1, 
        zorder=100
    )

    ax_bot.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_bot.transAxes,
            fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')
    # cbar = fig.colorbar(im, ax=ax_bot, label=r"FFT($\phi$) [arb.]")
    cax = fig.add_subplot(gs[subplot_idx, 6])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label=r"FFT($\phi$) [arb.]")

    phase_fft_matrix_0_slice = phase_fft_matrix[:, 0]
    # avg_num = 1
    # phase_fft_matrix_0_slice = np.mean(phase_fft_matrix[:, 0:0+avg_num], axis=1)
    ax_long.plot(
        freq_axis,
        phase_fft_matrix_0_slice,
        label=f"Cooldown {subplot_idx+1}",
        color='dodgerblue' if subplot_idx == 0 else 'C1',
        linewidth=1.5,          # line width
        marker='o',           # dot marker
        markersize=2          # dot size (pt)
    )
ax_long.set_xlabel("Frequency [GHz]")
ax_long.set_xlim(display_freq_range)
ax_long.set_ylabel(r"FFT($\phi$) [arb.]")
ax_long.set_ylim([0, None])
ax_long.legend(ncol=5, fontsize=text_box_font_size, loc="upper right")
ax_long.text(abcd_x /2 +0.018, abcd_y + 0.02, r'\textbf{c}', transform=ax_long.transAxes,
            fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')




# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, f"Fig_thermal_cycle.png")
if save_fig:
    plt.savefig(file_path, dpi=200, bbox_inches = 'tight')
    print(f"Figure saved to: {file_path}")
plt.show()
plt.close()
