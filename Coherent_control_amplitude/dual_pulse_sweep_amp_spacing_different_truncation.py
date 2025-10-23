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
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['figure.titlesize'] = 14

script_dir = os.path.dirname(os.path.abspath(__file__))

fig_save_dir = os.path.join(script_dir, 'analysis_plots')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)

#region functions

def compute_fft_GHz(trace, time_spacing_us):
    time_spacing = time_spacing_us * 1e-6

    # FFT and frequencies
    fft_vals = np.fft.fft(trace)
    freqs = np.fft.fftfreq(len(trace), d=time_spacing)
    
    # Convert frequency to GHz
    freqs_GHz = freqs / 1e9
    fft_magnitude = np.abs(np.fft.fftshift(fft_vals))
    freqs_GHz_shifted = np.fft.fftshift(freqs_GHz)
    return freqs_GHz_shifted, fft_magnitude

def find_FWHM_and_plot(pulse_width=308, save_plot=False):
    pulse_width_us = pulse_width / 9830.4 # conversion using QICK synthesis clock frequency
    time_array = np.linspace(0, 1, 512)
    pulse = np.zeros_like(time_array)
    start_idx = int((0.5 - pulse_width_us / 2) * 512)
    end_idx = int((0.5 + pulse_width_us / 2) * 512)
    pulse[start_idx:end_idx] = 20
    
    freqs_GHz, fft_mag = compute_fft_GHz(pulse, 1/512)
    abs_diff = np.abs(fft_mag - np.max(fft_mag) / 2)
    FWHM_idx_0, FWHM_idx_1 = np.argsort(abs_diff)[:2]

    if save_plot:
        fig, ax = plt.subplots(figsize=(2, 3))
        ax.plot(fft_mag, freqs_GHz, c='indigo')
        
        plt.vlines(x=fft_mag[FWHM_idx_0], ymin=freqs_GHz[FWHM_idx_0], ymax=freqs_GHz[FWHM_idx_1], colors='indigo')

        x_pos = fft_mag[FWHM_idx_0]
        y_start = freqs_GHz[FWHM_idx_0]+ 0.003
        y_end = freqs_GHz[FWHM_idx_1] - 0.003

        # Plot arrow-like vertical line with variable thickness (linewidth) and arrowheads
        plt.annotate(
            '', 
            xy=(x_pos, y_end), 
            xytext=(x_pos, y_start),
            arrowprops=dict(
                arrowstyle='<->',      # double-headed arrow
                color='indigo',
                linewidth=2            # thickness of the arrow
            )
        )

        ax.set_xlabel("Magnitude [arb.]")
        ax.set_ylim([-0.1, 0.1])
        ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
        ax.set_ylabel("Frequency [GHz]")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_label_coords(0.6, 1.2)

        fig_save_dir = os.path.join(script_dir, 'analysis_plots')
        os.makedirs(fig_save_dir, exist_ok=True)
        file_path = os.path.join(fig_save_dir, f"bandwidth_only_full.png")
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {file_path}")

    return np.abs(freqs_GHz[FWHM_idx_1]- freqs_GHz[FWHM_idx_0])

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

#endregion


matrix_data_dir = os.path.join(script_dir, 'matrix_npy_save_folder')

save_fig = True

fig = plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(
    4, 7,
    width_ratios=[5, 0.3, 0.2, 3.5, 5, 0.3, 0.2],  # [plot1, gap, plot2, gap, cbar2]
    height_ratios=[1, 1, 1, 1],     # 4 rows if needed
    wspace=0,
    hspace=0.2
)

#region plot params
pulse_mark_line_width = 1.5
pulse_mark_line_color = 'black'
pulse_mark_line_style = '--'
line_plot_line_width = 1.2
slicing_line_width = 1.5
slicing_line_color = 'black'
slicing_line_style = '--'
slicing_line_alpha = 0.5
slicing_line_dashes = (6,10)
display_freq_range = [3, 5]
display_tau_range = [0,0.8]
frequency_ticks = [3.0, 3.5, 4.0, 4.5, 5.0]
amp_ticks = [0, 7500, 15000, 22500]
tau_color = plt.cm.inferno(215)
left_plot_color = 'cornflowerblue'
text_box_x = 0.96
text_box_y = 0.09
text_box_font_size = 16
# max_tau = 0.5
slice_frequency = 3657
slice_idx = slice_frequency - 3000 # actual freq - 3000
i_x = 0.97
i_y = 0.95
i_ii_size = 20 #27
i_ii_color = 'white'
abcd_x = -0.28 # -0.47
abcd_y = 1.2 # 1.17
abcd_size = 23 #30

#endregion

#region subplot Row 1, left
# ---------------- Row 1 left: freq spectrum ----------------
sample = "AlOx"
freq = 3657
npy_name_prefix = f"{sample}_HFSS_calibrated_3-5_GHz"
display_offset = 45
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()
ax = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 2])

IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix.shape[0]])/1e3  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96

im = ax.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=0)

# ax.set_xlabel(rf"Time [$\mu$s]")
ax.set_ylabel("Frequency [GHz]", labelpad=17)
# ax.set_xlim([0,1])
ax.set_xlim(display_tau_range)
ax.set_ylim(display_freq_range)
ax.axhline(
    y=freq*1e-3,
    color=slicing_line_color,
    linestyle=slicing_line_style,
    linewidth=slicing_line_width,
    alpha=slicing_line_alpha,
    dashes=slicing_line_dashes
)

ax.text(
    0.96, 0.09,
    rf"AlOx",
    transform=ax.transAxes,
    fontsize=text_box_font_size,
    ha='right', va='bottom',
    color='black',
    bbox=dict(
        facecolor='white',
        alpha=0.5,
        boxstyle='round,pad=0.3',
        edgecolor='none'
    )
)

pulse_end_idx_2nd = 75.5
pulse_start_idx_2nd = pulse_end_idx_2nd - 17.32
ax.axvline(x=pulse_start_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

ax.text(abcd_x, abcd_y, r'\textbf{a}', transform=ax.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')
ax.text(i_x, i_y, r'\textbf{i}', transform=ax.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

#endregion

#region subplot Row 1, right
# ---------------- Row 1 right: freq spectrum ----------------
sample = "111_oxide"
freq = 4254 #3657
npy_name_prefix = f"{sample}_HFSS_calibrated_3-5_GHz"
display_offset = 45
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()
ax = fig.add_subplot(gs[0, 4])
cax = fig.add_subplot(gs[0, 6])

IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix.shape[0]])/1e3  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96

im = ax.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=2,
)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=0)

# ax.set_xlabel(rf"Time [$\mu$s]")
ax.set_ylabel("Frequency [GHz]", labelpad=17)
ax.set_xlim(display_tau_range)
ax.set_ylim(display_freq_range)
ax.axhline(
    y=freq*1e-3,
    color=slicing_line_color,
    linestyle=slicing_line_style,
    linewidth=slicing_line_width,
    alpha=slicing_line_alpha,
    dashes=slicing_line_dashes
)

ax.text(
    0.96, 0.09,
    rf"SiOx",
    transform=ax.transAxes,
    fontsize=text_box_font_size,
    ha='right', va='bottom',
    color='black',
    bbox=dict(
        facecolor='white',
        alpha=0.5,
        boxstyle='round,pad=0.3',
        edgecolor='none'
    )
)

pulse_end_idx_2nd = 75.5
pulse_start_idx_2nd = pulse_end_idx_2nd - 17.32
ax.axvline(x=pulse_start_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

ax.text(abcd_x + 0.0, abcd_y, r'\textbf{b}', transform=ax.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')
ax.text(i_x, i_y, r'\textbf{i}', transform=ax.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

#endregion

# region subplot Row 2, left

# ---------------- Row 4 left:  spacing 3 ----------------
spacing = 0.01
sample = "AlOx"
freq = 3657
npy_name_prefix = f"{sample}_308_freq_{freq}_spacing_{spacing}"
display_offset = 45
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()
uncalibrated_pulse_amp_list = np.arange(0, 30000, 300).tolist()
ax = fig.add_subplot(gs[1, 0])
cax = fig.add_subplot(gs[1, 2])

IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = uncalibrated_pulse_amp_list  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96
im = ax.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=0)

# ax.set_xlabel(rf"Time [$\mu$s]")
ax.set_ylabel("Amplitude [arb.]")
ax.set_yticks(amp_ticks)
ax.set_xlim(display_tau_range)

ax.text(
    0.96, 0.09,
    rf"{int(spacing*1e3)} ns Spacing",
    transform=ax.transAxes,
    fontsize=text_box_font_size,
    ha='right', va='bottom',
    color='black',
    bbox=dict(
        facecolor='white',
        alpha=0.5,
        boxstyle='round,pad=0.3',
        edgecolor='none'
    )
)

pulse_end_idx_2nd = 75.5
pulse_start_idx_2nd = pulse_end_idx_2nd - 17.32
ax.axvline(x=pulse_start_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

pulse_end_idx_1st = pulse_start_idx_2nd - spacing * 552.96 - 0.5
pulse_start_idx_1st = pulse_end_idx_1st - 17.32
ax.axvline(x=pulse_start_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

# ax.text(abcd_x, abcd_y, r'\textbf{a}', transform=ax.transAxes,
#              fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')
ax.text(i_x, i_y, r'\textbf{ii}', transform=ax.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color=i_ii_color)

#endregion

#region subplot Row 2, right

# ---------------- Row 4 right:  spacing 3 ----------------
spacing = 0.01
sample = "111_oxide"
freq = 4254 #3657
npy_name_prefix = f"{sample}_308_freq_{freq}_spacing_{spacing}"
display_offset = 45
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()
uncalibrated_pulse_amp_list = np.arange(0, 30000, 300).tolist()
ax = fig.add_subplot(gs[1, 4])
cax = fig.add_subplot(gs[1, 6])

IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = uncalibrated_pulse_amp_list  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96
im = ax.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=2,
)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=0)

# ax.set_xlabel(rf"Time [$\mu$s]")
ax.set_ylabel("Amplitude [arb.]")
ax.set_yticks(amp_ticks)
ax.set_xlim(display_tau_range)

ax.text(
    0.96, 0.09,
    rf"{int(spacing*1e3)} ns Spacing",
    transform=ax.transAxes,
    fontsize=text_box_font_size,
    ha='right', va='bottom',
    color='black',
    bbox=dict(
        facecolor='white',
        alpha=0.5,
        boxstyle='round,pad=0.3',
        edgecolor='none'
    )
)

pulse_end_idx_2nd = 75.5
pulse_start_idx_2nd = pulse_end_idx_2nd - 17.32
ax.axvline(x=pulse_start_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

pulse_end_idx_1st = pulse_start_idx_2nd - spacing * 552.96 - 0.5
pulse_start_idx_1st = pulse_end_idx_1st - 17.32
ax.axvline(x=pulse_start_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

# ax.text(abcd_x + 0.05, abcd_y, r'\textbf{b}', transform=ax.transAxes,
#              fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')
ax.text(i_x, i_y, r'\textbf{ii}', transform=ax.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color=i_ii_color)

#endregion

#region subplot Row 3, left

# ---------------- Row 4 left:  spacing 3 ----------------
spacing = 0.03
sample = "AlOx"
freq = 3657
npy_name_prefix = f"{sample}_308_freq_{freq}_spacing_{spacing}"
display_offset = 45
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()
uncalibrated_pulse_amp_list = np.arange(0, 30000, 300).tolist()
ax = fig.add_subplot(gs[2, 0])
cax = fig.add_subplot(gs[2, 2])

IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = uncalibrated_pulse_amp_list  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96
im = ax.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=0)

# ax.set_xlabel(rf"Time [$\mu$s]")
ax.set_ylabel("Amplitude [arb.]")
ax.set_yticks(amp_ticks)
ax.set_xlim(display_tau_range)

ax.text(
    0.96, 0.09,
    rf"{int(spacing*1e3)} ns Spacing",
    transform=ax.transAxes,
    fontsize=text_box_font_size,
    ha='right', va='bottom',
    color='black',
    bbox=dict(
        facecolor='white',
        alpha=0.5,
        boxstyle='round,pad=0.3',
        edgecolor='none'
    )
)

pulse_end_idx_2nd = 75.5
pulse_start_idx_2nd = pulse_end_idx_2nd - 17.32
ax.axvline(x=pulse_start_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

pulse_end_idx_1st = pulse_start_idx_2nd - spacing * 552.96 - 0.5
pulse_start_idx_1st = pulse_end_idx_1st - 17.32
ax.axvline(x=pulse_start_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

ax.text(i_x, i_y, r'\textbf{iii}', transform=ax.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color=i_ii_color)

#endregion

#region subplot Row 3, right

# ---------------- Row 4 right:  spacing 3 ----------------
spacing = 0.03
sample = "111_oxide"
freq = 4254 #3657
npy_name_prefix = f"{sample}_308_freq_{freq}_spacing_{spacing}"
display_offset = 45
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()
uncalibrated_pulse_amp_list = np.arange(0, 30000, 300).tolist()
ax = fig.add_subplot(gs[2, 4])
cax = fig.add_subplot(gs[2, 6])

IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = uncalibrated_pulse_amp_list  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96
im = ax.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=2,
)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=0)

# ax.set_xlabel(rf"Time [$\mu$s]")
ax.set_ylabel("Amplitude [arb.]")
ax.set_yticks(amp_ticks)
ax.set_xlim(display_tau_range)

ax.text(
    0.96, 0.09,
    rf"{int(spacing*1e3)} ns Spacing",
    transform=ax.transAxes,
    fontsize=text_box_font_size,
    ha='right', va='bottom',
    color='black',
    bbox=dict(
        facecolor='white',
        alpha=0.5,
        boxstyle='round,pad=0.3',
        edgecolor='none'
    )
)

pulse_end_idx_2nd = 75.5
pulse_start_idx_2nd = pulse_end_idx_2nd - 17.32
ax.axvline(x=pulse_start_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

pulse_end_idx_1st = pulse_start_idx_2nd - spacing * 552.96 - 0.5
pulse_start_idx_1st = pulse_end_idx_1st - 17.32
ax.axvline(x=pulse_start_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

ax.text(i_x, i_y, r'\textbf{iii}', transform=ax.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color=i_ii_color)

#endregion

#region subplot Row 4, left

# ---------------- Row 4 left:  spacing 3 ----------------
spacing = 0.05
sample = "AlOx"
freq = 3657
npy_name_prefix = f"{sample}_308_freq_{freq}_spacing_{spacing}"
display_offset = 45
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()
uncalibrated_pulse_amp_list = np.arange(0, 30000, 300).tolist()
ax = fig.add_subplot(gs[3, 0])
cax = fig.add_subplot(gs[3, 2])

IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = uncalibrated_pulse_amp_list  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96
im = ax.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=0)

ax.set_xlabel(rf"Time [$\mu$s]")
ax.set_ylabel("Amplitude [arb.]")
ax.set_yticks(amp_ticks)
ax.set_xlim(display_tau_range)

ax.text(
    0.96, 0.09,
    rf"{int(spacing*1e3)} ns Spacing",
    transform=ax.transAxes,
    fontsize=text_box_font_size,
    ha='right', va='bottom',
    color='black',
    bbox=dict(
        facecolor='white',
        alpha=0.5,
        boxstyle='round,pad=0.3',
        edgecolor='none'
    )
)

pulse_end_idx_2nd = 75.5
pulse_start_idx_2nd = pulse_end_idx_2nd - 17.32
ax.axvline(x=pulse_start_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

pulse_end_idx_1st = pulse_start_idx_2nd - spacing * 552.96 - 0.5
pulse_start_idx_1st = pulse_end_idx_1st - 17.32
ax.axvline(x=pulse_start_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

ax.text(i_x, i_y, r'\textbf{iv}', transform=ax.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color=i_ii_color)

#endregion

#region subplot Row 4, right

# ---------------- Row 4 right:  spacing 3 ----------------
spacing = 0.05
sample = "111_oxide"
freq = 4254 #3657
npy_name_prefix = f"{sample}_308_freq_{freq}_spacing_{spacing}"
display_offset = 45
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()
uncalibrated_pulse_amp_list = np.arange(0, 30000, 300).tolist()
ax = fig.add_subplot(gs[3, 4])
cax = fig.add_subplot(gs[3, 6])

IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = uncalibrated_pulse_amp_list  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96
im = ax.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=2,
)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=0)

ax.set_xlabel(rf"Time [$\mu$s]")
ax.set_ylabel("Amplitude [arb.]")
ax.set_yticks(amp_ticks)
ax.set_xlim(display_tau_range)

ax.text(
    0.96, 0.09,
    rf"{int(spacing*1e3)} ns Spacing",
    transform=ax.transAxes,
    fontsize=text_box_font_size,
    ha='right', va='bottom',
    color='black',
    bbox=dict(
        facecolor='white',
        alpha=0.5,
        boxstyle='round,pad=0.3',
        edgecolor='none'
    )
)

pulse_end_idx_2nd = 75.5
pulse_start_idx_2nd = pulse_end_idx_2nd - 17.32
ax.axvline(x=pulse_start_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_2nd/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

pulse_end_idx_1st = pulse_start_idx_2nd - spacing * 552.96 - 0.5
pulse_start_idx_1st = pulse_end_idx_1st - 17.32
ax.axvline(x=pulse_start_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax.axvline(x=pulse_end_idx_1st/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)

ax.text(i_x, i_y, r'\textbf{iv}', transform=ax.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color=i_ii_color)

#endregion

# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, f"Fig_sweep_spacing_and_amp.png")
if save_fig:
    plt.savefig(file_path, dpi=200, bbox_inches = 'tight')
    print(f"Figure saved to: {file_path}")
plt.show()
plt.close()
