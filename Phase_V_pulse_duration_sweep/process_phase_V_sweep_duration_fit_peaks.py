import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from datetime import datetime
import time
import matplotlib
# matplotlib.use('Agg')  # Use the 'Agg' backend, which does not require a display environment.
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


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
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['figure.titlesize'] = 14

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

matrix_data_dir = os.path.join(script_dir, 'matrix_npy')

save_fig = True

display_offset = 25
transient_processing_offset = 110
transient_processing_start_location = display_offset + transient_processing_offset
include_pulse = False

pulse_width_list = np.arange(50, 2101, 50)
pulse_frequency_list = np.arange(4100, 4600, 1).tolist()

display_freq_range = [4.1, 4.6]
display_freq_ticks = [4.1, 4.2, 4.3, 4.4, 4.5, 4.6]
line_plot_line_width = 1.2
slicing_line_width = 1.5
slice_and_box_color_1 = 'lime'
slice_and_box_color_2 = 'cyan'
vertical_slicing_line_width = 2
slicing_line_color = 'white'
slicing_line_style = '--'
slicing_line_alpha = 1
slicing_line_dashes = (6,6)
slicing_V_line_style = '--'
slicing_V_line_width = 4
slicing_V_line_alpha = 1
slicing_V_line_dashes = (3.8,3.8)
phase_V_fft_display_range = [0, 100]
text_box_x = 0.96
text_box_y = 0.09
text_box_font_size = 16
i_x = 0.97
i_y = 0.95
i_ii_size = 20 #27
i_ii_color = 'white'
abcd_x = -0.16 # -0.47
abcd_y = 1.17 # 1.17
abcd_size = 23 #30

print("processing data collection ...")

full_record_mag_log_matricies = []
full_record_mag_log_TP_matricies = []
full_record_mag_log_TP_fft_matricies = []
full_record_phase_TP_fft_matricies = []

for pulse_width in tqdm(pulse_width_list):

    IQ_matrix = np.load(os.path.join(matrix_data_dir, rf"{pulse_width}_IQ_avg_matrix_0.npy"))
    I_matrix = IQ_matrix[0]
    Q_matrix = IQ_matrix[1]
    mag_matrix = np.abs(I_matrix + 1j *Q_matrix)
    phase_matrix = np.angle(I_matrix + 1j *Q_matrix)
    mag_log_matrix = np.log10(mag_matrix + 0.01)

    # truncated whitespace before the pulse
    mag_matrix = mag_matrix[:, display_offset:]
    mag_log_matrix = mag_log_matrix[:, display_offset:]
    phase_matrix = phase_matrix[:, display_offset:]

    # transient processing region, with the pulse and post pulse artifacts (excessively) removed
    # mag fft and phase fft will only be computed for the TP region
    mag_TP_matrix = mag_matrix[:, transient_processing_start_location:]
    mag_log_TP_matrix = mag_log_matrix[:, transient_processing_start_location:]
    phase_TP_matrix = phase_matrix[:, transient_processing_start_location:]

    # initialize the fft and g2 matricies
    mag_log_TP_fft_matrix = []
    phase_TP_fft_matrix = []

    # process fft and g2
    for idx, pulse_frequency in enumerate(pulse_frequency_list):

        mag_TP = mag_TP_matrix[idx]
        intensity = mag_TP**2
        mag_log_TP = mag_log_TP_matrix[idx]
        phase_TP = phase_TP_matrix[idx]

        fft_freq_MHz, mag_log_TP_fft = fft_custom(mag_log_TP, 1/552.96)
        mag_log_TP_fft_matrix.append(mag_log_TP_fft)
        phase_fft_freq_MHz, phase_TP_fft = fft_custom(phase_TP, 1/552.96)
        phase_TP_fft_matrix.append(phase_TP_fft)

    mag_log_TP_fft_matrix = np.array(mag_log_TP_fft_matrix)
    phase_TP_fft_matrix = np.array(phase_TP_fft_matrix)
    fft_freq_MHz = np.array(fft_freq_MHz)
    phase_fft_freq_MHz = np.array(phase_fft_freq_MHz)

    full_record_mag_log_matricies.append(mag_log_matrix)
    full_record_mag_log_TP_matricies.append(mag_log_TP_matrix)
    full_record_mag_log_TP_fft_matricies.append(mag_log_TP_fft_matrix)
    full_record_phase_TP_fft_matricies.append(phase_TP_fft_matrix)

full_record_mag_log_matricies = np.array(full_record_mag_log_matricies)
full_record_mag_log_TP_matricies = np.array(full_record_mag_log_TP_matricies)
full_record_mag_log_TP_fft_matricies = np.array(full_record_mag_log_TP_fft_matricies)
full_record_phase_TP_fft_matricies = np.array(full_record_phase_TP_fft_matricies)

print("data processing complete, now ploting...")


fig = plt.figure(figsize=(12, 14))  # Adjust width to accommodate both plots
gs = gridspec.GridSpec(
    5, 7,
    width_ratios=[7, 1, 7, 1, 7, 0.3, .2],
    wspace=0,
    hspace=0.4  # Increase or decrease for more/less vertical spacing
)


for subplot_idx, pulse_width in enumerate([200, 500, 1950]):

    data_location_idx = np.where(pulse_width_list == pulse_width)[0][0] 

    mag_log_matrix = full_record_mag_log_matricies[data_location_idx]
    mag_log_TP_matrix = full_record_mag_log_TP_matricies[data_location_idx]
    mag_log_TP_fft_matrix = full_record_mag_log_TP_fft_matricies[data_location_idx]
    phase_TP_fft_matrix = full_record_phase_TP_fft_matricies[data_location_idx]

    # ---------------- 1st plot magnitude ----------------

    # ---------------- Right subplot: Color map ----------------
    ax_right = fig.add_subplot(gs[0, subplot_idx * 2])
    cax = fig.add_subplot(gs[0, subplot_idx * 2 + 2]) if subplot_idx == 2 else None

    time_axis = (np.arange(mag_log_matrix.shape[1]) if include_pulse else np.arange(mag_log_TP_matrix.shape[1])) / 552.96
    freq_axis = np.array(pulse_frequency_list)/1e3
    im = ax_right.imshow(
        (mag_log_matrix if include_pulse else mag_log_TP_matrix).T,
        aspect='auto',
        extent=[freq_axis[0], freq_axis[-1], time_axis[0], time_axis[-1]],
        origin='lower',
        cmap='inferno',
        interpolation='none',
        vmin=-2,
        vmax=3,
    )
    ax_right.set_ylim([0, 0.8])
    ax_right.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax_right.set_xlim(display_freq_range)
    ax_right.set_xlabel('Frequency [GHz]')
    ax_right.set_ylabel(rf'Time [$\mu$s]') if subplot_idx==0 else None

    if subplot_idx == 2:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=7)


    # ax_right.text(
    #     0.96, 0.09,
    #     rf"{round(pulse_width/ 9.8304 / 10) * 10} ns",
    #     transform=ax_right.transAxes,
    #     fontsize=text_box_font_size,
    #     ha='right', va='bottom',
    #     color='black',
    #     bbox=dict(
    #         facecolor='white',
    #         alpha=0.5,
    #         boxstyle='round,pad=0.3',
    #         edgecolor='none'
    #     )
    # )

    ax_right.text(
        0.05, 0.91,  # near top-left corner in Axes coords
        rf"{round(pulse_width/ 9.8304 / 10) * 10} ns",
        transform=ax_right.transAxes,
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

    ax_right.axhline(y=0, color=slice_and_box_color_1, linestyle=slicing_V_line_style, linewidth=slicing_V_line_width, dashes=slicing_V_line_dashes, alpha = slicing_V_line_alpha, zorder=100)

    ax_right.text(abcd_x, abcd_y, r'\textbf{a}', transform=ax_right.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black') if subplot_idx==0 else None
    labels = [r'\textbf{i}', r'\textbf{ii}', r'\textbf{iii}']
    ax_right.text(i_x, i_y, labels[subplot_idx], transform=ax_right.transAxes,
            fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')
    
    if subplot_idx == 2:
        rec_x_center_1 = 4.202
        rec_x_width_1 = 0.04
        rect_1 = patches.Rectangle(
            (rec_x_center_1 - rec_x_width_1 / 2, 0.05), rec_x_width_1, 0.4,            # (x,y), width, height
            linewidth=0.7,
            edgecolor="white",
            facecolor="none",          # transparent fill
            linestyle="-"             # dashed outline
        )

        rec_x_center_2 = 4.351
        rec_x_width_2 = 0.04
        rect_2 = patches.Rectangle(
            (rec_x_center_2 - rec_x_width_2 / 2, 0.05), rec_x_width_2, 0.4,             # (x,y), width, height
            linewidth=0.7,
            edgecolor="magenta",
            facecolor="none",          # transparent fill
            linestyle="-"             # dashed outline
        )
        ax_right.add_patch(rect_1)
        ax_right.add_patch(rect_2)


    


    # ---------------- 2nd row fft of phase ----------------

    # ---------------- Right subplot: Color map ----------------
    ax_right = fig.add_subplot(gs[1, subplot_idx * 2])
    cax = fig.add_subplot(gs[1, subplot_idx * 2 + 2]) if subplot_idx == 2 else None

    im = ax_right.imshow(
        phase_TP_fft_matrix.T,
        aspect='auto',
        extent=[freq_axis[0], freq_axis[-1], fft_freq_MHz[0], fft_freq_MHz[-1]],
        origin='lower',
        cmap='inferno',
        interpolation='none',
        vmin=0,
        vmax=300,
    )

    

    ax_right.axhline(y=0, color=slice_and_box_color_2, linestyle=slicing_V_line_style, linewidth=slicing_V_line_width, dashes=slicing_V_line_dashes, alpha = slicing_V_line_alpha, zorder=100)
    ax_right.set_ylim(phase_V_fft_display_range)
    # ax_right.set_ylim([0, 50])
    ax_right.set_xlim(display_freq_range)
    ax_right.set_xlabel('Frequency [GHz]')
    ax_right.set_ylabel(rf'FFT Freq. [MHz]') if subplot_idx==0 else None
    if subplot_idx == 2:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"FFT($\phi$) [arb.]", labelpad=10)

    center = 4.78
    band_wdith = 0.038
    pick_time=200*7/8

    ax_right.text(
        0.05, 0.91,  # near top-left corner in Axes coords
        rf"{round(pulse_width/ 9.8304 / 10) * 10} ns",
        transform=ax_right.transAxes,
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

    ax_right.text(abcd_x, abcd_y, r'\textbf{b}', transform=ax_right.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black') if subplot_idx==0 else None
    labels = [r'\textbf{i}', r'\textbf{ii}', r'\textbf{iii}']
    ax_right.text(i_x, i_y, labels[subplot_idx], transform=ax_right.transAxes,
            fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

            



# --------------3rd row, mag 0th slice--------------
ax_right = fig.add_subplot(gs[2, 0:5])
cax = fig.add_subplot(gs[2, 6])

zero_slice_mag_log_TP = full_record_mag_log_TP_matricies[:,:,0]

y_axis = pulse_width_list / 9830.4
im = ax_right.imshow(
    zero_slice_mag_log_TP,
    aspect='auto',
    extent=[freq_axis[0], freq_axis[-1], y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    # vmin=0,
    # vmax=300,
)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=10)
ax_right.set_xlim(display_freq_range)
ax_right.set_ylim([None, 0.2])
ax_right.set_ylabel(rf"Pulse Duration [$\mu$s]")
ax_right.set_xlabel('Frequency [GHz]')


for spine in ax_right.spines.values():
    spine.set_linestyle((0, (6, 10)))  # dashed: 6 on, 4 off
    spine.set_linewidth(2)
    spine.set_edgecolor(slice_and_box_color_1)

ax_right.text(-0.047, abcd_y, r'\textbf{c}', transform=ax_right.transAxes,
            fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')



# --------------forth row, phase fft 0 slice--------------

ax_right = fig.add_subplot(gs[3, 0:5])
cax = fig.add_subplot(gs[3, 6])

zero_slice_phase_TP_fft_matricies = full_record_phase_TP_fft_matricies[:,:,0]

im = ax_right.imshow(
    zero_slice_phase_TP_fft_matricies,
    aspect='auto',
    extent=[freq_axis[0], freq_axis[-1], y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=0,
    vmax=300,
)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"FFT($\phi$) [arb.]", labelpad=10)
ax_right.set_ylim([None, 0.2])
ax_right.set_ylabel(rf"Pulse Duration [$\mu$s]")
ax_right.set_xlim(display_freq_range)
ax_right.set_xlabel('Frequency [GHz]')

for spine in ax_right.spines.values():
    spine.set_linestyle((0, (6, 10)))  # dashed: 6 on, 4 off
    spine.set_linewidth(2)
    spine.set_edgecolor(slice_and_box_color_2)

ax_right.text(-0.047, abcd_y, r'\textbf{d}', transform=ax_right.transAxes,
            fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')


# --------------fifth row, phase fft 0 slice avg--------------

ax_right = fig.add_subplot(gs[4, 0:5])

zero_slice_phase_TP_fft_matricies_avg = np.mean(zero_slice_phase_TP_fft_matricies, axis=0)

peaks, _ = find_peaks(zero_slice_phase_TP_fft_matricies_avg, prominence=1, distance=1, height=120)
ax_right.plot(freq_axis, zero_slice_phase_TP_fft_matricies_avg, color="C1")
# ax_right.plot(freq_axis[peaks], zero_slice_phase_TP_fft_matricies_avg[peaks], 'o')
offset = 40  # y-units, adjust to your scale
ax_right.plot(
    freq_axis[peaks],
    zero_slice_phase_TP_fft_matricies_avg[peaks] + offset,  # move marker up
    marker='v',          
    markersize=6,        
    markerfacecolor='grey',
    markeredgecolor='none',
    linestyle='None',    
    label=f"Peaks (n={len(peaks)})"
)

print(f"{len(peaks)} peaks found.")

ax_right.set_xlim(display_freq_range)
ax_right.set_ylim([0, 1000])
ax_right.set_ylabel(r"FFT($\phi$) [arb.]")
ax_right.set_xlabel('Frequency [GHz]')

ax_right.text(-0.047, abcd_y, r'\textbf{e}', transform=ax_right.transAxes,
            fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

# # --- Save both the image data and the averaged line ---
# save_path = os.path.join(matrix_data_dir, 'phase_fft_slice_and_avg.npz')
# np.savez_compressed(
#     save_path,
#     # axes
#     freq_axis=freq_axis,            # shape (Nf,)
#     y_axis=y_axis,                  # shape (Ny,)
#     display_freq_range=np.array(display_freq_range),

#     # panel (e): imshow matrix (Ny x Nf)
#     phase_fft_slice=zero_slice_phase_TP_fft_matricies,  # from full_record_phase_TP_fft_matricies[:,:,0]
#     imshow_cmap='inferno',
#     imshow_origin='lower',
#     imshow_vmin=np.float32(0.0),
#     imshow_vmax=np.float32(300.0),

#     # panel (f): averaged line over y
#     avg_line=zero_slice_phase_TP_fft_matricies_avg      # shape (Nf,)
# )

# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, f"Fig_phase_V_sweep_width_include_pulse.png" if include_pulse else f"Fig_phase_V_sweep_width_fit_peaks.png")
if save_fig:
    plt.savefig(file_path, dpi=200, bbox_inches = 'tight')
    print(f"Figure saved to: {file_path}")
plt.show()
plt.close()