import os, sys
sys.path.append(r"C:\Users\chris\Desktop\local test scripts\BCTDS_processing_pipeline\post_processing_helper") #import helpers
from mag_and_phase_fft import *
from MAD_threshold_peak_finding import *
from seperate_pulse_transient_region import *
from transmission_and_lifetime import *
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
from pathlib import Path
# import duckdb
# sys.path.append(r"S:\fitzlab\code\BlueFors Log DB\scripts")
# from read_available_log_value import *


# #### Preamble
# # Set the font globally to Helvetica
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['figure.titlesize'] = 14

script_dir = os.path.dirname(os.path.abspath(__file__))
fig_save_dir = os.path.join(script_dir, 'phase_FFT_analysis')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)

matrix_data_dir = os.path.join(script_dir, 'npz_data_base')

save_fig = True


include_pulse = True

display_freq_range = [3.3, 3.8]
# display_freq_ticks = [4.1, 4.2, 4.3, 4.4, 4.5, 4.6]
line_plot_line_width = 0.5
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



pulse_width = 308
cooldown_npz_list = ["Cooldown_1_Oct_8", "Cooldown_2_Oct_11", "Cooldown_2_Oct_16", "Cooldown_2_Oct_19"]
cooldown_note = ["Cooldown 1, Oct 8th 2025", "Cooldown 2, Oct 11th 2025", "Cooldown 2, Oct 16th 2025", "Cooldown 2, Oct 19th 2025"]
letter_list = [r'\textbf{a}', r'\textbf{b}', r'\textbf{c}', r'\textbf{d}']


fig = plt.figure(figsize=(9,10))  # Adjust width to accommodate both plots
gs = gridspec.GridSpec(
    4, 7,
    width_ratios=[7, 0.3, .2, 4, 7, 0.3, .2],
    wspace=0,
    hspace=0.2  # Increase or decrease for more/less vertical spacing
)

for subplot_idx, npz_prefix in enumerate(cooldown_npz_list):

    npz_path = os.path.join(matrix_data_dir, f"FM_Shipley_{npz_prefix}.npz")
    with np.load(npz_path) as f: print({k: (f[k].shape, f[k].dtype) for k in f.files})
    
    with np.load(npz_path) as z:
        IQ_matrix = z["IQ_avg_matrix"]
        pulse_frequency_list = z["pulse_freq_array"].tolist()
        time_stamp_list = z["time_stamp_list"]
        T_still = z["T_still"]
        T_mxc = z["T_mxc"]
        mask_still = ~np.isnan(T_still) 
        T_still_avg = float(T_still[mask_still].mean()) if mask_still.any() else np.nan
        mask_mxc = ~np.isnan(T_mxc) 
        T_mxc_avg = float(T_mxc[mask_mxc].mean()) if mask_mxc.any() else np.nan

        # print(time_stamp_list[0])



    display_offset = 90 #90
    pulse_start_offset = 13
    pulse_end_offset = 30
    transient_processing_offset = pulse_end_offset

    mag_matrix, mag_log_matrix, phase_matrix = truncate_display_offset(IQ_matrix, display_offset)
    mag_TP_matrix, mag_log_TP_matrix, phase_TP_matrix = extract_TP_region(IQ_matrix, display_offset, transient_processing_offset)
    mag_PR_matrix, mag_log_PR_matrix, phase_PR_matrix = extract_pulse_region(IQ_matrix, display_offset, pulse_start_offset, pulse_end_offset)



    # interp = True
    k_sigma = 4

    fft_freq_axis, phase_TP_fft_matrix = matrix_fft(phase_TP_matrix)
    # interpolated_freq_axis, interpolated_fft_freq_axis, interpolated_phase_TP_fft_matrix = matrix_fft_interpolated_biaxial(phase_TP_matrix, pulse_frequency_list, spacing_MHz=1.0)
    mag_log_fft_freq_MHz, mag_log_TP_fft_matrix = matrix_fft(mag_log_TP_matrix)
    mag_log_TP_fft_log_matrix = np.log10(mag_log_TP_fft_matrix + 0.01)
    # print(np.shape(interpolated_phase_TP_fft_matrix))




    # ---------------- left: magnitude ----------------
    ax_right = fig.add_subplot(gs[subplot_idx, 0])
    cax = fig.add_subplot(gs[subplot_idx, 2])

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
    ax_right.set_xlabel('Frequency [GHz]') if subplot_idx==3 else None
    ax_right.set_ylabel(rf'Time [$\mu$s]')


    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=7)

    ax_right.text(
        0.05, 0.91,  # near top-left corner in Axes coords
        rf"{cooldown_note[subplot_idx]}",
        transform=ax_right.transAxes,
        fontsize=12,
        ha='left', va='top',
        color='black',
        bbox=dict(
            facecolor='white',
            alpha=0.5,
            boxstyle='round,pad=0.3',
            edgecolor='none'
        )
    )

    if include_pulse:
        ax_right.axhline(y=pulse_start_offset/552.96, color="black", linestyle=slicing_V_line_style, linewidth=2, dashes=(9,9), alpha = 0.5, zorder=100)
        ax_right.axhline(y=transient_processing_offset/552.96, color="black", linestyle=slicing_V_line_style, linewidth=2, dashes=(9,9), alpha = 0.5, zorder=100)

    ax_right.text(abcd_x, abcd_y, letter_list[subplot_idx], transform=ax_right.transAxes,
                fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')
    ax_right.text(i_x, i_y, r"\textbf{i}", transform=ax_right.transAxes,
                fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='white')
    

    # ---------------- right: fft of phase ----------------
    ax_right = fig.add_subplot(gs[subplot_idx, 4])
    cax = fig.add_subplot(gs[subplot_idx, 6])

    im = ax_right.imshow(
        phase_TP_fft_matrix.T,
        aspect='auto',
        extent=[freq_axis[0], freq_axis[-1], fft_freq_axis[0], fft_freq_axis[-1]],
        origin='lower',
        cmap='inferno',
        interpolation='none',
        vmin=0,
        vmax=300,
    )

    ax_right.set_ylim(phase_V_fft_display_range)
    # ax_right.set_ylim([0, 50])
    ax_right.set_xlim(display_freq_range)
    ax_right.set_xlabel('Frequency [GHz]') if subplot_idx==3 else None
    ax_right.set_ylabel(rf'FFT Freq. [MHz]')

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"FFT($\phi$) [arb.]", labelpad=10)

    ax_right.text(
        0.05, 0.91,  # near top-left corner in Axes coords
        (f"Still: {T_still_avg:.2f} K, MXC: {T_mxc_avg:.2f} K"
            if (not np.isnan(T_mxc_avg) and T_mxc_avg >= 1)
            else f"Still: {T_still_avg:.2f} K, MXC: {T_mxc_avg*1000:.2f} mK"),
        transform=ax_right.transAxes,
        fontsize=12,
        ha='left', va='top',
        color='black',
        bbox=dict(
            facecolor='white',
            alpha=0.5,
            boxstyle='round,pad=0.3',
            edgecolor='none'
        )
    )

    center = 4.78
    band_wdith = 0.038
    pick_time=200*7/8

    # ax_right.text(abcd_x, abcd_y, r'\textbf{c}', transform=ax_right.transAxes,
    #             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')
    # labels = [r'\textbf{i}', r'\textbf{ii}', r'\textbf{iii}']
    # ax_right.text(i_x, i_y, labels[subplot_idx], transform=ax_right.transAxes,
    #         fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')       
    ax_right.text(i_x, i_y, r"\textbf{ii}", transform=ax_right.transAxes,
                fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='white')         



# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, f"Fig_Shipley_4_point_compare.png")
if save_fig:
    plt.savefig(file_path, dpi=600, bbox_inches = 'tight')
    print(f"Figure saved to: {file_path}")
plt.show()
plt.close()