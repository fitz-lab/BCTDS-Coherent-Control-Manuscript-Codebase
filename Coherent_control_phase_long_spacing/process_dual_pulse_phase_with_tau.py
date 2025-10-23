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
fig_save_dir = os.path.join(script_dir, 'phase_FFT_analysis')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)

matrix_data_dir = os.path.join(script_dir, 'npz_data_base')

save_fig = True

display_freq_range = [3.0, 5.0]
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

reps = [0]
pulse_frequency_list = [3400] 
pulse_spacing_us_list =  [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# pulse_spacing_us_list =  [0.2, 0.1, 0.05, 0.03, 0.01]
pulse_phase_list = np.arange(0, 361, 3).tolist()

pulse_width = 308



for pulse_frequency in tqdm(pulse_frequency_list):

    fig = plt.figure(figsize=(7,7))  # Adjust width to accommodate both plots
    gs = gridspec.GridSpec(
        9, 3,
        width_ratios=[7, 0.3, .2],
        wspace=0,
        hspace=0.1  # Increase or decrease for more/less vertical spacing
    )

    for spacing_idx, pulse_spacing_us in tqdm(enumerate(pulse_spacing_us_list)):

        exp_ID = "Shipley_phase_8mK_20251012_132853"
        npz_name_prefix = f"{pulse_width}_freq_{pulse_frequency}_spacing_{pulse_spacing_us}_ID_{exp_ID}"

        npz_path = os.path.join(matrix_data_dir, f"{npz_name_prefix}.npz")
        with np.load(npz_path) as f: print({k: (f[k].shape, f[k].dtype) for k in f.files})
        
        with np.load(npz_path) as z:
            IQ_matrix = z["IQ_avg_matrix"]
            pulse_phase_list = z["pulse_phase_array"].tolist()
            time_stamp_list = z["time_stamp_list"]

        # T_still, T_mxc, T_still_avg, T_mxc_avg = fetch_still_and_mxc(time_stamp_list)


        display_offset = 11 #90
        pulse_start_offset = 432
        pulse_end_offset = 432 + 17
        transient_processing_offset = pulse_end_offset

        mag_matrix, mag_log_matrix, phase_matrix = truncate_display_offset(IQ_matrix, display_offset)
        mag_TP_matrix, mag_log_TP_matrix, phase_TP_matrix = extract_TP_region(IQ_matrix, display_offset, transient_processing_offset)

        # k_sigma = 4


        # ---------------- 1st colomn 1st row magnitude ----------------

        # ---------------- Right subplot: Color map ----------------
        ax_right = fig.add_subplot(gs[spacing_idx, 0])
        

        time_axis = (np.arange(mag_log_matrix.shape[1])) / 552.96
        phase_axis = np.array(pulse_phase_list)
        im = ax_right.imshow(
            mag_log_matrix,
            aspect='auto',
            extent=[time_axis[0], time_axis[-1], phase_axis[0], phase_axis[-1]],
            origin='lower',
            cmap='inferno',
            interpolation='none',
            vmin=-2,
            vmax=3,
        )
        ax_right.set_xlim([0, 1.6])
        # ax_right.set_xticks([0, 0.4, 0.8, 1.2, 1.6])
        ax_right.set_xticks([])
        ax_right.set_ylim([0, 360])
        ax_right.set_yticks([0, 180])
        # ax_right.set_xlabel('Frequency [GHz]')
        # ax_right.set_xlabel(rf'Time [$\mu$s]')
        ax_right.set_ylabel(rf'Phase [deg]', labelpad=12) if spacing_idx == 4 else None

        if spacing_idx == 8:
            cax = fig.add_subplot(gs[0:9, 2])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=7)
            ax_right.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
            ax_right.set_xlabel(rf'Time [$\mu$s]')

        # temp_line = (
        #     f"Still: {T_still_avg:.2f} K, MXC: {T_mxc_avg:.2f} K"
        #     if (not np.isnan(T_mxc_avg) and T_mxc_avg >= 1)
        #     else f"Still: {T_still_avg:.2f} K, MXC: {T_mxc_avg*1000:.2f} mK"
        # )

        # label = rf"{exp_ID}  {temp_line}"
        # label = rf"{exp_ID}"
        label = rf"{int(pulse_spacing_us*1000)} ns Spacing"

        ax_right.text(
            0.025, 0.8,  # near top-left corner in Axes coords
            label,
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


        if spacing_idx==8:
            # ax_right.axvline(x=pulse_start_offset/552.96, color="black", linestyle=slicing_V_line_style, linewidth=2, dashes=(3,3), alpha = 0.5, zorder=100)
            # ax_right.axvline(x=transient_processing_offset/552.96, color="black", linestyle=slicing_V_line_style, linewidth=2, dashes=(3,3), alpha = 0.5, zorder=100)
            tau_us_array, CI_tau_lower, CI_tau_upper, xc_array, CI_xc_lower, CI_xc_upper = lifetime_fit_matrix(
                mag_log_TP_matrix, time_spacing_us=1/552.96
            )
            print(rf"Single pulse tau: {(np.mean(tau_us_array) * 1e3):.3f} $\pm$ {(np.std(tau_us_array, ddof=1) * 1e3):.3f} ns")

        


    # ---------------- Figure save and show ----------------
    file_path = os.path.join(fig_save_dir, f"Fig_long_pulse_spacing.png")
    if save_fig:
        plt.savefig(file_path, dpi=200, bbox_inches = 'tight')
        print(f"Figure saved to: {file_path}")
    plt.show()
    plt.close()