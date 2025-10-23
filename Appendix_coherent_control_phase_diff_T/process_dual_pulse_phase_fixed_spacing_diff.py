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
fig_save_dir = os.path.join(script_dir, 'phase_FFT_analysis_fixed_spacing')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)



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
abcd_x = -0.10 # -0.47
abcd_y = 1.4 # 1.17
abcd_size = 23 #30

reps = [0]
# pulse_spacing_us_list =  [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
pulse_spacing_us_list =  [0.1]
pulse_phase_list = np.arange(0, 361, 3).tolist()

pulse_width = 308
pulse_frequency = 3400 #3400, 3560
# temperature_list = [8, 95, 200, 307, 400, 600, 750]
# exp_ID_list = ["Shipley_phase_8mK_20251012_132853", "Shipley_phase_95mK_20251012_221708", "Shipley_phase_200mK_20251013_130527", 
#                "Shipley_phase_307mK_20251013_212945", "Shipley_phase_400mK_20251014_124917", "Shipley_phase_600mK_20251014_220010", 
#                "Shipley_phase_750mK_20251015_110827"]

temperature_list = [750, 600, 400, 307, 200, 95, 8]
exp_ID_list = ["Shipley_phase_750mK_20251015_110827", "Shipley_phase_600mK_20251014_220010", "Shipley_phase_400mK_20251014_124917", 
               "Shipley_phase_307mK_20251013_212945", "Shipley_phase_200mK_20251013_130527", "Shipley_phase_95mK_20251012_221708", 
               "Shipley_phase_8mK_20251012_132853"]


for spacing_idx, pulse_spacing_us in tqdm(enumerate(pulse_spacing_us_list)):

    fig = plt.figure(figsize=(12,6))  # Adjust width to accommodate both plots
    gs = gridspec.GridSpec(
        len(temperature_list), 7,
        width_ratios=[7, 0.3, .2, 2.8, 7, 0.3, .2],
        height_ratios=[1,1,1,1,1,1,1],
        wspace=0,
        hspace=0.1  # Increase or decrease for more/less vertical spacing
    )


    # ---------------- Diff Avg ----------------
    # ax = fig.add_subplot(gs[8, 0:5])

    for temp_idx, temperature in enumerate(temperature_list):

        exp_ID = exp_ID_list[temp_idx]
        npz_name_prefix = f"{pulse_width}_freq_{pulse_frequency}_spacing_{pulse_spacing_us}_ID_{exp_ID}"

        matrix_data_dir = os.path.join(script_dir, f'npz_data_base_{temperature}mK')
        npz_path = os.path.join(matrix_data_dir, f"{npz_name_prefix}.npz")
        with np.load(npz_path) as f: print({k: (f[k].shape, f[k].dtype) for k in f.files})
        
        with np.load(npz_path) as z:
            IQ_matrix = z["IQ_avg_matrix"]
            pulse_phase_list = z["pulse_phase_array"].tolist()
            time_stamp_list = z["time_stamp_list"]

        # T_still, T_mxc, T_still_avg, T_mxc_avg = fetch_still_and_mxc(time_stamp_list)


        display_offset = 320 #90
        pulse_start_offset = 13 + 67
        pulse_end_offset = 30 + 67
        transient_processing_offset = pulse_end_offset

        mag_matrix, mag_log_matrix, phase_matrix = truncate_display_offset(IQ_matrix, display_offset)

        # k_sigma = 4



        # ---------------- Right subplot: Color map ----------------
        ax_right = fig.add_subplot(gs[temp_idx, 4])
        

        time_axis = (np.arange(mag_log_matrix.shape[1])) / 552.96
        phase_axis = np.array(pulse_phase_list)

        mag_log_matrix_diff_from_0 = mag_log_matrix - mag_log_matrix[0]
        mag_log_matrix_diff_from_avg = mag_log_matrix - np.mean(mag_log_matrix, axis=0)
        im = ax_right.imshow(
            mag_log_matrix_diff_from_avg,
            aspect='auto',
            extent=[time_axis[0], time_axis[-1], phase_axis[0], phase_axis[-1]],
            origin='lower',
            cmap='coolwarm',
            interpolation='none',
            vmin=-0.5,
            vmax=0.5,
        )
        ax_right.set_xlim([0, 1.0])
        # ax_right.set_xticks([0, 0.4, 0.8, 1.2, 1.6])
        ax_right.set_xticks([])
        ax_right.set_ylim([0, 360])
        ax_right.set_yticks([0, 180])
        # ax_right.set_xlabel('Frequency [GHz]')
        # ax_right.set_xlabel(rf'Time [$\mu$s]')
        ax_right.set_ylabel(rf'Phase [deg]') if temp_idx == 3 else None

        if temp_idx == len(temperature_list) - 1:
            cax = fig.add_subplot(gs[0:len(temperature_list), 6])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=2)
            ax_right.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_right.set_xlabel(rf'Time [$\mu$s]')

        label = rf"{temperature} mK"

        ax_right.text(
            0.03, 0.85,  # near top-left corner in Axes coords
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

        ax_right.text(abcd_x, abcd_y, r'\textbf{b}', transform=ax_right.transAxes,
        fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black') if temp_idx == 0 else None



        # ---------------- Left subplot: Color map ----------------
        ax_left = fig.add_subplot(gs[temp_idx, 0])
        
        im = ax_left.imshow(
            mag_log_matrix,
            aspect='auto',
            extent=[time_axis[0], time_axis[-1], phase_axis[0], phase_axis[-1]],
            origin='lower',
            cmap='inferno',
            interpolation='none',
            vmin=-2,
            vmax=3,
        )
        ax_left.set_xlim([0, 1.0])
        ax_left.set_xticks([])
        ax_left.set_ylim([0, 360])
        ax_left.set_yticks([0, 180])
        ax_left.set_ylabel(rf'Phase [deg]') if temp_idx == 3 else None

        if temp_idx == len(temperature_list) - 1:
            cax = fig.add_subplot(gs[0:len(temperature_list), 2])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=2)
            ax_left.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_left.set_xlabel(rf'Time [$\mu$s]')

        ax_left.text(
            0.03, 0.85,  # near top-left corner in Axes coords
            label,
            transform=ax_left.transAxes,
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
        colors = plt.cm.coolwarm(np.linspace(0, 0.4, len(temperature_list)))
        # phase_diff_avg = np.mean(np.abs(mag_log_matrix_diff_from_0), axis=0)
        phase_diff_avg = np.mean(np.abs(mag_log_matrix_diff_from_avg), axis=0)
        phase_diff_avg = np.abs(mag_log_matrix_diff_from_avg)[int(len(mag_log_matrix_diff_from_avg)/2)]
        time_slice_idx = 232
        # ax_right.axvline(x=time_slice_idx/552.96, color="black", linestyle="--", linewidth=1.2, dashes=(3,6), alpha = 0.4, zorder=100)
        time_slice_array = mag_log_matrix_diff_from_avg[:, time_slice_idx]

        ax_left.text(abcd_x, abcd_y, r'\textbf{a}', transform=ax_left.transAxes,
        fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black') if temp_idx == 0 else None

        # ax.plot(time_axis, phase_diff_avg, lw=0.2, color=colors[temp_idx], label=f"{temperature} mK")
        # ax.plot(phase_axis, time_slice_array, lw=1, color=colors[temp_idx], label=f"{temperature} mK")
        # print(np.shape(time_slice_array))
        # print(np.shape(phase_axis))
        # time.sleep(100000)
    
    # ax.set_xlim([0, 1.0])
    # ax.set_xlim([0, 360])
    # ax.set_xticks(range(0, 361, 60))
    # ax.set_ylabel(r"Log$_{10}$(A) [arb.]")
    # ax.set_xlabel("Phase [deg]")
    # ax.legend(
    #     loc='upper left', bbox_to_anchor=(1.01, 1),
    #     fontsize=12, framealpha=1.0
    # )

    # ---------------- Figure save and show ----------------
    file_path = os.path.join(fig_save_dir, f"Fig_phase_control_temp.png")
    if save_fig:
        plt.savefig(file_path, dpi=200, bbox_inches = 'tight')
        print(f"Figure saved to: {file_path}")
    plt.show()
    plt.close()