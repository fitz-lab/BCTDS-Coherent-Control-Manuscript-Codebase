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
import matplotlib.image as mpimg
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
fig_save_dir = os.path.join(script_dir, 'temperature_dependence')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)

process_result_npz_directory = os.path.join(script_dir, 'processed_result_npz')

# matrix_data_dir = os.path.join(script_dir, 'npz_data_base')

save_fig = True


include_pulse = True

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
abcd_x = -0.05 # -0.47
abcd_y = 1.4 #1.17
abcd_size = 23 #30



pulse_width_list = [308]
reps = range(0, 51, 1)
# reps = range(142, 240, 1)
# reps = [0]

pulse_width = 308

freq_axis_list = []
mag_log_matrix_list = []
time_axis_list = []
tau_us_array_list = []
xc_array_list = []
average_transmission_list = []
time_stamp_list_list = []
T_still_avg_list = []
T_mxc_avg_list = []
phase_TP_fft_matrix_list = []
interpolated_freq_axis_list = []
interpolated_fft_freq_axis_list = []
V_fit_avg_phase_TP_fft_matricies_list = []

for pulse_width in tqdm(pulse_width_list):

    for rep in tqdm(reps):

        exp_ID = "FM_Shipley_cont_HFSS_calib_20251008_174837"
        npz_name_prefix = f"{pulse_width}_rep_{rep}_ID_{exp_ID}"

        npz_path = os.path.join(process_result_npz_directory, f"processed_lifetime_{npz_name_prefix}.npz")
        # with np.load(npz_path) as f: print({k: (f[k].shape, f[k].dtype) for k in f.files})

        # time.sleep(1000000)

        with np.load(npz_path, allow_pickle=True) as z:
            freq_axis_list=z["freq_axis"]
            mag_log_matrix_list.append(z["mag_log_matrix"])
            time_axis_list=z["time_axis"]
            tau_us_array_list.append(z["tau_us_array"])
            xc_array_list.append(z["xc_array"])
            average_transmission_list.append(z["average_transmission"])
            time_stamp_list_list.append(z["time_stamp_list"])  # may be object/strings
            phase_TP_fft_matrix_list.append(z["phase_TP_fft_matrix"])
            interpolated_freq_axis_list=z["interpolated_freq_axis"]
            interpolated_fft_freq_axis_list=z["interpolated_fft_freq_axis"]
            V_fit_avg_phase_TP_fft_matricies_list.append(z["V_fit_avg_phase_TP_fft_matricies"])

            T_still_list = z["T_still"]
            T_mxc_list = z["T_mxc"]
            mask_still = ~np.isnan(T_still_list)
            T_still_avg = float(T_still_list[mask_still].mean()) if mask_still.any() else np.nan
            mask_mxc = ~np.isnan(T_mxc_list)
            T_mxc_avg = float(T_mxc_list[mask_mxc].mean()) if mask_mxc.any() else np.nan
            T_still_avg_list.append(T_still_avg)
            T_mxc_avg_list.append(T_mxc_avg)

freq_axis = np.array(freq_axis_list)
mag_log_matrix_list = np.array(mag_log_matrix_list)
time_axis = np.array(time_axis_list)
tau_us_array = np.array(tau_us_array_list)
xc_array = np.array(xc_array_list)
average_transmission = np.array(average_transmission_list)
T_still_avg_list = np.array(T_still_avg_list)
T_mxc_avg_list = np.array(T_mxc_avg_list)
phase_TP_fft_matrix_list = np.array(phase_TP_fft_matrix_list)
interpolated_freq_axis = np.array(interpolated_freq_axis_list)
interpolated_fft_freq_axis = np.array(interpolated_fft_freq_axis_list)
V_fit_avg_phase_TP_fft_matricies = np.array(V_fit_avg_phase_TP_fft_matricies_list)
data_label_axis = range(len(tau_us_array))

# convert timestamps to hours

time_stamp_list = np.array(time_stamp_list_list, dtype='datetime64[s]')
hours = (time_stamp_list - time_stamp_list[0,0]) / np.timedelta64(1, 'h')
time_stamp_matrix = np.asmatrix(hours)
time_stamp_start_array = np.asarray(time_stamp_matrix)[:, 0] 
# print(np.shape(time_stamp_start_array))
# print(np.diff(time_stamp_start_array))
# print(time_stamp_start_array)
# time.sleep(100000)

# ---------------- 1st Image: log mag spectrum and tau fits at several slices  ----------------

fig = plt.figure(figsize=(12,8))  # Adjust width to accommodate both plots
gs = gridspec.GridSpec(
    4, 7,
    width_ratios=[7, 2.5, 7, 2.5, 7, 0.5, .4],
    height_ratios=[1, 3, 0.5, 4],
    wspace=0,
    hspace=0.1  # Increase or decrease for more/less vertical spacing
)

pulse_start_offset = 13
pulse_end_offset = 30
transient_processing_offset = pulse_end_offset

exp_slice_list = [0, 5, 45]
label_letter_list = [r'\textbf{a}', r'\textbf{b}', r'\textbf{c}']

for slice_plot_idx, exp_slice in enumerate(exp_slice_list): 
    mag_log_matrix = mag_log_matrix_list[exp_slice]
    tau_us = tau_us_array[exp_slice]

    print(np.max(tau_us))
    MXC_temp = T_mxc_avg_list[exp_slice]

    # ---------------- log magnitude ----------------
    ax_right = fig.add_subplot(gs[1, 2 * slice_plot_idx + 0])

    im = ax_right.imshow(
        mag_log_matrix.T,
        aspect='auto',
        extent=[freq_axis[0], freq_axis[-1], time_axis[0], time_axis[-1]],
        origin='lower',
        cmap='inferno',
        interpolation='none',
        vmin=-2,
        vmax=3,
    )
    ax_right.set_ylim([0, 0.8])
    ax_right.set_yticks([0, 0.2, 0.4, 0.6])
    ax_right.set_xlim(display_freq_range)
    ax_right.set_xlabel('Frequency [GHz]')
    ax_right.set_ylabel(rf'Time [$\mu$s]') if slice_plot_idx==0 else None

    if slice_plot_idx == 2:
        
        cax = fig.add_subplot(gs[1, 2 * slice_plot_idx + 2])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=7)

    ax_right.text(
        0.05, 0.91,  # near top-left corner in Axes coords
        rf"Index {exp_slice}, T_MXC: {MXC_temp:.2f} K",
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

    ax_right.text(i_x, i_y, r'\textbf{ii}', transform=ax_right.transAxes,
        fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

    if include_pulse:
        ax_right.axhline(y=pulse_start_offset/552.96, color="black", linestyle=slicing_V_line_style, linewidth=2, dashes=(9,9), alpha = 0.5, zorder=100)
        ax_right.axhline(y=transient_processing_offset/552.96, color="black", linestyle=slicing_V_line_style, linewidth=2, dashes=(9,9), alpha = 0.5, zorder=100)

    # ax_right.text(abcd_x, abcd_y, r'\textbf{a}', transform=ax_right.transAxes,
    #             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

    # ---------------- tau fit----------------
    ax_right = fig.add_subplot(gs[0, 2 * slice_plot_idx + 0])

    ax_right.plot(freq_axis, tau_us, label=r'$\tau$ ($\mu$s)', linewidth=line_plot_line_width)

    # ax_right.set_xlabel("Frequency (GHz)")
    ax_right.set_ylabel(r'$\tau$ [$\mu$s]') if slice_plot_idx == 0 else None
    ax_right.set_xlim(display_freq_range)
    ax_right.set_xticks([])
    ax_right.set_ylim([0, 0.15])

    ax_right.text(abcd_x, abcd_y, label_letter_list[slice_plot_idx], transform=ax_right.transAxes,
            fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')
    
    ax_right.text(i_x, i_y - 0.06, r'\textbf{i}', transform=ax_right.transAxes,
        fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')


# ---------------- temeperature, time, and, tau color plot ----------------

ax_left = fig.add_subplot(gs[3, 0])
# ax_left.plot(T_still_avg_list, data_label_axis, label="T_Still", c="#FECB00")
ax_left.plot(T_mxc_avg_list, data_label_axis, label="T_MXC", c="#0065FF")

ax_left.set_xscale('log')
all_T = np.r_[T_mxc_avg_list].astype(float)
pos = all_T[(all_T > 0) & np.isfinite(all_T)]
xmin = pos.min()
xmax = pos.max()
# ax_left.set_xlim(xmin*0.8, xmax*1.1)
ax_left.set_xlim(xmin*0.8, 100)
ax_left.set_ylim([0, 50])
ax_left.set_xlabel('Temperature [K]')
ax_left.set_ylabel('Data Set Index')
# ax_left.legend()
ax_left.grid(True, which='both', alpha=0.3)

h  = 6.62607015e-34        # Planck constant [J·s]
kB = 1.380649e-23          # Boltzmann constant [J/K]

# h f = kB T
ax_left.axvline(x=h * 3e9 / kB, color="black", linestyle=slicing_V_line_style, linewidth=1, dashes=(9,9), alpha = 0.5, zorder=1)
ax_left.axvline(x=h * 5e9 / kB, color="black", linestyle=slicing_V_line_style, linewidth=1, dashes=(9,9), alpha = 0.5, zorder=1)
# print(h * 3e9 / kB)
# print(h * 5e9 / kB)

def avg_photon_population(f_Hz, T_K):
    h  = 6.62607015e-34        # Planck constant [J·s]
    kB = 1.380649e-23          # Boltzmann constant [J/K]
    x = h * f_Hz / (kB * T_K)
    return 1/(np.exp(x) - 1)

# print(avg_photon_population(f_Hz=3e9, T_K=0.01))
# print(avg_photon_population(f_Hz=3e9, T_K=1))

# print(avg_photon_population(f_Hz=5e9, T_K=0.01))
# print(avg_photon_population(f_Hz=5e9, T_K=1))

ax_left.text(abcd_x, 1.12, r'\textbf{d}', transform=ax_left.transAxes,
        fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')






n = 10  # tick every n points
ax_time = ax_left.twinx()
i = slice(None, None, n)

ax_time.set(
    yticks=data_label_axis[i],
    yticklabels=[f"{h:.1f}" for h in time_stamp_start_array[i]],
    ylim=ax_left.get_ylim(),
    ylabel="Time [h]",
)
ax_time.grid(False)



ax_right = fig.add_subplot(gs[3, 2:5])
cax = fig.add_subplot(gs[3, 6])

im = ax_right.imshow(
    tau_us_array,
    aspect='auto',
    extent=[freq_axis[0], freq_axis[-1], data_label_axis[0], data_label_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=0,
    vmax=0.15,
)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(rf"$\tau$[$\mu$s]", labelpad=10)

ax_right.set_xlim(display_freq_range)
ax_right.set_yticks([]) 
ax_right.set_ylim([0, data_label_axis[-1]])
ax_right.set_xlabel('Frequency [GHz]')




for slice_plot_idx, exp_slice in enumerate(exp_slice_list):
    
    abc_size = 20
    if slice_plot_idx==2:
        x = T_mxc_avg_list[exp_slice] - 10
        y = data_label_axis[exp_slice]
        ax_left.text(x -12, y+1, label_letter_list[slice_plot_idx], fontsize=abc_size, fontweight='bold', ha='right', va='center', zorder=6)

        x = T_mxc_avg_list[exp_slice]
        y = data_label_axis[exp_slice] + 1
        ax_left.plot(T_mxc_avg_list[exp_slice], data_label_axis[exp_slice] + 1, marker='v', markersize=8,
            markerfacecolor='C1', markeredgecolor='none',
            linestyle='None', zorder=5)

        ax_right.axhline(y=data_label_axis[exp_slice] + 0.2, color="white", linestyle=slicing_V_line_style, linewidth=2, dashes=(9,9), alpha = 0.5, zorder=100)
        ax_right.text(3.05, data_label_axis[exp_slice] + 0.25, label_letter_list[slice_plot_idx], 
                      c = "white", fontsize=abc_size, fontweight='bold', ha='center', va='bottom', zorder=100)
    else:
        x = T_mxc_avg_list[exp_slice]
        y = data_label_axis[exp_slice]
        ax_left.plot(T_mxc_avg_list[exp_slice], data_label_axis[exp_slice] + 1, marker='v', markersize=8,
                 markerfacecolor='C1', markeredgecolor='none',
                 linestyle='None', zorder=5)
        ax_left.text(x, y + 3, label_letter_list[slice_plot_idx], fontsize=abc_size, fontweight='bold', ha='center', va='bottom', zorder=6)
        ax_right.axhline(y=data_label_axis[exp_slice] + 0.2, color="white", linestyle=slicing_V_line_style, linewidth=2, dashes=(9,9), alpha = 0.5, zorder=100)
        ax_right.text(3.05, data_label_axis[exp_slice] + 0.25, label_letter_list[slice_plot_idx], 
                      c = "white", fontsize=abc_size, fontweight='bold', ha='center', va='bottom', zorder=100)

ax_right.text(abcd_x/2, 1.12, r'\textbf{e}', transform=ax_right.transAxes,
        fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, "Fig_temp_dependence.png")
if save_fig:
    plt.savefig(file_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    print(f"Figure saved to: {file_path}")
plt.show()
plt.close()