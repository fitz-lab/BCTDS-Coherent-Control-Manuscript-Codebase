import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "post_processing_helper"))

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
# fig_save_dir = os.path.join(script_dir, 'phase_FFT_analysis')
# if not os.path.isdir(fig_save_dir):
#     os.mkdir(fig_save_dir)

process_result_npz_directory = os.path.join(script_dir, 'processed_result_npz')
if not os.path.isdir(process_result_npz_directory):
    os.mkdir(process_result_npz_directory)

matrix_data_dir = os.path.join(script_dir, 'npz_data_base_with_T')

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
abcd_x = -0.16 # -0.47
abcd_y = 1.17 # 1.17
abcd_size = 23 #30



pulse_width_list = [308]
reps = range(0, 106, 1)
# reps = range(142, 240, 1)
# reps = [0]

pulse_width = 308

for pulse_width in tqdm(pulse_width_list):

    for rep in tqdm(reps):

        exp_ID = "FM_Shipley_cont_HFSS_calib_20251008_174837"
        npz_name_prefix = f"{pulse_width}_rep_{rep}_ID_{exp_ID}"

        npz_path = os.path.join(matrix_data_dir, f"{npz_name_prefix}_with_T.npz")
        with np.load(npz_path) as f: print({k: (f[k].shape, f[k].dtype) for k in f.files})

        processed_data_npz_path = os.path.join(process_result_npz_directory, f"processed_lifetime_{npz_name_prefix}.npz")
        
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


        display_offset = 90 #90
        pulse_start_offset = 13
        pulse_end_offset = 30
        transient_processing_offset = pulse_end_offset

        mag_matrix, mag_log_matrix, phase_matrix = truncate_display_offset(IQ_matrix, display_offset)
        mag_TP_matrix, mag_log_TP_matrix, phase_TP_matrix = extract_TP_region(IQ_matrix, display_offset, transient_processing_offset)
        mag_PR_matrix, mag_log_PR_matrix, phase_PR_matrix = extract_pulse_region(IQ_matrix, display_offset, pulse_start_offset, pulse_end_offset)



        # interp = True
        k_sigma = 4

        interpolated_freq_axis, interpolated_fft_freq_axis, phase_TP_fft_matrix = matrix_fft_interpolated_biaxial(phase_TP_matrix, pulse_frequency_list, spacing_MHz=1.0)
        mag_log_fft_freq_MHz, mag_log_TP_fft_matrix = matrix_fft(mag_log_TP_matrix)
        mag_log_TP_fft_log_matrix = np.log10(mag_log_TP_fft_matrix + 0.01)
        # print(np.shape(phase_TP_fft_matrix))

        



        fig = plt.figure(figsize=(12,8))  # Adjust width to accommodate both plots
        gs = gridspec.GridSpec(
            2, 7,
            width_ratios=[7, 0.3, .2, 3, 7, 0.3, .2],
            wspace=0,
            hspace=0.2  # Increase or decrease for more/less vertical spacing
        )



        # ---------------- 1st colomn 1st row magnitude ----------------

        # ---------------- Right subplot: Color map ----------------
        ax_right = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 2])

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
        # ax_right.set_xlabel('Frequency [GHz]')
        ax_right.set_ylabel(rf'Time [$\mu$s]')


        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=7)

        ax_right.text(
            0.05, 0.91,  # near top-left corner in Axes coords
            rf"{exp_ID}",
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

        ax_right.text(abcd_x, abcd_y, r'\textbf{a}', transform=ax_right.transAxes,
                    fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')
        


        # ---------------- 1st colomn 2nd transmission and lifetime ----------------

        # ---------------- Right subplot: Color map ----------------
        ax_right = fig.add_subplot(gs[1, 0])

        # --- lifetime & xc with CIs (unchanged) ---
        tau_us_array, CI_tau_lower, CI_tau_upper, xc_array, CI_xc_lower, CI_xc_upper = lifetime_fit_matrix(
            mag_log_TP_matrix, time_spacing_us=1/552.96
        )
        freq_axis = np.array(pulse_frequency_list[: mag_log_TP_matrix.shape[0]])/1e3
        tau_line, = ax_right.plot(freq_axis, tau_us_array, label=r'$\tau$ ($\mu$s)', linewidth=line_plot_line_width)
        mask_tau = np.isfinite(CI_tau_lower) & np.isfinite(CI_tau_upper)
        ax_right.fill_between(freq_axis[mask_tau], CI_tau_lower[mask_tau], CI_tau_upper[mask_tau],
                            color=tau_line.get_color(), alpha=0.5, linewidth=0, zorder=tau_line.get_zorder()-1)

        xc_line, = ax_right.plot(freq_axis, xc_array, label=r'$x_c$ ($\mu$s)', linewidth=line_plot_line_width)
        mask_xc = np.isfinite(CI_xc_lower) & np.isfinite(CI_xc_upper)
        ax_right.fill_between(freq_axis[mask_xc], CI_xc_lower[mask_xc], CI_xc_upper[mask_xc],
                            color=xc_line.get_color(), alpha=0.5, linewidth=0, zorder=xc_line.get_zorder()-1)

        ax_right.set_xlabel("Frequency (GHz)")
        ax_right.set_ylabel(r'$\tau$ ($\mu$s)')
        ax_right.set_xlim(display_freq_range)
        ax_right.set_ylim([0, 0.8])

        # --- NEW: twin axis for average transmission ---
        ax_S21 = ax_right.twinx() 
        average_transmission = np.mean(mag_log_PR_matrix, axis=1)
        # print(np.shape(average_transmission))
        S21_line, = ax_S21.plot(freq_axis, average_transmission, label='Transmission', linewidth=line_plot_line_width, c='green')
        ax_S21.set_ylabel('Transmission (arb.)')

        ax_right.text(abcd_x, abcd_y, r'\textbf{b}', transform=ax_right.transAxes,
                fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')


        # ---------------- 2nd colomn 1st row fft of phase ----------------

        # ---------------- Right subplot: Color map ----------------
        ax_right = fig.add_subplot(gs[0, 4])
        cax = fig.add_subplot(gs[0, 6])

        im = ax_right.imshow(
            phase_TP_fft_matrix.T,
            aspect='auto',
            extent=[interpolated_freq_axis[0], interpolated_freq_axis[-1], interpolated_fft_freq_axis[0], interpolated_fft_freq_axis[-1]],
            origin='lower',
            cmap='inferno',
            interpolation='none',
            vmin=0,
            vmax=300,
        )

        ax_right.set_ylim(phase_V_fft_display_range)
        # ax_right.set_ylim([0, 50])
        ax_right.set_xlim(display_freq_range)
        ax_right.set_xlabel('Frequency [GHz]')
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

        ax_right.text(abcd_x, abcd_y, r'\textbf{c}', transform=ax_right.transAxes,
                    fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')
        # labels = [r'\textbf{i}', r'\textbf{ii}', r'\textbf{iii}']
        # ax_right.text(i_x, i_y, labels[subplot_idx], transform=ax_right.transAxes,
        #         fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')                


        # --------------2nd colomn 2nd row, phase fft 0 slice avg--------------

        ax_right = fig.add_subplot(gs[1, 4])

        # print(np.shape(phase_TP_fft_matrix))
        V_fit_avg_phase_TP_fft_matricies = average_V_diagonals(phase_TP_fft_matrix, k_max=50)

        peaks, _ = find_peaks(V_fit_avg_phase_TP_fft_matricies, prominence=1, distance=1, height=180)


        thresh = noise_threshold_mad(V_fit_avg_phase_TP_fft_matricies, k=k_sigma)
        peaks, _ = find_peaks(V_fit_avg_phase_TP_fft_matricies, prominence=1, distance=1, height=thresh)
        out = peak_count_uncertainty_neffN(N_obs=len(peaks), N=len(V_fit_avg_phase_TP_fft_matricies), k=k_sigma)
        # print(freq_axis[peaks])
        # print(f"N ≈ {out['N_true']:.1f} ± {out['err']:.1f}  (E_FP={out['E_FP']:.2f})")

        ax_right.plot(interpolated_freq_axis, V_fit_avg_phase_TP_fft_matricies, color="C1")

        ymin, ymax = ax_right.get_ylim()
        ax_right.axhspan(0, thresh, facecolor="0.7", alpha=0.25, zorder=0)

        # ax_right.plot(freq_axis[peaks], V_fit_avg_phase_TP_fft_matricies[peaks], 'o')
        offset = 10 #40
        ax_right.plot(
            interpolated_freq_axis[peaks],
            V_fit_avg_phase_TP_fft_matricies[peaks] + offset,  # move marker up
            marker='v',          
            markersize=6,        
            markerfacecolor='grey',
            markeredgecolor='none',
            linestyle='None',    
            label=rf"Peaks (n={len(peaks)} ± {out['err']:.1f}), threshold: {k_sigma} $\sigma$"
        )

        # print(f"{len(peaks)} peaks found.")

        ax_right.set_xlim(display_freq_range)
        ax_right.set_ylim([0, None])
        ax_right.set_ylabel(r"FFT($\phi$) [arb.]")
        ax_right.set_xlabel('Frequency [GHz]')

        ax_right.text(abcd_x, abcd_y, r'\textbf{d}', transform=ax_right.transAxes,
                    fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

        ax_right.text(
            0.05, 0.91,  # near top-left corner in Axes coords
            rf"Peaks (n={len(peaks)} ± {out['err']:.1f}), threshold: {k_sigma} $\sigma$",
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


        # # ---------------- Figure save and show ----------------
        # file_path = os.path.join(fig_save_dir, f"{pulse_width}_rep_{rep}_ID_{exp_ID}.png")
        # if save_fig:
        #     plt.savefig(file_path, dpi=600, bbox_inches = 'tight')
        #     print(f"Figure saved to: {file_path}")
        # # plt.show()
        plt.close()

        time_stamp_list = np.asarray(time_stamp_list, dtype='datetime64[s]')
        np.savez(
            processed_data_npz_path,
            freq_axis=freq_axis,
            mag_log_matrix=mag_log_matrix,
            time_axis=time_axis,
            tau_us_array=tau_us_array,
            xc_array=xc_array,
            average_transmission=average_transmission,
            time_stamp_list=time_stamp_list,
            T_still=T_still,
            T_mxc=T_mxc,
            phase_TP_fft_matrix=phase_TP_fft_matrix,
            interpolated_freq_axis=interpolated_freq_axis,
            interpolated_fft_freq_axis=interpolated_fft_freq_axis,
            V_fit_avg_phase_TP_fft_matricies=V_fit_avg_phase_TP_fft_matricies
            )