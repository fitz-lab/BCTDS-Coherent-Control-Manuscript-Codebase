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
from matplotlib.widgets import Slider

def fit_piecewise_slider(trace_list, time_spacing_us, show_slider=True):
    N = len(trace_list)
    t = np.arange(len(trace_list[0])) * time_spacing_us
    tau_array = np.full(N, np.nan)
    fit_curves = []

    # --- Piecewise model: continuous linear -> flat ---
    def piecewise_model_continuous(t, x_c, k, b):
        return np.where(t < x_c, k * t + b, k * x_c + b)

    # --- Fit all traces ---
    for trace in trace_list:
        if not isinstance(trace, np.ndarray):
            fit_curves.append(None)
            continue
        try:
            x_c0 = t[len(t)//2]
            k0 = (trace[5] - trace[0]) / (t[5] - t[0])
            b0 = trace[0]
            p0 = [x_c0, k0, b0]
            bounds = ([t[0], -np.inf, -np.inf], [t[-1], np.inf, np.inf])
            popt, _ = curve_fit(piecewise_model_continuous, t, trace, p0=p0, bounds=bounds)
            tau_array[len(fit_curves)] = -np.log10(np.e)/popt[1] * 1000 # Save x_c as effective decay time
            fit_curves.append((popt, piecewise_model_continuous(t, *popt)))
        except Exception:
            fit_curves.append((None, None))

    if not show_slider:
        return tau_array

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.subplots_adjust(bottom=0.3)

    first_trace = trace_list[0]
    trace_line, = ax.plot(t, first_trace, label="Data", color="black")
    first_fit = fit_curves[0][1] if fit_curves[0][1] is not None else np.full_like(t, np.nan)
    fit_line, = ax.plot(t, first_fit, label="Fit", color="red")

    title = ax.set_title(f"Trace at {3000:.1f} MHz")
    equation_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            fontsize=10, verticalalignment='top')

    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)

    # --- Slider setup ---
    slider_ax = plt.axes([0.2, 0.15, 0.6, 0.03])
    idx_slider = Slider(
        ax=slider_ax,
        label='Index',
        valmin=0,
        valmax=N - 1,
        valinit=0,
        valstep=1,
    )

    # --- Update function ---

    def update(val):
        idx = int(idx_slider.val)
        trace = trace_list[idx]
        trace_line.set_ydata(trace)

        popt, fit_vals = fit_curves[idx]
        if popt is not None:
            fit_line.set_ydata(fit_vals)
            fit_line.set_visible(True)

            # Refit to get A, tau for annotation
            try:
                x_c, k, b = popt
                C = k * x_c + b
                eq_str = (
                    rf"$y(t) = {k:.2f} \cdot t + {b:.2f}, \quad t < {x_c:.2f}$" + "\n" +
                    rf"$y(t) = {C:.2f}, \quad t \ge {x_c:.2f}$"
                )
                equation_text.set_text(eq_str)
            except Exception:
                equation_text.set_text('')
        else:
            fit_line.set_visible(False)
            equation_text.set_text('')

        title.set_text(f"Trace at {3000 + idx:.1f} MHz")
        fig.canvas.draw_idle()

    idx_slider.on_changed(update)

    # Non-blocking display; resume after closing
    plt.show(block=False)
    input("Press Enter to continue...")
    plt.close(fig)
    plt.pause(0.001)

    return tau_array

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

script_dir = os.path.dirname(os.path.abspath(__file__))

fig_save_dir = os.path.join(script_dir, 'analysis_plots')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)

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


matrix_data_dir = os.path.join(script_dir, 'matrix_npy')

save_fig = True

fig = plt.figure(figsize=(7, 9))

gs = gridspec.GridSpec(
    nrows=3,
    ncols=2,
    wspace=0.04,
    hspace=0.3,
    width_ratios=[1, 5]
)

pulse_start_idx = 11
pulse_end_idx = 28
line_plot_line_width = 0.8
pulse_mark_line_width = 1.5
pulse_mark_line_color = 'black'
pulse_mark_line_style = '--'
display_freq_range = [3, 5]
display_tau_range = [0,200]
display_tau_ticks = [40, 160]
frequency_ticks = [3.0, 3.5, 4.0, 4.5, 5.0]
# tau_color = plt.cm.inferno(215)
tau_color = '#F57C1F'
transmission_color = 'indigo'
text_box_x = 0.96
text_box_y = 0.09
text_box_font_size = 20
show_fit_slices = False
i_x = 0.9
i_y = 0.95
ii_x = 0.97
ii_y = 0.95
i_ii_size = 27
abcd_x = -0.47
abcd_y = 1.15
abcd_size = 30


# ---------------- FIRST ROW PLOTS ----------------

# ---------------- Process Data ----------------

npy_name_prefix = "2nm_AlOx_IQ_avg_matrix_07_03"
display_offset = 25
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()

IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:display_offset+900]
mag_avg_log_matrix = mag_avg_log_matrix[200::5] 
mag_avg_log_matrix = mag_avg_log_matrix[:200, :] 
pulse_frequency_list = np.arange(3200, 4200, 5).tolist()
print(np.shape(mag_avg_log_matrix))

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_subplot(gs[0, 1])

time_axis = np.arange(mag_avg_log_matrix.shape[1])/552.96
freq_axis = np.array(pulse_frequency_list) /1e3
im = ax_right.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)
ax_right.axvline(x=pulse_start_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.axvline(x=pulse_end_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.set_xlim([0, 0.8])
ax_right.set_ylim([3.2, 4.2])
# ax_right.set_yticks(frequency_ticks)
ax_right.set_ylabel("Frequency [GHz]")

# ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
#              fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

ax_right.text(
    text_box_x, text_box_y, "Measurement 1",  # position in axes coordinates
    transform=ax_right.transAxes,
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



# ---------------- 2nd ROW PLOTS ----------------

# ---------------- Process Data ----------------

npy_name_prefix = "2nm_AlOx_IQ_avg_matrix_07_10"
display_offset = 93
pulse_frequency_list = np.arange(3200, 4200, 5).tolist()

IQ_avg_matrix_2 = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}.npy"))
I_avg_matrix_2 = IQ_avg_matrix_2[0]
Q_avg_matrix_2 = IQ_avg_matrix_2[1]
mag_avg_matrix_2 = np.abs(I_avg_matrix_2 + 1j *Q_avg_matrix_2)
mag_avg_log_matrix_2 = np.log10(mag_avg_matrix_2 + 0.01)
mag_avg_log_matrix_2 = mag_avg_log_matrix_2[:, display_offset:display_offset+900]
print(np.shape(mag_avg_log_matrix_2))

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_subplot(gs[1, 1])

time_axis = np.arange(mag_avg_log_matrix_2.shape[1])/552.96
freq_axis = np.array(pulse_frequency_list) /1e3
im = ax_right.imshow(
    mag_avg_log_matrix_2,
    aspect='auto',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)
ax_right.axvline(x=pulse_start_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.axvline(x=pulse_end_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.set_xlim([0, 0.8])
ax_right.set_ylim([3.2, 4.2])
# ax_right.set_yticks(frequency_ticks)
ax_right.set_ylabel("Frequency [GHz]")
# ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
#              fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

ax_right.text(
    text_box_x, text_box_y, "Measurement 2",  # position in axes coordinates
    transform=ax_right.transAxes,
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

# ---------------- THIRD ROW PLOTS ----------------

# ---------------- Process Data ----------------

pulse_frequency_list = np.arange(3000, 5000, 1).tolist()

mag_avg_log_matrix_diff = mag_avg_log_matrix_2 - mag_avg_log_matrix
print(f'diff: {np.shape(mag_avg_log_matrix_diff)}')

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_subplot(gs[2, 1])

im = ax_right.imshow(
    mag_avg_log_matrix_diff,
    aspect='auto',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    origin='lower',
    cmap='coolwarm',
    interpolation='none',
    vmin=-1.2,
    vmax=1.2,
)
ax_right.axvline(x=pulse_start_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.axvline(x=pulse_end_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.set_xlim([0, 0.8])
ax_right.set_ylim([3.2, 4.2])
ax_right.set_ylabel("Frequency [GHz]")
# ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
#              fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')
ax_right.set_xlabel(r"Time [$\mu$s]")
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

ax_right.text(
    text_box_x, text_box_y, r"Difference",  # position in axes coordinates
    transform=ax_right.transAxes,
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


# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, f"Fig_same_cooldown_compare.png")
if save_fig:
    plt.savefig(file_path, dpi=300, bbox_inches = 'tight')
    print(f"Figure saved to: {file_path}")
plt.show()
plt.close()
