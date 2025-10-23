from scipy.signal import firwin, filtfilt, hilbert
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import time
import matplotlib
# matplotlib.use('Agg')  # Use the 'Agg' backend, which does not require a display environment.
from mpl_toolkits.axes_grid1 import make_axes_locatable

def add_box_edge_arrows(ax, lw=2, style="-|>", head=16, overshoot=0.07):
    # Hide spines/ticks
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # Bottom arrow: left -> right, end past the box
    ax.annotate("",
        xy=(1+overshoot, 0), xycoords="axes fraction",
        xytext=(0, 0),      textcoords="axes fraction",
        arrowprops=dict(arrowstyle=style, lw=lw, color="black",
                        mutation_scale=head),
        clip_on=False, zorder=5
    )

    # Left arrow: bottom -> top, end past the box
    ax.annotate("",
        xy=(0, 1+overshoot), xycoords="axes fraction",
        xytext=(0, 0),       textcoords="axes fraction",
        arrowprops=dict(arrowstyle=style, lw=lw, color="black",
                        mutation_scale=head),
        clip_on=False, zorder=5
    )
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
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 25
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 16

script_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.join(script_dir, 'matrix_npy')

save_plot = True

# Create directory to save plots
fig_save_dir = os.path.join(script_dir, 'analysis_plots')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)

# Sampling setup
fs = 5000  # Sampling frequency (Hz)
t = np.linspace(0, 1, fs, endpoint=False)

# === Filter Design ===
cutoff = 30.0       # Low-pass cutoff frequency (Hz)
numtaps = 201       # Number of taps in FIR filter
fir_coeff = firwin(numtaps, cutoff, fs=fs, window='hamming')

# === Reference frequency for homodyne ===
f_ref = 50  # Hz

# === Column 1: Single frequency ===
f0 = 50  # Signal frequency matches reference
signal1 = np.sin(2 * np.pi * f0 * t)
ref1 = np.cos(2 * np.pi * f_ref * t)
homodyned1 = signal1 * ref1
homodyned1_filt = filtfilt(fir_coeff, [1.0], homodyned1)
phase1 = np.angle(hilbert(homodyned1_filt)+ np.pi/4)  # Wrapped phase

# === Column 2: Multiple frequencies ===
f0 = 50
f1 = 70
f2 = 90  # Detuned components
A0 = 1
A1 = 0.5
A2 = 0.3
signal2 = (
    A0 * np.sin(2 * np.pi * f0 * t) +
    A1 * np.sin(2 * np.pi * f1 * t + np.pi/4) +
    A2 * np.sin(2 * np.pi * f2 * t + np.pi/2)
)
ref2 = np.cos(2 * np.pi * f_ref * t)
homodyned2 = signal2 * ref2
homodyned2_filt = filtfilt(fir_coeff, [1.0], homodyned2)
phase2 = np.angle(hilbert(homodyned2_filt))  # Wrapped phase

# === Plotting ===
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(4, 3, 
    # height_ratios=[1, 1],  # second row slightly taller
    width_ratios=[1, 0.4, 0.7],
    hspace=0.6,             # vertical space between rows
    wspace=0.4
)

plot_lw = 1.5
# Common y-limits
x_signal_lim = (0.1, 0.3)
y_signal_lim = (-2, 2)
y_phase_lim = (-np.pi, np.pi)

# Left Column: Single Frequency
ax_top_1 = fig.add_subplot(gs[0:2, 0])
ax_top_1.plot(t, signal1, color='black', lw=plot_lw)
# ax_top_1.set_title('Single Freq Signal')
ax_top_1.set_ylim(y_signal_lim)
ax_top_1.set_xlim(x_signal_lim)
add_box_edge_arrows(ax_top_1)

ax_top_2 = fig.add_subplot(gs[0, 2])
line2, = ax_top_2.plot(t, homodyned1_filt, color='blue', label='Amplitude', lw=plot_lw)
ax_top_2.set_ylim(y_signal_lim)
ax_top_2.set_xlim(x_signal_lim)
add_box_edge_arrows(ax_top_2)

ax_top_3 = fig.add_subplot(gs[1, 2])
line3, = ax_top_3.plot(t, phase1, color='red', label='Phase (rad)', lw=plot_lw)
ax_top_3.set_ylim(y_phase_lim)
ax_top_3.set_xlim(x_signal_lim)
add_box_edge_arrows(ax_top_3)

# Right Column: Multi-Frequency
ax_bot_1 = fig.add_subplot(gs[2:4, 0])
ax_bot_1.plot(t, signal2, color='black', lw=plot_lw)
# ax_bot_1.set_title('Multi-Freq Signal')
ax_bot_1.set_ylim(y_signal_lim)
ax_bot_1.set_xlim(x_signal_lim)
add_box_edge_arrows(ax_bot_1)

ax_bot_2 = fig.add_subplot(gs[2, 2])
line2, = ax_bot_2.plot(t, homodyned2_filt, color='blue', label='Amplitude', lw=plot_lw)
ax_bot_2.set_ylim(y_signal_lim)
ax_bot_2.set_xlim(x_signal_lim)
add_box_edge_arrows(ax_bot_2)

# Twin axis for phase
ax_bot_3 = fig.add_subplot(gs[3, 2])
line3, = ax_bot_3.plot(t, phase2, color='red', label='Phase (rad)', lw=plot_lw)
ax_bot_3.set_ylim(y_phase_lim)
ax_bot_3.set_xlim(x_signal_lim)
add_box_edge_arrows(ax_bot_3)

file_path = os.path.join(fig_save_dir, f"3_osc.png")
if save_plot:
    plt.savefig(file_path, dpi=600, bbox_inches='tight',)
    print(f"Figure saved to: {file_path}")
plt.show()
plt.close()
