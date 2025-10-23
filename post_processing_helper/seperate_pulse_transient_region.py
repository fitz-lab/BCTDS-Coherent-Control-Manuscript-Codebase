import numpy as np

def truncate_display_offset(IQ_matrix, display_offset):
    I_matrix = IQ_matrix[0]
    Q_matrix = IQ_matrix[1]
    mag_matrix = np.abs(I_matrix + 1j *Q_matrix)
    phase_matrix = np.angle(I_matrix + 1j *Q_matrix)
    mag_log_matrix = np.log10(mag_matrix + 0.01)

    # print(np.shape(mag_log_matrix))

    # truncated whitespace before the pulse
    mag_matrix = mag_matrix[:, display_offset:]
    mag_log_matrix = mag_log_matrix[:, display_offset:]
    phase_matrix = phase_matrix[:, display_offset:]
    return mag_matrix, mag_log_matrix, phase_matrix

def extract_TP_region(IQ_matrix, display_offset, transient_processing_offset):
    I_matrix = IQ_matrix[0]
    Q_matrix = IQ_matrix[1]
    mag_matrix = np.abs(I_matrix + 1j *Q_matrix)
    phase_matrix = np.angle(I_matrix + 1j *Q_matrix)
    mag_log_matrix = np.log10(mag_matrix + 0.01)

    # print(np.shape(mag_log_matrix))

    # transient processing region, with the pulse and post pulse artifacts (oftern excessively) removed
    # mag fft and phase fft will only be computed for the TP region
    transient_processing_start_location = display_offset + transient_processing_offset

    mag_TP_matrix = mag_matrix[:, transient_processing_start_location:]
    mag_log_TP_matrix = mag_log_matrix[:, transient_processing_start_location:]
    phase_TP_matrix = phase_matrix[:, transient_processing_start_location:]
    return mag_TP_matrix, mag_log_TP_matrix, phase_TP_matrix

def extract_pulse_region(IQ_matrix, display_offset, pulse_start_offset, pulse_end_offset):
    I_matrix = IQ_matrix[0]
    Q_matrix = IQ_matrix[1]
    mag_matrix = np.abs(I_matrix + 1j *Q_matrix)
    phase_matrix = np.angle(I_matrix + 1j *Q_matrix)
    mag_log_matrix = np.log10(mag_matrix + 0.01)

    pulse_start_location = display_offset + pulse_start_offset
    pulse_end_location = display_offset + pulse_end_offset

    mag_PR_matrix = mag_matrix[:, pulse_start_location:pulse_end_location]
    mag_log_PR_matrix = mag_log_matrix[:, pulse_start_location:pulse_end_location]
    phase_PR_matrix = phase_matrix[:, pulse_start_location:pulse_end_location]
    return mag_PR_matrix, mag_log_PR_matrix, phase_PR_matrix