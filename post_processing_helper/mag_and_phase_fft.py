import numpy as np
import os

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

def matrix_fft(data_matrix):
    data_matrix = np.array(data_matrix)
    data_fft_matrix = []

    for idx in range(len(data_matrix)):
        data_array = data_matrix[idx]
        fft_freq_MHz, data_array_fft = fft_custom(data_array, 1/552.96)
        data_fft_matrix.append(data_array_fft)

    fft_freq_MHz = np.array(fft_freq_MHz)
    data_fft_matrix = np.array(data_fft_matrix)

    # print(fft_freq_MHz)

    return fft_freq_MHz, data_fft_matrix


def matrix_fft_interpolated(data_matrix, spacing_MHz=1.0):
    if spacing_MHz <= 0:
        raise ValueError("spacing_MHz must be positive.")

    data_matrix = np.array(data_matrix)
    data_fft_matrix = []

    for idx in range(len(data_matrix)):
        data_array = data_matrix[idx]
        fft_freq_MHz, data_array_fft = fft_custom(data_array, 1/552.96)
        data_fft_matrix.append(data_array_fft)

    # Convert to arrays
    fft_freq_MHz = np.array(fft_freq_MHz)
    data_fft_matrix = np.array(data_fft_matrix)

    # Build uniformly spaced frequency axis within the original range
    eps = 1e-9 * spacing_MHz
    new_freq_MHz = np.arange(fft_freq_MHz.min(), fft_freq_MHz.max() + eps, spacing_MHz)

    # Interpolate each row to the new axis (linear interpolation)
    data_fft_matrix = np.array([
        np.interp(new_freq_MHz, fft_freq_MHz, row) for row in data_fft_matrix
    ])

    # print(new_freq_MHz)

    return new_freq_MHz, data_fft_matrix

def matrix_fft_interpolated_biaxial(data_matrix, freq_axis_original, spacing_MHz=1.0):
    """
    Interpolate a (N_freq, N_fft) matrix onto uniform grids along BOTH axes (real-only).

    Parameters
    ----------
    data_matrix : array-like, shape (N_freq, N_fft)
        Rows correspond to original frequencies in `freq_axis_original`.
        Columns correspond to FFT frequencies returned by `fft_custom`.
    freq_axis_original : array-like, shape (N_freq,)
        Original frequency axis for rows (e.g., np.arange(f0, f1+step, step)).
    spacing_MHz : float
        Desired uniform spacing (MHz) for BOTH axes.

    Returns
    -------
    interpolated_freq_axis : ndarray, shape (N_interp_freq,)
    interpolated_fft_freq_axis : ndarray, shape (N_interp_fft,)
    data_fft_matrix : ndarray, shape (N_interp_freq, N_interp_fft), dtype float
    """
    if spacing_MHz <= 0:
        raise ValueError("spacing_MHz must be positive.")

    data_matrix = np.asarray(data_matrix)
    freq_axis_original = np.asarray(freq_axis_original)

    if data_matrix.ndim != 2:
        raise ValueError("data_matrix must be 2D (N_freq, N_fft).")
    if freq_axis_original.ndim != 1:
        raise ValueError("freq_axis_original must be 1D.")
    if data_matrix.shape[0] != freq_axis_original.size:
        raise ValueError(
            f"len(freq_axis_original) ({freq_axis_original.size}) must equal "
            f"data_matrix.shape[0] ({data_matrix.shape[0]})."
        )
    if not np.all(np.diff(freq_axis_original) > 0):
        raise ValueError("freq_axis_original must be strictly increasing.")

    # 1) Per-row FFT via your fft_custom, then interpolate each row onto uniform FFT grid
    data_fft_rows = []
    fft_freq_axis = None
    for i in range(data_matrix.shape[0]):
        fft_freq_MHz, data_row_fft = fft_custom(data_matrix[i], 1/552.96)
        if fft_freq_axis is None:
            fft_freq_axis = np.asarray(fft_freq_MHz)
        data_fft_rows.append(np.asarray(data_row_fft, dtype=float))

    data_fft_rows = np.vstack(data_fft_rows)  # (N_freq, N_fft_original)

    eps_fft = 1e-9 * spacing_MHz
    interpolated_fft_freq_axis = np.arange(
        fft_freq_axis.min(), fft_freq_axis.max() + eps_fft, spacing_MHz
    )

    # Interpolate along FFT axis for each row (real-only)
    n_rows = data_fft_rows.shape[0]
    n_fft_new = interpolated_fft_freq_axis.size
    data_fft_interp_fft = np.empty((n_rows, n_fft_new), dtype=float)
    for i in range(n_rows):
        data_fft_interp_fft[i] = np.interp(
            interpolated_fft_freq_axis, fft_freq_axis, data_fft_rows[i]
        )

    # 2) Interpolate along row axis to uniform freq grid
    eps_row = 1e-9 * spacing_MHz
    interpolated_freq_axis = np.arange(
        freq_axis_original.min(), freq_axis_original.max() + eps_row, spacing_MHz
    )

    n_rows_new = interpolated_freq_axis.size
    data_fft_matrix = np.empty((n_rows_new, n_fft_new), dtype=float)
    for j in range(n_fft_new):
        col = data_fft_interp_fft[:, j]
        data_fft_matrix[:, j] = np.interp(
            interpolated_freq_axis, freq_axis_original, col
        )
    print(np.shape(interpolated_freq_axis))
    print(np.shape(interpolated_fft_freq_axis))
    print(np.shape(data_fft_matrix))
    return interpolated_freq_axis / 1e3, interpolated_fft_freq_axis, data_fft_matrix


def average_V_diagonals(A, k_max=None):
    """
    A: 2D array-like with shape (n, m)
    k_max: optional int, maximum diagonal depth (number of columns to include - 1).
           If None, goes to the maximum possible (m-1).
    Returns:
        avgs: 1D array of length n, the average V-diagonal value for each start (i,0)
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    n, m = A.shape

    if k_max is None or k_max > m - 1:
        k_max = m - 1

    avgs = np.empty(n, dtype=float)

    for i in range(n):
        total = A[i, 0]
        cnt = 1

        for k in range(1, k_max + 1):
            r1 = i - k    # upper diagonal
            r2 = i + k    # lower diagonal
            c  = k

            if 0 <= r1 < n:
                total += A[r1, c]
                cnt += 1
            if 0 <= r2 < n:
                total += A[r2, c]
                cnt += 1

        avgs[i] = total / cnt

    return avgs