import numpy as np

EPSILON = 1e-12


def discrete_convolution(h, x):
    h = np.asarray(h, dtype=float)
    x = np.asarray(x, dtype=float)

    M = len(h)
    N = len(x)
    y_len = M + N - 1
    y = np.zeros(y_len, dtype=float)

    for n in range(y_len):
        acc = 0.0
        for k in range(M):
            x_idx = n - k
            if 0 <= x_idx < N:
                acc += h[k] * x[x_idx]
        y[n] = acc

    return y


def cross_correlation_direct(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    Ny = len(y)
    Nx = len(x)
    lags = np.arange(-(Nx - 1), Ny, dtype=int)
    result = np.zeros(len(lags), dtype=float)

    for i, lag in enumerate(lags):
        acc = 0.0
        for k in range(Ny):
            x_idx = k - lag
            if 0 <= x_idx < Nx:
                acc += y[k] * x[x_idx]
        result[i] = acc

    return result, lags


def cross_correlation_via_convolution(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    x_reversed = x[::-1]
    conv_result = discrete_convolution(y, x_reversed)
    lags = np.arange(-(len(x) - 1), len(y), dtype=int)

    return conv_result, lags


def _window(M: int, window_type: str) -> np.ndarray:
    n = np.arange(M, dtype=float)

    if window_type == "prostokątne":
        return np.ones(M)
    if window_type == "hamminga":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / max(M - 1, 1))
    if window_type == "hanninga":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / max(M - 1, 1))
    if window_type == "blackmana":
        return (
            0.42
            - 0.5 * np.cos(2 * np.pi * n / max(M - 1, 1))
            + 0.08 * np.cos(4 * np.pi * n / max(M - 1, 1))
        )

    raise ValueError(f"Nieznane okno: {window_type}")


def design_lowpass_fir(
    M: int, f0_hz: float, fs_hz: float, window_type: str = "hamminga"
):
    if M < 2:
        raise ValueError("Rząd filtru M musi być >= 2")
    if fs_hz <= 0:
        raise ValueError("Częstotliwość próbkowania fs musi być > 0")

    f0_hz = min(max(float(f0_hz), 0.0), fs_hz / 2.0)
    n = np.arange(M, dtype=float)
    alpha = (M - 1) / 2.0

    h_ideal = np.zeros(M, dtype=float)
    norm_cutoff = f0_hz / fs_hz

    for i, ni in enumerate(n):
        k = ni - alpha
        if abs(k) < EPSILON:
            h_ideal[i] = 2.0 * norm_cutoff
        else:
            h_ideal[i] = np.sin(2.0 * np.pi * norm_cutoff * k) / (np.pi * k)

    w = _window(M, window_type)
    h = h_ideal * w

    return h, h_ideal, w


def transform_to_bandpass(h_lp):
    h_lp = np.asarray(h_lp, dtype=float)
    n = np.arange(len(h_lp))
    s = 2.0 * np.sin(np.pi * n / 2.0)
    return h_lp * s, s


def transform_to_highpass(h_lp):
    h_lp = np.asarray(h_lp, dtype=float)
    n = np.arange(len(h_lp))
    s = (-1.0) ** n
    return h_lp * s, s


def filter_signal_with_fir(x, h):
    return discrete_convolution(h, x)


def magnitude_response_db(h, fs_hz: float, n_fft: int = 4096):
    h = np.asarray(h, dtype=float)
    H = np.fft.rfft(h, n=n_fft)
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)
    mag = np.abs(H)
    mag_db = 20.0 * np.log10(np.maximum(mag, EPSILON))
    return freq, mag, mag_db


def estimate_delay_and_distance(
    correlation, lags, fs_hz: float, wave_speed: float
):
    correlation = np.asarray(correlation, dtype=float)
    lags = np.asarray(lags, dtype=int)

    idx = int(np.argmax(correlation))
    best_lag = int(lags[idx])
    delay_s = best_lag / fs_hz
    distance_m = (wave_speed * delay_s) / 2.0

    return {
        "peak_index": idx,
        "peak_lag": best_lag,
        "delay_s": float(delay_s),
        "distance_m": float(distance_m),
        "peak_value": float(correlation[idx]),
    }
