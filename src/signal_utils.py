import numpy as np


EPSILON = 1e-12


def _match_lengths(y1, y2):
    min_len = min(len(y1), len(y2))
    return np.asarray(y1)[:min_len], np.asarray(y2)[:min_len]


def signal_add(y1, y2):
    a, b = _match_lengths(y1, y2)
    return a + b


def signal_subtract(y1, y2):
    a, b = _match_lengths(y1, y2)
    return a - b


def signal_multiply(y1, y2):
    a, b = _match_lengths(y1, y2)
    return a * b


def signal_divide(y1, y2, epsilon=1e-10):
    a, b = _match_lengths(y1, y2)
    b_safe = np.where(b == 0, epsilon, b)
    return a / b_safe


def calculate_signal_metrics(signal):
    y = np.asarray(signal, dtype=float)
    power = np.mean(np.square(y))
    return {
        "mean": float(np.mean(y)),
        "mean_abs": float(np.mean(np.abs(y))),
        "rms": float(np.sqrt(power)),
        "variance": float(np.var(y)),
        "power": float(power),
    }


def get_periodic_analysis_data(x_vals, y_vals, signals, signal_registry):
    info = {
        "used_full_periods": False,
        "full_periods": None,
        "period": 0,
        "reason": "",
    }

    if x_vals is None or y_vals is None or len(y_vals) == 0:
        info["reason"] = "Brak danych wejściowych."
        return x_vals, y_vals, info

    active_signals = [signal for signal in signals if "data" in signal]
    if len(active_signals) != 1:
        info["reason"] = "Dla sygnalu zlozonego statystyki i histogram liczone sa z calego zakresu."
        return x_vals, y_vals, info

    signal_data = active_signals[0]
    signal_config = signal_registry.get(signal_data.get("type"), {})

    if not (signal_config.get("is_periodic", False) and signal_config.get("is_continuous", False)):
        info["reason"] = "Sygnał nie jest jednocześnie okresowy i ciągły."
        return x_vals, y_vals, info

    period = signal_data.get("params", {}).get("T")
    if period is None or period <= 0:
        info["reason"] = "Brak poprawnego okresu T."
        return x_vals, y_vals, info

    duration = float(x_vals[-1] - x_vals[0])
    full_periods = int(np.floor(duration / period + 1e-12))

    if full_periods <= 0:
        info["period"] = period
        info["full_periods"] = 0
        info["reason"] = "W wybranym zakresie czasu nie ma pełnego okresu."
        return x_vals[:0], y_vals[:0], info

    end_time = float(x_vals[0] + full_periods * period)
    mask = x_vals <= (end_time + 1e-12)

    info["used_full_periods"] = True
    info["full_periods"] = full_periods
    info["period"] = period
    return x_vals[mask], y_vals[mask], info


def generate_sine_wave(frequency_hz, amplitude, phase_rad, t_start, duration_s, fs_hz):
    sample_count = max(2, int(duration_s * fs_hz))
    t = np.linspace(t_start, t_start + duration_s, sample_count, endpoint=False)
    y = amplitude * np.sin(2 * np.pi * frequency_hz * (t - t_start) + phase_rad)
    return t, y


def uniform_sample(t_reference, y_reference, fs_hz, t_start=None, t_end=None):
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")

    t_ref = np.asarray(t_reference, dtype=float)
    y_ref = np.asarray(y_reference, dtype=float)

    if t_start is None:
        t_start = float(t_ref[0])
    if t_end is None:
        t_end = float(t_ref[-1])

    step = 1.0 / fs_hz
    t_samples = np.arange(t_start, t_end + EPSILON, step)
    y_samples = np.interp(t_samples, t_ref, y_ref)
    return t_samples, y_samples


def quantize_uniform(y, bits, mode="round", min_value=None, max_value=None):
    if bits < 1:
        raise ValueError("bits must be >= 1")

    signal = np.asarray(y, dtype=float)
    levels = 2**bits

    if min_value is None:
        min_value = float(np.min(signal))
    if max_value is None:
        max_value = float(np.max(signal))

    if max_value <= min_value:
        max_value = min_value + EPSILON

    step = (max_value - min_value) / (levels - 1)
    normalized = (signal - min_value) / step

    if mode == "truncate":
        q_idx = np.floor(normalized)
    elif mode == "round":
        q_idx = np.rint(normalized)
    else:
        raise ValueError("mode must be 'truncate' or 'round'")

    q_idx = np.clip(q_idx, 0, levels - 1)
    quantized = min_value + q_idx * step

    return quantized, {
        "bits": bits,
        "levels": levels,
        "min": min_value,
        "max": max_value,
        "step": step,
        "mode": mode,
    }


def reconstruct_zoh(t_samples, y_samples, t_target):
    ts = np.asarray(t_samples, dtype=float)
    ys = np.asarray(y_samples, dtype=float)
    tt = np.asarray(t_target, dtype=float)

    idx = np.searchsorted(ts, tt, side="right") - 1
    idx = np.clip(idx, 0, len(ts) - 1)
    return ys[idx]


def reconstruct_foh(t_samples, y_samples, t_target):
    ts = np.asarray(t_samples, dtype=float)
    ys = np.asarray(y_samples, dtype=float)
    tt = np.asarray(t_target, dtype=float)
    return np.interp(tt, ts, ys, left=ys[0], right=ys[-1])


def reconstruct_sinc(t_samples, y_samples, t_target, fs_hz=None, neighbors=16):
    ts = np.asarray(t_samples, dtype=float)
    ys = np.asarray(y_samples, dtype=float)
    tt = np.asarray(t_target, dtype=float)

    if fs_hz is None:
        dt = np.diff(ts)
        fs_hz = 1.0 / np.median(dt)

    neighbors = max(1, int(neighbors))
    reconstructed = np.zeros_like(tt)

    for i, t in enumerate(tt):
        center = np.searchsorted(ts, t)
        start = max(0, center - neighbors)
        end = min(len(ts), center + neighbors + 1)

        t_local = ts[start:end]
        y_local = ys[start:end]
        reconstructed[i] = np.sum(y_local * np.sinc((t - t_local) * fs_hz))

    return reconstructed


def mse(reference, estimate):
    r, e = _match_lengths(reference, estimate)
    return float(np.mean((r - e) ** 2))


def snr(reference, estimate):
    r, e = _match_lengths(reference, estimate)
    noise = r - e
    signal_power = np.mean(r**2)
    noise_power = np.mean(noise**2)

    if noise_power <= EPSILON:
        return float("inf")
    if signal_power <= EPSILON:
        return float("-inf")

    return float(10 * np.log10(signal_power / noise_power))


def psnr(reference, estimate, peak=None):
    r, e = _match_lengths(reference, estimate)
    err = mse(r, e)
    if err <= EPSILON:
        return float("inf")

    if peak is None:
        peak = float(np.max(np.abs(r)))

    if peak <= EPSILON:
        return float("-inf")

    return float(10 * np.log10((peak**2) / err))


def max_difference(reference, estimate):
    r, e = _match_lengths(reference, estimate)
    return float(np.max(np.abs(r - e)))


def quality_metrics(reference, estimate):
    return {
        "mse": mse(reference, estimate),
        "snr": snr(reference, estimate),
        "psnr": psnr(reference, estimate),
        "md": max_difference(reference, estimate),
    }


def theoretical_quantization_snr(bits):
    return float(6.02 * bits + 1.76)


def enob_from_snr(snr_db):
    return float((snr_db - 1.76) / 6.02)


def run_sampling_experiment(reference_t, reference_y, fs_values, reconstruction_method, sinc_neighbors=16):
    results = []

    for fs in fs_values:
        t_s, y_s = uniform_sample(reference_t, reference_y, fs_hz=fs)

        if reconstruction_method == "zoh":
            y_r = reconstruct_zoh(t_s, y_s, reference_t)
        elif reconstruction_method == "foh":
            y_r = reconstruct_foh(t_s, y_s, reference_t)
        elif reconstruction_method == "sinc":
            y_r = reconstruct_sinc(
                t_s,
                y_s,
                reference_t,
                fs_hz=fs,
                neighbors=sinc_neighbors,
            )
        else:
            raise ValueError("reconstruction_method must be: zoh, foh, sinc")

        metrics = quality_metrics(reference_y, y_r)
        metrics["fs"] = float(fs)
        results.append(metrics)

    return results


def run_quantization_experiment(y_samples, bits_values, mode):
    results = []

    y_arr = np.asarray(y_samples, dtype=float)
    min_value = float(np.min(y_arr))
    max_value = float(np.max(y_arr))

    for bits in bits_values:
        y_q, _ = quantize_uniform(
            y_arr,
            bits=bits,
            mode=mode,
            min_value=min_value,
            max_value=max_value,
        )
        snr_measured = snr(y_arr, y_q)
        snr_theory = theoretical_quantization_snr(bits)

        results.append(
            {
                "bits": int(bits),
                "snr_measured": snr_measured,
                "snr_theory": snr_theory,
                "enob": enob_from_snr(snr_measured),
                "mse": mse(y_arr, y_q),
            }
        )

    return results


def alias_frequency(f0_hz, fs_hz, k):
    return float(f0_hz + k * fs_hz)

