from __future__ import annotations

import numpy as np

FPR_HZ = 16.0


def time_axis(length: int, fpr_hz: float = FPR_HZ) -> np.ndarray:
    if length < 2:
        raise ValueError("Signal length must be >= 2")
    if fpr_hz <= 0:
        raise ValueError("Sampling frequency must be > 0")
    return np.arange(length, dtype=np.float64) / float(fpr_hz)


def generate_signal_s1(
    length: int,
    amplitude: float = 1.0,
    fpr_hz: float = FPR_HZ,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """S1: uniform noise in range <-A, A>."""
    t = time_axis(length, fpr_hz)
    rng = np.random.default_rng(seed)
    signal = amplitude * rng.uniform(-1.0, 1.0, size=length)
    return t, signal.astype(np.float64)


def generate_signal_s2(
    length: int,
    amplitude: float = 1.0,
    fpr_hz: float = FPR_HZ,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """S2: gaussian noise with sigma = A / 3."""
    t = time_axis(length, fpr_hz)
    rng = np.random.default_rng(seed)
    signal = rng.normal(loc=0.0, scale=amplitude / 3.0, size=length)
    return t, signal.astype(np.float64)


def generate_signal_s3(
    length: int,
    amplitude: float = 1.0,
    period: float = 1.0,
    fpr_hz: float = FPR_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """S3: sinusoidal signal y[n] = A * sin((2*pi/T) * t[n])."""
    if period <= 0:
        raise ValueError("Period must be > 0")

    t = time_axis(length, fpr_hz)
    signal = amplitude * np.sin((2.0 * np.pi / period) * t)
    return t, signal.astype(np.float64)


def with_imaginary_component(
    t: np.ndarray,
    real_signal: np.ndarray,
    imag_mode: str = "zero",
    imag_amplitude: float = 1.0,
    imag_frequency_hz: float = 1.0,
    seed: int | None = 123,
) -> np.ndarray:
    real_part = np.asarray(real_signal, dtype=np.float64)

    if imag_mode == "zero":
        imag_part = np.zeros_like(real_part)
    elif imag_mode == "sine":
        imag_part = imag_amplitude * np.sin(2.0 * np.pi * imag_frequency_hz * t)
    elif imag_mode == "gaussian":
        rng = np.random.default_rng(seed)
        imag_part = rng.normal(0.0, imag_amplitude / 3.0, size=real_part.shape[0])
    else:
        raise ValueError("imag_mode must be one of: 'zero', 'sine', 'gaussian'.")

    return real_part.astype(np.complex128) + 1j * imag_part.astype(np.complex128)
