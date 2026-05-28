from __future__ import annotations

import numpy as np

from .common import ensure_power_of_two
from .fourier import fft_dit


def dct_ii(signal: np.ndarray) -> np.ndarray:
    """DCT-II from direct definition (O(N^2), unnormalized)."""
    x = np.asarray(signal, dtype=float)
    n = x.shape[0]

    k = np.arange(n)[:, None]
    m = np.arange(n)[None, :]

    kernel = np.cos(np.pi / n * (m + 0.5) * k)
    return kernel @ x


def fct_ii(signal: np.ndarray) -> np.ndarray:
    """Fast DCT-II using FFT relation (requires N = 2^n)."""
    x = np.asarray(signal, dtype=float)
    n = x.shape[0]
    ensure_power_of_two(n)

    mirrored = np.concatenate([x, x[::-1]])
    spectrum = fft_dit(mirrored.astype(np.complex128))

    k = np.arange(n)
    twiddle = np.exp(-1j * np.pi * k / (2.0 * n))
    dct_values = 0.5 * np.real(spectrum[:n] * twiddle)

    return dct_values.astype(np.float64)
