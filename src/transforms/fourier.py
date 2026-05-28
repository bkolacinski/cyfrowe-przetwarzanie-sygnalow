from __future__ import annotations

import math

import numpy as np

from .common import bit_reversal_indices, ensure_power_of_two


def dft(signal: np.ndarray) -> np.ndarray:
    """Discrete Fourier Transform computed directly from definition (O(N^2))."""
    x = np.asarray(signal, dtype=np.complex128)
    n = x.shape[0]

    result = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        acc = 0.0j
        for m in range(n):
            angle = -2.0 * math.pi * k * m / n
            acc += x[m] * complex(math.cos(angle), math.sin(angle))
        result[k] = acc

    return result


def fft_dit(signal: np.ndarray) -> np.ndarray:
    """Radix-2 FFT (Decimation in Time), iterative, bit-reversal + butterfly."""
    x = np.asarray(signal, dtype=np.complex128)
    n = x.shape[0]
    ensure_power_of_two(n)

    indices = bit_reversal_indices(n)
    a = x[np.array(indices, dtype=int)].copy()

    stage_size = 2
    while stage_size <= n:
        half = stage_size // 2
        twiddle_base = np.exp(-2j * np.pi / stage_size)

        for block_start in range(0, n, stage_size):
            twiddle = 1.0 + 0.0j
            for j in range(half):
                top = block_start + j
                bottom = top + half

                t = twiddle * a[bottom]
                u = a[top]

                a[top] = u + t
                a[bottom] = u - t
                twiddle *= twiddle_base

        stage_size *= 2

    return a


def fft_dif(signal: np.ndarray) -> np.ndarray:
    """Radix-2 FFT (Decimation in Frequency), iterative, butterfly + bit-reversal."""
    x = np.asarray(signal, dtype=np.complex128)
    n = x.shape[0]
    ensure_power_of_two(n)

    a = x.copy()
    stage_size = n

    while stage_size >= 2:
        half = stage_size // 2
        twiddle_base = np.exp(-2j * np.pi / stage_size)

        for block_start in range(0, n, stage_size):
            twiddle = 1.0 + 0.0j
            for j in range(half):
                top = block_start + j
                bottom = top + half

                u = a[top]
                v = a[bottom]

                a[top] = u + v
                a[bottom] = (u - v) * twiddle
                twiddle *= twiddle_base

        stage_size //= 2

    indices = bit_reversal_indices(n)
    return a[np.array(indices, dtype=int)]
