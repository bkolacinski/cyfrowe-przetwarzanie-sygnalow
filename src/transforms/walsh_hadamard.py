from __future__ import annotations

import numpy as np

from .common import ensure_power_of_two


def hadamard_matrix(length: int) -> np.ndarray:
    """Sylvester Hadamard matrix (length must be 2^n)."""
    ensure_power_of_two(length, "length")

    matrix = np.array([[1.0]], dtype=np.float64)
    while matrix.shape[0] < length:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])

    return matrix


def walsh_hadamard_transform(signal: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Walsh-Hadamard transform using explicit matrix multiplication."""
    x = np.asarray(signal, dtype=np.complex128)
    n = x.shape[0]

    h = hadamard_matrix(n)
    result = h @ x

    if normalize:
        result = result / np.sqrt(n)

    return result


def fwht(signal: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Fast Walsh-Hadamard transform (butterfly)."""
    x = np.asarray(signal, dtype=np.complex128)
    n = x.shape[0]
    ensure_power_of_two(n)

    a = x.copy()
    step = 1

    while step < n:
        block = step * 2
        for start in range(0, n, block):
            for offset in range(step):
                i = start + offset
                j = i + step
                u = a[i]
                v = a[j]
                a[i] = u + v
                a[j] = u - v
        step = block

    if normalize:
        a = a / np.sqrt(n)

    return a
