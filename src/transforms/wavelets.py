from __future__ import annotations

import math

import numpy as np

from .common import ensure_power_of_two

DAUBECHIES_COEFFICIENTS: dict[str, np.ndarray] = {
    "db2": np.array(
        [
            (1 + math.sqrt(3)) / (4 * math.sqrt(2)),
            (3 + math.sqrt(3)) / (4 * math.sqrt(2)),
            (3 - math.sqrt(3)) / (4 * math.sqrt(2)),
            (1 - math.sqrt(3)) / (4 * math.sqrt(2)),
        ],
        dtype=np.float64,
    ),
    "db3": np.array(
        [
            0.332670552950,
            0.806891509311,
            0.459877502118,
            -0.135011020010,
            -0.085441273882,
            0.035226291885,
        ],
        dtype=np.float64,
    ),
    "db4": np.array(
        [
            -0.010597401785069032,
            0.0328830116668852,
            0.030841381835560764,
            -0.18703481171909309,
            -0.027983769416859854,
            0.6308807679298587,
            0.7148465705529157,
            0.2303778133088964,
        ],
        dtype=np.float64,
    ),
    "db6": np.array(
        [
            -0.0010773010853084799,
            0.004777257511010651,
            0.0005538422011614961,
            -0.03158203931748603,
            0.027522865530016288,
            0.09750160558732248,
            -0.12976686756709563,
            -0.22626469396543983,
            0.3152503517091982,
            0.7511339080215775,
            0.4946238903984533,
            0.11154074335010947,
        ],
        dtype=np.float64,
    ),
    "db8": np.array(
        [
            -0.00011747678412476953,
            0.0006754494064505693,
            -0.00039174037337694705,
            -0.004870352993451574,
            0.008746094047405777,
            0.013981027917398282,
            -0.044088253930794755,
            -0.017369301001807547,
            0.12874742662047847,
            0.0004724845739124,
            -0.2840155429615469,
            -0.015829105256349306,
            0.5853546836542067,
            0.6756307362972898,
            0.3128715909144659,
            0.05441584224308161,
        ],
        dtype=np.float64,
    ),
}


def _analysis_filters(wavelet_name: str) -> tuple[np.ndarray, np.ndarray]:
    key = wavelet_name.lower()
    if key not in DAUBECHIES_COEFFICIENTS:
        raise ValueError(f"Unsupported wavelet '{wavelet_name}'.")

    dec_lo = DAUBECHIES_COEFFICIENTS[key]
    length = dec_lo.shape[0]
    signs = (-1.0) ** np.arange(length)
    dec_hi = signs * dec_lo[::-1]
    return dec_lo, dec_hi


def _synthesis_filters(wavelet_name: str) -> tuple[np.ndarray, np.ndarray]:
    # For the periodic convolution convention used in dwt_single_level
    # reconstruction uses the same H/G coefficients.
    dec_lo, dec_hi = _analysis_filters(wavelet_name)
    return dec_lo, dec_hi


def dwt_single_level(
    signal: np.ndarray, wavelet_name: str = "db4"
) -> tuple[np.ndarray, np.ndarray]:
    """Single-level periodic DWT using analysis filters H/G."""
    x = np.asarray(signal, dtype=np.complex128)
    n = x.shape[0]

    if n % 2 != 0:
        raise ValueError("Signal length for DWT must be even.")

    dec_lo, dec_hi = _analysis_filters(wavelet_name)
    filter_len = dec_lo.shape[0]

    half = n // 2
    approx = np.zeros(half, dtype=np.complex128)
    detail = np.zeros(half, dtype=np.complex128)

    for k in range(half):
        a_acc = 0.0j
        d_acc = 0.0j
        for i in range(filter_len):
            idx = (2 * k + i) % n
            sample = x[idx]
            a_acc += dec_lo[i] * sample
            d_acc += dec_hi[i] * sample
        approx[k] = a_acc
        detail[k] = d_acc

    return approx, detail


def idwt_single_level(
    approx: np.ndarray, detail: np.ndarray, wavelet_name: str = "db4"
) -> np.ndarray:
    """Single-level inverse periodic DWT."""
    a = np.asarray(approx, dtype=np.complex128)
    d = np.asarray(detail, dtype=np.complex128)

    if a.shape[0] != d.shape[0]:
        raise ValueError("Approximation and detail vectors must have same length.")

    n_half = a.shape[0]
    n = 2 * n_half

    rec_lo, rec_hi = _synthesis_filters(wavelet_name)
    filter_len = rec_lo.shape[0]

    reconstructed = np.zeros(n, dtype=np.complex128)

    for k in range(n_half):
        for i in range(filter_len):
            idx = (2 * k + i) % n
            reconstructed[idx] += rec_lo[i] * a[k] + rec_hi[i] * d[k]

    return reconstructed


def dwt_multilevel(
    signal: np.ndarray, levels: int, wavelet_name: str = "db4"
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Multilevel DWT decomposition. Returns final approx + details list."""
    if levels < 1:
        raise ValueError("levels must be >= 1")

    x = np.asarray(signal, dtype=np.complex128)
    ensure_power_of_two(x.shape[0], "signal length")

    current = x
    details: list[np.ndarray] = []

    for _ in range(levels):
        if current.shape[0] % 2 != 0:
            raise ValueError("Signal length is not divisible by 2 at current level.")
        current, detail = dwt_single_level(current, wavelet_name)
        details.append(detail)

    return current, details


def idwt_multilevel(
    approx: np.ndarray, details: list[np.ndarray], wavelet_name: str = "db4"
) -> np.ndarray:
    """Inverse multilevel DWT reconstruction."""
    current = np.asarray(approx, dtype=np.complex128)

    for detail in reversed(details):
        current = idwt_single_level(current, detail, wavelet_name)

    return current
