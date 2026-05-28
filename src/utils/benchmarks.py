from __future__ import annotations

import time
from typing import Callable

import numpy as np

from transforms.dct import dct_ii, fct_ii
from transforms.fourier import dft, fft_dif, fft_dit
from transforms.walsh_hadamard import fwht, walsh_hadamard_transform


def _benchmark_call(
    function: Callable[[np.ndarray], np.ndarray],
    signal: np.ndarray,
    repeats: int,
) -> float:
    times: list[float] = []

    for _ in range(repeats):
        start = time.perf_counter()
        function(signal)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    return float(np.mean(times))


def benchmark_transforms(
    signal: np.ndarray,
    n_values: list[int],
    include_numpy: bool = True,
    max_direct_n: int = 256,
) -> dict[str, list[float]]:
    """Benchmark classic and fast DSP transforms for selected N values.

    Returned times are in milliseconds.
    """
    x = np.asarray(signal, dtype=np.complex128)

    results: dict[str, list[float]] = {
        "N": [],
        "DFT": [],
        "FFT_DIT": [],
        "FFT_DIF": [],
        "NUMPY_FFT": [],
        "DCT_II": [],
        "FCT_II": [],
        "WHT": [],
        "FWHT": [],
    }

    for n in n_values:
        n = int(n)
        if n < 2 or n > x.shape[0]:
            continue

        sample = x[:n]
        sample_real = np.real(sample)

        repeats_fast = 5 if n <= 256 else 3
        repeats_slow = 3 if n <= 128 else 1

        results["N"].append(n)

        if n <= max_direct_n:
            results["DFT"].append(_benchmark_call(dft, sample, repeats_slow))
            results["DCT_II"].append(_benchmark_call(dct_ii, sample_real, repeats_slow))
            results["WHT"].append(
                _benchmark_call(walsh_hadamard_transform, sample, repeats_slow)
            )
        else:
            results["DFT"].append(float("nan"))
            results["DCT_II"].append(float("nan"))
            results["WHT"].append(float("nan"))

        results["FFT_DIT"].append(_benchmark_call(fft_dit, sample, repeats_fast))
        results["FFT_DIF"].append(_benchmark_call(fft_dif, sample, repeats_fast))
        results["FCT_II"].append(_benchmark_call(fct_ii, sample_real, repeats_fast))
        results["FWHT"].append(_benchmark_call(fwht, sample, repeats_fast))

        if include_numpy:
            results["NUMPY_FFT"].append(
                _benchmark_call(lambda s: np.fft.fft(s), sample, repeats_fast)
            )
        else:
            results["NUMPY_FFT"].append(float("nan"))

    return results
