from __future__ import annotations

import numpy as np

from transforms.dct import dct_ii, fct_ii
from transforms.fourier import dft, fft_dif, fft_dit
from transforms.walsh_hadamard import fwht, walsh_hadamard_transform
from transforms.wavelets import dwt_multilevel, idwt_multilevel


def _max_abs_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference)
    est = np.asarray(estimate)
    return float(np.max(np.abs(ref - est)))


def _validate_fourier(signal: np.ndarray) -> dict[str, float]:
    reference = np.fft.fft(signal)
    return {
        "DFT vs numpy.fft": _max_abs_error(reference, dft(signal)),
        "FFT DIT vs numpy.fft": _max_abs_error(reference, fft_dit(signal)),
        "FFT DIF vs numpy.fft": _max_abs_error(reference, fft_dif(signal)),
    }


def _validate_dct(signal_real: np.ndarray) -> dict[str, float]:
    slow = dct_ii(signal_real)
    fast = fct_ii(signal_real)

    result = {
        "DCT-II vs FCT-II": _max_abs_error(slow, fast),
    }

    try:
        from scipy.fft import dct as scipy_dct  # type: ignore

        scipy_ref = (
            np.asarray(scipy_dct(signal_real, type=2, norm=None), dtype=np.float64)
            * 0.5
        )
        result["DCT-II vs scipy.fft.dct"] = _max_abs_error(slow, scipy_ref)
        result["FCT-II vs scipy.fft.dct"] = _max_abs_error(fast, scipy_ref)
    except Exception:
        # scipy is optional in this project
        pass

    return result


def _validate_walsh(signal: np.ndarray) -> dict[str, float]:
    classical = walsh_hadamard_transform(signal)
    fast = fwht(signal)
    return {
        "Walsh-Hadamard (matrix) vs FWHT": _max_abs_error(classical, fast),
    }


def _validate_wavelets(signal: np.ndarray) -> dict[str, float]:
    results: dict[str, float] = {}
    for name in ("db4", "db6", "db8"):
        approx, details = dwt_multilevel(signal, levels=2, wavelet_name=name)
        reconstructed = idwt_multilevel(approx, details, wavelet_name=name)
        results[f"DWT/IDWT reconstruction ({name})"] = _max_abs_error(
            signal, reconstructed
        )
    return results


def validate_all_transforms(
    signal: np.ndarray,
    tolerance: float = 1e-8,
) -> tuple[dict[str, float], dict[str, bool]]:
    x = np.asarray(signal, dtype=np.complex128)
    xr = np.real(x)

    errors: dict[str, float] = {}
    errors.update(_validate_fourier(x))
    errors.update(_validate_dct(xr))
    errors.update(_validate_walsh(x))
    errors.update(_validate_wavelets(x))

    status = {name: (err <= tolerance) for name, err in errors.items()}
    return errors, status
