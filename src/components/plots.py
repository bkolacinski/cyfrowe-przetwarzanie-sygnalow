from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def plot_complex_signal(
    t: np.ndarray,
    signal: np.ndarray,
    mode: str = "W1",
    title: str = "Sygnał zespolony",
):
    x = np.asarray(signal, dtype=np.complex128)
    time = np.asarray(t, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    if mode == "W2":
        axes[0].plot(time, np.abs(x), color="tab:blue")
        axes[0].set_ylabel("Moduł |x[n]|")
        axes[0].set_title(f"{title} - moduł")

        axes[1].plot(time, np.angle(x), color="tab:orange")
        axes[1].set_ylabel("Faza arg(x[n]) [rad]")
        axes[1].set_title(f"{title} - faza")
    else:
        axes[0].plot(time, np.real(x), color="tab:blue")
        axes[0].set_ylabel("Re{x[n]}")
        axes[0].set_title(f"{title} - część rzeczywista")

        axes[1].plot(time, np.imag(x), color="tab:orange")
        axes[1].set_ylabel("Im{x[n]}")
        axes[1].set_title(f"{title} - część urojona")

    axes[1].set_xlabel("Czas [s]")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_transform_components(indices: np.ndarray, values: np.ndarray, title: str):
    idx = np.asarray(indices)
    y = np.asarray(values, dtype=np.complex128)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(idx, np.real(y), label="Re", color="tab:blue")
    axes[0].plot(idx, np.imag(y), label="Im", color="tab:orange", alpha=0.8)
    axes[0].set_ylabel("Amplituda")
    axes[0].set_title(f"{title} - składowe")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(idx, np.abs(y), color="tab:green", label="|X[k]|")
    axes[1].set_ylabel("Moduł")
    axes[1].set_xlabel("Indeks")
    axes[1].set_title(f"{title} - moduł")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_frequency_spectrum(
    transform_values: np.ndarray,
    fs_hz: float,
    title: str = "Widmo częstotliwościowe",
):
    values = np.asarray(transform_values, dtype=np.complex128)
    n = values.shape[0]

    freqs = np.fft.fftfreq(n, d=1.0 / fs_hz)
    positive = freqs >= 0

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].stem(
        freqs[positive],
        np.abs(values[positive]),
        linefmt="tab:blue",
        markerfmt=" ",
        basefmt=" ",
    )
    axes[0].set_ylabel("|X(f)|")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(
        freqs[positive],
        np.angle(values[positive]),
        linefmt="tab:orange",
        markerfmt=" ",
        basefmt=" ",
    )
    axes[1].set_ylabel("arg(X(f)) [rad]")
    axes[1].set_xlabel("Częstotliwość [Hz]")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_benchmarks(results: dict[str, list[float]]):
    n_values = np.asarray(results.get("N", []), dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(n_values, results.get("DFT", []), "o-", label="DFT")
    axes[0].plot(n_values, results.get("FFT_DIT", []), "o-", label="FFT DIT")
    axes[0].plot(n_values, results.get("FFT_DIF", []), "o-", label="FFT DIF")
    if "NUMPY_FFT" in results:
        axes[0].plot(
            n_values,
            results.get("NUMPY_FFT", []),
            "o--",
            label="NumPy FFT",
            alpha=0.7,
        )
    axes[0].set_ylabel("Czas [ms]")
    axes[0].set_title("Benchmark: DFT / FFT")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(n_values, results.get("DCT_II", []), "o-", label="DCT-II")
    axes[1].plot(n_values, results.get("FCT_II", []), "o-", label="FCT-II")
    axes[1].set_ylabel("Czas [ms]")
    axes[1].set_title("Benchmark: DCT-II / FCT-II")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(n_values, results.get("WHT", []), "o-", label="WHT")
    axes[2].plot(n_values, results.get("FWHT", []), "o-", label="FWHT")
    axes[2].set_ylabel("Czas [ms]")
    axes[2].set_xlabel("Liczba próbek N")
    axes[2].set_title("Benchmark: Walsh-Hadamard")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    return fig


def plot_wavelet_details(approx: np.ndarray, details: list[np.ndarray], title: str):
    levels = len(details)
    fig, axes = plt.subplots(levels + 1, 1, figsize=(12, 2.8 * (levels + 1)))

    if levels == 0:
        axes = [axes]

    axes[0].plot(np.real(approx), color="tab:blue")
    axes[0].set_title(f"{title} - aproksymacja końcowa")
    axes[0].grid(True, alpha=0.3)

    for idx, detail in enumerate(details, start=1):
        axes[idx].plot(np.real(detail), color="tab:orange")
        axes[idx].set_title(f"Detal poziom {idx}")
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Indeks")
    fig.tight_layout()
    return fig
