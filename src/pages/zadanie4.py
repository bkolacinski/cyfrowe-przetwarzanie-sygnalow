from __future__ import annotations

import numpy as np
import streamlit as st

from components.plots import (
    plot_benchmarks,
    plot_complex_signal,
    plot_frequency_spectrum,
    plot_transform_components,
    plot_wavelet_details,
)
from transforms.dct import dct_ii, fct_ii
from transforms.fourier import dft, fft_dif, fft_dit
from transforms.walsh_hadamard import fwht, walsh_hadamard_transform
from transforms.wavelets import dwt_multilevel, idwt_multilevel
from utils.benchmarks import benchmark_transforms
from utils.complex_io import (
    export_complex_to_csv,
    load_complex_signal,
    save_complex_signal,
)
from utils.signal_generation import (
    FPR_HZ,
    generate_signal_s1,
    generate_signal_s2,
    generate_signal_s3,
    with_imaginary_component,
)
from utils.validation import validate_all_transforms

TRANSFORM_OPTIONS = [
    "DFT (definicja)",
    "FFT DIT",
    "FFT DIF",
    "DCT-II",
    "FCT-II",
    "Walsh-Hadamard (macierz)",
    "Walsh-Hadamard (szybka)",
    "Wavelet DB (DWT)",
]


def _next_power_length(signal: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.complex128)
    n = int(length)

    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]

    padded = np.zeros(n, dtype=np.complex128)
    padded[: x.shape[0]] = x
    return padded


def _generate_test_signal(
    kind: str,
    length: int,
    amplitude: float,
    period: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "S1":
        return generate_signal_s1(length=length, amplitude=amplitude, seed=seed)
    if kind == "S2":
        return generate_signal_s2(length=length, amplitude=amplitude, seed=seed)
    return generate_signal_s3(length=length, amplitude=amplitude, period=period)


def _transform_to_csv(indices: np.ndarray, values: np.ndarray) -> str:
    idx = np.asarray(indices)
    y = np.asarray(values, dtype=np.complex128)

    lines = ["index,real,imag,magnitude,phase"]
    for i, v in zip(idx, y):
        lines.append(
            f"{int(i)},{v.real:.10f},{v.imag:.10f},{np.abs(v):.10f},{np.angle(v):.10f}"
        )

    return "\n".join(lines)


def _run_transform(
    transform_name: str,
    signal: np.ndarray,
    wavelet_name: str,
    wavelet_levels: int,
    inverse: bool = False,
):
    if transform_name == "DFT (definicja)":
        return "spectrum", dft(signal, inverse=inverse)
    if transform_name == "FFT DIT":
        return "spectrum", fft_dit(signal, inverse=inverse)
    if transform_name == "FFT DIF":
        return "spectrum", fft_dif(signal, inverse=inverse)
    if transform_name == "DCT-II":
        return "real", dct_ii(np.real(signal))
    if transform_name == "FCT-II":
        return "real", fct_ii(np.real(signal))
    if transform_name == "Walsh-Hadamard (macierz)":
        return "spectrum", walsh_hadamard_transform(signal)
    if transform_name == "Walsh-Hadamard (szybka)":
        return "spectrum", fwht(signal)

    approx, details = dwt_multilevel(
        signal, levels=wavelet_levels, wavelet_name=wavelet_name
    )
    reconstructed = idwt_multilevel(approx, details, wavelet_name=wavelet_name)
    reconstruction_error = float(np.max(np.abs(signal - reconstructed)))
    return "wavelet", (approx, details, reconstructed, reconstruction_error)


def _reference_error(
    transform_name: str, signal: np.ndarray, result: np.ndarray, inverse: bool = False
) -> str:
    try:
        if transform_name in {"DFT (definicja)", "FFT DIT", "FFT DIF"}:
            if inverse:
                # No direct numpy equivalent for "inverse DFT" without normalising differently
                # But we can use ifft
                reference = np.fft.ifft(signal)
            else:
                reference = np.fft.fft(signal)
            return f"max|X-X_ref| = {np.max(np.abs(result - reference)):.3e}"

        if transform_name in {"DCT-II", "FCT-II"}:
            try:
                from scipy.fft import dct as scipy_dct  # type: ignore

                reference = (
                    np.asarray(
                        scipy_dct(np.real(signal), type=2, norm=None), dtype=np.float64
                    )
                    * 0.5
                )
                return (
                    f"max|X-X_ref| (SciPy) = {np.max(np.abs(result - reference)):.3e}"
                )
            except Exception:
                other = (
                    fct_ii(np.real(signal))
                    if transform_name == "DCT-II"
                    else dct_ii(np.real(signal))
                )
                return f"max|X-X_pair| = {np.max(np.abs(result - other)):.3e}"

        if transform_name == "Walsh-Hadamard (macierz)":
            other = fwht(signal)
            return f"max|X-X_fast| = {np.max(np.abs(result - other)):.3e}"

        if transform_name == "Walsh-Hadamard (szybka)":
            other = walsh_hadamard_transform(signal)
            return f"max|X-X_classic| = {np.max(np.abs(result - other)):.3e}"

        return "Brak referencji bibliotekowej dla tej transformacji."
    except Exception as exc:
        return f"Nie udało się policzyć porównania: {exc}"


def _build_signal_for_benchmark(
    signal_kind: str,
    amplitude: float,
    period: float,
    seed: int,
    max_length: int,
    imag_mode: str,
    imag_amplitude: float,
    imag_frequency_hz: float,
) -> np.ndarray:
    t_full, y_full = _generate_test_signal(
        kind=signal_kind,
        length=max_length,
        amplitude=amplitude,
        period=period,
        seed=seed,
    )
    return with_imaginary_component(
        t=t_full,
        real_signal=y_full,
        imag_mode=imag_mode,
        imag_amplitude=imag_amplitude,
        imag_frequency_hz=imag_frequency_hz,
        seed=seed + 1,
    )


st.title("Zadanie 4 - Transformacje sygnałowe i benchmarki")
st.caption(
    "Implementacje własne: DFT, FFT DIT/DIF, DCT-II/FCT-II, Walsh-Hadamard, "
    "FWHT oraz wavelety DB4/DB6/DB8."
)

st.sidebar.header("Konfiguracja sygnału")
n_power = st.sidebar.slider("n (N = 2^n)", min_value=1, max_value=10, value=7)
length_n = int(2**n_power)

signal_kind = st.sidebar.selectbox("Sygnał testowy", ["S1", "S2", "S3"], index=2)
amplitude = st.sidebar.number_input("Amplituda A", value=1.0, step=0.1)
period = st.sidebar.number_input("Okres T (dla S3)", min_value=0.1, value=1.0, step=0.1)
seed = int(st.sidebar.number_input("Seed", value=42, step=1))

st.sidebar.caption(
    f"Częstotliwość próbkowania zgodnie z wymaganiem: fpr = {FPR_HZ:.0f} Hz"
)

imag_mode = st.sidebar.selectbox(
    "Składowa urojona",
    ["zero", "sine", "gaussian"],
    format_func=lambda x: {"zero": "Brak", "sine": "Sinus", "gaussian": "Szum Gaussa"}[
        x
    ],
)
imag_amplitude = st.sidebar.number_input("A_im", value=1.0, step=0.1)
imag_frequency_hz = st.sidebar.number_input(
    "f_im [Hz]", min_value=0.1, value=1.0, step=0.1
)

uploaded_file = st.sidebar.file_uploader(
    "Wczytaj sygnał zespolony (.bin)", type=["bin"]
)

if uploaded_file is not None:
    try:
        t_signal, x_signal, metadata = load_complex_signal(uploaded_file)
        x_signal = _next_power_length(x_signal, length_n)
        t_signal = np.arange(length_n, dtype=float) / FPR_HZ
        st.sidebar.success("Wczytano sygnał z pliku.")
        if metadata:
            st.sidebar.caption(f"Metadane: {metadata}")
    except Exception as exc:
        st.sidebar.error(f"Błąd wczytywania: {exc}")
        t_signal, y_real = _generate_test_signal(
            signal_kind, length_n, amplitude, period, seed
        )
        x_signal = with_imaginary_component(
            t_signal,
            y_real,
            imag_mode=imag_mode,
            imag_amplitude=imag_amplitude,
            imag_frequency_hz=imag_frequency_hz,
            seed=seed + 1,
        )
else:
    t_signal, y_real = _generate_test_signal(
        signal_kind, length_n, amplitude, period, seed
    )
    x_signal = with_imaginary_component(
        t_signal,
        y_real,
        imag_mode=imag_mode,
        imag_amplitude=imag_amplitude,
        imag_frequency_hz=imag_frequency_hz,
        seed=seed + 1,
    )

st.sidebar.divider()

signal_bin = save_complex_signal(
    t_signal,
    x_signal,
    metadata={
        "kind": signal_kind,
        "N": length_n,
        "fpr_hz": FPR_HZ,
        "imag_mode": imag_mode,
    },
)

st.sidebar.download_button(
    label="Pobierz sygnał zespolony (.bin)",
    data=signal_bin,
    file_name="zadanie4_signal_complex.bin",
    mime="application/octet-stream",
)

st.sidebar.download_button(
    label="Pobierz sygnał zespolony (.csv)",
    data=export_complex_to_csv(t_signal, x_signal),
    file_name="zadanie4_signal_complex.csv",
    mime="text/csv",
)

st.subheader("1) Wizualizacja sygnału zespolonego")
view_mode = st.radio("Tryb wizualizacji", ["W1", "W2"], horizontal=True)
st.pyplot(
    plot_complex_signal(t_signal, x_signal, mode=view_mode, title="Sygnał wejściowy"),
    width="stretch",
)

st.subheader("2) Transformacje")
selected_transform = st.selectbox("Wybierz transformację", TRANSFORM_OPTIONS)

is_fourier = selected_transform in {"DFT (definicja)", "FFT DIT", "FFT DIF"}
inverse_transform = False
if is_fourier:
    inverse_transform = st.checkbox("Odwrotna transformacja (IDFT/IFFT)", value=False)

compare_reference = st.checkbox("Porównaj z implementacją referencyjną", value=True)

wavelet_name = st.selectbox("Wavelet", ["db2", "db3", "db4", "db6", "db8"], index=0)
wavelet_levels = st.slider(
    "Poziomy dekompozycji wavelet",
    min_value=1,
    max_value=min(5, n_power),
    value=min(3, n_power),
)

transform_type, transform_result = _run_transform(
    transform_name=selected_transform,
    signal=x_signal,
    wavelet_name=wavelet_name,
    wavelet_levels=wavelet_levels,
    inverse=inverse_transform,
)

if transform_type == "wavelet":
    approx, details, reconstructed, reconstruction_error = transform_result
    st.pyplot(
        plot_wavelet_details(approx, details, title=f"DWT {wavelet_name.upper()}"),
        width="stretch",
    )
    st.metric("Błąd rekonstrukcji max|x-x̂|", f"{reconstruction_error:.3e}")

    st.download_button(
        label="Pobierz rekonstrukcję wavelet (.csv)",
        data=export_complex_to_csv(t_signal, reconstructed),
        file_name="zadanie4_wavelet_reconstruction.csv",
        mime="text/csv",
    )
else:
    y_transformed = np.asarray(transform_result, dtype=np.complex128)
    st.pyplot(
        plot_transform_components(
            np.arange(length_n), y_transformed, title=selected_transform
        ),
        width="stretch",
    )

    if selected_transform in {"DFT (definicja)", "FFT DIT", "FFT DIF"}:
        st.pyplot(
            plot_frequency_spectrum(
                y_transformed,
                fs_hz=FPR_HZ,
                title=f"Widmo częstotliwościowe - {selected_transform}{' (Odwrotna)' if inverse_transform else ''}",
            ),
            width="stretch",
        )

    if compare_reference:
        st.info(
            _reference_error(
                selected_transform, x_signal, y_transformed, inverse=inverse_transform
            )
        )

    st.download_button(
        label="Pobierz wynik transformacji (.csv)",
        data=_transform_to_csv(np.arange(length_n), y_transformed),
        file_name="zadanie4_transform.csv",
        mime="text/csv",
    )

st.subheader("3) Benchmarki")
col_b1, col_b2 = st.columns(2)
with col_b1:
    min_pow = st.slider("n_min", min_value=1, max_value=9, value=1)
with col_b2:
    default_max_pow = max(min_pow + 1, min(9, n_power))
    max_pow = st.slider(
        "n_max", min_value=min_pow + 1, max_value=10, value=default_max_pow
    )

include_numpy = st.checkbox("Pokaż także NumPy FFT", value=True)

if st.button("Uruchom benchmark", use_container_width=True):
    n_values = [2**p for p in range(min_pow, max_pow + 1)]
    max_bench_len = max(n_values)

    bench_signal = _build_signal_for_benchmark(
        signal_kind=signal_kind,
        amplitude=amplitude,
        period=period,
        seed=seed,
        max_length=max_bench_len,
        imag_mode=imag_mode,
        imag_amplitude=imag_amplitude,
        imag_frequency_hz=imag_frequency_hz,
    )

    benchmark_results = benchmark_transforms(
        signal=bench_signal,
        n_values=n_values,
        include_numpy=include_numpy,
        max_direct_n=256,
    )
    st.session_state["zad4_benchmark_results"] = benchmark_results

if "zad4_benchmark_results" in st.session_state:
    benchmark_results = st.session_state["zad4_benchmark_results"]
    st.pyplot(plot_benchmarks(benchmark_results), width="stretch")
    st.dataframe(benchmark_results, use_container_width=True)

st.subheader("4) Walidacja numeryczna")
tolerance = st.number_input(
    "Tolerancja błędu", min_value=1e-12, max_value=1e-2, value=1e-8, format="%e"
)

if st.button("Uruchom walidację", use_container_width=True):
    errors, status = validate_all_transforms(x_signal, tolerance=float(tolerance))
    st.session_state["zad4_validation_errors"] = errors
    st.session_state["zad4_validation_status"] = status

if "zad4_validation_errors" in st.session_state:
    errors = st.session_state["zad4_validation_errors"]
    status = st.session_state["zad4_validation_status"]

    rows = []
    for name, err in errors.items():
        rows.append(
            {
                "Test": name,
                "max|error|": err,
                "Status": "OK" if status[name] else "BŁĄD",
            }
        )

    st.dataframe(rows, use_container_width=True)

st.divider()
st.caption(
    "Dla sprawozdania: aplikacja umożliwia wygenerowanie wykresów sygnału, "
    "wyników transformacji, widm oraz porównań szybkości algorytmów."
)
