import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import functions as fn
import signal_utils as su

SIGNAL_REGISTRY = {
    "S1": {
        "name": "Szum o rozkładzie jednostajnym (S1)",
        "func": fn.generate_uniform_noise,
        "params": ["A", "t1", "d", "fs"],
        "is_periodic": False,
<<<<<<< HEAD
        "is_continuous": True,
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    },
    "S2": {
        "name": "Szum gaussowski (S2)",
        "func": fn.generate_gaussian_noise,
        "params": ["A", "t1", "d", "fs"],
        "is_periodic": False,
<<<<<<< HEAD
        "is_continuous": True,
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    },
    "S3": {
        "name": "Sygnał sinusoidalny (S3)",
        "func": fn.generate_sine,
        "params": ["A", "T", "t1", "d", "fs"],
        "is_periodic": True,
<<<<<<< HEAD
        "is_continuous": True,
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    },
    "S4": {
        "name": "Sygnał sinusoidalny wyprostowany jednopołówkowo (S4)",
        "func": fn.generate_sine_wyprostowany_jednopolowkowo,
        "params": ["A", "T", "t1", "d", "fs"],
        "is_periodic": True,
<<<<<<< HEAD
        "is_continuous": True,
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    },
    "S5": {
        "name": "Sygnał sinusoidalny wyprostowany dwupołówkowo (S5)",
        "func": fn.generate_sine_wyprostowany_dwupolowkowo,
        "params": ["A", "T", "t1", "d", "fs"],
        "is_periodic": True,
<<<<<<< HEAD
        "is_continuous": True,
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    },
    "S6": {
        "name": "Sygnał prostokątny (S6)",
        "func": fn.generate_square,
        "params": ["A", "T", "t1", "d", "kw", "fs"],
        "is_periodic": True,
<<<<<<< HEAD
        "is_continuous": True,
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    },
    "S7": {
        "name": "Sygnał prostokątny symetryczny (S7)",
        "func": fn.generate_symmetric_square,
        "params": ["A", "T", "t1", "d", "kw", "fs"],
        "is_periodic": True,
<<<<<<< HEAD
        "is_continuous": True,
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    },
    "S8": {
        "name": "Sygnał trójkątny (S8)",
        "func": fn.generate_triangle,
        "params": ["A", "T", "t1", "d", "kw", "fs"],
        "is_periodic": True,
<<<<<<< HEAD
        "is_continuous": True,
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    },
    "S9": {
        "name": "Skok jednostkowy (S9)",
        "func": fn.unit_step,
        "params": ["A", "t1", "d", "ts", "fs"],
        "is_periodic": False,
<<<<<<< HEAD
        "is_continuous": True,
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    },
    "S10": {
        "name": "Impuls jednostkowy (S10)",
        "func": fn.generate_unit_impulse,
        "params": ["A", "ns", "n1", "l", "f"],
        "is_periodic": False,
<<<<<<< HEAD
        "is_continuous": False,
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    },
    "S11": {
        "name": "Szum impulsowy (S11)",
        "func": fn.generate_impulse_noise,
        "params": ["A", "t1", "d", "f", "p", "fs"],
        "is_periodic": False,
<<<<<<< HEAD
        "is_continuous": False,
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    },
}


def reconstruct_signal(t_samples, y_samples, t_target, method, sinc_neighbors):
<<<<<<< HEAD
    if len(y_samples) == 0:
        return np.zeros_like(t_target)

    if len(y_samples) == 1 or len(t_samples) < 2:
        return np.full_like(t_target, y_samples[0], dtype=float)

=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    if method == "ZOH":
        return su.reconstruct_zoh(t_samples, y_samples, t_target)
    if method == "FOH":
        return su.reconstruct_foh(t_samples, y_samples, t_target)
    return su.reconstruct_sinc(
        t_samples,
        y_samples,
        t_target,
        fs_hz=1.0 / np.mean(np.diff(t_samples)),
        neighbors=sinc_neighbors,
    )
<<<<<<< HEAD
=======


def render_signal_params(required_params, defaults):
    params = {}

    if "A" in required_params:
        params["A"] = st.sidebar.number_input(
            "Amplituda (A)",
            value=float(defaults.get("A", 1.0)),
            step=0.1,
        )

    if "T" in required_params:
        params["T"] = st.sidebar.number_input(
            "Okres podstawowy (T)",
            min_value=0.1,
            value=float(defaults.get("T", 1.0)),
            step=0.1,
        )

    if "kw" in required_params:
        params["kw"] = st.sidebar.slider(
            "Współczynnik wypełnienia (kw)",
            0.0,
            1.0,
            float(defaults.get("kw", 0.5)),
        )

    if "ts" in required_params:
        params["ts"] = st.sidebar.number_input(
            "Punkt skoku (ts)",
            value=float(defaults.get("ts", 0.5)),
            step=0.1,
        )

    if "ns" in required_params:
        params["ns"] = st.sidebar.number_input(
            "Numer próbki impulsu (ns)",
            value=int(defaults.get("ns", 0)),
            step=1,
        )

    if "n1" in required_params:
        params["n1"] = st.sidebar.number_input(
            "Numer pierwszej próbki (n1)",
            value=int(defaults.get("n1", 0)),
            step=1,
        )

    if "l" in required_params:
        params["l"] = st.sidebar.number_input(
            "Liczba próbek (l)",
            min_value=1,
            value=int(defaults.get("l", 100)),
            step=1,
        )

    if "f" in required_params:
        params["f"] = st.sidebar.number_input(
            "Częstotliwość próbkowania dyskretnego (f)",
            min_value=1.0,
            value=float(defaults.get("f", 1000.0)),
            step=10.0,
        )

    if "p" in required_params:
        params["p"] = st.sidebar.slider(
            "Prawdopodobieństwo impulsu (p)",
            0.0,
            1.0,
            float(defaults.get("p", 0.1)),
        )

    return params
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a


st.title("Zadanie 2 - Próbkowanie, kwantyzacja i rekonstrukcja")

<<<<<<< HEAD
st.sidebar.header("Konfiguracja sygnału wejściowego")
signal_keys = list(SIGNAL_REGISTRY.keys())
signal_names = [SIGNAL_REGISTRY[k]["name"] for k in signal_keys]
selected_signal_name = st.sidebar.selectbox(
    "Wybierz sygnał/szum", signal_names
=======
st.sidebar.header("Konfiguracja sygnału")
signal_keys = list(SIGNAL_REGISTRY.keys())
signal_names = [SIGNAL_REGISTRY[k]["name"] for k in signal_keys]
selected_signal_name = st.sidebar.selectbox(
    "Wybierz sygnał/szum", signal_names, index=2
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
)
selected_signal_key = signal_keys[signal_names.index(selected_signal_name)]
signal_config = SIGNAL_REGISTRY[selected_signal_key]
required_params = signal_config["params"]

<<<<<<< HEAD
# Parametry wspólne
amplitude = 1.0
if "A" in required_params:
    amplitude = st.sidebar.number_input("Amplituda A", value=1.0, step=0.1)

t_start = st.sidebar.number_input(
    "Czas początkowy t1 [s]", value=0.0, step=0.1
)
=======
t_start = st.sidebar.number_input("Czas początkowy t1 [s]", value=0.0, step=0.1)
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
duration_s = st.sidebar.number_input(
    "Czas trwania d [s]", min_value=0.05, value=1.0, step=0.05
)
fs_ref_hz = st.sidebar.number_input(
    "Referencyjne fs_ref [Hz]", min_value=100.0, value=20000.0, step=100.0
)

<<<<<<< HEAD
signal_params = {"A": amplitude}

if "T" in required_params:
    signal_params["T"] = st.sidebar.number_input(
        "Okres podstawowy T [s]", min_value=0.01, value=0.1, step=0.01
    )

if "kw" in required_params:
    signal_params["kw"] = st.sidebar.slider(
        "Współczynnik wypełnienia kw", 0.01, 0.99, 0.5
    )

if "ts" in required_params:
    signal_params["ts"] = st.sidebar.number_input(
        "Punkt skoku ts [s]",
        min_value=t_start,
        max_value=t_start + duration_s,
        value=t_start + duration_s / 2,
        step=0.01,
    )

if "ns" in required_params:
    signal_params["ns"] = st.sidebar.number_input(
        "Numer próbki impulsu ns", value=0, step=1
    )

if "n1" in required_params:
    signal_params["n1"] = st.sidebar.number_input(
        "Numer pierwszej próbki n1", value=0, step=1
    )

if "l" in required_params:
    signal_params["l"] = st.sidebar.number_input(
        "Liczba próbek l", min_value=1, value=100, step=1
    )

if "f" in required_params:
    signal_params["f"] = st.sidebar.number_input(
        "Częstotliwość sygnału dyskretnego f [Hz]",
        min_value=1.0,
        value=1000.0,
        step=10.0,
    )

if "p" in required_params:
    signal_params["p"] = st.sidebar.slider(
        "Prawdopodobieństwo impulsu p", 0.0, 1.0, 0.1
    )

# Parametry globalne generatorów z zadania 1
signal_params["t1"] = t_start
signal_params["d"] = duration_s
signal_params["fs"] = fs_ref_hz
=======
signal_param_defaults = {
    "A": 1.0,
    "T": 0.1,
    "kw": 0.5,
    "ts": t_start + duration_s / 2.0,
    "ns": 0,
    "n1": 0,
    "l": 200,
    "f": 1000.0,
    "p": 0.1,
}

signal_params = render_signal_params(required_params, signal_param_defaults)

params_with_globals = signal_params.copy()
params_with_globals["t1"] = t_start
params_with_globals["d"] = duration_s
params_with_globals["fs"] = fs_ref_hz

t_ref, y_ref = signal_config["func"](params_with_globals)
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a

st.sidebar.header("A/C i C/A")
fs_sample_hz = st.sidebar.number_input(
    "fs próbkowania [Hz]", min_value=1.0, value=1000.0, step=10.0
)
bits = st.sidebar.slider("Liczba bitów", min_value=1, max_value=16, value=8)
quant_mode = st.sidebar.selectbox(
    "Tryb kwantyzacji",
    options=["round", "truncate"],
    format_func=lambda x: "Zaokrąglanie" if x == "round" else "Obcięcie",
<<<<<<< HEAD
)
reconstruction_method = st.sidebar.selectbox(
    "Metoda rekonstrukcji", ["ZOH", "FOH", "SINC"]
)
sinc_neighbors = st.sidebar.slider(
    "SINC: liczba sąsiadów", min_value=4, max_value=64, value=16, step=4
)

# Generacja sygnału wejściowego (zestaw funkcji jak w Zadaniu 1)
t_ref, y_ref = signal_config["func"](signal_params)

# Próbkowanie do analizy/rekonstrukcji
sample_start = float(t_ref[0])
sample_end = float(t_ref[-1])
if sample_end <= sample_start:
    sample_end = sample_start + 1.0 / fs_sample_hz
=======
)
reconstruction_method = st.sidebar.selectbox(
    "Metoda rekonstrukcji", ["ZOH", "FOH", "SINC"]
)
sinc_neighbors = st.sidebar.slider(
    "SINC: liczba sąsiadów", min_value=4, max_value=64, value=16, step=4
)
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a

t_samples, y_samples = su.uniform_sample(
    t_reference=t_ref,
    y_reference=y_ref,
    fs_hz=fs_sample_hz,
<<<<<<< HEAD
    t_start=sample_start,
    t_end=sample_end,
=======
    t_start=float(t_ref[0]),
    t_end=float(t_ref[-1]),
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
)

y_quantized, q_meta = su.quantize_uniform(
    y_samples, bits=bits, mode=quant_mode
)
y_reconstructed = reconstruct_signal(
    t_samples,
    y_samples,
    t_ref,
    reconstruction_method,
    sinc_neighbors,
)
y_reconstructed_q = reconstruct_signal(
    t_samples,
    y_quantized,
    t_ref,
    reconstruction_method,
    sinc_neighbors,
)

sampling_metrics = su.quality_metrics(y_ref, y_reconstructed)
quantization_metrics = su.quality_metrics(y_samples, y_quantized)

st.subheader("Przebiegi: sygnał wejściowy i rekonstrukcja")
fig_sig, ax_sig = plt.subplots(figsize=(12, 4))
<<<<<<< HEAD

if signal_config["is_continuous"]:
    ax_sig.plot(t_ref, y_ref, label="Oryginał (referencja)", linewidth=1.5)
    ax_sig.plot(
        t_ref,
        y_reconstructed,
        label=f"Rekonstrukcja {reconstruction_method}",
        linewidth=1.2,
    )
else:
    ax_sig.scatter(t_ref, y_ref, label="Oryginał (referencja)", s=18)
    ax_sig.scatter(
        t_ref,
        y_reconstructed,
        label=f"Rekonstrukcja {reconstruction_method}",
        s=14,
        alpha=0.7,
    )

ax_sig.scatter(
    t_samples, y_samples, label="Próbki", color="black", s=18, zorder=3
)
=======
ax_sig.plot(t_ref, y_ref, label="Oryginał (referencja)", linewidth=1.5)
ax_sig.plot(
    t_ref,
    y_reconstructed,
    label=f"Rekonstrukcja {reconstruction_method}",
    linewidth=1.2,
)
ax_sig.scatter(t_samples, y_samples, label="Próbki", color="black", s=18, zorder=3)
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
ax_sig.set_xlabel("Czas [s]")
ax_sig.set_ylabel("Amplituda")
ax_sig.grid(True, alpha=0.3)
ax_sig.legend(loc="upper right")
fig_sig.tight_layout()
st.pyplot(fig_sig, width="stretch")

st.subheader("Efekt kwantyzacji")
fig_q, ax_q = plt.subplots(figsize=(12, 4))
if reconstruction_method == "ZOH":
    ax_q.step(
        t_ref,
        y_reconstructed,
        where="post",
        label="Przed kwantyzacją (po rekonstrukcji)",
        linewidth=1.2,
    )
    ax_q.step(
        t_ref,
        y_reconstructed_q,
        where="post",
        label=f"Po kwantyzacji ({quant_mode}, po rekonstrukcji)",
        linewidth=1.2,
    )
else:
<<<<<<< HEAD
    if signal_config["is_continuous"]:
        ax_q.plot(
            t_ref,
            y_reconstructed,
            label="Przed kwantyzacją (po rekonstrukcji)",
            linewidth=1.2,
        )
        ax_q.plot(
            t_ref,
            y_reconstructed_q,
            label=f"Po kwantyzacji ({quant_mode}, po rekonstrukcji)",
            linewidth=1.2,
        )
    else:
        ax_q.scatter(
            t_ref,
            y_reconstructed,
            label="Przed kwantyzacją (po rekonstrukcji)",
            s=14,
        )
        ax_q.scatter(
            t_ref,
            y_reconstructed_q,
            label=f"Po kwantyzacji ({quant_mode}, po rekonstrukcji)",
            s=14,
            alpha=0.7,
        )

=======
    ax_q.plot(
        t_ref,
        y_reconstructed,
        label="Przed kwantyzacją (po rekonstrukcji)",
        linewidth=1.2,
    )
    ax_q.plot(
        t_ref,
        y_reconstructed_q,
        label=f"Po kwantyzacji ({quant_mode}, po rekonstrukcji)",
        linewidth=1.2,
    )
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
ax_q.scatter(
    t_samples,
    y_quantized,
    label="Próbki skwantyzowane",
    color="black",
    s=12,
    alpha=0.55,
)
ax_q.set_xlabel("Czas [s]")
ax_q.set_ylabel("Amplituda")
ax_q.grid(True, alpha=0.3)
ax_q.legend(loc="upper right")
fig_q.tight_layout()
st.pyplot(fig_q, width="stretch")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("### Miary jakości próbkowania")
    st.metric("MSE", f"{sampling_metrics['mse']:.6e}")
    st.metric("SNR [dB]", f"{sampling_metrics['snr']:.3f}")
    st.metric("PSNR [dB]", f"{sampling_metrics['psnr']:.3f}")
    st.metric("MD", f"{sampling_metrics['md']:.6f}")

with col_b:
    st.markdown("### Miary jakości kwantyzacji")
    st.metric("MSE", f"{quantization_metrics['mse']:.6e}")
    st.metric("SNR [dB]", f"{quantization_metrics['snr']:.3f}")
    st.metric("PSNR [dB]", f"{quantization_metrics['psnr']:.3f}")
    st.metric("MD", f"{quantization_metrics['md']:.6f}")
<<<<<<< HEAD
    st.caption(
        f"Poziomy: {q_meta['levels']}, krok kwantyzacji: {q_meta['step']:.6f}"
    )
=======
    st.caption(f"Poziomy: {q_meta['levels']}, krok kwantyzacji: {q_meta['step']:.6f}")
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a

# Eksperyment fs/f0 ma sens dla sygnałów okresowych z parametrem T
show_fs_experiment = signal_config["is_periodic"] and "T" in required_params

<<<<<<< HEAD
if show_fs_experiment:
    st.divider()
    st.subheader("Eksperyment 1: jakość a relacja fs/f0")

    f0_hz = 1.0 / signal_params["T"]
=======
f0_hz = None
if "T" in signal_params and signal_params["T"] > 0:
    f0_hz = 1.0 / signal_params["T"]

if f0_hz is None:
    st.info(
        "Dla wybranego sygnału brak parametru okresu T, więc wykres fs/f0 nie jest dostępny."
    )
else:
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
    exp_fs_min = st.number_input(
        "fs_min [Hz]", min_value=1.0, value=max(10.0, f0_hz), step=10.0
    )
    exp_fs_max = st.number_input(
        "fs_max [Hz]",
        min_value=exp_fs_min + 1.0,
        value=max(exp_fs_min + 10.0, 8.0 * f0_hz),
        step=10.0,
    )
<<<<<<< HEAD
    exp_fs_count = st.slider(
        "Liczba punktów fs", min_value=4, max_value=40, value=12
    )
=======
    exp_fs_count = st.slider("Liczba punktów fs", min_value=4, max_value=40, value=12)
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a

    fs_values = np.linspace(exp_fs_min, exp_fs_max, exp_fs_count)
    sampling_exp = su.run_sampling_experiment(
        reference_t=t_ref,
        reference_y=y_ref,
        fs_values=fs_values,
        reconstruction_method=reconstruction_method.lower(),
        sinc_neighbors=sinc_neighbors,
    )

    ratio = np.array([row["fs"] / f0_hz for row in sampling_exp])
    snr_vals = np.array([row["snr"] for row in sampling_exp])
    mse_vals = np.array([row["mse"] for row in sampling_exp])

    fig_exp1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ratio, snr_vals, marker="o")
    ax1.set_xlabel("fs / f0")
    ax1.set_ylabel("SNR [dB]")
    ax1.set_title("SNR vs fs/f0")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ratio, mse_vals, marker="o", color="crimson")
    ax2.set_xlabel("fs / f0")
    ax2.set_ylabel("MSE")
    ax2.set_yscale("log")
    ax2.set_title("MSE vs fs/f0")
    ax2.grid(True, alpha=0.3)
    fig_exp1.tight_layout()
    st.pyplot(fig_exp1, width="stretch")
<<<<<<< HEAD
else:
    st.info(
        "Eksperyment fs/f0 jest dostępny dla sygnałów okresowych z parametrem T."
    )
=======
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a

st.divider()
st.subheader("Eksperyment 2: kwantyzacja, teoria SNR i ENOB")

bits_min, bits_max = st.slider(
    "Zakres bitów", min_value=1, max_value=16, value=(2, 12)
)
bits_values = list(range(bits_min, bits_max + 1))
quant_exp = su.run_quantization_experiment(
    y_samples, bits_values, mode=quant_mode
)

bits_arr = np.array([row["bits"] for row in quant_exp])
snr_measured = np.array([row["snr_measured"] for row in quant_exp])
snr_theory = np.array([row["snr_theory"] for row in quant_exp])
enob_arr = np.array([row["enob"] for row in quant_exp])

fig_exp2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
ax3.plot(bits_arr, snr_measured, marker="o", label="SNR zmierzone")
ax3.plot(
<<<<<<< HEAD
    bits_arr,
    snr_theory,
    marker="x",
    linestyle="--",
    label="SNR teoria 6.02b+1.76",
=======
    bits_arr, snr_theory, marker="x", linestyle="--", label="SNR teoria 6.02b+1.76"
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
)
ax3.set_xlabel("Liczba bitów")
ax3.set_ylabel("SNR [dB]")
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper left")

ax4.plot(bits_arr, enob_arr, marker="o", color="purple")
ax4.set_xlabel("Liczba bitów")
ax4.set_ylabel("ENOB")
ax4.grid(True, alpha=0.3)
fig_exp2.tight_layout()
st.pyplot(fig_exp2, width="stretch")

st.divider()
st.subheader("Aliasing (przypadki obowiązkowe, dla sinusoidy)")

mandatory_cases = [
    (100.0, 1000.0),
    (440.0, 22050.0),
    (220.0, 44100.0),
]

case_label = st.selectbox(
    "Wybierz przypadek (f0, fs)",
    [f"f0={f0:.0f} Hz, fs={fs:.0f} Hz" for f0, fs in mandatory_cases],
<<<<<<< HEAD
)
case_idx = [
    f"f0={f0:.0f} Hz, fs={fs:.0f} Hz" for f0, fs in mandatory_cases
].index(case_label)
=======
)
case_idx = [f"f0={f0:.0f} Hz, fs={fs:.0f} Hz" for f0, fs in mandatory_cases].index(
    case_label
)
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
case_f0, case_fs = mandatory_cases[case_idx]

k_values = st.multiselect(
    "Wartości k", options=list(range(-4, 5)), default=[-2, -1, 1, 2]
)
amp_values_text = st.text_input(
    "Amplitudy dla fd (oddzielone przecinkami)", value="0.5,1.0,1.5"
)
amp_values = [
    float(item.strip()) for item in amp_values_text.split(",") if item.strip()
]

alias_duration = st.number_input(
    "Czas prezentacji aliasingu [s]", min_value=0.01, value=0.05, step=0.01
)
alias_ref_fs = st.number_input(
<<<<<<< HEAD
    "fs referencyjne aliasingu [Hz]",
    min_value=1000.0,
    value=100000.0,
    step=1000.0,
=======
    "fs referencyjne aliasingu [Hz]", min_value=1000.0, value=100000.0, step=1000.0
>>>>>>> 29bbe3edc0d59c46a7f688b177586e66cbd5fd8a
)

t_alias_ref, y_alias_ref = su.generate_sine_wave(
    case_f0, 1.0, 0.0, 0.0, alias_duration, alias_ref_fs
)
t_alias_s, y_alias_s = su.uniform_sample(t_alias_ref, y_alias_ref, case_fs)

rows = []
fig_alias, ax_alias = plt.subplots(figsize=(12, 4))
ax_alias.scatter(t_alias_s, y_alias_s, label=f"f0={case_f0:.0f} Hz", s=26)

for k in k_values:
    fd = su.alias_frequency(case_f0, case_fs, k)
    t_fd_ref, y_fd_ref = su.generate_sine_wave(
        fd, 1.0, 0.0, 0.0, alias_duration, alias_ref_fs
    )
    t_fd_s, y_fd_s = su.uniform_sample(t_fd_ref, y_fd_ref, case_fs)
    md = su.max_difference(y_alias_s, y_fd_s)
    rows.append({"k": k, "fd [Hz]": fd, "MD względem f0": md})
    ax_alias.scatter(t_fd_s, y_fd_s, s=16, alpha=0.6, label=f"fd=f0+{k}fs")

ax_alias.set_xlabel("Czas [s]")
ax_alias.set_ylabel("Próbki")
ax_alias.grid(True, alpha=0.3)
ax_alias.legend(loc="upper right", ncols=2)
fig_alias.tight_layout()
st.pyplot(fig_alias, width="stretch")
st.dataframe(rows, width="stretch")

st.markdown("#### Wpływ amplitudy fd przy stałym f0")
amp_k = st.slider(
    "Wybierz k do porównania amplitud", min_value=-5, max_value=5, value=1
)
fd_amp = su.alias_frequency(case_f0, case_fs, amp_k)

fig_amp, ax_amp = plt.subplots(figsize=(12, 4))
for amp in amp_values:
    t_fd_ref, y_fd_ref = su.generate_sine_wave(
        fd_amp, amp, 0.0, 0.0, alias_duration, alias_ref_fs
    )
    t_fd_s, y_fd_s = su.uniform_sample(t_fd_ref, y_fd_ref, case_fs)
    ax_amp.scatter(t_fd_s, y_fd_s, s=18, label=f"A={amp}")

ax_amp.set_title(
    f"Próbki dla fd={fd_amp:.2f} Hz (f0={case_f0:.0f} Hz, fs={case_fs:.0f} Hz)"
)
ax_amp.set_xlabel("Czas [s]")
ax_amp.set_ylabel("Amplituda")
ax_amp.grid(True, alpha=0.3)
ax_amp.legend(loc="upper right")
fig_amp.tight_layout()
st.pyplot(fig_amp, width="stretch")

st.divider()
st.subheader("Aliasing dla wybranego sygnału (rekonstrukcja SINC)")

if "T" not in required_params:
    st.info(
        "Dla wybranego sygnału brak parametru okresu T, więc nie można ustawić częstotliwości oryginalnej. "
        "Wybierz sygnał okresowy (S3–S8), aby uruchomić demonstrację aliasingu."
    )
else:
    alias_amplitude = st.number_input(
        "Aliasing: amplituda A",
        value=float(signal_params.get("A", 1.0)),
        step=0.1,
        key="alias_amplitude_input",
    )
    alias_f0_hz = st.number_input(
        "Aliasing: częstotliwość oryginalna f0 [Hz]",
        min_value=0.1,
        value=float(1.0 / signal_params["T"]),
        step=1.0,
        key="alias_f0_hz_input",
    )
    alias_fs_hz = st.number_input(
        "Aliasing: częstotliwość próbkowania fs [Hz]",
        min_value=1.0,
        value=20.0,
        step=1.0,
        key="alias_fs_hz_input",
    )
    alias_t_start = st.number_input(
        "Aliasing: punkt startowy t1 [s]",
        value=0.0,
        step=0.01,
        key="alias_t_start_input",
    )
    alias_duration_s = st.number_input(
        "Aliasing: długość trwania d [s]",
        min_value=0.01,
        value=0.3,
        step=0.01,
        key="alias_duration_input",
    )
    alias_sinc_neighbors = st.slider(
        "Aliasing: SINC liczba sąsiadów",
        min_value=4,
        max_value=128,
        value=32,
        step=4,
        key="alias_sinc_neighbors_input",
    )

    alias_fs_ref_hz = max(2000.0, 200.0 * alias_f0_hz)

    alias_params = signal_params.copy()
    alias_params["A"] = alias_amplitude
    alias_params["T"] = 1.0 / alias_f0_hz
    alias_params["t1"] = alias_t_start
    alias_params["d"] = alias_duration_s
    alias_params["fs"] = alias_fs_ref_hz

    t_alias_ref, y_alias_ref = signal_config["func"](alias_params)

    t_alias_samples, y_alias_samples = su.uniform_sample(
        t_reference=t_alias_ref,
        y_reference=y_alias_ref,
        fs_hz=alias_fs_hz,
        t_start=float(t_alias_ref[0]),
        t_end=float(t_alias_ref[-1]),
    )

    y_alias_reconstructed_sinc = su.reconstruct_sinc(
        t_alias_samples,
        y_alias_samples,
        t_alias_ref,
        fs_hz=alias_fs_hz,
        neighbors=alias_sinc_neighbors,
    )

    fig_alias_custom, ax_alias_custom = plt.subplots(figsize=(12, 4))
    ax_alias_custom.plot(
        t_alias_ref,
        y_alias_ref,
        label="Oryginalna funkcja",
        linewidth=1.6,
    )
    ax_alias_custom.scatter(
        t_alias_samples,
        y_alias_samples,
        label="Próbki",
        color="black",
        s=22,
        zorder=3,
    )
    ax_alias_custom.plot(
        t_alias_ref,
        y_alias_reconstructed_sinc,
        label="Rekonstrukcja SINC",
        linewidth=1.2,
        color="crimson",
    )

    ax_alias_custom.set_title(
        f"{signal_config['name']} | f0={alias_f0_hz:.2f} Hz, fs={alias_fs_hz:.2f} Hz"
    )
    ax_alias_custom.set_xlabel("Czas [s]")
    ax_alias_custom.set_ylabel("Amplituda")
    ax_alias_custom.grid(True, alpha=0.3)
    ax_alias_custom.legend(loc="upper right")
    fig_alias_custom.tight_layout()
    st.pyplot(fig_alias_custom, width="stretch")
