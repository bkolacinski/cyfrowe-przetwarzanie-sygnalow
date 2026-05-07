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
        "is_continuous": True,
    },
    "S2": {
        "name": "Szum gaussowski (S2)",
        "func": fn.generate_gaussian_noise,
        "params": ["A", "t1", "d", "fs"],
        "is_periodic": False,
        "is_continuous": True,
    },
    "S3": {
        "name": "Sygnał sinusoidalny (S3)",
        "func": fn.generate_sine,
        "params": ["A", "T", "t1", "d", "fs"],
        "is_periodic": True,
        "is_continuous": True,
    },
    "S4": {
        "name": "Sygnał sinusoidalny wyprostowany jednopołówkowo (S4)",
        "func": fn.generate_sine_wyprostowany_jednopolowkowo,
        "params": ["A", "T", "t1", "d", "fs"],
        "is_periodic": True,
        "is_continuous": True,
    },
    "S5": {
        "name": "Sygnał sinusoidalny wyprostowany dwupołówkowo (S5)",
        "func": fn.generate_sine_wyprostowany_dwupolowkowo,
        "params": ["A", "T", "t1", "d", "fs"],
        "is_periodic": True,
        "is_continuous": True,
    },
    "S6": {
        "name": "Sygnał prostokątny (S6)",
        "func": fn.generate_square,
        "params": ["A", "T", "t1", "d", "kw", "fs"],
        "is_periodic": True,
        "is_continuous": True,
    },
    "S7": {
        "name": "Sygnał prostokątny symetryczny (S7)",
        "func": fn.generate_symmetric_square,
        "params": ["A", "T", "t1", "d", "kw", "fs"],
        "is_periodic": True,
        "is_continuous": True,
    },
    "S8": {
        "name": "Sygnał trójkątny (S8)",
        "func": fn.generate_triangle,
        "params": ["A", "T", "t1", "d", "kw", "fs"],
        "is_periodic": True,
        "is_continuous": True,
    },
    "S9": {
        "name": "Skok jednostkowy (S9)",
        "func": fn.unit_step,
        "params": ["A", "t1", "d", "ts", "fs"],
        "is_periodic": False,
        "is_continuous": True,
    },
    "S10": {
        "name": "Impuls jednostkowy (S10)",
        "func": fn.generate_unit_impulse,
        "params": ["A", "ns", "n1", "l", "f"],
        "is_periodic": False,
        "is_continuous": False,
    },
    "S11": {
        "name": "Szum impulsowy (S11)",
        "func": fn.generate_impulse_noise,
        "params": ["A", "t1", "d", "f", "p", "fs"],
        "is_periodic": False,
        "is_continuous": False,
    },
}


def reconstruct_signal(t_samples, y_samples, t_target, method, sinc_neighbors):
    if len(y_samples) == 0:
        return np.zeros_like(t_target)

    if len(y_samples) == 1 or len(t_samples) < 2:
        return np.full_like(t_target, y_samples[0], dtype=float)

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


def render_signal_params(required_params, t_start, duration_s, fs_ref_hz):
    params = {}

    if "A" in required_params:
        params["A"] = st.sidebar.number_input(
            "Amplituda (A)",
            value=1.0,
            step=0.1,
        )

    if "T" in required_params:
        params["T"] = st.sidebar.number_input(
            "Okres podstawowy (T)",
            min_value=0.1,
            value=1.0,
            step=0.1,
        )

    if "kw" in required_params:
        params["kw"] = st.sidebar.slider(
            "Współczynnik wypełnienia (kw)",
            0.0,
            1.0,
            0.5,
        )

    if "ts" in required_params:
        params["ts"] = st.sidebar.number_input(
            "Punkt skoku (ts)",
            min_value=t_start,
            max_value=t_start + duration_s,
            value=t_start + duration_s / 2,
            step=0.1,
        )

    if "ns" in required_params:
        params["ns"] = st.sidebar.number_input(
            "Numer próbki impulsu (ns)",
            value=0,
            step=1,
        )

    if "n1" in required_params:
        params["n1"] = st.sidebar.number_input(
            "Numer pierwszej próbki (n1)",
            value=0,
            step=1,
        )

    if "l" in required_params:
        params["l"] = st.sidebar.number_input(
            "Liczba próbek (l)",
            min_value=1,
            value=100,
            step=1,
        )

    if "f" in required_params:
        params["f"] = st.sidebar.number_input(
            "Częstotliwość próbkowania dyskretnego (f)",
            min_value=1.0,
            value=float(fs_ref_hz),
            step=10.0,
        )

    if "p" in required_params:
        params["p"] = st.sidebar.slider(
            "Prawdopodobieństwo impulsu (p)",
            0.0,
            1.0,
            0.1,
        )

    return params


def render_sinc_summary_plot(
    signal_name,
    signal_config,
    signal_params,
    t_start,
    duration_s,
    t_ref,
    y_ref,
    t_samples,
    y_samples,
    fs_sample_hz,
    sinc_neighbors,
):
    y_sinc = reconstruct_signal(
        t_samples,
        y_samples,
        t_ref,
        method="SINC",
        sinc_neighbors=sinc_neighbors,
    )

    amplitude = signal_params.get("A")
    period = signal_params.get("T")
    original_freq_hz = None
    if period is not None and period > 0:
        original_freq_hz = 1.0 / period
    elif signal_params.get("f") is not None and signal_params.get("f") > 0:
        original_freq_hz = float(signal_params.get("f"))

    info_parts = [
        f"A={amplitude:.3f}" if amplitude is not None else "A=brak",
        (
            f"f0={original_freq_hz:.3f} Hz"
            if original_freq_hz is not None
            else "f0=nieokreślona"
        ),
        f"fs={fs_sample_hz:.3f} Hz",
        f"t1={t_start:.3f} s",
        f"T={period:.3f} s" if period is not None else "T=brak",
        f"d={duration_s:.3f} s",
    ]

    fig, ax = plt.subplots(figsize=(12, 4))

    if signal_config["is_continuous"]:
        ax.plot(t_ref, y_ref, label="Funkcja oryginalna", linewidth=1.5)
    else:
        ax.scatter(t_ref, y_ref, label="Funkcja oryginalna", s=16)

    ax.scatter(
        t_samples,
        y_samples,
        label="Miejsca próbkowania",
        color="black",
        s=20,
        zorder=3,
    )
    ax.plot(
        t_ref,
        y_sinc,
        label="Rekonstrukcja SINC",
        linewidth=1.2,
        color="tab:orange",
    )

    ax.set_title(f"{signal_name} | " + " | ".join(info_parts))
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Amplituda")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    st.pyplot(fig, width="stretch")


st.title("Zadanie 2 - Próbkowanie, kwantyzacja i rekonstrukcja")

st.sidebar.header("Konfiguracja sygnału wejściowego")
signal_keys = list(SIGNAL_REGISTRY.keys())
signal_names = [SIGNAL_REGISTRY[k]["name"] for k in signal_keys]
selected_signal_name = st.sidebar.selectbox(
    "Wybierz sygnał/szum", signal_names, index=2
)
selected_signal_key = signal_keys[signal_names.index(selected_signal_name)]
signal_config = SIGNAL_REGISTRY[selected_signal_key]
required_params = signal_config["params"]

t_start = st.sidebar.number_input(
    "Czas początkowy t1 [s]", value=0.0, step=0.1
)
duration_s = st.sidebar.number_input(
    "Czas trwania d [s]", min_value=0.05, value=1.0, step=0.05
)
fs_ref_hz = st.sidebar.number_input(
    "Referencyjne fs_ref [Hz]", min_value=100.0, value=20000.0, step=100.0
)

signal_params = render_signal_params(
    required_params, t_start, duration_s, fs_ref_hz
)
params_with_globals = signal_params.copy()
params_with_globals["t1"] = t_start
params_with_globals["d"] = duration_s
params_with_globals["fs"] = fs_ref_hz

t_ref, y_ref = signal_config["func"](params_with_globals)

st.sidebar.header("A/C i C/A")
fs_sample_hz = st.sidebar.number_input(
    "fs próbkowania [Hz]", min_value=1.0, value=1000.0, step=10.0
)
bits = st.sidebar.slider("Liczba bitów", min_value=1, max_value=16, value=8)
quant_mode = st.sidebar.selectbox(
    "Tryb kwantyzacji",
    options=["round", "truncate"],
    format_func=lambda x: "Zaokrąglanie" if x == "round" else "Obcięcie",
)
reconstruction_method = st.sidebar.selectbox(
    "Metoda rekonstrukcji", ["ZOH", "FOH", "SINC"]
)
sinc_neighbors = st.sidebar.slider(
    "SINC: liczba sąsiadów", min_value=4, max_value=64, value=16, step=4
)

sample_start = float(t_ref[0])
sample_end = float(t_ref[-1])
if sample_end <= sample_start:
    sample_end = sample_start + 1.0 / fs_sample_hz

t_samples, y_samples = su.uniform_sample(
    t_reference=t_ref,
    y_reference=y_ref,
    fs_hz=fs_sample_hz,
    t_start=sample_start,
    t_end=sample_end,
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
    st.caption(
        f"Poziomy: {q_meta['levels']}, krok kwantyzacji: {q_meta['step']:.6f}"
    )

show_fs_experiment = (
    signal_config["is_periodic"]
    and "T" in required_params
    and signal_params.get("T", 0) > 0
)
if show_fs_experiment:
    st.divider()
    st.subheader("Eksperyment 1: jakość a relacja fs/f0")

    f0_hz = 1.0 / signal_params["T"]
    exp_fs_min = st.number_input(
        "fs_min [Hz]", min_value=1.0, value=max(10.0, f0_hz), step=10.0
    )
    exp_fs_max = st.number_input(
        "fs_max [Hz]",
        min_value=exp_fs_min + 1.0,
        value=max(exp_fs_min + 10.0, 8.0 * f0_hz),
        step=10.0,
    )
    exp_fs_count = st.slider(
        "Liczba punktów fs", min_value=4, max_value=40, value=12
    )

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
else:
    st.info(
        "Eksperyment fs/f0 jest dostępny dla sygnałów okresowych z parametrem T."
    )

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
    bits_arr,
    snr_theory,
    marker="x",
    linestyle="--",
    label="SNR teoria 6.02b+1.76",
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
st.subheader("Aliasing (przypadki obowiązkowe)")

mandatory_cases = [
    (100.0, 1000.0),
    (440.0, 22050.0),
    (220.0, 44100.0),
]

case_labels = [f"f0={f0:.0f} Hz, fs={fs:.0f} Hz" for f0, fs in mandatory_cases]
case_label = st.selectbox("Wybierz przypadek (f0, fs)", case_labels)
case_idx = case_labels.index(case_label)
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
    "fs referencyjne aliasingu [Hz]",
    min_value=1000.0,
    value=100000.0,
    step=1000.0,
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
st.subheader("Podsumowanie: oryginał, próbkowanie i rekonstrukcja SINC")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1:
    summary_a = st.number_input(
        "Amplituda A",
        value=float(signal_params.get("A", 1.0)),
        step=0.1,
        key="summary_a",
    )
with col_p2:
    summary_fs_sample = st.number_input(
        "Częstotliwość próbkowania fs [Hz]",
        min_value=1.0,
        value=float(fs_sample_hz),
        step=10.0,
        key="summary_fs_sample",
    )
with col_p3:
    summary_t_start = st.number_input(
        "Czas początkowy t1 [s]",
        value=float(t_start),
        step=0.1,
        key="summary_t_start",
    )

summary_f = signal_params.get("f")
summary_t = signal_params.get("T")

with col_p4:
    if "T" in required_params and signal_params.get("T", 0) > 0:
        default_f0 = 1.0 / float(signal_params["T"])
        summary_f0 = st.number_input(
            "Częstotliwość funkcji f [Hz]",
            min_value=0.001,
            value=float(default_f0),
            step=0.1,
            key="summary_f0",
        )
        summary_t = 1.0 / summary_f0
    elif "f" in required_params and summary_f is not None and summary_f > 0:
        summary_f0 = st.number_input(
            "Częstotliwość funkcji f [Hz]",
            min_value=0.001,
            value=float(summary_f),
            step=0.1,
            key="summary_f0",
        )
        summary_f = summary_f0
    else:
        st.markdown("Częstotliwość funkcji: nie dotyczy")

summary_signal_params = signal_params.copy()
summary_signal_params["A"] = float(summary_a)
if summary_t is not None:
    summary_signal_params["T"] = float(summary_t)
if "f" in required_params and summary_f is not None:
    summary_signal_params["f"] = float(summary_f)

summary_params_with_globals = summary_signal_params.copy()
summary_params_with_globals["t1"] = float(summary_t_start)
summary_params_with_globals["d"] = float(duration_s)
summary_params_with_globals["fs"] = float(fs_ref_hz)

t_ref_summary, y_ref_summary = signal_config["func"](
    summary_params_with_globals
)

summary_sample_start = float(t_ref_summary[0])
summary_sample_end = float(t_ref_summary[-1])
if summary_sample_end <= summary_sample_start:
    summary_sample_end = summary_sample_start + 1.0 / float(summary_fs_sample)

t_samples_summary, y_samples_summary = su.uniform_sample(
    t_reference=t_ref_summary,
    y_reference=y_ref_summary,
    fs_hz=float(summary_fs_sample),
    t_start=summary_sample_start,
    t_end=summary_sample_end,
)

render_sinc_summary_plot(
    signal_name=selected_signal_name,
    signal_config=signal_config,
    signal_params=summary_signal_params,
    t_start=float(summary_t_start),
    duration_s=duration_s,
    t_ref=t_ref_summary,
    y_ref=y_ref_summary,
    t_samples=t_samples_summary,
    y_samples=y_samples_summary,
    fs_sample_hz=float(summary_fs_sample),
    sinc_neighbors=sinc_neighbors,
)
