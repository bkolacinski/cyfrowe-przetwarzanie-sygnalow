import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import signal_utils as su


def reconstruct_signal(t_samples, y_samples, t_target, method, sinc_neighbors):
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


st.title("Zadanie 2 - Próbkowanie, kwantyzacja i rekonstrukcja")

st.sidebar.header("Konfiguracja sygnału")
amplitude = st.sidebar.number_input("Amplituda A", value=1.0, step=0.1)
f0_hz = st.sidebar.number_input("Częstotliwość sygnału f0 [Hz]", min_value=1.0, value=10.0, step=10.0)
phase_rad = st.sidebar.number_input("Faza [rad]", value=0.0, step=0.1)
t_start = st.sidebar.number_input("Czas początkowy t1 [s]", value=0.0, step=0.1)
duration_s = st.sidebar.number_input("Czas trwania d [s]", min_value=0.05, value=1.0, step=0.05)
fs_ref_hz = st.sidebar.number_input("Referencyjne fs_ref [Hz]", min_value=100.0, value=20000.0, step=100.0)

st.sidebar.header("A/C i C/A")
fs_sample_hz = st.sidebar.number_input("fs próbkowania [Hz]", min_value=1.0, value=1000.0, step=10.0)
bits = st.sidebar.slider("Liczba bitów", min_value=1, max_value=16, value=8)
quant_mode = st.sidebar.selectbox(
	"Tryb kwantyzacji",
	options=["round", "truncate"],
	format_func=lambda x: "Zaokrąglanie" if x == "round" else "Obcięcie",
)
reconstruction_method = st.sidebar.selectbox("Metoda rekonstrukcji", ["ZOH", "FOH", "SINC"])
sinc_neighbors = st.sidebar.slider("SINC: liczba sąsiadów", min_value=4, max_value=64, value=16, step=4)

t_ref, y_ref = su.generate_sine_wave(
	frequency_hz=f0_hz,
	amplitude=amplitude,
	phase_rad=phase_rad,
	t_start=t_start,
	duration_s=duration_s,
	fs_hz=fs_ref_hz,
)

t_samples, y_samples = su.uniform_sample(
	t_reference=t_ref,
	y_reference=y_ref,
	fs_hz=fs_sample_hz,
	t_start=t_start,
	t_end=t_start + duration_s,
)

y_quantized, q_meta = su.quantize_uniform(y_samples, bits=bits, mode=quant_mode)
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

st.subheader("Przebiegi: sygnał oryginalny i rekonstrukcja")
fig_sig, ax_sig = plt.subplots(figsize=(12, 4))
ax_sig.plot(t_ref, y_ref, label="Oryginał (referencja)", linewidth=1.5)
ax_sig.plot(t_ref, y_reconstructed, label=f"Rekonstrukcja {reconstruction_method}", linewidth=1.2)
ax_sig.scatter(t_samples, y_samples, label="Próbki", color="black", s=18, zorder=3)
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
ax_q.scatter(t_samples, y_quantized, label="Próbki skwantyzowane", color="black", s=12, alpha=0.55)
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
	st.caption(f"Poziomy: {q_meta['levels']}, krok kwantyzacji: {q_meta['step']:.6f}")

st.divider()
st.subheader("Eksperyment 1: jakość a relacja fs/f0")

exp_fs_min = st.number_input("fs_min [Hz]", min_value=1.0, value=max(10.0, f0_hz), step=10.0)
exp_fs_max = st.number_input("fs_max [Hz]", min_value=exp_fs_min + 1.0, value=max(exp_fs_min + 10.0, 8.0 * f0_hz), step=10.0)
exp_fs_count = st.slider("Liczba punktów fs", min_value=4, max_value=40, value=12)

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

st.divider()
st.subheader("Eksperyment 2: kwantyzacja, teoria SNR i ENOB")

bits_min, bits_max = st.slider("Zakres bitów", min_value=1, max_value=16, value=(2, 12))
bits_values = list(range(bits_min, bits_max + 1))
quant_exp = su.run_quantization_experiment(y_samples, bits_values, mode=quant_mode)

bits_arr = np.array([row["bits"] for row in quant_exp])
snr_measured = np.array([row["snr_measured"] for row in quant_exp])
snr_theory = np.array([row["snr_theory"] for row in quant_exp])
enob_arr = np.array([row["enob"] for row in quant_exp])

fig_exp2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
ax3.plot(bits_arr, snr_measured, marker="o", label="SNR zmierzone")
ax3.plot(bits_arr, snr_theory, marker="x", linestyle="--", label="SNR teoria 6.02b+1.76")
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

case_label = st.selectbox(
	"Wybierz przypadek (f0, fs)",
	[f"f0={f0:.0f} Hz, fs={fs:.0f} Hz" for f0, fs in mandatory_cases],
)
case_idx = [f"f0={f0:.0f} Hz, fs={fs:.0f} Hz" for f0, fs in mandatory_cases].index(case_label)
case_f0, case_fs = mandatory_cases[case_idx]

k_values = st.multiselect("Wartości k", options=list(range(-4, 5)), default=[-2, -1, 1, 2])
amp_values_text = st.text_input("Amplitudy dla fd (oddzielone przecinkami)", value="0.5,1.0,1.5")
amp_values = [float(item.strip()) for item in amp_values_text.split(",") if item.strip()]

alias_duration = st.number_input("Czas prezentacji aliasingu [s]", min_value=0.01, value=0.05, step=0.01)
alias_ref_fs = st.number_input("fs referencyjne aliasingu [Hz]", min_value=1000.0, value=100000.0, step=1000.0)

t_alias_ref, y_alias_ref = su.generate_sine_wave(case_f0, 1.0, 0.0, 0.0, alias_duration, alias_ref_fs)
t_alias_s, y_alias_s = su.uniform_sample(t_alias_ref, y_alias_ref, case_fs)

rows = []
fig_alias, ax_alias = plt.subplots(figsize=(12, 4))
ax_alias.scatter(t_alias_s, y_alias_s, label=f"f0={case_f0:.0f} Hz", s=26)

for k in k_values:
	fd = su.alias_frequency(case_f0, case_fs, k)
	t_fd_ref, y_fd_ref = su.generate_sine_wave(fd, 1.0, 0.0, 0.0, alias_duration, alias_ref_fs)
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
amp_k = st.slider("Wybierz k do porównania amplitud", min_value=-5, max_value=5, value=1)
fd_amp = su.alias_frequency(case_f0, case_fs, amp_k)

fig_amp, ax_amp = plt.subplots(figsize=(12, 4))
for amp in amp_values:
	t_fd_ref, y_fd_ref = su.generate_sine_wave(fd_amp, amp, 0.0, 0.0, alias_duration, alias_ref_fs)
	t_fd_s, y_fd_s = su.uniform_sample(t_fd_ref, y_fd_ref, case_fs)
	ax_amp.scatter(t_fd_s, y_fd_s, s=18, label=f"A={amp}")

ax_amp.set_title(f"Próbki dla fd={fd_amp:.2f} Hz (f0={case_f0:.0f} Hz, fs={case_fs:.0f} Hz)")
ax_amp.set_xlabel("Czas [s]")
ax_amp.set_ylabel("Amplituda")
ax_amp.grid(True, alpha=0.3)
ax_amp.legend(loc="upper right")
fig_amp.tight_layout()
st.pyplot(fig_amp, width="stretch")
