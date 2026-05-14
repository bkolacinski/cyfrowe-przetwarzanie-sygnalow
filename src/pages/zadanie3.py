import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import dsp_logic as dsp
import signal_factory as sf


def _build_signal(name: str, prefix: str):
    st.markdown(f"**{prefix}: {name}**")
    common = {
        "t1": st.number_input(
            f"{prefix} - t1 [s]", value=0.0, step=0.1, key=f"{prefix}_t1"
        ),
        "d": st.number_input(
            f"{prefix} - d [s]",
            min_value=0.05,
            value=1.0,
            step=0.05,
            key=f"{prefix}_d",
        ),
        "fs": st.number_input(
            f"{prefix} - fs [Hz]",
            min_value=10.0,
            value=1000.0,
            step=10.0,
            key=f"{prefix}_fs",
        ),
    }

    params = {}
    config = sf.SIGNAL_REGISTRY[name]

    if "A" in config["params"]:
        params["A"] = st.number_input(
            f"{prefix} - A", value=1.0, step=0.1, key=f"{prefix}_A"
        )
    if "T" in config["params"]:
        params["T"] = st.number_input(
            f"{prefix} - T [s]",
            min_value=0.01,
            value=0.2,
            step=0.01,
            key=f"{prefix}_T",
        )
    if "kw" in config["params"]:
        params["kw"] = st.slider(
            f"{prefix} - kw", 0.05, 0.95, 0.5, key=f"{prefix}_kw"
        )

    t, y = sf.generate_named_signal(name, common, params)
    return t, y, common


def module_convolution_and_correlation():
    st.header("Moduł I: Splot i Korelacja")

    c1, c2 = st.columns(2)
    signal_names = list(sf.SIGNAL_REGISTRY.keys())

    with c1:
        x_name = st.selectbox(
            "Sygnał x(n)", signal_names, index=0, key="m1_x_name"
        )
        tx, x, _ = _build_signal(x_name, "x")

    with c2:
        h_name = st.selectbox(
            "Sygnał h(n)", signal_names, index=1, key="m1_h_name"
        )
        th, h, _ = _build_signal(h_name, "h")

    add_noise = st.checkbox("Add Noise", value=True, key="m1_noise_enabled")
    noise_level = st.slider(
        "Noise Level", 0.0, 1.0, 0.05, 0.01, key="m1_noise_level"
    )

    x_n = sf.add_noise(x, add_noise, noise_level)
    h_n = sf.add_noise(h, add_noise, noise_level)

    conv = dsp.discrete_convolution(h_n, x_n)
    corr_direct, lags = dsp.cross_correlation_direct(h_n, x_n)
    corr_conv, _ = dsp.cross_correlation_via_convolution(h_n, x_n)

    idx_direct = np.arange(len(corr_direct))
    idx_conv = np.arange(len(corr_conv))

    st.subheader("Wyniki obliczeń")
    st.write(
        f"Długość x: {len(x_n)}, długość h: {len(h_n)}, długość splotu: {len(conv)}"
    )
    st.write(
        "Korelacja jest przeindeksowana: tablice wynikowe startują od indeksu 0, a wektor lagów jest udostępniony osobno."
    )
    st.write(
        f"Maks. różnica |R_direct - R_conv|: {np.max(np.abs(corr_direct - corr_conv)):.3e}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(tx, x_n, label="x(n)")
    axes[0, 0].set_title("Sygnał x(n)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(th, h_n, label="h(n)", color="tab:orange")
    axes[0, 1].set_title("Sygnał h(n)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(np.arange(len(conv)), conv, color="tab:green")
    axes[1, 0].set_title("Splot dyskretny (h*x)(n)")
    axes[1, 0].set_xlabel("n")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(idx_direct, corr_direct, label="Korelacja bezpośrednia")
    axes[1, 1].plot(idx_conv, corr_conv, "--", label="Korelacja przez splot")
    axes[1, 1].set_title("Korelacja wzajemna (indeksowane od 0)")
    axes[1, 1].set_xlabel("indeks tablicy")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig, width="stretch")

    st.subheader("Mapa indeks tablicy -> lag")
    preview_len = min(15, len(lags))
    st.dataframe(
        {
            "indeks_tablicy": np.arange(preview_len),
            "lag": lags[:preview_len],
            "R_direct": corr_direct[:preview_len],
            "R_conv": corr_conv[:preview_len],
        },
        use_container_width=True,
    )


def module_fir_filtering():
    st.header("Moduł II: Filtry FIR (SOI) i filtracja")

    signal_names = list(sf.SIGNAL_REGISTRY.keys())
    in_name = st.selectbox(
        "Sygnał wejściowy", signal_names, index=0, key="m2_in_name"
    )
    t_in, x_in, common = _build_signal(in_name, "input")

    add_noise = st.checkbox("Add Noise", value=True, key="m2_noise_enabled")
    noise_level = st.slider(
        "Noise Level", 0.0, 1.0, 0.05, 0.01, key="m2_noise_level"
    )
    x_proc = sf.add_noise(x_in, add_noise, noise_level)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        M = st.slider(
            "Rząd filtru M", min_value=11, max_value=301, value=61, step=2
        )
    with c2:
        f0 = st.number_input(
            "Częstotliwość odcięcia f0 [Hz]",
            min_value=1.0,
            value=120.0,
            step=1.0,
        )
    with c3:
        window_name = st.selectbox(
            "Okno",
            ["prostokątne", "hamminga", "hanninga", "blackmana"],
            index=1,
        )
    with c4:
        filter_type = st.selectbox(
            "Typ filtru",
            ["dolnoprzepustowy", "środkowoprzepustowy", "górnoprzepustowy"],
            index=0,
        )

    fs_hz = common["fs"]
    h_lp, h_ideal, win = dsp.design_lowpass_fir(
        M=M, f0_hz=f0, fs_hz=fs_hz, window_type=window_name
    )

    if filter_type == "dolnoprzepustowy":
        h = h_lp
        modulation = np.ones_like(h)
    elif filter_type == "środkowoprzepustowy":
        h, modulation = dsp.transform_to_bandpass(h_lp)
    else:
        h, modulation = dsp.transform_to_highpass(h_lp)

    y_out = dsp.filter_signal_with_fir(x_proc, h)

    f, mag, mag_db = dsp.magnitude_response_db(h, fs_hz=fs_hz)

    fig1, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(t_in, x_proc)
    axes[0, 0].set_title("Sygnał wejściowy (z opcjonalnym szumem)")
    axes[0, 0].set_xlabel("t [s]")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(np.arange(len(h)), h, label="h(n)")
    axes[0, 1].plot(np.arange(len(h_ideal)), h_ideal, "--", label="h_ideal(n)")
    axes[0, 1].plot(np.arange(len(win)), win, ":", label="w(n)")
    axes[0, 1].set_title("Odpowiedź impulsowa i okno")
    axes[0, 1].set_xlabel("n")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(np.arange(len(y_out)), y_out, color="tab:green")
    axes[1, 0].set_title("Sygnał po filtracji (realizacja przez splot)")
    axes[1, 0].set_xlabel("n")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(np.arange(len(modulation)), modulation, color="tab:purple")
    axes[1, 1].set_title("Sygnał modulujący s(n) użyty do transformacji")
    axes[1, 1].set_xlabel("n")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig1, width="stretch")

    st.subheader("Moduł transmitancji filtru")
    fr1, fr2 = st.columns(2)

    with fr1:
        fig_lin, ax_lin = plt.subplots(figsize=(6, 3.5))
        ax_lin.plot(f, mag)
        ax_lin.set_title("|H(f)| - skala liniowa")
        ax_lin.set_xlabel("f [Hz]")
        ax_lin.set_ylabel("|H(f)|")
        ax_lin.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_lin, width="stretch")

    with fr2:
        fig_db, ax_db = plt.subplots(figsize=(6, 3.5))
        ax_db.plot(f, mag_db)
        ax_db.set_title("|H(f)| [dB] - skala logarytmiczna")
        ax_db.set_xlabel("f [Hz]")
        ax_db.set_ylabel("20log10|H(f)| [dB]")
        ax_db.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_db, width="stretch")


def module_distance_sensor():
    st.header("Moduł III: Symulacja korelacyjnego czujnika odległości")

    c1, c2, c3 = st.columns(3)
    with c1:
        unit_time = st.number_input(
            "Jednostka czasowa Δt [s]", min_value=0.001, value=0.1, step=0.001
        )
        simulation_step = st.number_input(
            "Krok symulacji k", min_value=0, value=10, step=1
        )
        object_velocity = st.number_input(
            "Prędkość obiektu [m/s]", value=1.0, step=0.1
        )
    with c2:
        initial_distance = st.number_input(
            "Początkowa odległość S0 [m]", min_value=0.1, value=15.0, step=0.1
        )
        wave_speed = st.number_input(
            "Prędkość fali w ośrodku V [m/s]",
            min_value=1.0,
            value=340.0,
            step=1.0,
        )
        fs_hz = st.number_input(
            "Częstotliwość próbkowania fs [Hz]",
            min_value=100.0,
            value=8000.0,
            step=100.0,
        )
    with c3:
        tx_len = st.slider(
            "Długość bufora sygnału sondującego",
            min_value=128,
            max_value=4096,
            value=1024,
            step=64,
        )
        rx_len = st.slider(
            "Długość bufora odbiornika",
            min_value=128,
            max_value=8192,
            value=2048,
            step=64,
        )
        attenuation = st.slider(
            "Tłumienie echa",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
        )

    f1 = st.number_input(
        "Składowa 1: f1 [Hz]", min_value=10.0, value=410.0, step=10.0
    )
    f2 = st.number_input(
        "Składowa 2: f2 [Hz]", min_value=10.0, value=870.0, step=10.0
    )

    add_noise = st.checkbox("Add Noise", value=True, key="m3_noise_enabled")
    noise_level = st.slider(
        "Noise Level", 0.0, 1.0, 0.08, 0.01, key="m3_noise_level"
    )

    current_time = simulation_step * unit_time
    true_distance = initial_distance + object_velocity * current_time
    true_delay_s = (2.0 * true_distance) / wave_speed
    delay_samples = int(round(true_delay_s * fs_hz))

    _, probe_one = sf.generate_two_tone_probe(
        fs_hz=fs_hz, length=tx_len, f1_hz=f1, f2_hz=f2
    )

    repeats = int(np.ceil(rx_len / tx_len))
    probe_periodic = np.tile(probe_one, repeats)[:rx_len]

    echo = (
        sf.shift_signal(probe_periodic, delay_samples=delay_samples)
        * attenuation
    )
    echo = sf.add_noise(echo, enabled=add_noise, noise_level=noise_level)

    corr, lags = dsp.cross_correlation_direct(echo, probe_periodic)
    estimate = dsp.estimate_delay_and_distance(
        corr, lags, fs_hz=fs_hz, wave_speed=wave_speed
    )

    est_delay = max(estimate["delay_s"], 0.0)
    est_distance = max(estimate["distance_m"], 0.0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rzeczywista odległość [m]", f"{true_distance:.3f}")
    m2.metric("Estymowana odległość [m]", f"{est_distance:.3f}")
    m3.metric("Rzeczywiste opóźnienie [s]", f"{true_delay_s:.6f}")
    m4.metric("Estymowane opóźnienie [s]", f"{est_delay:.6f}")

    st.caption(
        f"Wykryty lag maksimum korelacji: {estimate['peak_lag']} próbek (index tablicy: {estimate['peak_index']})"
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

    axes[0].plot(np.arange(len(probe_periodic)), probe_periodic)
    axes[0].set_title("Sygnał oryginalny (okresowy, wieloskładowy)")
    axes[0].set_xlabel("n")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(len(echo)), echo, color="tab:orange")
    axes[1].set_title("Sygnał opóźniony i zaszumiony")
    axes[1].set_xlabel("n")
    axes[1].grid(True, alpha=0.3)

    corr_idx = np.arange(len(corr))
    axes[2].plot(corr_idx, corr, color="tab:green", label="R_yx")
    axes[2].scatter(
        [estimate["peak_index"]],
        [estimate["peak_value"]],
        color="red",
        zorder=5,
        label="Maksimum",
    )
    axes[2].set_title("Korelacja wzajemna z zaznaczonym maksimum")
    axes[2].set_xlabel("indeks tablicy korelacji")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    st.pyplot(fig, width="stretch")


st.title("Zadanie 3 - Splot, FIR i czujnik korelacyjny")
module_choice = st.sidebar.radio(
    "Wybierz moduł",
    [
        "I - Splot i Korelacja",
        "II - Filtry FIR",
        "III - Czujnik odległości",
    ],
)

if module_choice.startswith("I -"):
    module_convolution_and_correlation()
elif module_choice.startswith("II -"):
    module_fir_filtering()
else:
    module_distance_sensor()
