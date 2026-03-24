import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import functions as fn
import signal_io

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


def signal_add(y1, y2):
    min_len = min(len(y1), len(y2))
    return y1[:min_len] + y2[:min_len]


def signal_subtract(y1, y2):
    min_len = min(len(y1), len(y2))
    return y1[:min_len] - y2[:min_len]


def signal_multiply(y1, y2):
    min_len = min(len(y1), len(y2))
    return y1[:min_len] * y2[:min_len]


def signal_divide(y1, y2):
    min_len = min(len(y1), len(y2))
    y2_safe = np.where(y2[:min_len] == 0, 1e-10, y2[:min_len])
    return y1[:min_len] / y2_safe


def save_signal(t, y, metadata):
    return signal_io.save_signal(t, y, metadata)


def load_signal(file_buffer):
    return signal_io.load_signal(file_buffer)


def get_analysis_data(x_vals, y_vals, signals):
    info = {
        "used_full_periods": False,
        "full_periods": None,
        "period": 0,
        "reason": "",
    }

    if x_vals is None or y_vals is None or len(y_vals) == 0:
        info["reason"] = "Brak danych wejściowych."
        return x_vals, y_vals, info

    active_signals = [signal for signal in signals if "data" in signal]
    if len(active_signals) != 1:
        info["reason"] = (
            "Dla sygnału złożonego statystyki i histogram liczone są z całego zakresu."
        )
        return x_vals, y_vals, info

    signal_data = active_signals[0]
    signal_config = SIGNAL_REGISTRY.get(signal_data.get("type"), {})

    if not (
        signal_config.get("is_periodic", False)
        and signal_config.get("is_continuous", False)
    ):
        info["reason"] = "Sygnał nie jest jednocześnie okresowy i ciągły."
        return x_vals, y_vals, info

    period = signal_data.get("params", {}).get("T")
    if period is None or period <= 0:
        info["reason"] = "Brak poprawnego okresu T."
        return x_vals, y_vals, info

    duration = float(x_vals[-1] - x_vals[0])
    full_periods = int(np.floor(duration / period + 1e-12))

    if full_periods <= 0:
        info["period"] = period
        info["full_periods"] = 0
        info["reason"] = "W wybranym zakresie czasu nie ma pełnego okresu."
        return x_vals[:0], y_vals[:0], info

    end_time = float(x_vals[0] + full_periods * period)
    mask = x_vals <= (end_time + 1e-12)
    t_for_analysis = x_vals[mask]
    y_for_analysis = y_vals[mask]

    info["used_full_periods"] = True
    info["full_periods"] = full_periods
    info["period"] = period
    return t_for_analysis, y_for_analysis, info


def calculate_signal_metrics(signal):
    y = np.asarray(signal)
    power = np.mean(np.square(y))
    return {
        "mean": np.mean(y),
        "mean_abs": np.mean(np.abs(y)),
        "rms": np.sqrt(power),
        "variance": np.var(y),
        "power": power,
    }


def render_signal_params(signal_idx, signal_data):
    signal_keys = list(SIGNAL_REGISTRY.keys())
    signal_names = [SIGNAL_REGISTRY[k]["name"] for k in signal_keys]

    current_type = signal_data.get("type", "S1")
    current_idx = (
        signal_keys.index(current_type) if current_type in signal_keys else 0
    )

    selected_signal_name = st.sidebar.selectbox(
        "Wybierz sygnał/szum",
        signal_names,
        index=current_idx,
        key=f"signal_select_{signal_idx}",
    )

    selected_key = signal_keys[signal_names.index(selected_signal_name)]
    current_signal_config = SIGNAL_REGISTRY[selected_key]
    required_params = current_signal_config["params"]

    params = {}

    if "A" in required_params:
        params["A"] = st.sidebar.number_input(
            "Amplituda (A)",
            value=signal_data.get("params", {}).get("A", 1.0),
            step=0.1,
            key=f"A_{signal_idx}",
        )

    if "T" in required_params:
        params["T"] = st.sidebar.number_input(
            "Okres podstawowy (T)",
            min_value=0.1,
            value=signal_data.get("params", {}).get("T", 1.0),
            step=0.1,
            key=f"T_{signal_idx}",
        )

    if "kw" in required_params:
        params["kw"] = st.sidebar.slider(
            "Współczynnik wypełnienia (kw)",
            0.0,
            1.0,
            signal_data.get("params", {}).get("kw", 0.5),
            key=f"kw_{signal_idx}",
        )

    if "ts" in required_params:
        params["ts"] = st.sidebar.number_input(
            "Punkt skoku (ts)",
            min_value=st.session_state.global_t1,
            max_value=st.session_state.global_t1 + st.session_state.global_d,
            value=signal_data.get("params", {}).get("ts", 0.5),
            step=0.1,
            key=f"ts_{signal_idx}",
        )

    if "ns" in required_params:
        params["ns"] = st.sidebar.number_input(
            "Numer próbki impulsu (ns)",
            value=int(signal_data.get("params", {}).get("ns", 0)),
            step=1,
            key=f"ns_{signal_idx}",
        )

    if "n1" in required_params:
        params["n1"] = st.sidebar.number_input(
            "Numer pierwszej próbki (n1)",
            value=int(signal_data.get("params", {}).get("n1", 0)),
            step=1,
            key=f"n1_{signal_idx}",
        )

    if "l" in required_params:
        params["l"] = st.sidebar.number_input(
            "Liczba próbek (l)",
            min_value=1,
            value=int(signal_data.get("params", {}).get("l", 100)),
            step=1,
            key=f"l_{signal_idx}",
        )

    if "f" in required_params:
        params["f"] = st.sidebar.number_input(
            "Częstotliwość próbkowania dyskretnego (f)",
            min_value=1.0,
            value=float(
                signal_data.get("params", {}).get(
                    "f", st.session_state.global_fs
                )
            ),
            step=10.0,
            key=f"f_{signal_idx}",
        )

    if "p" in required_params:
        params["p"] = st.sidebar.slider(
            "Prawdopodobieństwo impulsu (p)",
            0.0,
            1.0,
            float(signal_data.get("params", {}).get("p", 0.1)),
            key=f"p_{signal_idx}",
        )

    return selected_key, params, current_signal_config


st.title("Zadanie 1 - Generacja sygnału i szumu")

if "signals" not in st.session_state:
    st.session_state.signals = [
        {"type": "S1", "params": {}, "operation": None}
    ]

if "histogram_bins" not in st.session_state:
    st.session_state.histogram_bins = 10

if "global_fs" not in st.session_state:
    st.session_state.global_fs = 1000.0

if "prev_global_fs" not in st.session_state:
    st.session_state.prev_global_fs = 1000.0

if "global_t1" not in st.session_state:
    st.session_state.global_t1 = 0.0

if "prev_global_t1" not in st.session_state:
    st.session_state.prev_global_t1 = 0.0

if "global_d" not in st.session_state:
    st.session_state.global_d = 5.0

if "prev_global_d" not in st.session_state:
    st.session_state.prev_global_d = 5.0

if "last_loaded_file_id" not in st.session_state:
    st.session_state.last_loaded_file_id = None

st.sidebar.header("Ustawienia globalne")
st.session_state.global_fs = st.sidebar.number_input(
    "Częstotliwość próbkowania (fs)",
    min_value=1.0,
    value=st.session_state.global_fs,
    step=10.0,
    key="global_fs_input",
)

st.session_state.global_t1 = st.sidebar.number_input(
    "Czas początkowy (t1)",
    value=st.session_state.global_t1,
    step=0.1,
    key="global_t1_input",
)

st.session_state.global_d = st.sidebar.number_input(
    "Czas trwania (d)",
    min_value=0.1,
    value=st.session_state.global_d,
    step=0.5,
    key="global_d_input",
)

fs_changed = st.session_state.prev_global_fs != st.session_state.global_fs
t1_changed = st.session_state.prev_global_t1 != st.session_state.global_t1
d_changed = st.session_state.prev_global_d != st.session_state.global_d
global_params_changed = fs_changed or t1_changed or d_changed

if st.sidebar.button("Reset", use_container_width=True):
    for key in list(st.session_state.keys()):
        if (
            key.startswith("signal_select_")
            or key.startswith("A_")
            or key.startswith("T_")
            or key.startswith("kw_")
            or key.startswith("ts_")
            or key.startswith("ns_")
            or key.startswith("n1_")
            or key.startswith("l_")
            or key.startswith("f_")
            or key.startswith("p_")
        ):
            del st.session_state[key]
    st.session_state.signals = [
        {"type": "S1", "params": {}, "operation": None}
    ]
    st.session_state.global_fs = 100.0
    st.session_state.prev_global_fs = 100.0
    st.session_state.global_t1 = 0.0
    st.session_state.prev_global_t1 = 0.0
    st.session_state.global_d = 5.0
    st.session_state.prev_global_d = 5.0
    st.session_state.histogram_bins = 10
    st.rerun()

st.sidebar.divider()

for i, signal_data in enumerate(st.session_state.signals):
    st.sidebar.header(f"Parametry Sygnału {i + 1}")

    if i > 0:
        operation_names = {
            "add": "Dodawanie",
            "subtract": "Odejmowanie",
            "multiply": "Mnożenie",
            "divide": "Dzielenie",
        }
        current_op = signal_data.get("operation", "add")
        st.sidebar.caption(
            f"Operacja: {operation_names.get(current_op, 'Brak')}"
        )

    if signal_data.get("type") == "LOADED":
        st.sidebar.info("Wczytany sygnał z pliku")
        if "data" in signal_data:
            t_l, y_l = signal_data["data"]
            st.sidebar.caption(
                f"Próbki: {len(y_l)}, zakres: {t_l[0]:.3f} – {t_l[-1]:.3f} s"
            )
        if i > 0:
            if st.sidebar.button(f"Usuń sygnał {i + 1}", key=f"remove_{i}"):
                st.session_state.signals.pop(i)
                st.rerun()
        st.sidebar.divider()
        continue

    selected_key, params, config = render_signal_params(i, signal_data)

    needs_update = (
        signal_data.get("type") != selected_key
        or signal_data.get("params") != params
        or global_params_changed
    )

    if needs_update:
        generator_func = config["func"]
        params_with_globals = params.copy()
        params_with_globals["fs"] = st.session_state.global_fs
        params_with_globals["t1"] = st.session_state.global_t1
        params_with_globals["d"] = st.session_state.global_d
        t, y = generator_func(params_with_globals)
        st.session_state.signals[i]["type"] = selected_key
        st.session_state.signals[i]["params"] = params.copy()
        st.session_state.signals[i]["data"] = (t, y)

if global_params_changed:
    st.session_state.prev_global_fs = st.session_state.global_fs
    st.session_state.prev_global_t1 = st.session_state.global_t1
    st.session_state.prev_global_d = st.session_state.global_d

    if i > 0:
        if st.sidebar.button(f"Usuń sygnał {i + 1}", key=f"remove_{i}"):
            st.session_state.signals.pop(i)
            st.rerun()

    st.sidebar.divider()


result_t = None
result_y = None
signal_labels = []

for i, signal_data in enumerate(st.session_state.signals):
    if "data" not in signal_data:
        continue

    t, y = signal_data["data"]
    signal_type = signal_data["type"]

    # Handle loaded signals
    if signal_type == "LOADED":
        signal_name = "Wczytany sygnał"
    else:
        signal_name = SIGNAL_REGISTRY[signal_type]["name"]

    if i == 0:
        result_t = t
        result_y = y
        signal_labels.append(signal_name)
    else:
        operation = signal_data.get("operation", "add")
        if operation == "add":
            result_y = signal_add(result_y, y)
            signal_labels.append(f"+ {signal_name}")
        elif operation == "subtract":
            result_y = signal_subtract(result_y, y)
            signal_labels.append(f"- {signal_name}")
        elif operation == "multiply":
            result_y = signal_multiply(result_y, y)
            signal_labels.append(f"× {signal_name}")
        elif operation == "divide":
            result_y = signal_divide(result_y, y)
            signal_labels.append(f"÷ {signal_name}")

        if result_t is not None and result_y is not None:
            result_t = result_t[: len(result_y)]

analysis_t, analysis_y, analysis_info = get_analysis_data(
    result_t, result_y, st.session_state.signals
)

active_signals = [
    signal for signal in st.session_state.signals if "data" in signal
]
only_discrete_signals = bool(active_signals) and all(
    signal["type"] in SIGNAL_REGISTRY
    and not SIGNAL_REGISTRY[signal["type"]].get("is_continuous", True)
    for signal in active_signals
)

plot_height = 4.0
signal_figsize = (8, plot_height)
hist_figsize = (4, plot_height)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Wykres wartości od czasu")
    fig, ax = plt.subplots(figsize=signal_figsize)

    if result_t is not None and result_y is not None:
        combined_label = " ".join(signal_labels)
        if only_discrete_signals:
            ax.scatter(
                result_t,
                result_y,
                label=combined_label,
                color="royalblue",
                s=16,
            )
        else:
            ax.plot(
                result_t,
                result_y,
                label=combined_label,
                color="royalblue",
                linewidth=1.5,
            )

    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Wartość")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")

with col2:
    st.subheader("Histogram")

    fig_hist, ax_hist = plt.subplots(figsize=hist_figsize)
    current_hist_bins = st.session_state.histogram_bins

    if analysis_y is not None and len(analysis_y) > 0:
        counts, edges, patches = ax_hist.hist(
            analysis_y,
            bins=current_hist_bins,
            color="skyblue",
            edgecolor="black",
        )
        ax_hist.set_xlabel("Wartość")
        ax_hist.set_ylabel("Liczebność")
        ax_hist.grid(True, alpha=0.3)

        num_bins = len(edges) - 1
        if num_bins <= 7:
            tick_positions = edges
        else:
            step = max(1, num_bins // 7)
            tick_indices = list(range(0, len(edges), step))
            if tick_indices[-1] != len(edges) - 1:
                tick_indices.append(len(edges) - 1)
            tick_positions = edges[tick_indices]

        ax_hist.set_xticks(tick_positions)
        ax_hist.set_xticklabels(
            [f"{val:.2f}" for val in tick_positions], rotation=45, ha="right"
        )
    else:
        ax_hist.text(
            0.5,
            0.5,
            "Brak danych\ndo histogramu",
            ha="center",
            va="center",
            transform=ax_hist.transAxes,
        )
        ax_hist.set_xlabel("Wartość")
        ax_hist.set_ylabel("Liczebność")

    fig_hist.tight_layout()
    st.pyplot(fig_hist, width="stretch")

    st.select_slider(
        "Liczba przedziałów histogramu",
        options=np.linspace(5, 20, num=16, dtype=int),
        key="histogram_bins",
    )

st.divider()

st.subheader("Działania na sygnałach")

op_col1, op_col2, op_col3, op_col4 = st.columns(4)

with op_col1:
    if st.button("Dodaj sygnał", use_container_width=True):
        st.session_state.signals.append(
            {"type": "S1", "params": {}, "operation": "add"}
        )
        st.rerun()

with op_col2:
    if st.button("Odejmij sygnał", use_container_width=True):
        st.session_state.signals.append(
            {"type": "S1", "params": {}, "operation": "subtract"}
        )
        st.rerun()

with op_col3:
    if st.button("Pomnóż sygnał", use_container_width=True):
        st.session_state.signals.append(
            {"type": "S1", "params": {}, "operation": "multiply"}
        )
        st.rerun()

with op_col4:
    if st.button("Podziel sygnał", use_container_width=True):
        st.session_state.signals.append(
            {"type": "S1", "params": {}, "operation": "divide"}
        )
        st.rerun()

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Zapis / Odczyt")
    if st.button("Zapisz sygnał do pliku (BIN)"):
        if result_t is not None and result_y is not None:
            save_signal(result_t, result_y, {})

    uploaded_file = st.file_uploader("Wczytaj sygnał", type="bin")
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_loaded_file_id != file_id:
            result = load_signal(uploaded_file)
            if result and result[0] is not None:
                st.session_state.last_loaded_file_id = file_id
                st.rerun()
            else:
                st.error("❌ Nie udało się wczytać danych z pliku")

    if (
        st.session_state.signals
        and st.session_state.signals[0].get("type") == "LOADED"
    ):
        loaded_signal = st.session_state.signals[0]
        if "data" in loaded_signal:
            t_loaded, y_loaded = loaded_signal["data"]
            st.success(f"Sygnał wczytany pomyślnie!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Liczba próbek", len(y_loaded))
                st.metric(
                    "Zakres czasowy", f"{t_loaded[0]:.3f} - {t_loaded[-1]:.3f}"
                )
            with col2:
                st.metric("Min wartość", f"{y_loaded.min():.3f}")
                st.metric("Max wartość", f"{y_loaded.max():.3f}")

with c2:
    st.markdown("### Parametry wyliczone")
    if analysis_y is not None and len(analysis_y) > 0:
        metrics = calculate_signal_metrics(analysis_y)
        st.text(f"Liczba próbek do obliczeń: {len(analysis_y)}")
        st.text(f"Wartość średnia: {metrics['mean']:.6f}")
        st.text(f"Wartość średnia bezwzględna: {metrics['mean_abs']:.6f}")
        st.text(f"Wartość skuteczna (RMS): {metrics['rms']:.6f}")
        st.text(f"Wariancja: {metrics['variance']:.6f}")
        st.text(f"Moc średnia: {metrics['power']:.6f}")

        if analysis_info["used_full_periods"]:
            st.caption(
                "Do obliczeń i histogramu użyto "
                f"{analysis_info['full_periods']} pełnych okresów (T={analysis_info['period']})."
            )
        elif analysis_info["reason"]:
            st.caption(analysis_info["reason"])
    else:
        st.text("Brak danych do obliczeń")
        if analysis_info["reason"]:
            st.caption(analysis_info["reason"])
