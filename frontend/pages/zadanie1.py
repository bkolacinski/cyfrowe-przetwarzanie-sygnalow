import functions as fn
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

SIGNAL_REGISTRY = {
    "S1": {
        "name": "Szum o rozkładzie jednostajnym (S1)",
        "func": fn.generate_uniform_noise,
        "params": ["A", "t1", "d", "fs"],
    },
    "S2": {
        "name": "Szum gaussowski (S2)",
        "func": fn.generate_gaussian_noise,
        "params": ["A", "t1", "d", "fs"],
    },
    "S3": {
        "name": "Sygnał sinusoidalny (S3)",
        "func": fn.generate_sine,
        "params": ["A", "T", "t1", "d", "fs"],
    },
    "S4": {
        "name": "Sygnał sinusoidalny wyprostowany jednopołówkowo (S4)",
        "func": fn.generate_sine_wyprostowany_jednopolowkowo,
        "params": ["A", "T", "t1", "d", "fs"],
    },
    "S5": {
        "name": "Sygnał sinusoidalny wyprostowany dwupołówkowo (S5)",
        "func": fn.generate_sine_wyprostowany_dwupolowkowo,
        "params": ["A", "T", "t1", "d", "fs"],
    },
    "S6": {
        "name": "Sygnał prostokątny (S6)",
        "func": fn.generate_square,
        "params": ["A", "T", "t1", "d", "kw", "fs"],
    },
    "S7": {
        "name": "Sygnał prostokątny symetryczny (S7)",
        "func": fn.generate_symmetric_square,
        "params": ["A", "T", "t1", "d", "kw", "fs"],
    },
    "S8": {
        "name": "Sygnał trójkątny (S8)",
        "func": fn.generate_triangle,
        "params": ["A", "T", "t1", "d", "kw", "fs"],
    },
    "S9": {
        "name": "Skok jednostkowy (S9)",
        "func": fn.unit_step,
        "params": ["A", "t1", "d", "ts", "fs"],
    },
    "S10": {
        "name": "Impuls jednostkowy (S10)",
        "func": lambda x: (np.array([0]), np.array([0])),
        "params": ["A", "ns", "n1", "l", "fs"],
    },
    "S11": {
        "name": "Szum impulsowy (S11)",
        "func": lambda x: (np.array([0]), np.array([0])),
        "params": ["A", "t1", "d", "f", "p", "fs"],
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
    pass


def load_signal(file_buffer):
    pass


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

st.sidebar.header("Ustawienia globalne")
st.session_state.global_fs = st.sidebar.number_input(
    "Częstotliwość próbkowania (Hz)",
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

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Wykres wartości od czasu")
    fig, ax = plt.subplots(figsize=(8, 4))

    if result_t is not None and result_y is not None:
        combined_label = " ".join(signal_labels)
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

    st.pyplot(fig)

with col2:
    st.subheader("Histogram")

    fig_hist, ax_hist = plt.subplots(figsize=(4, 4))

    if result_y is not None:
        counts, edges, patches = ax_hist.hist(
            result_y,
            bins=st.session_state.histogram_bins,
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

    plt.tight_layout()

    st.pyplot(fig_hist)

    st.session_state.histogram_bins = st.select_slider(
        "Liczba przedziałów histogramu",
        options=np.linspace(5, 20, num=16, dtype=int),
        value=st.session_state.histogram_bins,
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
    if uploaded_file:
        load_signal(uploaded_file)

with c2:
    st.markdown("### Parametry wyliczone")
    if result_y is not None:
        st.text(f"Liczba próbek: {len(result_y)}")
        st.text(f"Min: {np.min(result_y):.2f}")
        st.text(f"Max: {np.max(result_y):.2f}")
    else:
        st.text("Brak danych")
