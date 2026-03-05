import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def get_time_vector(t1, d, fs):
    num_samples = int(d * fs)
    t = np.linspace(t1, t1 + d, num_samples, endpoint=False)
    return t


def generate_sine(params):
    A = params["A"]
    T = params["T"]
    t1 = params["t1"]
    d = params["d"]
    fs = params["fs"]

    t = get_time_vector(t1, d, fs)
    y = A * np.sin((2 * np.pi / T) * (t - t1))

    return t, y


def generate_uniform_noise(params):
    A = params["A"]
    t1 = params["t1"]
    d = params["d"]
    fs = params["fs"]

    t = get_time_vector(t1, d, fs)
    y = A * np.random.uniform(-1, 1, len(t))

    return t, y


SIGNAL_REGISTRY = {
    "S1": {
        "name": "Szum o rozkładzie jednostajnym (S1)",
        "func": generate_uniform_noise,
        "params": ["A", "t1", "d", "fs"],
    },
    "S2": {
        "name": "Szum gaussowski (S2)",
        "func": lambda x: (np.array([0]), np.array([0])),
        "params": ["A", "t1", "d", "fs"],
    },
    "S3": {
        "name": "Sygnał sinusoidalny (S3)",
        "func": generate_sine,
        "params": ["A", "T", "t1", "d", "fs"],
    },
    "S4": {
        "name": "Sygnał sinusoidalny wyprostowany jednopołówkowo (S4)",
        "func": lambda x: (np.array([0]), np.array([0])),
        "params": ["A", "T", "t1", "d", "fs"],
    },
    "S5": {
        "name": "Sygnał sinusoidalny wyprostowany dwupołówkowo (S5)",
        "func": lambda x: (np.array([0]), np.array([0])),
        "params": ["A", "T", "t1", "d", "fs"],
    },
    "S6": {
        "name": "Sygnał prostokątny (S6)",
        "func": lambda x: (np.array([0]), np.array([0])),
        "params": ["A", "T", "t1", "d", "kw", "fs"],
    },
    "S7": {
        "name": "Sygnał prostokątny symetryczny (S7)",
        "func": lambda x: (np.array([0]), np.array([0])),
        "params": ["A", "T", "t1", "d", "kw", "fs"],
    },
    "S8": {
        "name": "Sygnał trójkątny (S8)",
        "func": lambda x: (np.array([0]), np.array([0])),
        "params": ["A", "T", "t1", "d", "kw", "fs"],
    },
    "S9": {
        "name": "Skok jednostkowy (S9)",
        "func": lambda x: (np.array([0]), np.array([0])),
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


def signal_add(s1, s2):
    pass


def signal_subtract(s1, s2):
    pass


def signal_multiply(s1, s2):
    pass


def signal_divide(s1, s2):
    pass


def save_signal(t, y, metadata):
    pass


def load_signal(file_buffer):
    pass


st.title("Zadanie 1 - Generacja sygnału i szumu")

st.sidebar.header("Parametry Sygnału")

signal_keys = list(SIGNAL_REGISTRY.keys())
signal_names = [SIGNAL_REGISTRY[k]["name"] for k in signal_keys]

selected_signal_name = st.sidebar.selectbox("Wybierz sygnał/szum", signal_names)

selected_key = signal_keys[signal_names.index(selected_signal_name)]
current_signal_config = SIGNAL_REGISTRY[selected_key]
required_params = current_signal_config["params"]

params = {}

st.sidebar.subheader("Ustawienia")

if "fs" in required_params:
    params["fs"] = st.sidebar.number_input(
        "Częstotliwość próbkowania (Hz)", min_value=1.0, value=100.0, step=10.0
    )

if "A" in required_params:
    params["A"] = st.sidebar.number_input("Amplituda (A)", value=1.0, step=0.1)

if "T" in required_params:
    params["T"] = st.sidebar.number_input(
        "Okres podstawowy (T)", min_value=0.1, value=1.0, step=0.1
    )

if "t1" in required_params:
    params["t1"] = st.sidebar.number_input("Czas początkowy (t1)", value=0.0, step=0.1)

if "d" in required_params:
    params["d"] = st.sidebar.number_input(
        "Czas trwania (d)", min_value=0.1, value=5.0, step=0.5
    )

if "kw" in required_params:
    params["kw"] = st.sidebar.slider("Współczynnik wypełnienia (kw)", 0.0, 1.0, 0.5)

if "histogram_bins" not in st.session_state:
    st.session_state.histogram_bins = 10

if (
    "signal_params" not in st.session_state
    or st.session_state.get("signal_params") != params
    or st.session_state.get("signal_type") != selected_key
):
    generator_func = current_signal_config["func"]
    t, y = generator_func(params)
    st.session_state.signal_data = (t, y)
    st.session_state.signal_params = params.copy()
    st.session_state.signal_type = selected_key
else:
    t, y = st.session_state.signal_data

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Wykres wartości od czasu")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, y, label=selected_signal_name, color="royalblue", linewidth=1.5)
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Wartość")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper left")
    ax.axhline(0, color="black", linewidth=0.8)

    st.pyplot(fig)

with col2:
    st.subheader("Histogram")

    fig_hist, ax_hist = plt.subplots(figsize=(4, 4))

    counts, edges, patches = ax_hist.hist(
        y, bins=st.session_state.histogram_bins, color="skyblue", edgecolor="black"
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

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### Zapis / Odczyt")
    st.markdown("Zapisz sygnał")
    if st.button("Zapisz sygnał do pliku (BIN)"):
        save_signal(t, y, params)

    uploaded_file = st.file_uploader("Wczytaj sygnał", type="bin")
    if uploaded_file:
        load_signal(uploaded_file)

with c2:
    st.markdown("### Działania na sygnałach")
    st.caption("Funkcjonalność w przygotowaniu (wymaga wczytania drugiego sygnału)")
    op_col1, op_col2 = st.columns(2)
    with op_col1:
        st.button("(D1) Dodawanie", disabled=True)
        st.button("(D2) Odejmowanie", disabled=True)
    with op_col2:
        st.button("(D3) Mnożenie", disabled=True)
        st.button("(D4) Dzielenie", disabled=True)

with c3:
    st.markdown("### Parametry wyliczone")
    st.text(f"Liczba próbek: {len(y)}")
    st.text(f"Min: {np.min(y):.2f}")
    st.text(f"Max: {np.max(y):.2f}")
