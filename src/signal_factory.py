import numpy as np

import functions as legacy_fn

EPSILON = 1e-12


# Reużycie generatorów z poprzednich zadań (bez duplikowania implementacji)
generate_sine = legacy_fn.generate_sine
generate_square = legacy_fn.generate_square
generate_triangle = legacy_fn.generate_triangle
generate_uniform_noise = legacy_fn.generate_uniform_noise
generate_gaussian_noise = legacy_fn.generate_gaussian_noise


SIGNAL_REGISTRY = {
    "Sinus": {
        "func": generate_sine,
        "params": ["A", "T"],
    },
    "Prostokątny": {
        "func": generate_square,
        "params": ["A", "T", "kw"],
    },
    "Trójkątny": {
        "func": generate_triangle,
        "params": ["A", "T", "kw"],
    },
    "Szum jednostajny": {
        "func": generate_uniform_noise,
        "params": ["A"],
    },
    "Szum gaussowski": {
        "func": generate_gaussian_noise,
        "params": ["A"],
    },
}


def generate_named_signal(
    name: str, common_params: dict, specific_params: dict
):
    if name not in SIGNAL_REGISTRY:
        raise ValueError(f"Unknown signal name: {name}")

    config = SIGNAL_REGISTRY[name]
    params = {**common_params, **specific_params}
    return config["func"](params)


def generate_two_tone_probe(
    fs_hz: float,
    length: int,
    f1_hz: float,
    f2_hz: float,
    a1: float = 1.0,
    a2: float = 0.6,
    phase2_rad: float = np.pi / 4,
):
    n = np.arange(length)
    t = n / fs_hz
    y = a1 * np.sin(2 * np.pi * f1_hz * t) + a2 * np.sin(
        2 * np.pi * f2_hz * t + phase2_rad
    )
    return t, y


def add_noise(
    y: np.ndarray, enabled: bool = True, noise_level: float = 0.05
) -> np.ndarray:
    signal = np.asarray(y, dtype=float)
    if not enabled:
        return signal
    return signal + np.random.normal(0.0, noise_level, size=len(signal))


def shift_signal(
    y: np.ndarray, delay_samples: int, fill_value: float = 0.0
) -> np.ndarray:
    signal = np.asarray(y, dtype=float)
    delay_samples = int(delay_samples)

    if delay_samples <= 0:
        return signal.copy()

    shifted = np.full_like(signal, fill_value, dtype=float)
    if delay_samples < len(signal):
        shifted[delay_samples:] = signal[:-delay_samples]
    return shifted
