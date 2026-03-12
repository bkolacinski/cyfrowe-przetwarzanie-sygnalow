import numpy as np


def get_time_vector(t1, d, fs):
    num_samples = int(d * fs)
    t = np.linspace(t1, t1 + d, num_samples, endpoint=True)
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


def generate_sine_wyprostowany_jednopolowkowo(params):
    A = params["A"]
    T = params["T"]
    t1 = params["t1"]
    d = params["d"]
    fs = params["fs"]

    t = get_time_vector(t1, d, fs)
    y_sin = A * np.sin((2 * np.pi / T) * (t - t1))
    y = np.where(
        y_sin > 0,
        y_sin,
        0,
    )

    return t, y


def generate_sine_wyprostowany_dwupolowkowo(params):
    A = params["A"]
    T = params["T"]
    t1 = params["t1"]
    d = params["d"]
    fs = params["fs"]

    t = get_time_vector(t1, d, fs)
    y_sin = A * np.sin((2 * np.pi / T) * (t - t1))
    y = np.where(
        y_sin > 0,
        y_sin,
        -y_sin,
    )

    return t, y


def generate_square(params):
    A = params["A"]
    T = params["T"]
    t1 = params["t1"]
    d = params["d"]
    kw = params["kw"]
    fs = params["fs"]

    time = get_time_vector(t1, d, fs)

    # Dla każdego czasu obliczamy pozycję w okresie
    t_relative = time - t1  # Czas względem t1
    t_in_period = np.mod(t_relative, T)  # Pozycja w aktualnym okresie (0 do T)

    # Sygnał jest A gdy t_in_period < kw*T, w przeciwnym razie 0
    y = np.where(t_in_period < kw * T, A, 0)

    return time, y


def generate_symmetric_square(params):
    A = params["A"]
    T = params["T"]
    t1 = params["t1"]
    d = params["d"]
    kw = params["kw"]
    fs = params["fs"]

    time = get_time_vector(t1, d, fs)

    # Dla każdego czasu obliczamy pozycję w okresie
    t_relative = time - t1  # Czas względem t1
    t_in_period = np.mod(t_relative, T)  # Pozycja w aktualnym okresie (0 do T)

    # Sygnał jest A gdy t_in_period < kw*T, w przeciwnym razie 0
    y = np.where(t_in_period < kw * T, A, -A)

    return time, y


def generate_uniform_noise(params):
    A = params["A"]
    t1 = params["t1"]
    d = params["d"]
    fs = params["fs"]

    t = get_time_vector(t1, d, fs)
    y = A * np.random.uniform(-1, 1, len(t))

    return t, y


def generate_gaussian_noise(params):
    A = params["A"]
    t1 = params["t1"]
    d = params["d"]
    fs = params["fs"]

    t = get_time_vector(t1, d, fs)
    y = np.random.normal(loc=0.0, scale=A / 3, size=len(t))

    return t, y


def unit_step(params):
    A = params["A"]
    t1 = params["t1"]
    d = params["d"]
    fs = params["fs"]
    ts = params["ts"]

    t = get_time_vector(t1, d, fs)

    y = np.where(t > ts, A, 0)

    return t, y


def generate_triangle(params):
    A = params["A"]
    T = params["T"]
    t1 = params["t1"]
    d = params["d"]
    kw = params["kw"]
    fs = params["fs"]

    time = get_time_vector(t1, d, fs)

    # Obliczamy pozycję w okresie
    t_relative = time - t1
    t_in_period = np.mod(t_relative, T)

    # Faza wzrostu: od 0 do kw*T
    # y = (A / (kw*T)) * t_in_period
    y_rising = (A / (kw * T)) * t_in_period

    # Faza spadku: od kw*T do T
    # y = (-A / (T*(1-kw))) * (t_in_period - kw*T) + A
    y_falling = (-A / (T * (1 - kw))) * (t_in_period - kw * T) + A

    # Wybieramy odpowiednią fazę w zależności od pozycji w okresie
    y = np.where(t_in_period < kw * T, y_rising, y_falling)

    return time, y
