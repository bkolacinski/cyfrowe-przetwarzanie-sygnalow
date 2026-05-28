from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import msgpack
import numpy as np


def save_complex_signal(
    t: np.ndarray,
    signal: np.ndarray,
    metadata: dict | None = None,
) -> bytes:
    if metadata is None:
        metadata = {}

    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(signal, dtype=np.complex128)

    payload = {
        "version": "2.0",
        "format": "complex-msgpack",
        "t": t_arr.tolist(),
        "y_real": np.real(y_arr).tolist(),
        "y_imag": np.imag(y_arr).tolist(),
        "metadata": metadata,
    }

    return msgpack.packb(payload, use_bin_type=True)


def load_complex_signal(
    file_bytes: bytes | bytearray | BytesIO | BinaryIO,
) -> tuple[np.ndarray, np.ndarray, dict]:
    raw_bytes: bytes
    if isinstance(file_bytes, BytesIO):
        raw_bytes = file_bytes.getvalue()
    elif isinstance(file_bytes, (bytes, bytearray)):
        raw_bytes = bytes(file_bytes)
    else:
        raw_bytes = bytes(file_bytes.read())

    payload = msgpack.unpackb(raw_bytes, raw=False)

    t = np.asarray(payload.get("t", []), dtype=np.float64)
    y_real = np.asarray(payload.get("y_real", []), dtype=np.float64)
    y_imag = np.asarray(payload.get("y_imag", []), dtype=np.float64)
    metadata = payload.get("metadata", {})

    if t.size == 0 or y_real.size == 0:
        raise ValueError("Signal file is empty or invalid.")

    if not (t.size == y_real.size == y_imag.size):
        raise ValueError("Signal vectors t, y_real and y_imag must have equal length.")

    y = y_real.astype(np.complex128) + 1j * y_imag.astype(np.complex128)
    return t, y, metadata


def export_complex_to_csv(t: np.ndarray, signal: np.ndarray) -> str:
    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(signal, dtype=np.complex128)

    lines = ["time,real,imag,magnitude,phase"]
    for ti, yi in zip(t_arr, y_arr):
        lines.append(
            f"{ti:.10f},{yi.real:.10f},{yi.imag:.10f},{np.abs(yi):.10f},{np.angle(yi):.10f}"
        )

    return "\n".join(lines)
