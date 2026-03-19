import msgpack
import numpy as np
import streamlit as st


def save_signal(t, y, metadata=None):
    if metadata is None:
        metadata = {}

    data = {
        "t": t.tolist() if isinstance(t, np.ndarray) else list(t),
        "y": y.tolist() if isinstance(y, np.ndarray) else list(y),
        "metadata": metadata,
        "version": "1.0",
        "format": "msgpack",
    }

    packed_data = msgpack.packb(data, use_bin_type=True)

    st.download_button(
        label="Pobierz plik sygnału (.bin)",
        data=packed_data,
        file_name="signal.bin",
        mime="application/octet-stream",
        help="Pobierz sygnał w formacie binarnym (msgpack)",
    )

    st.success(f"Sygnał przygotowany do pobrania ({len(y)} próbek)")

    return packed_data


def load_signal(file_buffer):
    try:
        file_bytes = file_buffer.read()

        data = msgpack.unpackb(file_bytes, raw=False)

        if not isinstance(data, dict):
            raise ValueError("Invalid file format: expected dictionary")

        t = np.array(data.get("t", []), dtype=np.float64)
        y = np.array(data.get("y", []), dtype=np.float64)
        metadata = data.get("metadata", {})

        if len(t) == 0 or len(y) == 0:
            raise ValueError("Empty signal data")

        if len(t) != len(y):
            raise ValueError(
                f"Length mismatch: t has {len(t)} samples, y has {len(y)} samples"
            )

        st.session_state.signals = [
            {
                "type": "LOADED",
                "params": metadata,
                "operation": None,
                "data": (t, y),
            }
        ]

        return t, y, metadata

    except msgpack.exceptions.ExtraData as e:
        st.error("❌ Błąd formatu pliku: plik zawiera dodatkowe dane")
        return None, None, None
    except msgpack.exceptions.UnpackException as e:
        st.error("❌ Błąd rozpakowania: plik nie jest w formacie msgpack")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Błąd podczas wczytywania pliku: {str(e)}")
        return None, None, None


def export_signal_to_text(t, y, filename="signal.txt"):
    import io

    output = io.StringIO()
    output.write("# Time,Value\n")

    for time_val, signal_val in zip(t, y):
        output.write(f"{time_val},{signal_val}\n")

    text_data = output.getvalue()

    st.download_button(
        label="📄 Pobierz jako TXT/CSV",
        data=text_data,
        file_name=filename,
        mime="text/plain",
    )

    return text_data
