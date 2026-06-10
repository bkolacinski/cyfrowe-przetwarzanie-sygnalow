from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transforms.dct import dct_ii, fct_ii
from transforms.fourier import dft, fft_dif, fft_dit
from transforms.walsh_hadamard import fwht, walsh_hadamard_transform
from transforms.wavelets import dwt_multilevel, idwt_multilevel


class TestZadanie4Transforms(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)

    def test_dft_matches_numpy_fft(self):
        for n in (2, 4, 8, 16, 32):
            x = self.rng.normal(size=n) + 1j * self.rng.normal(size=n)

            # Forward
            reference = np.fft.fft(x)
            estimate = dft(x)
            self.assertLessEqual(np.max(np.abs(reference - estimate)), 1e-9)

            # Inverse
            roundtrip = dft(estimate, inverse=True)
            self.assertLessEqual(np.max(np.abs(x - roundtrip)), 1e-9)

    def test_fft_dit_and_dif_match_numpy_fft(self):
        for n in (2, 4, 8, 16, 64, 256):
            x = self.rng.normal(size=n) + 1j * self.rng.normal(size=n)

            # Forward
            reference = np.fft.fft(x)
            dit_fwd = fft_dit(x)
            dif_fwd = fft_dif(x)
            self.assertLessEqual(np.max(np.abs(reference - dit_fwd)), 1e-9)
            self.assertLessEqual(np.max(np.abs(reference - dif_fwd)), 1e-9)

            # Inverse
            dit_inv = fft_dit(dit_fwd, inverse=True)
            dif_inv = fft_dif(dif_fwd, inverse=True)
            self.assertLessEqual(np.max(np.abs(x - dit_inv)), 1e-9)
            self.assertLessEqual(np.max(np.abs(x - dif_inv)), 1e-9)

    def test_dct_ii_matches_fct_ii(self):
        for n in (2, 4, 8, 16, 64):
            x = self.rng.normal(size=n)
            slow = dct_ii(x)
            fast = fct_ii(x)
            self.assertLessEqual(np.max(np.abs(slow - fast)), 1e-9)

    def test_walsh_matches_fwht(self):
        for n in (2, 4, 8, 16, 32):
            x = self.rng.normal(size=n) + 1j * self.rng.normal(size=n)
            classic = walsh_hadamard_transform(x)
            fast = fwht(x)
            self.assertLessEqual(np.max(np.abs(classic - fast)), 1e-9)

    def test_wavelet_roundtrip_all(self):
        n = 128
        x = self.rng.normal(size=n) + 1j * self.rng.normal(size=n)

        for wavelet in ("db2", "db3", "db4", "db6", "db8"):
            approx, details = dwt_multilevel(x, levels=3, wavelet_name=wavelet)
            reconstructed = idwt_multilevel(approx, details, wavelet_name=wavelet)
            self.assertLessEqual(np.max(np.abs(x - reconstructed)), 1e-8)


if __name__ == "__main__":
    unittest.main()
