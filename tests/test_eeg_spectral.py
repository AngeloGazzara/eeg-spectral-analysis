"""Sanity tests on synthetic signals. Run with:  python -m unittest -v"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eeg_spectral import (
    band_powers,
    correlogram_psd,
    interpolate_map,
    load_eeg_bin,
    periodogram,
    relative_band_powers,
    to_grid,
    welch_psd,
)
from eeg_spectral.spectral import autocorrelation

FS = 512.0
N = 7680
T = np.arange(N) / FS


def sine(freq=10.0):
    return np.sin(2 * np.pi * freq * T)


def white(seed=0):
    return np.random.default_rng(seed).standard_normal(N)


class LoaderTests(unittest.TestCase):
    def test_multiplexed_layout(self):
        data = np.random.default_rng(1).standard_normal((N, 25)).astype("<f4")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.bin"
            data.tofile(path)  # row-major: all channels of sample 0, then sample 1, ...
            eeg = load_eeg_bin(path)
        self.assertEqual(eeg.shape, (N, 25))
        np.testing.assert_allclose(eeg, data)

    def test_wrong_size_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.bin"
            np.zeros(100, dtype="<f4").tofile(path)
            with self.assertRaises(ValueError):
                load_eeg_bin(path)


class EstimatorTests(unittest.TestCase):
    def test_periodogram_parseval(self):
        x = white() - white().mean()
        f, p = periodogram(x, FS, nfft=20480)
        df = f[1] - f[0]
        self.assertAlmostEqual(p.sum() * df, np.mean(x**2), delta=1e-6 * np.mean(x**2))

    def test_welch_power_close_to_variance(self):
        x = white(2)
        f, p = welch_psd(x, FS)
        df = f[1] - f[0]
        self.assertAlmostEqual(p.sum() * df / np.var(x), 1.0, delta=0.05)

    def test_all_methods_find_the_sine(self):
        x = sine(10.0)
        for f, p in (periodogram(x, FS), welch_psd(x, FS), correlogram_psd(x, FS)):
            self.assertAlmostEqual(f[np.argmax(p)], 10.0, delta=0.5)

    def test_autocorrelation_matches_brute_force(self):
        x = np.random.default_rng(3).standard_normal(64)
        lags, r = autocorrelation(x, 5, biased=False)
        for m, value in zip(lags, r):
            m = abs(int(m))
            expected = np.dot(x[m:], x[: len(x) - m]) / (len(x) - m)
            self.assertAlmostEqual(value, expected, places=10)

    def test_correlogram_grid(self):
        f, p = correlogram_psd(white(), FS, max_lag=256)
        self.assertEqual(len(f), 257)
        self.assertAlmostEqual(f[1] - f[0], 1.0)
        self.assertAlmostEqual(f[-1], FS / 2)


class BandTests(unittest.TestCase):
    def test_sine_in_alpha(self):
        f, p = periodogram(sine(10.0), FS, nfft=20480)
        bp = band_powers(f, p)
        self.assertGreater(bp["alpha"], 0.95)
        self.assertLessEqual(sum(bp.values()), 1.0 + 1e-12)

    def test_white_noise_proportional_to_bandwidth(self):
        f, p = welch_psd(white(4), FS)
        bp = band_powers(f, p)
        self.assertAlmostEqual(bp["alpha"], 7.0 / (FS / 2), delta=0.01)

    def test_edges_not_double_counted(self):
        f = np.arange(0, 40, 1.0)
        bp = band_powers(f, np.ones_like(f))
        counted = sum(round(v * f.size) for v in bp.values())
        self.assertEqual(counted, 3 + 3 + 7 + 7 + 9)  # bins 1-2-3 | 4-5-6 | 7-13 | 14-20 | 21-29

    def test_relative_band_powers_include_high(self):
        bp = relative_band_powers(sine(40.0), FS)
        self.assertGreater(bp["high"], 0.9)


class MapTests(unittest.TestCase):
    def test_grid_is_row_major(self):
        z = to_grid(np.arange(1, 26))
        np.testing.assert_array_equal(z[0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(z[:, 0], [1, 6, 11, 16, 21])

    def test_interpolation_passes_through_nodes(self):
        z = np.random.default_rng(5).random((5, 5))
        xi, yi, zi = interpolate_map(z)
        self.assertEqual(zi.shape, (81, 41))
        self.assertAlmostEqual(zi[0, 0], z[0, 0])
        self.assertAlmostEqual(zi[40, 20], z[2, 2])
        self.assertAlmostEqual(zi[-1, -1], z[4, 4])


if __name__ == "__main__":
    unittest.main()
