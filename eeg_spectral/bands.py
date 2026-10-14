"""Power distribution over the fundamental EEG rhythms."""

from __future__ import annotations

import numpy as np
from scipy import signal

#: EEG rhythms, in Hz. Bands are half-open: ``lo <= f < hi``.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 3.5),
    "theta": (3.5, 7.0),
    "alpha": (7.0, 14.0),
    "beta1": (14.0, 21.0),
    "beta2": (21.0, 30.0),
}

_EPS = 1e-9  # tolerance for floating point frequency grids


def band_powers(f: np.ndarray, pxx: np.ndarray, bands=BANDS) -> dict[str, float]:
    """Fraction of the *total* spectral power falling in each band.

    The total is the sum over the whole spectrum (0 ... fs/2), so the
    fractions do not add up to 1: the rest lies outside the bands. Using
    half-open intervals guarantees that no frequency bin is counted twice,
    and works whatever the frequency grid is (band edges need not be grid
    points).
    """
    f = np.asarray(f)
    pxx = np.asarray(pxx)
    total = pxx.sum()
    return {
        name: float(pxx[(f >= lo - _EPS) & (f < hi - _EPS)].sum() / total)
        for name, (lo, hi) in bands.items()
    }


def relative_band_powers(
    x: np.ndarray, fs: float, nperseg: int = 4096, bands=BANDS
) -> dict[str, float]:
    """Relative power per band from a single-segment periodogram.

    Uses a rectangular window of ``nperseg`` samples (no overlap, no zero
    padding), i.e. only the first ``nperseg`` samples of ``x``. Besides the
    bands, the key ``"high"`` holds the fraction above the last band edge
    (>= 30 Hz) up to Nyquist.
    """
    x = np.asarray(x, dtype=float)
    f, pxx = signal.welch(
        x,
        fs=fs,
        window=np.ones(nperseg),
        nperseg=nperseg,
        noverlap=0,
        nfft=nperseg,
        detrend=False,
        scaling="density",
    )
    out = band_powers(f, pxx, bands)
    top = max(hi for _, hi in bands.values())
    out["high"] = float(pxx[f >= top - _EPS].sum() / pxx.sum())
    return out
