"""Non-parametric power spectral density estimators.

All estimators return ``(f, pxx)`` with ``f`` in Hz (one-sided, 0 ... fs/2).
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def make_window(spec, n: int) -> np.ndarray:
    """Symmetric window of length ``n``.

    ``spec`` is a scipy window name (``"boxcar"``, ``"hamming"``) or a tuple
    such as ``("kaiser", 5)``. Symmetric windows are used on purpose, to
    match the usual ``hamming(n)`` / ``kaiser(n, beta)`` definitions.
    """
    return signal.get_window(spec, n, fftbins=False)


def periodogram(x: np.ndarray, fs: float, nfft: int | None = None):
    """Classic periodogram: rectangular window over the whole record.

    ``nfft > len(x)`` zero-pads the record, which interpolates the spectrum
    (finer frequency grid) without improving the true resolution ~ fs/len(x).
    """
    x = np.asarray(x, dtype=float)
    return signal.welch(
        x,
        fs=fs,
        window=np.ones(x.size),
        nperseg=x.size,
        noverlap=0,
        nfft=nfft or x.size,
        detrend=False,
        scaling="density",
    )


def welch_psd(
    x: np.ndarray,
    fs: float,
    nperseg: int = 1024,
    noverlap: int = 512,
    window="hamming",
    nfft: int | None = None,
):
    """Welch's method: averaged, windowed, overlapping periodograms."""
    x = np.asarray(x, dtype=float)
    return signal.welch(
        x,
        fs=fs,
        window=make_window(window, nperseg),
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft or nperseg,
        detrend=False,
        scaling="density",
    )


def autocorrelation(x: np.ndarray, max_lag: int, biased: bool = False):
    """Sample autocorrelation for lags ``-max_lag ... +max_lag``.

    ``biased=False`` divides by ``N - |m|`` (MATLAB ``xcorr(..., 'unbiased')``),
    ``biased=True`` divides by ``N``.

    Returns ``(lags, r)``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    full = signal.correlate(x, x, mode="full", method="fft")  # lags -(n-1)..(n-1)
    lags = np.arange(-max_lag, max_lag + 1)
    r = full[n - 1 + lags]
    r = r / n if biased else r / (n - np.abs(lags))
    return lags, r


def correlogram_psd(
    x: np.ndarray, fs: float, max_lag: int = 256, window="hamming"
):
    """Blackman-Tukey correlogram.

    The unbiased autocorrelation is computed for lags ``-max_lag ... max_lag``
    (``2*max_lag + 1`` values); the first one is dropped to obtain a sequence
    of ``2*max_lag`` values, which is multiplied by a window of the same
    length and Fourier transformed. The magnitude is returned, which removes
    the linear phase caused by lag 0 not being the first element.

    The result is in arbitrary units (use it normalised or for relative
    band powers). The frequency grid has step ``fs / (2*max_lag)``.
    """
    _, r = autocorrelation(x, max_lag, biased=False)
    r = r[1:]
    w = make_window(window, r.size)
    spectrum = np.abs(np.fft.fft(r * w))
    f = np.arange(r.size // 2 + 1) * fs / r.size
    return f, spectrum[: r.size // 2 + 1]
