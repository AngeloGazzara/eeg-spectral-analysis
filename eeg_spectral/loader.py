"""Loading of the multiplexed binary EEG recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_eeg_bin(
    path: str | Path,
    n_channels: int = 25,
    fs: float = 512.0,
    duration_s: float = 15.0,
    dtype: str = "<f4",
) -> np.ndarray:
    """Read a multiplexed binary EEG file.

    The samples are stored one *time instant* after the other: first the
    value of every channel at sample 1, then every channel at sample 2, and
    so on (ch1[0], ch2[0], ..., chN[0], ch1[1], ...).

    Parameters
    ----------
    path : path of the ``.BIN`` file.
    n_channels : number of channels (electrodes).
    fs : sampling frequency in Hz.
    duration_s : duration of the recording in seconds.
    dtype : numpy dtype of one sample (default: little-endian float32).

    Returns
    -------
    eeg : array of shape ``(n_samples, n_channels)``; one column per channel.
    """
    n_samples = int(round(fs * duration_s))
    expected = n_samples * n_channels
    raw = np.fromfile(path, dtype=dtype, count=expected)
    if raw.size != expected:
        raise ValueError(
            f"{path}: expected {expected} samples "
            f"({n_channels} channels x {n_samples}), found {raw.size}. "
            "Check n_channels, fs, duration_s and dtype."
        )
    return raw.reshape(n_samples, n_channels).astype(np.float64)
