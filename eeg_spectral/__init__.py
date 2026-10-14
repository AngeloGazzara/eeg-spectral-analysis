"""Non-parametric spectral estimation and scalp maps for multichannel EEG."""

from .bands import BANDS, band_powers, relative_band_powers
from .loader import load_eeg_bin
from .maps import interpolate_map, plot_scalp_map, to_grid
from .spectral import autocorrelation, correlogram_psd, periodogram, welch_psd

__all__ = [
    "BANDS",
    "autocorrelation",
    "band_powers",
    "correlogram_psd",
    "interpolate_map",
    "load_eeg_bin",
    "periodogram",
    "plot_scalp_map",
    "relative_band_powers",
    "to_grid",
    "welch_psd",
]
