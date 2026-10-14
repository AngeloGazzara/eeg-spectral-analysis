#!/usr/bin/env python3
"""Lab 4 - non-parametric spectral estimation of a 25-channel EEG recording.

For the selected channels the script
  1. plots the signal during one second of the recording,
  2. estimates the PSD with the periodogram, Welch's method and the
     correlogram (four lag windows),
  3. prints / saves the relative power in the delta, theta, alpha, beta1 and
     beta2 bands,
and then, using all channels, draws scalp maps of the RMS value and of the
relative power of each rhythm.

Usage:  python run_lab4.py --data data/EEG2501.BIN --out results
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")  # no GUI needed when only saving figures
import matplotlib.pyplot as plt
import numpy as np

from eeg_spectral import (
    BANDS,
    band_powers,
    correlogram_psd,
    load_eeg_bin,
    periodogram,
    plot_scalp_map,
    relative_band_powers,
    to_grid,
    welch_psd,
)

FS = 512.0  # sampling frequency [Hz]
N_CHANNELS = 25
DURATION_S = 15.0
PERIODOGRAM_STEP_HZ = 0.025  # apparent resolution of the zero-padded periodogram
WELCH_NPERSEG = 1024
WELCH_NOVERLAP = 512
CORRELOGRAM_MAX_LAG = 256  # -> 512 autocorrelation values
CORRELOGRAM_WINDOWS = {
    "rectangular": "boxcar",
    "Hamming": "hamming",
    "Kaiser beta=5": ("kaiser", 5),
    "Kaiser beta=12": ("kaiser", 12),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/EEG2501.BIN"),
                   help="multiplexed float32 EEG file (default: %(default)s)")
    p.add_argument("--out", type=Path, default=Path("results"),
                   help="output folder for figures and CSV (default: %(default)s)")
    p.add_argument("--channels", type=int, nargs="+", default=[7, 9],
                   help="1-based channels to analyse in detail (default: 7 9)")
    p.add_argument("--start", type=float, default=10.0,
                   help="start [s] of the 1 s time-domain excerpt (default: 10)")
    p.add_argument("--show", action="store_true", help="also open the figures")
    return p.parse_args()


def analyse_channel(x: np.ndarray, fs: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """PSD of one channel with every method. Returns {label: (f, pxx)}."""
    x = x - x.mean()
    out = {}
    out["Periodogram (0.025 Hz grid)"] = periodogram(x, fs, nfft=int(round(fs / PERIODOGRAM_STEP_HZ)))
    out["Welch (1024, 50% overlap, Hamming)"] = welch_psd(
        x, fs, WELCH_NPERSEG, WELCH_NOVERLAP, "hamming", nfft=x.size
    )
    for label, win in CORRELOGRAM_WINDOWS.items():
        out[f"Correlogram ({label})"] = correlogram_psd(x, fs, CORRELOGRAM_MAX_LAG, win)
    return out


def plot_time_excerpt(eeg, channels, start_s, out_dir):
    i0, i1 = int(start_s * FS), int((start_s + 1) * FS)
    t = np.arange(i0, i1) / FS
    fig, axes = plt.subplots(len(channels), 1, sharex=True, figsize=(8, 2.4 * len(channels)))
    for ax, ch in zip(np.atleast_1d(axes), channels):
        ax.plot(t, eeg[i0:i1, ch - 1])
        ax.set_title(f"Channel {ch}")
        ax.set_ylabel("amplitude")
    ax.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out_dir / "time_domain.png", dpi=110)


def plot_spectra(ch, results, out_dir):
    labels = list(results)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    for ax, label, colour in zip(axes[:2], labels[:2], ("tab:blue", "tab:red")):
        f, p = results[label]
        ax.plot(f, p / p.max(), color=colour)
        ax.set_title(label)
    for label in labels[2:]:
        f, p = results[label]
        axes[2].plot(f, p / p.max(), label=label.replace("Correlogram (", "").rstrip(")"))
    axes[2].set_title("Correlogram (512 autocorrelation values)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlim(0, 127)
        ax.set_ylim(0, 1)
        ax.set_ylabel("normalised PSD")
    axes[2].set_xlabel("frequency [Hz]")
    fig.suptitle(f"Channel {ch}")
    fig.tight_layout()
    fig.savefig(out_dir / f"spectra_ch{ch:02d}.png", dpi=110)


def plot_maps(eeg, out_dir):
    rms = eeg.std(axis=0, ddof=1)  # ddof=1: same normalisation as MATLAB std
    rel = [relative_band_powers(eeg[:, k], FS) for k in range(eeg.shape[1])]
    panels = [("RMS value", rms)] + [
        (f"Relative power - {name}", [r[name] for r in rel]) for name in BANDS
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 10))
    for ax, (title, values) in zip(axes.ravel(), panels):
        plot_scalp_map(to_grid(values), title, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "scalp_maps.png", dpi=110)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    eeg = load_eeg_bin(args.data, N_CHANNELS, FS, DURATION_S)
    print(f"Loaded {args.data}: {eeg.shape[0]} samples x {eeg.shape[1]} channels, fs = {FS:g} Hz")

    plot_time_excerpt(eeg, args.channels, args.start, args.out)

    rows = []
    for ch in args.channels:
        results = analyse_channel(eeg[:, ch - 1], FS)
        plot_spectra(ch, results, args.out)
        print(f"\nChannel {ch} - relative band power")
        for label, (f, p) in results.items():
            bp = band_powers(f, p)
            rows.append({"channel": ch, "method": label, **bp})
            values = "  ".join(f"{k}={v:.4f}" for k, v in bp.items())
            print(f"  {label:<38s} {values}")

    with open(args.out / "band_powers.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["channel", "method", *BANDS])
        writer.writeheader()
        writer.writerows(rows)

    plot_maps(eeg, args.out)
    print(f"\nFigures and band_powers.csv written to {args.out.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
