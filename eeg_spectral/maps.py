"""Colour-coded scalp maps of a quantity measured on a 5x5 electrode grid."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline

GRID_SHAPE = (5, 5)
X_NODES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
Y_NODES = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])


def to_grid(values) -> np.ndarray:
    """Arrange 25 per-channel values on the 5x5 grid.

    Channels are placed row by row: channels 1-5 form the first row,
    6-10 the second, and so on (row-major order).
    """
    return np.asarray(values, dtype=float).reshape(GRID_SHAPE)


def interpolate_map(z: np.ndarray, step: float = 0.1):
    """Cubic interpolation of the 5x5 grid onto a fine mesh.

    Returns ``(xi, yi, zi)`` where ``xi, yi`` are 2-D coordinate arrays
    (like ``meshgrid``) and ``zi`` has one row per ``y`` value.
    """
    xs = np.linspace(X_NODES[0], X_NODES[-1], int(round(np.ptp(X_NODES) / step)) + 1)
    ys = np.linspace(Y_NODES[0], Y_NODES[-1], int(round(np.ptp(Y_NODES) / step)) + 1)
    spline = RectBivariateSpline(Y_NODES, X_NODES, z, kx=3, ky=3, s=0)
    zi = spline(ys, xs)
    xi, yi = np.meshgrid(xs, ys)
    return xi, yi, zi


def plot_scalp_map(z: np.ndarray, title: str, ax=None, n_contours: int = 12):
    """Interpolated colour map with contour lines. Returns the axes."""
    if ax is None:
        _, ax = plt.subplots()
    xi, yi, zi = interpolate_map(z)
    cmap = matplotlib.colormaps["hot"].resampled(32)
    mesh = ax.pcolormesh(xi, yi, zi, shading="gouraud", cmap=cmap)
    ax.contour(xi, yi, zi, n_contours, colors="k", linewidths=0.6)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(mesh, ax=ax, shrink=0.8)
    return ax
