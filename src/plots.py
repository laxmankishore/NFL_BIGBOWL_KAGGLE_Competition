"""Shared figure style and Combine trace plots (static matplotlib for Kaggle)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
ACCENT = "#eb6834"

# Single-hue sequential ramp (blue 150 -> 700): magnitude = speed.
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed_blue", ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 9,
            "lines.linewidth": 2,
        }
    )


def plot_drill_trace(trace, title: str, break_frame: int | None = None, vmax: float = 9.0):
    """Two panels: x-y path coloured by speed, and speed vs time.

    ``trace`` is one event's frames (polars DataFrame) sorted by time.
    """
    x, y, s = (trace[c].to_numpy() for c in ("x", "y", "s"))
    t = np.arange(len(s)) * 0.1
    fig, (ax_path, ax_spd) = plt.subplots(
        1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=SPEED_CMAP,
                        norm=plt.Normalize(0, vmax), linewidth=3, capstyle="round")
    lc.set_array(s[:-1])
    ax_path.add_collection(lc)
    ax_path.plot(x[0], y[0], "o", color=INK_2, ms=8, mec=SURFACE, mew=2)
    ax_path.annotate("start", (x[0], y[0]), xytext=(6, 6), textcoords="offset points", color=INK_2)
    ax_path.set_aspect("equal", adjustable="datalim")
    ax_path.autoscale_view()
    ax_path.set_xlabel("x (yd)")
    ax_path.set_ylabel("y (yd)")
    ax_path.set_title(title)
    cb = fig.colorbar(lc, ax=ax_path, fraction=0.04, pad=0.02)
    cb.set_label("speed (yd/s)", color=INK_2)
    cb.outline.set_visible(False)

    ax_spd.plot(t, s, color=SERIES[0])
    ax_spd.set_xlabel("time since first frame (s)")
    ax_spd.set_ylabel("speed (yd/s)")
    ax_spd.set_ylim(0, max(vmax, s.max() * 1.05))
    ax_spd.set_title("Speed profile")

    if break_frame is not None:
        for ax, (bx, by) in ((ax_path, (x[break_frame], y[break_frame])),
                             (ax_spd, (t[break_frame], s[break_frame]))):
            ax.plot(bx, by, "o", color=ACCENT, ms=9, mec=SURFACE, mew=2, zorder=5)
        ax_spd.annotate("detected break", (t[break_frame], s[break_frame]),
                        xytext=(10, -14), textcoords="offset points", color=INK)
    fig.tight_layout()
    return fig
