from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

WIDTH = 7.4


@dataclass(frozen=True)
class Theme:
    mode: str
    fg: str
    muted: str
    grid: str
    surface: str
    panel: str
    series: tuple[str, ...]
    brand: str
    brand2: str
    seq: LinearSegmentedColormap = field(repr=False)
    seq2: LinearSegmentedColormap = field(repr=False)
    div: LinearSegmentedColormap = field(repr=False)
    brand_map: LinearSegmentedColormap = field(repr=False)

    @property
    def dark(self) -> bool:
        return self.mode == "dark"


def _cmap(name, colors):
    return LinearSegmentedColormap.from_list(name, colors, N=256)


LIGHT = Theme(
    mode="light",
    fg="#1f1d33",
    muted="#5f5c80",
    grid="#e4e3ee",
    surface="#ffffff",
    panel="#f4f3fa",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    brand="#574f9e",
    brand2="#3f8f2b",
    seq=_cmap("seq_l", ["#f5f9fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
    seq2=_cmap("seq2_l", ["#fff8f3", "#fbd9c4", "#f4a57c", "#eb6834", "#b8431a", "#7a2a0e"]),
    div=_cmap("div_l", ["#1c5cab", "#6da7ec", "#f0efec", "#ef8a73", "#b3261e"]),
    brand_map=_cmap("brand_l", ["#1c1a3e", "#574f9e", "#6cb44c", "#cfe3a5"]),
)

DARK = Theme(
    mode="dark",
    fg="#e6e5f2",
    muted="#a9a7c6",
    grid="#2e2c40",
    surface="#121119",
    panel="#1f1d2b",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    brand="#a59ef0",
    brand2="#8fd46b",
    seq=_cmap("seq_d", ["#16192a", "#152f57", "#1c5cab", "#3987e5", "#86b6ef", "#e3eefc"]),
    seq2=_cmap("seq2_d", ["#1b1515", "#4a2212", "#8f3a17", "#d95926", "#f4a57c", "#fde6d8"]),
    div=_cmap("div_d", ["#6da7ec", "#2c5f9e", "#383835", "#a8433a", "#f08a7e"]),
    brand_map=_cmap("brand_d", ["#2a2760", "#a59ef0", "#8fd46b", "#e8f5d0"]),
)

THEMES = {"light": LIGHT, "dark": DARK}


def style(t: Theme) -> None:
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 9.5,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "axes.labelsize": 9.5,
            "axes.edgecolor": t.muted,
            "axes.labelcolor": t.fg,
            "axes.titlecolor": t.fg,
            "axes.linewidth": 0.8,
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "xtick.color": t.muted,
            "ytick.color": t.muted,
            "xtick.labelcolor": t.muted,
            "ytick.labelcolor": t.muted,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": t.fg,
            "axes.grid": True,
            "grid.color": t.grid,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=list(t.series)),
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": t.fg,
            "savefig.transparent": True,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "image.cmap": "viridis",
            "mathtext.fontset": "dejavusans",
        }
    )


def figure(nrows: int = 1, ncols: int = 1, *, width: float = WIDTH, height: float | None = None, **kw):
    if height is None:
        height = width * (0.52 if nrows == 1 else 0.42 * nrows)
    return plt.subplots(nrows, ncols, figsize=(width, height), layout="constrained", **kw)


def blank(width: float = WIDTH, height: float = 3.5):
    return plt.figure(figsize=(width, height), layout="constrained")


def pane_3d(ax, t: Theme) -> None:
    pane = matplotlib.colors.to_rgba(t.panel)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane)
        axis._axinfo["grid"]["color"] = t.grid
    ax.tick_params(colors=t.muted, labelsize=7.5)


def map_axes(ax, t: Theme) -> None:
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(t.grid if t.dark else "#d6d5e3")


def save(name: str, draw) -> None:
    os.makedirs(ASSETS, exist_ok=True)
    for mode, t in THEMES.items():
        style(t)
        fig = draw(t)
        path = os.path.join(ASSETS, f"{name}_{mode}.png")
        fig.savefig(path)
        plt.close(fig)
        print("figure written to", os.path.relpath(path))
