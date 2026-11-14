"""One palette for every figure, in a light and a dark variant, with a Matplotlib style and a save helper.

The palette is eOx's: eight categorical colours in a fixed order, the chart chrome (surface, inks, grid, axis)
and the status colours, each chosen for a light and a dark surface and for colour-blind readers.

    import matplotlib.pyplot as plt
    from similitude import figures

    with plt.rc_context(figures.style("dark")):
        fig, ax = plt.subplots()
        for (name, ys), colour in zip(series.items(), figures.series_colors(len(series), "dark")):
            ax.plot(xs, ys, color=colour, label=name)
        figures.save(fig, "out/sweep/polar")      # writes polar.png and polar.svg

Three rules come with the palette:

- **Categorical colours go out in slot order and are never cycled.** A ninth series is an error; fold the rest
  into "Other" or split the figure. The order is what keeps neighbouring series apart for colour-blind readers.
- **A colour follows its series, not its rank.** Use `assign` once and reuse the mapping across figures, so a
  series keeps its colour when others are filtered out.
- **Status colours are reserved** for PASS, WARN and FAIL, and never stand in for a series.

On the light surface, slots 3, 4 and 5 (aqua, yellow, magenta) have less than 3:1 contrast with the surface, so
label those series directly or give the numbers in a table as well.

This module needs no Matplotlib to import. `style` and `save` are for Matplotlib figures; install it with the
`figures` extra.

The colours are ported from `THEMES` and `STATUS` in eOx's `src/eox/drawings.py` (MIT); see NOTICE. eOx uses
slots 1 to 5 of the reference palette it names; slots 6 to 8 are the rest of that palette.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

SLOTS = ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red")
MAX_SERIES = len(SLOTS)


class TooManySeriesError(ValueError):
    """More series than the palette has categorical colours."""


@dataclass(frozen=True)
class Theme:
    """The palette on one surface."""

    name: str
    surface: str  # the figure and axes background
    ink: str  # primary text: titles
    ink2: str  # secondary text: axis labels, legends
    muted: str  # tick labels, and a status with no colour of its own
    grid: str  # gridlines
    base: str  # the axis line
    series: tuple[str, ...]  # the categorical colours, in slot order
    status: MappingProxyType = field(hash=False)  # PASS, WARN and FAIL

    def status_color(self, status: str) -> str:
        """The colour for a check status; INFO, or any other status, is muted."""
        return self.status.get(str(status), self.muted)


_STATUS = MappingProxyType({"PASS": "#0ca30c", "WARN": "#fab219", "FAIL": "#d03b3b"})

LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    base="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    status=_STATUS,
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#f0efec",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    base="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    status=_STATUS,
)
THEMES = MappingProxyType({"light": LIGHT, "dark": DARK})


def theme(name: str | Theme = "light") -> Theme:
    """The light or the dark theme, by name."""
    if isinstance(name, Theme):
        return name
    if name not in THEMES:
        raise ValueError(f"the themes are {', '.join(THEMES)}, not {name!r}")
    return THEMES[name]


def series_colors(n: int, theme_name: str | Theme = "light") -> tuple[str, ...]:
    """The first `n` categorical colours, in slot order. More than eight is an error; they are never cycled."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"the number of series is a whole number, not {n!r}")
    if n > MAX_SERIES:
        raise TooManySeriesError(
            f"{n} series, but the palette has {MAX_SERIES} categorical colours and never cycles them; "
            'fold the smallest into "Other", or split the figure into small multiples'
        )
    return theme(theme_name).series[:n]


def assign(names: Iterable[str], theme_name: str | Theme = "light") -> dict[str, str]:
    """A colour for each named series, in slot order: the first name gets slot 1.

    Make the mapping once, from every series a set of figures can show, and reuse it, so a series keeps its
    colour whichever others are drawn beside it. A ninth name or a repeated name is an error.
    """
    names = list(names)
    repeated = sorted({str(n) for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"each series is named once; repeated: {', '.join(repeated)}")
    return dict(zip(names, series_colors(len(names), theme_name)))


def style(theme_name: str | Theme = "light") -> dict:
    """Matplotlib rcParams for a theme; use with `matplotlib.rc_context` or `matplotlib.rcParams.update`.

    Matplotlib's own colour cycle wraps round without saying so, which would hand a ninth series the first
    colour again. So the style's cycle holds slot 1 only: one series is drawn in it with no more ado, and several
    get their colours from `series_colors` or `assign`, which refuse a ninth.
    """
    from cycler import cycler  # comes with Matplotlib

    t = theme(theme_name)
    return {
        "figure.facecolor": t.surface,
        "figure.edgecolor": t.surface,
        "axes.facecolor": t.surface,
        "savefig.facecolor": t.surface,
        "savefig.edgecolor": t.surface,
        "text.color": t.ink,
        "axes.titlecolor": t.ink,
        "axes.labelcolor": t.ink2,
        "legend.labelcolor": t.ink2,
        "xtick.labelcolor": t.muted,
        "ytick.labelcolor": t.muted,
        "xtick.color": t.base,
        "ytick.color": t.base,
        "axes.edgecolor": t.base,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": t.grid,
        "grid.linewidth": 0.8,
        "axes.prop_cycle": cycler(color=[t.series[0]]),
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
        "patch.edgecolor": t.surface,
        "legend.frameon": False,
        "svg.hashsalt": "similitude",  # with no date in the file either, the same figure gives the same SVG
    }


def save(fig, path: str | os.PathLike, *, dpi: float = 200) -> tuple[Path, Path]:
    """Write a Matplotlib figure as PNG and as SVG in one call, and return the two paths.

    `path` is the files' common name without a suffix: `save(fig, "out/sweep/polar")` writes `polar.png` and
    `polar.svg` in `out/sweep/`, making the directory if need be. A `.png` or `.svg` suffix is dropped. The files
    keep the figure's own background. `dpi` is for the PNG.
    """
    import matplotlib

    path = Path(path)
    if path.suffix.lower() in (".png", ".svg"):
        path = path.with_suffix("")
    if not path.name:
        raise ValueError("save needs a file name to add .png and .svg to")
    png, svg = path.with_name(path.name + ".png"), path.with_name(path.name + ".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    background = {"facecolor": fig.get_facecolor(), "edgecolor": fig.get_facecolor()}
    fig.savefig(png, format="png", dpi=dpi, **background)
    with matplotlib.rc_context({"svg.hashsalt": matplotlib.rcParams["svg.hashsalt"] or "similitude"}):
        fig.savefig(svg, format="svg", metadata={"Date": None}, **background)
    return png, svg
