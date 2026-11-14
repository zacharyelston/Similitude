"""Figures: palette order, the ninth-series error, the Matplotlib style, and both files written."""

import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import matplotlib
import pytest
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.image import imread

from similitude import figures

# eOx's THEMES, from src/eox/drawings.py: slots 1 to 5 and the chart chrome.
EOX = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  base="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 base="#383835", series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]),
}  # fmt: skip


def a_figure(theme="light", lines=1):
    with matplotlib.rc_context(figures.style(theme)):
        fig = Figure(figsize=(4, 3))
        ax = fig.subplots()
        for i in range(lines):
            ax.plot([0, 1, 2], [i, i + 1, i + 3])
    return fig


# --- palette order


def test_the_categorical_palette_is_eight_colours_in_a_fixed_order():
    assert figures.SLOTS == ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red")
    assert figures.MAX_SERIES == 8
    assert figures.LIGHT.series == (
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
    )  # fmt: skip
    assert figures.DARK.series == (
        "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767",
    )  # fmt: skip


@pytest.mark.parametrize("name", ["light", "dark"])
def test_slots_1_to_5_and_the_chrome_are_eoxs(name):
    theme, eox = figures.THEMES[name], EOX[name]
    assert list(theme.series[:5]) == eox["series"]
    for role in ("surface", "ink", "ink2", "muted", "grid", "base"):
        assert getattr(theme, role) == eox[role]


def test_the_light_and_dark_variants_are_the_same_slots_stepped_for_each_surface():
    assert figures.THEMES["light"] is figures.LIGHT and figures.THEMES["dark"] is figures.DARK
    assert (figures.LIGHT.name, figures.DARK.name) == ("light", "dark")
    assert figures.LIGHT.surface != figures.DARK.surface
    for theme in figures.THEMES.values():
        colours = (*theme.series, theme.surface, theme.ink, theme.ink2, theme.muted, theme.grid, theme.base)
        assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in colours)
        assert len(set(theme.series)) == 8
    # Each slot keeps its hue between the variants: the largest of red, green and blue stays the largest.
    for light, dark in zip(figures.LIGHT.series, figures.DARK.series):
        assert max(range(3), key=to_rgb(light).__getitem__) == max(range(3), key=to_rgb(dark).__getitem__)


def test_a_theme_is_looked_up_by_name():
    assert figures.theme() is figures.LIGHT
    assert figures.theme("dark") is figures.DARK
    assert figures.theme(figures.DARK) is figures.DARK
    with pytest.raises(ValueError, match="the themes are light, dark, not 'sepia'"):
        figures.theme("sepia")


@pytest.mark.parametrize("name", ["light", "dark"])
def test_series_get_the_first_slots_in_order(name):
    palette = figures.THEMES[name].series
    for n in range(9):
        assert figures.series_colors(n, name) == palette[:n]
    assert figures.series_colors(3) == ("#2a78d6", "#eb6834", "#1baf7a")


# --- the ninth-series error


@pytest.mark.parametrize("n", [9, 10, 16, 100])
def test_a_ninth_series_is_an_error_not_a_cycled_colour(n):
    with pytest.raises(figures.TooManySeriesError, match=f"{n} series, but the palette has 8 categorical colours"):
        figures.series_colors(n)
    with pytest.raises(ValueError):  # it is a ValueError, for callers who don't know the name
        figures.series_colors(n, "dark")


@pytest.mark.parametrize("n", [-1, 2.0, "3", True, None])
def test_the_number_of_series_is_a_whole_number(n):
    with pytest.raises(ValueError, match="whole number"):
        figures.series_colors(n)


def test_assign_gives_each_named_series_its_slot():
    rigs = ["mast5", "gecko5", "spreader-pairs"]
    assert figures.assign(rigs) == {"mast5": "#2a78d6", "gecko5": "#eb6834", "spreader-pairs": "#1baf7a"}
    assert figures.assign(rigs, "dark") == {"mast5": "#3987e5", "gecko5": "#d95926", "spreader-pairs": "#199e70"}
    assert list(figures.assign(iter(rigs))) == rigs
    assert figures.assign([]) == {}


def test_a_series_keeps_its_colour_when_others_are_left_out_of_a_figure():
    colours = figures.assign(["mast5", "gecko5", "spreader-pairs", "pods"])
    shown = {name: colours[name] for name in ("gecko5", "pods")}  # a filtered figure draws from the same mapping
    assert shown == {"gecko5": "#eb6834", "pods": "#eda100"}


def test_a_ninth_name_or_a_repeated_name_is_an_error():
    eight = [f"rig {i}" for i in range(1, 9)]
    assert list(figures.assign(eight).values()) == list(figures.LIGHT.series)
    with pytest.raises(figures.TooManySeriesError, match="9 series"):
        figures.assign([*eight, "rig 9"])
    with pytest.raises(ValueError, match="repeated: mast5"):
        figures.assign(["mast5", "gecko5", "mast5"])


# --- status colours


def test_status_colours_are_eoxs_and_are_not_series_colours():
    for theme in figures.THEMES.values():
        assert theme.status_color("PASS") == "#0ca30c"
        assert theme.status_color("WARN") == "#fab219"
        assert theme.status_color("FAIL") == "#d03b3b"
        assert theme.status_color("INFO") == theme.muted == "#898781"
        assert not set(theme.status.values()) & set(theme.series)
    with pytest.raises(TypeError):
        figures.LIGHT.status["PASS"] = "#000000"


# --- the Matplotlib style


@pytest.mark.parametrize("name", ["light", "dark"])
def test_the_style_draws_a_figure_in_the_themes_chrome(name):
    theme = figures.THEMES[name]
    fig = a_figure(name)
    (ax,) = fig.axes
    assert to_hex(fig.get_facecolor()) == theme.surface
    assert to_hex(ax.get_facecolor()) == theme.surface
    assert to_hex(ax.spines["bottom"].get_edgecolor()) == theme.base
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert to_hex(ax.xaxis.label.get_color()) == theme.ink2
    assert to_hex(ax.title.get_color()) == theme.ink
    assert to_hex(ax.get_xticklabels()[0].get_color()) == theme.muted
    assert to_hex(ax.get_xgridlines()[0].get_color()) == theme.grid
    assert ax.lines[0].get_linewidth() == 2.0


def test_the_style_is_valid_rcparams_and_leaves_the_defaults_alone():
    before = dict(matplotlib.rcParams)
    for name in figures.THEMES:
        with matplotlib.rc_context(figures.style(name)):
            assert matplotlib.rcParams["figure.facecolor"] == figures.THEMES[name].surface
    assert dict(matplotlib.rcParams) == before


@pytest.mark.parametrize("name", ["light", "dark"])
def test_the_styles_own_cycle_never_hands_out_a_second_categorical_colour(name):
    theme = figures.THEMES[name]
    fig = a_figure(name, lines=9)  # nine series drawn with no colour given
    assert {to_hex(line.get_color()) for line in fig.axes[0].lines} == {theme.series[0]}


def test_series_drawn_with_assigned_colours_use_the_slots_in_order():
    with matplotlib.rc_context(figures.style("dark")):
        fig = Figure()
        ax = fig.subplots()
        for i, colour in enumerate(figures.series_colors(8, "dark")):
            ax.plot([0, 1], [i, i], color=colour)
    assert tuple(to_hex(line.get_color()) for line in ax.lines) == figures.DARK.series


# --- both files written


def test_save_writes_a_png_and_an_svg_in_one_call(tmp_path):
    png, svg = figures.save(a_figure(), tmp_path / "out" / "sweep" / "polar")
    assert (png, svg) == (tmp_path / "out" / "sweep" / "polar.png", tmp_path / "out" / "sweep" / "polar.svg")
    assert sorted(p.name for p in (tmp_path / "out" / "sweep").iterdir()) == ["polar.png", "polar.svg"]
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert ET.parse(svg).getroot().tag == "{http://www.w3.org/2000/svg}svg"


@pytest.mark.parametrize("given", ["polar", "polar.png", "polar.svg", "polar.SVG"])
def test_save_takes_the_name_with_or_without_a_suffix(tmp_path, given):
    png, svg = figures.save(a_figure(), str(tmp_path / given))
    assert (png.name, svg.name) == ("polar.png", "polar.svg")
    assert png.is_file() and svg.is_file()


def test_save_keeps_a_dot_that_isnt_a_figure_suffix(tmp_path):
    png, svg = figures.save(a_figure(), tmp_path / "polar_v1.2")
    assert (png.name, svg.name) == ("polar_v1.2.png", "polar_v1.2.svg")
    with pytest.raises(ValueError, match="needs a file name"):
        figures.save(a_figure(), "")


@pytest.mark.parametrize("name", ["light", "dark"])
def test_the_saved_files_keep_the_themes_background(tmp_path, name):
    theme = figures.THEMES[name]
    png, svg = figures.save(a_figure(name), tmp_path / "polar")  # saved outside the style's context
    pixels = imread(png)
    assert to_hex(pixels[0, 0][:3]) == theme.surface
    assert to_hex(pixels[-1, -1][:3]) == theme.surface
    assert f"fill: {theme.surface}" in svg.read_text()


def test_the_png_is_sized_by_dpi(tmp_path):
    png, _ = figures.save(a_figure(), tmp_path / "default")
    assert imread(png).shape[:2] == (600, 800)  # 3 in × 4 in at 200 dpi
    png, _ = figures.save(a_figure(), tmp_path / "small", dpi=50)
    assert imread(png).shape[:2] == (150, 200)


def test_the_same_figure_gives_the_same_files(tmp_path):
    first = [p.read_bytes() for p in figures.save(a_figure("dark", lines=3), tmp_path / "a")]
    second = [p.read_bytes() for p in figures.save(a_figure("dark", lines=3), tmp_path / "b")]
    assert first == second


# --- no Matplotlib needed for the palette


def test_the_palette_imports_without_matplotlib():
    script = (
        "import sys; sys.modules['matplotlib'] = None; sys.modules['cycler'] = None; "
        "from similitude import figures; print(figures.series_colors(2, 'dark'))"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "('#3987e5', '#d95926')"
