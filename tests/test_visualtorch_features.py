"""Tests for the visualtorch-inspired features:

- ``palette`` argument + ``build_layer_palette`` helper
- ``volumetric`` + ``style_variant`` rendering modes
- ``legend`` rendering
- ``size_by_shape`` layout transform
- isometric / stacked shape primitives
"""

from __future__ import annotations

import pytest

from modelvision import (
    PALETTES,
    Edge,
    LayerNode,
    ModelGraph,
    build_layer_palette,
)
from modelvision.core.palettes import resolve_palette
from modelvision.layout.shape_size import resize_by_shape
from modelvision.layout.vertical import layout_vertical
from modelvision.renderers.svg_renderer import render_svg
from modelvision.themes import get_theme

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------


def test_okabe_ito_is_available() -> None:
    assert "okabe_ito" in PALETTES
    assert len(PALETTES["okabe_ito"]) == 7


def test_resolve_palette_by_name() -> None:
    assert resolve_palette("okabe_ito") == PALETTES["okabe_ito"]


def test_resolve_palette_passthrough_list() -> None:
    custom = ["#ff0000", "#00ff00"]
    assert resolve_palette(custom) == custom


def test_resolve_palette_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown palette"):
        resolve_palette("no_such_palette_zzz")


def test_build_layer_palette_wraps_colors() -> None:
    # Palette with 2 colors + 12 assignments → cycles.
    p = build_layer_palette(["#111111", "#222222"])
    assert p["Conv2d"] == "#111111"
    assert p["Linear"] == "#222222"
    assert p["BatchNorm2d"] == "#111111"  # wrap-around


def test_build_layer_palette_wildcard() -> None:
    p = build_layer_palette("okabe_ito", wildcard="#000000")
    assert p["*"] == "#000000"


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


def _tiny_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            LayerNode(id="a", name="a", layer_type="Conv2d", framework="test", params=100),
            LayerNode(id="b", name="b", layer_type="ReLU", framework="test"),
            LayerNode(id="c", name="c", layer_type="Linear", framework="test", params=50),
        ],
        edges=[Edge(source_id="a", target_id="b"), Edge(source_id="b", target_id="c")],
    )


def test_legend_off_by_default() -> None:
    svg = render_svg(layout_vertical(_tiny_graph()), theme=get_theme("light"))
    assert 'class="mv-legend"' not in svg


def test_legend_lists_all_present_types() -> None:
    svg = render_svg(
        layout_vertical(_tiny_graph()),
        theme=get_theme("light"),
        layer_palette=build_layer_palette("okabe_ito"),
        legend=True,
    )
    assert 'class="mv-legend"' in svg
    # Every layer type present in the graph should appear once.
    for lt in ("Conv2d", "ReLU", "Linear"):
        assert f">{lt}<" in svg


def test_legend_skips_types_with_no_palette_entry() -> None:
    """Types with no palette entry (from theme *or* user) don't get legend rows."""
    import re

    from modelvision import Theme

    # A theme with an empty layer_palette so only the user's dict counts.
    bare = Theme(name="bare", layer_palette={})
    svg = render_svg(
        layout_vertical(_tiny_graph()),
        theme=bare,
        layer_palette={"Conv2d": "#ff0000"},  # only one entry, no wildcard
        legend=True,
    )
    match = re.search(r'<g class="mv-legend"[^>]*>(.*?)</g>', svg, re.S)
    assert match is not None
    legend_body = match.group(1)
    assert ">Conv2d<" in legend_body
    assert ">ReLU<" not in legend_body
    assert ">Linear<" not in legend_body


# ---------------------------------------------------------------------------
# Isometric + stacked shape primitives
# ---------------------------------------------------------------------------


def test_isometric_shape_draws_three_faces() -> None:
    from modelvision.core.style import NodeStyle

    svg = render_svg(
        layout_vertical(_tiny_graph()),
        theme=get_theme("light"),
        node_styles={"a": NodeStyle(shape="isometric")},
    )
    # Isometric emits 2 polygons (top + right) plus 1 rect (front) for that node.
    # Rough check: at least two <polygon> tags appear in the SVG.
    assert svg.count("<polygon") >= 2


def test_stacked_shape_draws_multiple_slices() -> None:
    from modelvision.core.style import NodeStyle

    svg = render_svg(
        layout_vertical(_tiny_graph()),
        theme=get_theme("light"),
        node_styles={"a": NodeStyle(shape="stacked")},
    )
    # Stacked emits ~6 slices — plenty of extra rects.
    assert svg.count("<rect") >= 6


def test_default_shape_applies_when_style_has_none() -> None:
    """``default_shape=`` promotes every un-shaped node to that shape."""
    svg = render_svg(
        layout_vertical(_tiny_graph()),
        theme=get_theme("light"),
        default_shape="isometric",
    )
    # Every node → 2 polygons apiece.
    assert svg.count("<polygon") >= 6


# ---------------------------------------------------------------------------
# Size-by-shape
# ---------------------------------------------------------------------------


def test_size_by_shape_widens_big_channel_layers() -> None:
    """A layer with 512 channels should render wider than one with 16."""
    graph = ModelGraph(
        nodes=[
            LayerNode(
                id="small",
                name="small",
                layer_type="Conv2d",
                framework="test",
                output_shape=("B", 16, 32, 32),
            ),
            LayerNode(
                id="big",
                name="big",
                layer_type="Conv2d",
                framework="test",
                output_shape=("B", 512, 8, 8),
            ),
        ],
    )
    resized = resize_by_shape(layout_vertical(graph))
    assert resized.boxes["big"].width > resized.boxes["small"].width


def test_size_by_shape_missing_shape_still_produces_valid_box() -> None:
    graph = ModelGraph(
        nodes=[
            LayerNode(id="a", name="a", layer_type="X", framework="t"),
            LayerNode(id="b", name="b", layer_type="X", framework="t"),
        ],
    )
    resized = resize_by_shape(layout_vertical(graph))
    assert resized.boxes["a"].width > 0
    assert resized.boxes["a"].height > 0


# ---------------------------------------------------------------------------
# End-to-end via public API
# ---------------------------------------------------------------------------


def test_render_volumetric_end_to_end() -> None:
    from modelvision import render

    torch = pytest.importorskip("torch")
    import torch.nn as nn

    svg = render(
        nn.Linear(4, 4),
        palette="okabe_ito",
        volumetric=True,
        legend=True,
    )
    assert "<polygon" in svg  # extruded faces
    assert 'class="mv-legend"' in svg


def test_render_unknown_style_variant_raises() -> None:
    from modelvision import render
    from modelvision.core.exceptions import RenderError

    torch = pytest.importorskip("torch")
    import torch.nn as nn

    with pytest.raises(RenderError, match="Unknown style_variant"):
        render(nn.Linear(4, 4), style_variant="not-a-real-mode")
