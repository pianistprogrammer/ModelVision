"""Small SVG renderer coverage additions — dashed edges, edge labels, long labels."""

from __future__ import annotations

from modelvision import Edge, LayerNode, ModelGraph, NodeStyle
from modelvision.layout.vertical import layout_vertical
from modelvision.renderers.svg_renderer import _human_params, _truncate, render_svg
from modelvision.themes import get_theme


def test_shared_edges_use_dashed_stroke() -> None:
    g = ModelGraph(
        nodes=[
            LayerNode(id="a", name="a", layer_type="X", framework="t"),
            LayerNode(id="b", name="b", layer_type="X", framework="t"),
        ],
        edges=[Edge(source_id="a", target_id="b", kind="shared", label="tied")],
    )
    svg = render_svg(layout_vertical(g), theme=get_theme("light"))
    assert "stroke-dasharray" in svg
    assert ">tied<" in svg


def test_long_label_is_truncated_and_full_kept_in_title() -> None:
    long_name = "very_long_layer_name_that_exceeds_the_thirty_character_limit"
    g = ModelGraph(nodes=[LayerNode(id="x", name=long_name, layer_type="Conv2d", framework="t")])
    svg = render_svg(layout_vertical(g), theme=get_theme("dark"))
    assert "…" in svg  # ellipsized short label
    # Full label preserved in the title tooltip.
    assert f"<title>{long_name}" in svg


def test_human_params_formatting() -> None:
    assert _human_params(50) == "50 params"
    assert _human_params(1500).endswith("K params")
    assert _human_params(2_500_000).endswith("M params")
    assert _human_params(3_500_000_000).endswith("B params")


def test_truncate_no_op_for_short_strings() -> None:
    short = "abc"
    result, full = _truncate(short, 10)
    assert result == short
    assert full == short


def test_node_style_font_size_and_weight_applied() -> None:
    g = ModelGraph(nodes=[LayerNode(id="a", name="a", layer_type="X", framework="t")])
    svg = render_svg(
        layout_vertical(g),
        theme=get_theme("light"),
        node_styles={"a": NodeStyle(font_size=20, font_weight="900")},
    )
    assert 'font-size="20"' in svg
    assert 'font-weight="900"' in svg


def test_output_shape_shown_in_subtitle() -> None:
    g = ModelGraph(
        nodes=[
            LayerNode(
                id="a",
                name="a",
                layer_type="Conv2d",
                framework="t",
                output_shape=("B", 3, 224, 224),
            )
        ]
    )
    svg = render_svg(layout_vertical(g), theme=get_theme("light"), show_shapes=True)
    assert "→" in svg  # arrow prefix for shape subtitle
