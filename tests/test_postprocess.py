"""Tests for the post-processing passes and new shape/badge rendering."""

from __future__ import annotations

import pytest

from modelvision import Edge, LayerNode, ModelGraph, ModelVisionWarning, NodeStyle
from modelvision.core.ir import SegmentGroup
from modelvision.core.postprocess import LARGE_MODEL_THRESHOLD, post_process
from modelvision.layout.vertical import layout_vertical
from modelvision.renderers.svg_renderer import render_svg
from modelvision.themes import get_theme

# ---------------------------------------------------------------------------
# Merge node insertion (PRD §6.2 "+ merge node")
# ---------------------------------------------------------------------------


def test_fan_in_inserts_merge_node() -> None:
    g = ModelGraph(
        nodes=[
            LayerNode(id="a", name="a", layer_type="Conv2d", framework="test"),
            LayerNode(id="b", name="b", layer_type="Conv2d", framework="test"),
            LayerNode(id="c", name="c", layer_type="ReLU", framework="test"),
        ],
        edges=[
            Edge(source_id="a", target_id="c"),
            Edge(source_id="b", target_id="c"),
        ],
    )
    out = post_process(g)
    merges = [n for n in out.nodes if n.layer_type == "Merge"]
    assert len(merges) == 1
    assert merges[0].name == "+"


def test_single_incoming_edge_no_merge() -> None:
    g = ModelGraph(
        nodes=[
            LayerNode(id="a", name="a", layer_type="X", framework="test"),
            LayerNode(id="b", name="b", layer_type="X", framework="test"),
        ],
        edges=[Edge(source_id="a", target_id="b")],
    )
    out = post_process(g)
    assert all(n.layer_type != "Merge" for n in out.nodes)


def test_shared_edges_do_not_trigger_merge() -> None:
    """Shared-weight edges are decoration and must not count as fan-in."""
    g = ModelGraph(
        nodes=[
            LayerNode(id="a", name="a", layer_type="X", framework="test"),
            LayerNode(id="b", name="b", layer_type="X", framework="test"),
            LayerNode(id="c", name="c", layer_type="X", framework="test"),
        ],
        edges=[
            Edge(source_id="a", target_id="c"),
            Edge(source_id="b", target_id="c", kind="shared"),
        ],
    )
    out = post_process(g)
    assert all(n.layer_type != "Merge" for n in out.nodes)


# ---------------------------------------------------------------------------
# Auto-collapse for >500-node models (PRD §6.4)
# ---------------------------------------------------------------------------


def test_large_model_auto_collapses_groups() -> None:
    n = LARGE_MODEL_THRESHOLD + 100
    nodes = [
        LayerNode(id=f"l{i}", name=f"l{i}", layer_type="Conv2d", framework="test", params=100)
        for i in range(n)
    ]
    g = ModelGraph(
        nodes=nodes,
        edges=[Edge(source_id=f"l{i}", target_id=f"l{i+1}") for i in range(n - 1)],
        groups=[SegmentGroup(id="layers", name="Layers", node_ids=[x.id for x in nodes])],
    )
    with pytest.warns(ModelVisionWarning, match="collapsing"):
        collapsed = post_process(g)
    assert len(collapsed.nodes) < 50
    # The collapsed placeholder inherits the aggregate param count.
    collapsed_node = next(n for n in collapsed.nodes if n.id == "layers")
    assert collapsed_node.params == 100 * (LARGE_MODEL_THRESHOLD + 100)


def test_large_model_expand_groups_bypasses_collapse() -> None:
    n = LARGE_MODEL_THRESHOLD + 50
    nodes = [
        LayerNode(id=f"l{i}", name=f"l{i}", layer_type="X", framework="test") for i in range(n)
    ]
    g = ModelGraph(
        nodes=nodes,
        edges=[Edge(source_id=f"l{i}", target_id=f"l{i+1}") for i in range(n - 1)],
        groups=[SegmentGroup(id="layers", name="Layers", node_ids=[x.id for x in nodes])],
    )
    out = post_process(g, expand_groups=True)
    assert len(out.nodes) == n


# ---------------------------------------------------------------------------
# Shape rendering (PRD §5.3.5 shape=)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["rect", "rounded_rect", "diamond", "cylinder", "parallelogram"])
def test_every_shape_variant_renders(shape: str) -> None:
    g = ModelGraph(
        nodes=[LayerNode(id="a", name="a", layer_type="X", framework="test")]
    )
    svg = render_svg(
        layout_vertical(g),
        theme=get_theme("light"),
        node_styles={"a": NodeStyle(shape=shape)},
    )
    assert svg.startswith("<?xml")
    # Rect/rounded_rect emit <rect>, diamond/parallelogram emit <polygon>,
    # cylinder emits <path>.
    if shape in {"rect", "rounded_rect"}:
        assert "<rect x=" in svg
    elif shape == "cylinder":
        assert "<path d=" in svg
    else:
        assert "<polygon points=" in svg


def test_symbolic_shape_dims_display_verbatim() -> None:
    g = ModelGraph(
        nodes=[
            LayerNode(
                id="a",
                name="a",
                layer_type="Conv2d",
                framework="test",
                output_shape=("B", 3, "H", "W"),
            )
        ]
    )
    svg = render_svg(layout_vertical(g), theme=get_theme("light"), show_shapes=True)
    assert "(B, 3, H, W)" in svg


# ---------------------------------------------------------------------------
# Badges (quantized, repeat)
# ---------------------------------------------------------------------------


def test_quantized_badge_rendered() -> None:
    node = LayerNode(
        id="a", name="a", layer_type="Conv2d", framework="test", attributes={"quantized": True}
    )
    g = ModelGraph(nodes=[node])
    svg = render_svg(layout_vertical(g), theme=get_theme("light"))
    assert 'class="mv-badge"' in svg
    assert ">Q<" in svg


def test_repeat_badge_rendered() -> None:
    node = LayerNode(
        id="a", name="a", layer_type="Attention", framework="test", attributes={"repeat": 12}
    )
    g = ModelGraph(nodes=[node])
    svg = render_svg(layout_vertical(g), theme=get_theme("dark"))
    assert "× 12" in svg


# ---------------------------------------------------------------------------
# Font handling
# ---------------------------------------------------------------------------


def test_embed_fonts_false_uses_system_font() -> None:
    from modelvision import Theme

    theme = Theme(name="t", font_family="CustomFont, serif")
    g = ModelGraph(nodes=[LayerNode(id="a", name="a", layer_type="X", framework="t")])
    svg = render_svg(layout_vertical(g), theme=theme, embed_fonts=False)
    assert "CustomFont" not in svg
    assert "system-ui" in svg


def test_embed_fonts_true_uses_theme_font() -> None:
    from modelvision import Theme

    theme = Theme(name="t", font_family="CustomFont, serif")
    g = ModelGraph(nodes=[LayerNode(id="a", name="a", layer_type="X", framework="t")])
    svg = render_svg(layout_vertical(g), theme=theme, embed_fonts=True)
    assert "CustomFont" in svg


# ---------------------------------------------------------------------------
# Large-output size warning
# ---------------------------------------------------------------------------


def test_render_warns_on_huge_svg(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from modelvision import _api

    # Lower the threshold so we don't need a real huge model.
    monkeypatch.setattr(_api, "_LARGE_SVG_BYTES", 100)
    with pytest.warns(ModelVisionWarning, match="large"):
        _api._warn_if_large("x" * 5000)
