"""Matplotlib renderer tests."""

from __future__ import annotations

import pytest

from modelvision import Edge, LayerNode, ModelGraph


def test_matplotlib_render_smoke() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")  # headless

    from modelvision.layout.vertical import layout_vertical
    from modelvision.renderers.matplotlib_renderer import render_matplotlib
    from modelvision.themes import get_theme

    graph = ModelGraph(
        nodes=[
            LayerNode(id="a", name="a", layer_type="Conv2d", framework="test"),
            LayerNode(id="b", name="b", layer_type="ReLU", framework="test"),
        ],
        edges=[Edge(source_id="a", target_id="b")],
    )
    laid = layout_vertical(graph)
    ax = render_matplotlib(laid, theme=get_theme("light"))
    # At least one text label was drawn.
    assert any(getattr(t, "get_text", lambda: "")() for t in ax.texts)
