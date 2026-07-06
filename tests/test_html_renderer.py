"""Tests for the HTML renderer — structure and data serialization."""

from __future__ import annotations

import json
import re

from modelvision import Edge, LayerNode, ModelGraph
from modelvision.layout.vertical import layout_vertical
from modelvision.renderers.html_renderer import render_html
from modelvision.themes import get_theme


def _graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            LayerNode(id="a", name="a", layer_type="Conv2d", framework="test", params=100),
            LayerNode(id="b", name="b", layer_type="ReLU", framework="test"),
        ],
        edges=[Edge(source_id="a", target_id="b")],
        metadata={"model_class": "TinyNet"},
    )


def test_html_wraps_svg_and_serializes_graph() -> None:
    laid = layout_vertical(_graph())
    html = render_html(laid, theme=get_theme("light"))
    assert html.startswith("<!doctype html>")
    assert "<svg" in html
    assert 'id="mv-graph"' in html
    # Extract the JSON blob and re-parse it.
    m = re.search(r'<script id="mv-graph"[^>]*>(.*?)</script>', html, re.S)
    assert m is not None
    data = json.loads(m.group(1))
    assert {n["id"] for n in data["nodes"]} == {"a", "b"}
    assert len(data["edges"]) == 1


def test_html_contains_interaction_bindings() -> None:
    laid = layout_vertical(_graph())
    html = render_html(laid, theme=get_theme("dark"))
    assert "wheel" in html  # pan/zoom listener
    assert "data-node-id" in html  # click-to-inspect targets
    assert "mv-inspector" in html
