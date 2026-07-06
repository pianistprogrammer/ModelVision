"""Tests for the SVG renderer — structure, determinism, edge cases."""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from modelvision import Edge, Group, LayerNode, ModelGraph, NodeStyle, Theme
from modelvision.layout.vertical import layout_vertical
from modelvision.renderers.svg_renderer import render_svg
from modelvision.themes import get_theme


def _tiny_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            LayerNode(id="a", name="a", layer_type="Conv2d", framework="torch", params=100),
            LayerNode(id="b", name="b", layer_type="ReLU", framework="torch"),
        ],
        edges=[Edge(source_id="a", target_id="b")],
    )


def _parse_svg(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def test_svg_is_well_formed() -> None:
    laid = layout_vertical(_tiny_graph())
    svg = render_svg(laid, theme=get_theme("light"))
    root = _parse_svg(svg)
    assert root.tag.endswith("svg")


def test_data_node_id_attributes_present() -> None:
    laid = layout_vertical(_tiny_graph())
    svg = render_svg(laid, theme=get_theme("dark"))
    assert 'data-node-id="a"' in svg
    assert 'data-node-id="b"' in svg


def test_output_is_deterministic() -> None:
    g = _tiny_graph()
    laid = layout_vertical(g)
    svg1 = render_svg(laid, theme=get_theme("light"))
    svg2 = render_svg(laid, theme=get_theme("light"))
    assert svg1 == svg2


def test_user_node_style_overrides_palette() -> None:
    laid = layout_vertical(_tiny_graph())
    svg = render_svg(
        laid,
        theme=get_theme("light"),
        node_styles={"a": NodeStyle(fill="#ff00ff")},
    )
    assert "#ff00ff" in svg


def test_group_regex_pattern_matches_nodes() -> None:
    g = _tiny_graph()
    laid = layout_vertical(g)
    svg = render_svg(
        laid,
        theme=get_theme("light"),
        groups=[Group(id="all", node_pattern_re=r"[ab]", fill="#00ff00")],
    )
    assert "#00ff00" in svg


def test_unknown_theme_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown theme"):
        get_theme("not-a-theme")


def test_theme_object_passes_through() -> None:
    t = Theme(name="custom", background="#010203")
    assert get_theme(t) is t
