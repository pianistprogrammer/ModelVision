"""Tests for the style resolution engine and Group validation."""

from __future__ import annotations

import pytest

from modelvision import Group, LayerNode, NodeStyle, StyleSpec, Theme
from modelvision.core.style import resolve_style


def _node(node_id: str = "layer", layer_type: str = "Conv2d") -> LayerNode:
    return LayerNode(id=node_id, name=node_id, layer_type=layer_type, framework="torch")


# ---------------------------------------------------------------------------
# Priority order — the resolver's single source of truth.
# ---------------------------------------------------------------------------


def test_theme_default_when_nothing_else_matches() -> None:
    theme = Theme(default_fill="#000000")
    style = resolve_style(_node(layer_type="Unknown"), theme=theme)
    assert style.fill == "#000000"


def test_theme_palette_beats_default() -> None:
    theme = Theme(default_fill="#000000", layer_palette={"Conv2d": "#111111"})
    style = resolve_style(_node(), theme=theme)
    assert style.fill == "#111111"


def test_user_palette_beats_theme_palette() -> None:
    theme = Theme(default_fill="#000000", layer_palette={"Conv2d": "#111111"})
    style = resolve_style(_node(), theme=theme, layer_palette={"Conv2d": "#222222"})
    assert style.fill == "#222222"


def test_wildcard_fallback() -> None:
    theme = Theme(default_fill="#000000")
    style = resolve_style(_node(layer_type="Weird"), theme=theme, layer_palette={"*": "#abcdef"})
    assert style.fill == "#abcdef"


def test_group_beats_palette() -> None:
    theme = Theme(default_fill="#000000", layer_palette={"Conv2d": "#111111"})
    style = resolve_style(
        _node(),
        theme=theme,
        groups=[Group(id="g", nodes=["layer"], fill="#333333")],
    )
    assert style.fill == "#333333"


def test_inspector_override_beats_group() -> None:
    node = _node()
    node.style_override = StyleSpec(fill="#444444")
    style = resolve_style(
        node,
        theme=Theme(),
        groups=[Group(id="g", nodes=["layer"], fill="#333333")],
    )
    assert style.fill == "#444444"


def test_node_styles_beats_everything() -> None:
    node = _node()
    node.style_override = StyleSpec(fill="#444444")
    style = resolve_style(
        node,
        theme=Theme(default_fill="#000000", layer_palette={"Conv2d": "#111111"}),
        layer_palette={"Conv2d": "#222222"},
        groups=[Group(id="g", nodes=["layer"], fill="#333333")],
        node_styles={"layer": NodeStyle(fill="#555555")},
    )
    assert style.fill == "#555555"


# ---------------------------------------------------------------------------
# Group pattern validation
# ---------------------------------------------------------------------------


def test_group_requires_exactly_one_selector() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Group(id="bad")
    with pytest.raises(ValueError, match="exactly one"):
        Group(id="bad", nodes=["a"], node_pattern="*")


def test_group_glob_pattern() -> None:
    g = Group(id="enc", node_pattern="features.*")
    assert g.matches("features.0")
    assert g.matches("features.1.conv")
    assert not g.matches("classifier.0")


def test_group_regex_pattern() -> None:
    g = Group(id="enc", node_pattern_re=r"encoder\.\d+")
    assert g.matches("encoder.0")
    assert g.matches("encoder.12")
    assert not g.matches("decoder.0")


# ---------------------------------------------------------------------------
# Merge idempotency
# ---------------------------------------------------------------------------


def test_merge_is_idempotent() -> None:
    a = StyleSpec(fill="#000000", stroke="#111111")
    b = StyleSpec(fill="#222222")
    once = a.merge(b)
    twice = once.merge(b)
    assert once == twice
