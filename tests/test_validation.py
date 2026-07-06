"""Tests for validation + accessibility helpers."""

from __future__ import annotations

import pytest

from modelvision import Group, LayerNode, ModelGraph, NodeStyle, Theme
from modelvision.core.exceptions import ModelVisionWarning
from modelvision.core.style import resolve_style
from modelvision.core.validation import (
    apply_accessibility,
    validate_groups,
    validate_node_styles,
)


def _graph(*ids: str) -> ModelGraph:
    return ModelGraph(
        nodes=[LayerNode(id=i, name=i, layer_type="X", framework="test") for i in ids],
    )


def test_validate_node_styles_flags_unknown_id() -> None:
    g = _graph("a", "b")
    with pytest.raises(ValueError, match="does not match any node ID"):
        validate_node_styles(g, {"nope": NodeStyle(fill="#000000")})


def test_validate_node_styles_ok_for_valid_ids() -> None:
    g = _graph("a", "b")
    # Should not raise.
    validate_node_styles(g, {"a": NodeStyle(fill="#000000")})


def test_validate_groups_strict_raises_on_overlap() -> None:
    g = _graph("a", "b", "c")
    groups = [
        Group(id="g1", nodes=["a", "b"]),
        Group(id="g2", nodes=["b", "c"]),
    ]
    with pytest.raises(ValueError, match="claimed by groups"):
        validate_groups(g, groups, strict=True)


def test_validate_groups_non_strict_warns() -> None:
    g = _graph("a", "b", "c")
    groups = [Group(id="g1", nodes=["a", "b"]), Group(id="g2", nodes=["b", "c"])]
    with pytest.warns(ModelVisionWarning, match="claimed"):
        validate_groups(g, groups, strict=False)


def test_accessibility_off_is_noop() -> None:
    g = _graph("a")
    assert apply_accessibility(
        g, mode=False, theme=Theme(), layer_palette=None, groups=None, node_styles=None
    ) is None


def test_accessibility_warns_on_low_contrast() -> None:
    g = _graph("a")
    # Light font on light fill fails AA.
    theme = Theme(default_fill="#eeeeee", font_color="#cccccc")
    with pytest.warns(ModelVisionWarning, match="WCAG AA"):
        apply_accessibility(
            g, mode=True, theme=theme, layer_palette=None, groups=None, node_styles=None
        )


def test_accessibility_enforce_bumps_font_color() -> None:
    node = LayerNode(id="a", name="a", layer_type="X", framework="test")
    g = ModelGraph(nodes=[node])
    theme = Theme(default_fill="#eeeeee", font_color="#cccccc")
    updated = apply_accessibility(
        g,
        mode="enforce",
        theme=theme,
        layer_palette=None,
        groups=None,
        node_styles=None,
    )
    assert updated is not None
    resolved = resolve_style(node, theme=theme, node_styles=updated)
    from modelvision.core.color import meets_wcag_aa

    assert meets_wcag_aa(resolved.font_color, "#eeeeee")
