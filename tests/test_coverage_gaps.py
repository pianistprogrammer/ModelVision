"""Additional targeted coverage tests — plug gaps that don't require extras."""

from __future__ import annotations

import pytest

from modelvision import Edge, LayerNode, ModelGraph
from modelvision.core._optional import require
from modelvision.core.exceptions import mv_warn
from modelvision.core.ir import SegmentGroup
from modelvision.core.validation import (
    apply_accessibility,
    validate_groups,
    validate_node_styles,
)


def test_require_raises_with_extras_hint() -> None:
    """A missing framework surfaces a friendly ``uv add`` hint."""
    with pytest.raises(ImportError, match="modelvision\\["):
        require("definitely_not_a_real_package_zzz")


def test_hierarchical_layout_falls_back_to_vertical() -> None:
    from modelvision.layout.hierarchical import layout_hierarchical

    g = ModelGraph(
        nodes=[
            LayerNode(id=str(i), name=str(i), layer_type="X", framework="test") for i in range(3)
        ],
        edges=[Edge(source_id="0", target_id="1"), Edge(source_id="1", target_id="2")],
    )
    laid = layout_hierarchical(g)
    # Vertical arrangement — y increases top-to-bottom.
    assert laid.boxes["0"].y < laid.boxes["1"].y < laid.boxes["2"].y


def test_segment_group_default_style_override_is_none() -> None:
    s = SegmentGroup(id="g", name="G", node_ids=["a"])
    assert s.style_override is None


def test_edge_default_kind_is_data() -> None:
    e = Edge(source_id="a", target_id="b")
    assert e.kind == "data"


def test_mv_warn_uses_modelvision_warning_category() -> None:
    from modelvision import ModelVisionWarning

    with pytest.warns(ModelVisionWarning):
        mv_warn("test message")


def test_validate_groups_no_op_for_empty() -> None:
    g = ModelGraph(nodes=[LayerNode(id="a", name="a", layer_type="X", framework="test")])
    validate_groups(g, None, strict=True)
    validate_groups(g, [], strict=True)


def test_validate_node_styles_no_op_for_empty() -> None:
    g = ModelGraph(nodes=[LayerNode(id="a", name="a", layer_type="X", framework="test")])
    validate_node_styles(g, None)


def test_accessibility_returns_same_dict_when_all_pass() -> None:
    from modelvision import Theme

    graph = ModelGraph(
        nodes=[LayerNode(id="a", name="a", layer_type="X", framework="test")],
    )
    good_theme = Theme(default_fill="#ffffff", font_color="#000000")
    result = apply_accessibility(
        graph, mode=True, theme=good_theme, layer_palette=None, groups=None, node_styles=None
    )
    assert result is None
