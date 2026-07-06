"""Tests for the ModelGraph IR dataclasses."""

from __future__ import annotations

from modelvision import Edge, LayerNode, ModelGraph, SegmentGroup


def test_layer_node_roundtrip() -> None:
    n = LayerNode(
        id="features.0",
        name="conv1",
        layer_type="Conv2d",
        framework="torch",
        params=448,
        attributes={"kernel_size": (3, 3)},
    )
    d = n.to_dict()
    assert d["id"] == "features.0"
    assert d["attributes"]["kernel_size"] == (3, 3)


def test_graph_helpers() -> None:
    g = ModelGraph(
        nodes=[
            LayerNode(id="a", name="a", layer_type="Linear", framework="torch"),
            LayerNode(id="b", name="b", layer_type="Linear", framework="torch"),
            LayerNode(id="c", name="c", layer_type="Linear", framework="torch"),
        ],
        edges=[Edge(source_id="a", target_id="b"), Edge(source_id="a", target_id="c")],
    )
    assert g.node_ids() == ["a", "b", "c"]
    assert g.in_degree("a") == 0
    assert g.in_degree("b") == 1
    assert g.out_degree("a") == 2


def test_segment_group_dict() -> None:
    s = SegmentGroup(id="enc", name="Encoder", node_ids=["a", "b"])
    assert s.to_dict() == {
        "id": "enc",
        "name": "Encoder",
        "node_ids": ["a", "b"],
        "style_override": None,
    }
