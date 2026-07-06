"""Tests for the ONNX inspector."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.onnx


def _make_tiny_onnx(path):  # type: ignore[no-untyped-def]
    """Build a tiny ONNX graph: input → Conv → Relu → output."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    input_ = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 3, 3, 3], [0.0] * (4 * 3 * 3 * 3)
    )
    conv_out = helper.make_tensor_value_info("conv_out", TensorProto.FLOAT, [1, 4, 6, 6])
    output_ = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 6, 6])
    conv = helper.make_node("Conv", ["x", "w"], ["conv_out"], name="conv1", kernel_shape=[3, 3])
    relu = helper.make_node("Relu", ["conv_out"], ["y"], name="relu1")
    graph = helper.make_graph(
        [conv, relu],
        "tiny",
        [input_],
        [output_],
        [weight],
        value_info=[conv_out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def test_onnx_inspector_reads_ops(tmp_path) -> None:  # type: ignore[no-untyped-def]
    onnx_path = tmp_path / "tiny.onnx"
    _make_tiny_onnx(onnx_path)

    from modelvision import inspect

    g = inspect(str(onnx_path))
    assert g.metadata["framework"] == "onnx"
    types = [n.layer_type for n in g.nodes]
    assert "Conv2d" in types  # remapped from ONNX "Conv"
    assert "ReLU" in types
    # Two nodes, one edge.
    assert len(g.nodes) == 2
    assert len(g.edges) == 1
    # Output shapes inferred.
    conv = next(n for n in g.nodes if n.layer_type == "Conv2d")
    assert conv.output_shape is not None
