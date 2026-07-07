"""ONNX inspector — the universal fallback path.

Uses ``onnx.load`` + ``onnx.shape_inference.infer_shapes`` — neither
executes the model. Op-type names map to canonical layer_type strings
so the theme's palette lookups work the same way as for framework-native
inspection.
"""

from __future__ import annotations

from typing import Any

from modelvision.core._optional import require
from modelvision.core.ids import sanitize, uniquify
from modelvision.core.ir import Edge, LayerNode, ModelGraph
from modelvision.inspectors.base import BaseInspector

# ONNX op → canonical layer_type. Anything not listed keeps its ONNX
# op-type verbatim, which the palette can still target via wildcard.
_OP_MAP: dict[str, str] = {
    "Conv": "Conv2d",
    "ConvTranspose": "ConvTranspose2d",
    "Gemm": "Linear",
    "MatMul": "Linear",
    "BatchNormalization": "BatchNorm2d",
    "LayerNormalization": "LayerNorm",
    "Relu": "ReLU",
    "Gelu": "GELU",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "MaxPool": "MaxPool2d",
    "AveragePool": "AvgPool2d",
    "GlobalAveragePool": "AdaptiveAvgPool2d",
    "Dropout": "Dropout",
    "Flatten": "Flatten",
    "Softmax": "Softmax",
    "Add": "Add",
    "Concat": "Concat",
    "Reshape": "Reshape",
    "Transpose": "Transpose",
    "Gather": "Embedding",
    "Attention": "Attention",
}


class ONNXInspector(BaseInspector):
    framework = "onnx"

    def can_handle(self, model: Any) -> bool:
        if isinstance(model, str):
            return model.lower().endswith(".onnx")
        return type(model).__module__.startswith("onnx.")

    def inspect(self, model: Any, **_: Any) -> ModelGraph:
        onnx = require("onnx")
        proto = onnx.load(model) if isinstance(model, str) else model
        try:
            proto = onnx.shape_inference.infer_shapes(proto)
        except Exception:
            pass  # shape inference is best-effort

        onnx_graph = proto.graph
        shape_by_output = _collect_shapes(onnx_graph)

        proposed = [
            sanitize(node.name or f"{node.op_type}_{i}") for i, node in enumerate(onnx_graph.node)
        ]
        ids = uniquify(proposed)

        # Map ONNX tensor name → producing node id.
        producer: dict[str, str] = {}
        for node, nid in zip(onnx_graph.node, ids, strict=True):
            for out in node.output:
                producer[out] = nid

        nodes: list[LayerNode] = []
        for node, nid in zip(onnx_graph.node, ids, strict=True):
            layer_type = _OP_MAP.get(node.op_type, node.op_type)
            out_shape = None
            if node.output and node.output[0] in shape_by_output:
                out_shape = shape_by_output[node.output[0]]
            attrs = {a.name: _attr_value(a) for a in node.attribute}
            nodes.append(
                LayerNode(
                    id=nid,
                    name=node.name or node.op_type,
                    layer_type=layer_type,
                    framework="onnx",
                    output_shape=out_shape,
                    attributes=attrs,
                )
            )

        edges: list[Edge] = []
        for node, nid in zip(onnx_graph.node, ids, strict=True):
            for inp in node.input:
                src = producer.get(inp)
                if src and src != nid:
                    edges.append(Edge(source_id=src, target_id=nid))

        return ModelGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                "framework": "onnx",
                "producer": onnx_graph.name or "onnx_model",
                "opset": [o.version for o in proto.opset_import][:1],
            },
        )


def _collect_shapes(graph: Any) -> dict[str, tuple[Any, ...]]:
    shapes: dict[str, tuple[Any, ...]] = {}
    for value_info in list(graph.value_info) + list(graph.output) + list(graph.input):
        tensor_type = value_info.type.tensor_type
        dims: list[Any] = []
        for d in tensor_type.shape.dim:
            if d.dim_value:
                dims.append(d.dim_value)
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append("?")
        shapes[value_info.name] = tuple(dims)
    return shapes


def _attr_value(attr: Any) -> Any:
    # ONNX attribute types: 1=FLOAT, 2=INT, 3=STRING, 6=FLOATS, 7=INTS, 8=STRINGS.
    if attr.type == 1:
        return attr.f
    if attr.type == 2:
        return attr.i
    if attr.type == 3:
        return attr.s.decode() if isinstance(attr.s, bytes) else attr.s
    if attr.type == 6:
        return list(attr.floats)
    if attr.type == 7:
        return list(attr.ints)
    if attr.type == 8:
        return [s.decode() if isinstance(s, bytes) else s for s in attr.strings]
    return None


__all__ = ["ONNXInspector"]
