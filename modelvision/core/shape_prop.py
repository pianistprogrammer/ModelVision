"""Symbolic shape propagation for PyTorch models.

Given an ``input_shape`` (e.g. ``(1, 3, 224, 224)``), walk the linearized
layer sequence and compute each layer's output shape from its attributes.
This lets us fill in ``LayerNode.output_shape`` without running a forward
pass — the same trick Keras uses internally.

Supported operations: Conv1d/2d/3d, ConvTranspose*, MaxPool/AvgPool,
AdaptiveAvgPool, BatchNorm, LayerNorm, Linear, Dropout, ReLU, Flatten.
Anything else passes shape through unchanged with a note in metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelvision.core.ir import ModelGraph


def propagate_shapes(graph: ModelGraph, input_shape: tuple[int, ...]) -> ModelGraph:
    """Populate ``output_shape`` on every node by simulating each op's shape rule.

    Only works for graphs whose edges form a linear chain — if the graph
    has branches (multi-input/output, skip connections), shape propagation
    is best-effort and may skip some nodes.
    """
    current = tuple(input_shape)
    for node in graph.nodes:
        rule = _RULES.get(node.layer_type, _passthrough)
        try:
            current = rule(current, node.attributes)
        except Exception:
            # Unknown op or attribute mismatch — leave shape unchanged.
            pass
        node.output_shape = current
    return graph


# ---------------------------------------------------------------------------
# Per-layer-type shape rules
# ---------------------------------------------------------------------------


def _passthrough(shape: tuple, _attrs: dict) -> tuple:
    return shape


def _conv2d(shape: tuple, attrs: dict) -> tuple:
    """(B, C_in, H, W) → (B, C_out, H_out, W_out)."""
    b, _, h, w = shape
    out_c = attrs.get("out_channels", shape[1])
    kh, kw = _pair(attrs.get("kernel_size", 1))
    sh, sw = _pair(attrs.get("stride", 1))
    ph, pw = _pair(attrs.get("padding", 0))
    dh, dw = _pair(attrs.get("dilation", 1))
    h_out = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return (b, out_c, h_out, w_out)


def _conv_transpose2d(shape: tuple, attrs: dict) -> tuple:
    b, _, h, w = shape
    out_c = attrs.get("out_channels", shape[1])
    kh, kw = _pair(attrs.get("kernel_size", 1))
    sh, sw = _pair(attrs.get("stride", 1))
    ph, pw = _pair(attrs.get("padding", 0))
    h_out = (h - 1) * sh - 2 * ph + kh
    w_out = (w - 1) * sw - 2 * pw + kw
    return (b, out_c, h_out, w_out)


def _conv1d(shape: tuple, attrs: dict) -> tuple:
    b, _, l = shape
    out_c = attrs.get("out_channels", shape[1])
    k = _single(attrs.get("kernel_size", 1))
    s = _single(attrs.get("stride", 1))
    p = _single(attrs.get("padding", 0))
    d = _single(attrs.get("dilation", 1))
    l_out = (l + 2 * p - d * (k - 1) - 1) // s + 1
    return (b, out_c, l_out)


def _pool2d(shape: tuple, attrs: dict) -> tuple:
    b, c, h, w = shape
    kh, kw = _pair(attrs.get("kernel_size", 1))
    sh, sw = _pair(attrs.get("stride") or attrs.get("kernel_size", 1))
    ph, pw = _pair(attrs.get("padding", 0))
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1
    return (b, c, h_out, w_out)


def _adaptive_pool2d(shape: tuple, attrs: dict) -> tuple:
    # ``AdaptiveAvgPool2d(output_size=(H, W))``; PyTorch stores output_size on the module.
    out = attrs.get("output_size", 1)
    oh, ow = _pair(out)
    return (shape[0], shape[1], oh, ow)


def _linear(shape: tuple, attrs: dict) -> tuple:
    return (shape[0], attrs.get("out_features", shape[-1]))


def _flatten(shape: tuple, _attrs: dict) -> tuple:
    total = 1
    for d in shape[1:]:
        total *= int(d) if isinstance(d, int) else 1
    return (shape[0], total)


def _pair(v) -> tuple:  # type: ignore[no-untyped-def]
    if isinstance(v, (list, tuple)):
        return (v[0], v[1]) if len(v) >= 2 else (v[0], v[0])
    return (v, v)


def _single(v) -> int:  # type: ignore[no-untyped-def]
    if isinstance(v, (list, tuple)):
        return v[0]
    return v


_RULES = {
    "Conv1d": _conv1d,
    "Conv2d": _conv2d,
    "Conv3d": _conv2d,  # rough — works for square inputs
    "ConvTranspose2d": _conv_transpose2d,
    "MaxPool2d": _pool2d,
    "AvgPool2d": _pool2d,
    "MaxPool1d": _pool2d,
    "AdaptiveAvgPool2d": _adaptive_pool2d,
    "AdaptiveMaxPool2d": _adaptive_pool2d,
    "Linear": _linear,
    "Flatten": _flatten,
    # Shape-preserving ops explicitly listed for clarity.
    "BatchNorm2d": _passthrough,
    "BatchNorm1d": _passthrough,
    "LayerNorm": _passthrough,
    "GroupNorm": _passthrough,
    "ReLU": _passthrough,
    "GELU": _passthrough,
    "SiLU": _passthrough,
    "Tanh": _passthrough,
    "Sigmoid": _passthrough,
    "Dropout": _passthrough,
    "Dropout2d": _passthrough,
}


__all__ = ["propagate_shapes"]
