"""Pastel theme — soft-hued diagrams, comfortable for slides."""

from __future__ import annotations

from modelvision.core.style import Theme

THEME = Theme(
    name="pastel",
    background="#fdfaf6",
    default_fill="#fbeaea",
    default_stroke="#8a7f8f",
    default_stroke_width=1.0,
    font_color="#3f3a44",
    font_family="Inter, system-ui, sans-serif",
    font_size=12,
    edge_color="#a89ba7",
    edge_width=1.0,
    group_fill="#f6efe8",
    group_stroke="#d9cec2",
    layer_palette={
        "Conv2d": "#b5d0e6",
        "Linear": "#d9c2e6",
        "BatchNorm2d": "#fce0b8",
        "LayerNorm": "#fce0b8",
        "ReLU": "#c8e6d0",
        "GELU": "#c8e6d0",
        "MaxPool2d": "#fbc9a8",
        "Dropout": "#e0dbe6",
        "Embedding": "#b8e0d2",
        "Attention": "#f5b5b5",
        "MultiheadAttention": "#f5b5b5",
        "*": "#e6dcd0",
    },
)
