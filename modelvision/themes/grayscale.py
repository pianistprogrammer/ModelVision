"""Grayscale theme — print-safe, journal-friendly."""

from __future__ import annotations

from modelvision.core.style import Theme

THEME = Theme(
    name="grayscale",
    background="#ffffff",
    default_fill="#e5e5e5",
    default_stroke="#333333",
    default_stroke_width=1.0,
    font_color="#111111",
    font_family="Inter, system-ui, sans-serif",
    font_size=12,
    edge_color="#555555",
    edge_width=1.0,
    group_fill="#f2f2f2",
    group_stroke="#999999",
    layer_palette={
        "Conv2d":       "#cccccc",
        "Linear":       "#b3b3b3",
        "BatchNorm2d":  "#e0e0e0",
        "LayerNorm":    "#e0e0e0",
        "ReLU":         "#f0f0f0",
        "MaxPool2d":    "#a0a0a0",
        "Dropout":      "#dcdcdc",
        "Embedding":    "#c0c0c0",
        "Attention":    "#888888",
        "MultiheadAttention": "#888888",
        "*":            "#d9d9d9",
    },
)
