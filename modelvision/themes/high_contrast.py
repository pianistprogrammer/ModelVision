"""High-contrast theme — maximum readability, accessibility-first."""

from __future__ import annotations

from modelvision.core.style import Theme

THEME = Theme(
    name="high_contrast",
    background="#ffffff",
    default_fill="#ffffff",
    default_stroke="#000000",
    default_stroke_width=2.0,
    font_color="#000000",
    font_family="Inter, system-ui, sans-serif",
    font_size=13,
    edge_color="#000000",
    edge_width=1.5,
    group_fill="#ffffff",
    group_stroke="#000000",
    layer_palette={
        "Conv2d":       "#0044aa",
        "Linear":       "#7700aa",
        "BatchNorm2d":  "#aa5500",
        "LayerNorm":    "#aa5500",
        "ReLU":         "#007744",
        "MaxPool2d":    "#aa3300",
        "Dropout":      "#666666",
        "Embedding":    "#006677",
        "Attention":    "#aa0033",
        "MultiheadAttention": "#aa0033",
        "*":            "#333333",
    },
)
