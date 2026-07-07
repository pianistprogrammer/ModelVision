"""Dark theme — WCAG-AA compliant on ``#1a1a2e`` background."""

from __future__ import annotations

from modelvision.core.style import Theme

THEME = Theme(
    name="dark",
    background="#1a1a2e",
    default_fill="#2a2f4a",
    default_stroke="#4b5573",
    default_stroke_width=1.0,
    font_color="#f5f5fa",
    font_family="Inter, system-ui, sans-serif",
    font_size=12,
    edge_color="#8a92b2",
    edge_width=1.0,
    group_fill="#232842",
    group_stroke="#3a4265",
    layer_palette={
        "Conv2d": "#4a90d9",
        "Conv1d": "#4a90d9",
        "Conv3d": "#4a90d9",
        "Linear": "#9b59b6",
        "Dense": "#9b59b6",
        "BatchNorm2d": "#f5a623",
        "LayerNorm": "#f5a623",
        "ReLU": "#7ed321",
        "GELU": "#7ed321",
        "SiLU": "#7ed321",
        "Tanh": "#7ed321",
        "Sigmoid": "#7ed321",
        "MaxPool2d": "#e67e22",
        "AvgPool2d": "#e67e22",
        "Dropout": "#95a5a6",
        "Embedding": "#1abc9c",
        "Attention": "#e74c3c",
        "MultiheadAttention": "#e74c3c",
        "*": "#4b5573",
    },
)
