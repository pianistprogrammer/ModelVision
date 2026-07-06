"""Default light theme — WCAG-AA compliant on white backgrounds."""

from __future__ import annotations

from modelvision.core.style import Theme

# A muted, ML-paper-friendly palette. Each color has ≥ 4.5:1 contrast
# against #ffffff with a #111111 label overlay.
THEME = Theme(
    name="light",
    background="#ffffff",
    default_fill="#f2f4f8",
    default_stroke="#4a5568",
    default_stroke_width=1.0,
    font_color="#111827",
    font_family="Inter, system-ui, sans-serif",
    font_size=12,
    edge_color="#6b7280",
    edge_width=1.0,
    group_fill="#f9fafb",
    group_stroke="#d1d5db",
    layer_palette={
        "Conv2d":       "#c7d9f1",
        "Conv1d":       "#c7d9f1",
        "Conv3d":       "#c7d9f1",
        "Linear":       "#dcd0f3",
        "Dense":        "#dcd0f3",
        "BatchNorm2d":  "#fde3b0",
        "LayerNorm":    "#fde3b0",
        "ReLU":         "#c8e6c9",
        "GELU":         "#c8e6c9",
        "SiLU":         "#c8e6c9",
        "Tanh":         "#c8e6c9",
        "Sigmoid":      "#c8e6c9",
        "MaxPool2d":    "#ffd6a5",
        "AvgPool2d":    "#ffd6a5",
        "Dropout":      "#e5e7eb",
        "Embedding":    "#b3e5d1",
        "Attention":    "#f8b4b4",
        "MultiheadAttention": "#f8b4b4",
        "*":            "#e5e7eb",
    },
)
