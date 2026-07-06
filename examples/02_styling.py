"""Example 02 — Full styling walkthrough.

Demonstrates every level of the 5-tier style resolver: global theme,
layer-type palette, group overrides, and per-node overrides. Also
shows shape variants and the accessibility check.

Run::

    python examples/02_styling.py
"""

from __future__ import annotations

import torch.nn as nn

import modelvision as mv


class TransformerBlock(nn.Module):
    """A single transformer encoder block — enough variety to style well."""

    def __init__(self, dim: int = 128, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.ffn(self.norm2(x))


def main() -> None:
    model = TransformerBlock()

    # -----------------------------------------------------------------
    # 1. Built-in themes.
    # -----------------------------------------------------------------
    for theme in ("light", "dark", "pastel", "grayscale", "high_contrast"):
        mv.render(model, f"02_theme_{theme}.svg", theme=theme)

    # -----------------------------------------------------------------
    # 2. Custom Theme object — override anything the built-in themes set.
    # -----------------------------------------------------------------
    brand = mv.Theme(
        name="brand",
        background="#0f172a",
        default_fill="#1e293b",
        default_stroke="#334155",
        font_color="#f1f5f9",
        font_family="JetBrains Mono, monospace",
        edge_color="#94a3b8",
        group_fill="#0b1220",
        group_stroke="#475569",
        layer_palette={
            "MultiheadAttention": "#ef4444",
            "LayerNorm": "#f59e0b",
            "Linear": "#8b5cf6",
            "GELU": "#22c55e",
            "*": "#334155",  # wildcard fallback for unmapped types
        },
    )
    mv.render(model, "02_brand_theme.svg", theme=brand)

    # -----------------------------------------------------------------
    # 3. Layer-type palette — the "colour by kind" recipe.
    # -----------------------------------------------------------------
    mv.render(
        model,
        "02_palette.svg",
        theme="light",
        layer_palette={
            "MultiheadAttention": "#e74c3c",
            "LayerNorm": "#f39c12",
            "Linear": "#9b59b6",
            "GELU": "#27ae60",
            "Dropout": "#95a5a6",
        },
    )

    # -----------------------------------------------------------------
    # 4. Groups — highlight semantic blocks. Three ways to select nodes.
    # -----------------------------------------------------------------
    mv.render(
        model,
        "02_groups.svg",
        groups=[
            # Explicit list.
            mv.Group(id="attn_block", nodes=["norm1", "attn"], fill="#fecaca"),
            # Glob pattern.
            mv.Group(id="ffn_block", node_pattern="ffn.*", fill="#bfdbfe"),
            # Regex pattern.
            mv.Group(id="norms", node_pattern_re=r"norm\d+", fill="#fef3c7"),
        ],
        # Groups can overlap when strict=False (warns instead of raises).
        strict=False,
    )

    # -----------------------------------------------------------------
    # 5. Per-node styles — full StyleSpec control per layer.
    # -----------------------------------------------------------------
    mv.render(
        model,
        "02_per_node.svg",
        node_styles={
            "attn": mv.NodeStyle(
                fill="#dc2626",
                stroke="#7f1d1d",
                stroke_width=3,
                shape="diamond",
                label="Multi-Head\nAttention",
                icon="🎯",
                glow=True,
            ),
            "ffn.0": mv.NodeStyle(shape="parallelogram", fill="#fbbf24"),
            "ffn.3": mv.NodeStyle(shape="parallelogram", fill="#fbbf24"),
            "norm1": mv.NodeStyle(shape="cylinder", fill="#a78bfa"),
            "norm2": mv.NodeStyle(shape="cylinder", fill="#a78bfa"),
        },
    )

    # -----------------------------------------------------------------
    # 6. Accessibility — warn or auto-adjust.
    # -----------------------------------------------------------------
    mv.render(
        model,
        "02_a11y_warn.svg",
        theme="pastel",
        accessibility_check=True,  # emits warnings for low-contrast nodes
    )
    mv.render(
        model,
        "02_a11y_enforce.svg",
        theme="pastel",
        accessibility_check="enforce",  # auto-bumps font colors until AA passes
    )

    print("wrote all 02_*.svg files")


if __name__ == "__main__":
    main()
