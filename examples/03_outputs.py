"""Example 03 — Every output format.

Shows how to produce SVG, PNG, PDF, HTML, an inline PIL image, and how
to draw into an existing matplotlib figure.

The PDF/PNG paths require the ``pdf`` extra (which pulls in cairosvg)::

    uv add "modelvision[pdf]"

Run::

    python examples/03_outputs.py
"""

from __future__ import annotations

import torch.nn as nn

import modelvision as mv


def tiny_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


def main() -> None:
    model = tiny_mlp()

    # -----------------------------------------------------------------
    # 1. SVG — the default, vector, publication-ready.
    # -----------------------------------------------------------------
    mv.render(model, "03_out.svg", theme="dark")

    # -----------------------------------------------------------------
    # 2. Interactive HTML — pan/zoom + click-to-inspect side panel.
    # -----------------------------------------------------------------
    mv.render(model, "03_out.html", theme="dark")

    # -----------------------------------------------------------------
    # 3. PNG — raster, useful for slides.
    # -----------------------------------------------------------------
    try:
        mv.render(model, "03_out.png", theme="dark", dpi=200)
        print("wrote 03_out.png")
    except ImportError as exc:
        print(f"PNG skipped — install the ``pdf`` extra: {exc}")

    # -----------------------------------------------------------------
    # 4. PDF — vector PDF for papers.
    # -----------------------------------------------------------------
    try:
        mv.render(model, "03_out.pdf", theme="light")
        print("wrote 03_out.pdf")
    except ImportError as exc:
        print(f"PDF skipped — install the ``pdf`` extra: {exc}")

    # -----------------------------------------------------------------
    # 5. Inline PIL Image — one-liner for Jupyter notebooks.
    # -----------------------------------------------------------------
    try:
        img = mv.render(model, inline=True, theme="light")
        img.save("03_out_from_pil.png")
        print(f"inline PIL image: {img.size}")
    except ImportError as exc:
        print(f"inline PIL skipped — install the ``pdf`` extra: {exc}")

    # -----------------------------------------------------------------
    # 6. Embed in a matplotlib figure alongside other plots.
    # -----------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        from modelvision.layout.vertical import layout_vertical
        from modelvision.renderers.matplotlib_renderer import render_matplotlib
        from modelvision.themes import get_theme

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Left: the diagram.
        graph = mv.inspect(model)
        render_matplotlib(layout_vertical(graph), theme=get_theme("light"), ax=axes[0])
        axes[0].set_title("Architecture")
        # Right: pretend loss curve to show side-by-side rendering.
        axes[1].plot([1, 0.5, 0.3, 0.2, 0.15, 0.12])
        axes[1].set_title("Training loss")
        axes[1].set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig("03_out_mpl.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("wrote 03_out_mpl.png")
    except ImportError:
        print("matplotlib not installed")

    # -----------------------------------------------------------------
    # 7. Return-as-string — useful if you're piping the SVG somewhere else.
    # -----------------------------------------------------------------
    svg_text = mv.render(model)  # output=None → returns the SVG string
    print(f"in-memory SVG: {len(svg_text):,} chars")


if __name__ == "__main__":
    main()
