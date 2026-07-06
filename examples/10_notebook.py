"""Example 10 — Jupyter / Colab inline display.

If you're in a notebook environment, ``mv.render(model, inline=True)``
returns a PIL Image that IPython displays inline automatically. This
script simulates the same flow outside a notebook.

Requires the ``pdf`` extra (for PIL rasterization)::

    uv add "modelvision[torch,pdf]"

Run::

    python examples/10_notebook.py

In a real notebook cell::

    import modelvision as mv
    from torchvision.models import resnet18
    mv.render(resnet18(), inline=True)   # ← displays inline
"""

from __future__ import annotations

import torch.nn as nn

import modelvision as mv


def main() -> None:
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )

    # ``inline=True`` returns a PIL Image (which Jupyter displays inline).
    try:
        img = mv.render(model, inline=True, theme="pastel")
        print(f"inline PIL image: {img.size} px")
        img.save("10_inline.png")
        print("wrote 10_inline.png")
    except ImportError as exc:
        print(f"inline display needs the ``pdf`` extra: {exc}")


if __name__ == "__main__":
    main()
