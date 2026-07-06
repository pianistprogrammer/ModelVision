"""Example 04 — Layouts side-by-side.

Renders the same model in every layout so you can pick the one that
fits your architecture and page format.

Run::

    python examples/04_layouts.py
"""

from __future__ import annotations

import torch.nn as nn

import modelvision as mv


class UNetBlock(nn.Module):
    """A branching encoder/decoder — a good stress test for radial and horizontal."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.bottleneck = nn.Conv2d(64, 128, 3, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(32, 1, 1)


def main() -> None:
    model = UNetBlock()

    for layout in ("vertical", "horizontal", "radial", "hierarchical"):
        mv.render(model, f"04_layout_{layout}.svg", layout=layout, theme="dark")
        print(f"wrote 04_layout_{layout}.svg")


if __name__ == "__main__":
    main()
