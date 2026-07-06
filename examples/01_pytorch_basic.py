"""Example 01 — Basic PyTorch model rendering.

The simplest end-to-end use case: define a model, render it to SVG.

Run with the ``torch`` extra installed::

    uv add "modelvision[torch]"
    python examples/01_pytorch_basic.py
"""

from __future__ import annotations

import torch.nn as nn

import modelvision as mv


class TinyCNN(nn.Module):
    """A minimal CNN — three convs, one classifier head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main() -> None:
    model = TinyCNN(num_classes=10)

    # 1. The one-liner — auto-detects the framework, uses the ``light`` theme.
    mv.render(model, "01_basic.svg")

    # 2. Peek inside — the same call returns the SVG string if no output path.
    svg = mv.render(model)
    print(f"SVG size: {len(svg):,} chars")

    # 3. Just want the graph? ``inspect`` gives you the IR.
    graph = mv.inspect(model)
    print(f"Model has {len(graph.nodes)} layers, {len(graph.groups)} groups")
    print(f"Total parameters: {graph.metadata['total_params']:,}")

    for node in graph.nodes[:3]:
        print(f"  {node.id:20s} {node.layer_type:15s} params={node.params}")
    print("  ...")


if __name__ == "__main__":
    main()
