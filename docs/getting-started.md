# Installation and Quick Start

## Install

Use the package variant that matches your workflow.

```bash
# Minimal core (recommended to start).
uv add modelvision

# Framework-specific extras.
uv add "modelvision[torch]"
uv add "modelvision[tensorflow]"
uv add "modelvision[jax]"
uv add "modelvision[huggingface]"
uv add "modelvision[sklearn]"

# Output extras for PDF and high-quality raster export.
uv add "modelvision[pdf]"

# Full install with all integrations.
uv add "modelvision[all]"
```

## First Python Render

```python
import torch.nn as nn
import modelvision as mv


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 10),
        )


mv.render(TinyNet(), output="tinynet.svg", theme="light", layout="flow",
          input_shape=(1, 3, 32, 32))
```

![TinyNet architecture diagram rendered by ModelVision](assets/sample_getting_started.png)

This creates an SVG architecture diagram from the model structure.

## First CLI Render

```bash
mvision render model.py MyNet -o tinynet.svg --theme dark
```

Use this command path when integrating with scripts, CI, or LLM-based
tool calling.

## Next Steps

- [Styling Guide](styling.md)
- [CLI Reference](cli.md)
- [Framework Guides](cookbook/index.md)
- [API Reference](api.md)
