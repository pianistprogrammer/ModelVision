# ModelVision

Framework-agnostic neural network architecture visualizer. Renders publication-quality, fully styleable diagrams for models built in **PyTorch, TensorFlow/Keras, JAX/Flax, JAX/Haiku, Hugging Face Transformers, scikit-learn, and ONNX** — without running a forward pass.

> **Status:** early development (Week 0 foundations landing; M1 next). See `PRD_ModelVision.md` for the full spec.

## Install

```bash
uv add modelvision                    # core only
uv add "modelvision[torch]"           # per-framework extras
uv add "modelvision[all]"             # everything
```

## Quick start

```python
import torchvision.models as models
import modelvision as mvision

model = models.vgg16()
mvision.render(
    model,
    output="vgg16.svg",
    theme="dark",
    layer_palette={
        "Conv2d":    "#4a90d9",
        "ReLU":      "#27ae60",
        "MaxPool2d": "#e67e22",
        "Linear":    "#9b59b6",
    },
)
```

## CLI

```bash
uvx modelvision model.py MyNet --output diagram.svg --theme dark
uvx modelvision model.onnx --output diagram.html
```

## Development

```bash
git clone https://github.com/pianistprogrammer/ModelVision.git
cd modelvision
uv sync --all-extras --extra dev
uv run pytest
```

## License

MIT
