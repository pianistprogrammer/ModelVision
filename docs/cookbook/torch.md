# PyTorch

## Basic model

```python
import torchvision.models as models
import modelvision as mv

mv.render(
    models.vgg16(),
    output="vgg16.svg",
    theme="dark",
    layer_palette={
        "Conv2d":    "#4a90d9",
        "ReLU":      "#27ae60",
        "MaxPool2d": "#e67e22",
        "Linear":    "#9b59b6",
        "Dropout":   "#7f8c8d",
    },
)
```

## Cross-scope edges (opt-in)

By default the PyTorch inspector emits edges between sibling modules
inside each parent container. To recover cross-scope edges (e.g. a
skip connection that jumps between `Sequential` blocks), pass
`symbolic_shapes=True`:

```python
mv.render(model, symbolic_shapes=True)
```

This uses `torch.fx.symbolic_trace` internally. If tracing fails (dynamic
control flow, unsupported ops), it falls back to the sequential edges
with a warning — never an error.

## Weight tying

Weight-tied modules (`self.a = shared; self.b = shared`) are detected
automatically. Each site is rendered as its own node, connected by a
dashed "shared" edge. Suppress with `show_shared_weights=False`.

## Wrappers

`DataParallel`, `DistributedDataParallel`, and `torch.compile` wrappers
are unwrapped automatically before inspection — you don't need to
call `.module` or `._orig_mod` yourself.
