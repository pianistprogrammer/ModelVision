# JAX

## Flax

```python
import jax.numpy as jnp
import flax.linen as nn
import modelvision as mv

class TinyNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(16, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(10)(x)
        return x

mv.render(TinyNet(), "model.svg", theme="light", palette="pastel",
          layout="flow", input_shape=(1, 32, 32, 3))
```

![JAX/Flax equivalent model architecture rendered by ModelVision](../assets/sample_jax.png)

ModelVision uses `module.tabulate` internally — a shape-init pass with
no data flow, no gradients, no user inputs. The RNG key is created
internally so callers don't need to thread keys.

## Haiku

Haiku support currently emits a placeholder node — full support ships
in a follow-up release.

## Block styles

Three `style_variant` values change how every node is drawn — `"flat"` (default 2D),
`"volumetric"` (3D isometric cuboids), and `"stacked"` (channel-slice slabs).
They work with any layout:

```python
mv.render(model, "flat.svg",       layout="vertical")
mv.render(model, "volumetric.svg", layout="vertical", style_variant="volumetric")
mv.render(model, "stacked.svg",    layout="vertical", style_variant="stacked")
mv.render(model, "flow.svg",       layout="flow",     input_shape=(1, 3, 32, 32))
```

See the [Styling guide](../styling.md#block-styles) for rendered examples of each variant.
