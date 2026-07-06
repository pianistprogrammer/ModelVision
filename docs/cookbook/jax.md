# JAX

## Flax

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import modelvision as mv

class MLP(nn.Module):
    hidden: int = 32
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x); x = nn.relu(x)
        return nn.Dense(2)(x)

mv.render(MLP(), input_shape=(1, 4), output="mlp.svg")
```

ModelVision uses `module.tabulate` internally — a shape-init pass with
no data flow, no gradients, no user inputs. The RNG key is created
internally so callers don't need to thread keys.

## Haiku

Haiku support currently emits a placeholder node — full support ships
in a follow-up release.
