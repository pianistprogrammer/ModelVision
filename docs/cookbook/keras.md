# Keras

## Sequential + Functional

```python
import tensorflow as tf
import modelvision as mv

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation="relu", input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10),
])
mv.render(model, "model.svg", theme="light", palette="pastel",
          layout="flow", input_shape=(1, 3, 32, 32))
```

![Keras equivalent model architecture rendered by ModelVision](../assets/sample_keras.png)

## Subclassed models

Subclassed models have unknown call graphs — ModelVision renders their
declared `.layers` list as a flat chain with a warning and a dashed
container border to signal the incompleteness.

## Lazy-built models

Pass `input_shape` and ModelVision will call `model.build(input_shape)`
before inspecting.

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
