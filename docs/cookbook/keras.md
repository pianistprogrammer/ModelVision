# Keras

## Sequential + Functional

```python
from tensorflow.keras.applications import ResNet50
import modelvision as mv

mv.render(
    ResNet50(),
    output="resnet50.html",
    groups=[
        mv.Group("stem",    node_pattern="conv1_*", fill="#1a237e"),
        mv.Group("stage_1", node_pattern="conv2_*", fill="#b71c1c"),
        mv.Group("stage_2", node_pattern="conv3_*", fill="#1b5e20"),
        mv.Group("head",    nodes=["avg_pool", "predictions"], fill="#4a148c"),
    ],
    layout="horizontal",
)
```

## Subclassed models

Subclassed models have unknown call graphs — ModelVision renders their
declared `.layers` list as a flat chain with a warning and a dashed
container border to signal the incompleteness.

## Lazy-built models

Pass `input_shape` and ModelVision will call `model.build(input_shape)`
before inspecting.
