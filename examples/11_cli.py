"""Example 11 — CLI walkthrough (this file is a bash reference, not runnable).

The ``mvision`` command is installed as a console script when you
``uv add modelvision``. Every argument below corresponds to a public
API parameter — the CLI is a thin wrapper.
"""

# Basic invocation — render a PyTorch class defined in a .py file.
"""
mvision path/to/model.py MyModel --output diagram.svg
"""

# Choose a theme and specify a per-layer palette.
"""
mvision path/to/model.py MyModel \\
    --output diagram.svg \\
    --theme dark \\
    --layer-palette "Conv2d=#4a90d9,Linear=#9b59b6,ReLU=#27ae60"
"""

# Interactive HTML output.
"""
mvision path/to/model.py MyModel --output diagram.html --theme high_contrast
"""

# Print a summary table and exit — no diagram is written.
"""
mvision path/to/model.py MyModel --summary
"""

# Constructor arguments as JSON.
"""
mvision path/to/model.py MyResNet --init-args '{"num_classes": 100}'
"""

# ONNX file — no class name needed.
"""
mvision path/to/model.onnx --output diagram.svg --theme pastel
"""

# Keras save file.
"""
mvision path/to/model.keras --output diagram.svg --theme grayscale
"""

# Horizontal layout for wide models.
"""
mvision path/to/model.py MyNet --output diagram.svg --layout horizontal
"""

# Turn on torch.fx symbolic tracing (attempts cross-scope edges).
"""
mvision path/to/model.py ResNet50 --output diagram.svg --symbolic-shapes
"""

# Lazy-built Keras models — provide an input shape.
"""
mvision path/to/model.keras --input-shape "1x3x224x224" --output diagram.svg
"""

# Under uv, all of these can be run without installing globally:
"""
uvx --from modelvision mvision path/to/model.py MyModel --output diagram.svg
uv run mvision path/to/model.py MyModel --output diagram.svg
"""

print("This file documents the CLI. Run ``mvision --help`` for the full flag list.")
