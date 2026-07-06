# API Reference

This page documents the public Python API exposed by ModelVision.

## Core Entry Points

- `render(model, ...)`: inspect and render a model diagram to SVG, PNG,
  PDF, or HTML.
- `inspect(model, ...)`: inspect a model and return graph metadata
  without rendering.

## Convenience Inspectors

Use framework-specific adapters when you want explicit control rather
than auto-detection:

- `from_torch(...)`
- `from_keras(...)`
- `from_jax(...)`
- `from_huggingface(...)`
- `from_sklearn(...)`
- `from_onnx(...)`

## Data Structures

- `ModelGraph`, `LayerNode`, `Edge`, `SegmentGroup`
- `StyleSpec`, `NodeStyle`, `Group`, `Theme`

## Full Reference

::: modelvision
    options:
      members:
        - render
        - inspect
        - from_torch
        - from_keras
        - from_jax
        - from_huggingface
        - from_sklearn
        - from_onnx
        - LayerNode
        - Edge
        - SegmentGroup
        - ModelGraph
        - StyleSpec
        - NodeStyle
        - Theme
        - Group
