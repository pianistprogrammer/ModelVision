# Changelog

All notable changes to this project will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning follows [SemVer](https://semver.org/).

## [Unreleased]

### Added

- **Core IR** (`ModelGraph`, `LayerNode`, `Edge`, `SegmentGroup`) and the
  five-level style resolver (`StyleSpec`, `Theme`, `NodeStyle`, `Group`).
- **PyTorch inspector** — direct `_modules` walk (preserves weight-tied
  duplicates), automatic unwrap for `DataParallel`, `DistributedDataParallel`,
  and `torch.compile`, per-layer-type attribute extraction, optional
  `torch.fx.symbolic_trace` cross-scope edges behind `symbolic_shapes=True`.
- **Keras inspector** — Sequential, Functional (via `_inbound_nodes`), and
  Subclassed (flat-chain fallback with warning); lazy-build via
  `input_shape`.
- **JAX inspector** — Flax `linen.tabulate` parsing with a structural
  fallback; Haiku placeholder pending further work.
- **HuggingFace inspector** — Path A delegates to the PyTorch inspector
  and folds repeated encoder/decoder layers; Path B synthesizes a
  canonical diagram from `PretrainedConfig` via architecture templates
  for BERT, RoBERTa, DistilBERT, ALBERT, GPT-2, GPT-Neo, GPT-J, LLaMA,
  Mistral, T5, BART, Whisper, ViT, and CLIP.
- **ONNX inspector** — universal fallback via `onnx.load` and
  `onnx.shape_inference.infer_shapes` (no execution); op-type remap to
  canonical layer names.
- **sklearn inspector** — `Pipeline`, `FeatureUnion`, `ColumnTransformer`,
  and `GridSearchCV.best_estimator_` traversal.
- **Layouts** — vertical (default), horizontal, radial, hierarchical
  (alias for vertical for now).
- **Renderers** — SVG (deterministic, `data-node-id` for downstream
  interaction), PNG (via cairosvg), PDF (via cairosvg), HTML
  (dependency-free inline pan/zoom + click-to-inspect), matplotlib
  (embed into an existing figure).
- **Themes** — light, dark, pastel, grayscale, high_contrast, each with
  a WCAG-tuned per-layer-type palette.
- **Accessibility** — `accessibility_check=True` warns on WCAG-AA
  violations; `accessibility_check="enforce"` auto-adjusts font colors.
- **CLI** — `mvision`/`python -m modelvision` supporting `.py` source
  files (with `--init-args` JSON), ONNX files, and Keras save files;
  `--summary` renders a `rich.Table`.
- **CI** — GitHub Actions matrix across Ubuntu/macOS/Windows × Python
  3.10/3.11/3.12 × per-framework extras, plus separate lint (ruff + mypy)
  and golden-SVG-regression jobs; PyPI release via OIDC trusted publisher.

### Notes

- All framework imports are lazy — `import modelvision` completes in
  under 200 ms and does not import torch/tensorflow/jax/etc.
- The public API is stable within a milestone; see PRD `PRD_ModelVision.md`
  for the full specification.
