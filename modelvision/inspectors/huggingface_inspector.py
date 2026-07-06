"""Hugging Face Transformers inspector.

Two paths:

- **Path A (delegate to torch)**: if the given object is an
  ``nn.Module`` (a loaded model instance), delegate to
  :class:`~modelvision.inspectors.torch_inspector.PyTorchInspector` and
  post-process — repeated encoder/decoder layers (``layer.0``,
  ``layer.1``, …) are folded into a single canonical block with a
  ``× N`` badge.
- **Path B (config-only)**: if the given object is a
  ``PretrainedConfig``, use the architecture template registry to
  construct a canonical diagram from
  ``config.num_hidden_layers`` / ``hidden_size`` / ``num_attention_heads``.

The template registry (:mod:`modelvision.inspectors.hf_templates`)
ships templates for the eight most-used architectures plus a generic
encoder / decoder / encoder-decoder fallback.
"""

from __future__ import annotations

from typing import Any

from modelvision.core._optional import require
from modelvision.core.exceptions import mv_warn
from modelvision.core.ir import ModelGraph
from modelvision.inspectors.base import BaseInspector
from modelvision.inspectors.hf_templates import build_from_config, is_config


class HuggingFaceInspector(BaseInspector):
    framework = "huggingface"

    def can_handle(self, model_or_config: Any) -> bool:
        mod = type(model_or_config).__module__
        return mod.startswith("transformers.")

    def inspect(
        self,
        model_or_config: Any,
        *,
        expand_groups: bool = False,
        **kwargs: Any,
    ) -> ModelGraph:
        # Config-only path.
        if is_config(model_or_config):
            return build_from_config(model_or_config)

        # Model-instance path — delegate to torch inspector, then fold.
        require("transformers")
        from modelvision.inspectors.torch_inspector import PyTorchInspector

        base_graph = PyTorchInspector().inspect(model_or_config, **kwargs)
        base_graph.metadata["framework"] = "huggingface"
        base_graph.metadata["model_class"] = type(model_or_config).__name__

        if not expand_groups:
            base_graph = _fold_repeated_blocks(base_graph, model_or_config)
        return base_graph


# ---------------------------------------------------------------------------
# Repeated-block folding
# ---------------------------------------------------------------------------


def _fold_repeated_blocks(graph: ModelGraph, model: Any) -> ModelGraph:
    """Collapse ``encoder.layer.{0,1,...,N-1}`` into a single ``layer.× N`` node.

    We only fold when we can prove (via ``model.config.num_hidden_layers``)
    exactly how many blocks are expected — otherwise we leave the graph alone.
    """
    cfg = getattr(model, "config", None)
    n_layers = getattr(cfg, "num_hidden_layers", None) if cfg else None
    if not n_layers:
        return graph

    # Find any parent path that appears repeated with numeric-index children.
    families: dict[str, list[str]] = {}
    for node in graph.nodes:
        parts = node.id.split(".")
        for i in range(len(parts) - 1):
            if parts[i].isdigit():
                parent = ".".join(parts[:i]) or "_root"
                families.setdefault(parent, []).append(node.id)
    if not families:
        return graph

    biggest_family = max(families, key=lambda k: len(families[k]))
    node_ids_in_family = set(families[biggest_family])
    if len(node_ids_in_family) < 2 or len(node_ids_in_family) < n_layers:
        return graph

    # Keep only layer.0's descendants; rename them and add a "×N" note.
    keep = [
        n
        for n in graph.nodes
        if n.id not in node_ids_in_family or f".{biggest_family}.0." in f".{n.id}."
        or n.id.startswith(f"{biggest_family}.0.")
    ]
    kept_ids = {n.id for n in keep}
    edges = [e for e in graph.edges if e.source_id in kept_ids and e.target_id in kept_ids]
    for n in keep:
        if n.id.startswith(f"{biggest_family}.0"):
            n.attributes = {**n.attributes, "repeat": n_layers}
    graph.nodes = keep
    graph.edges = edges
    graph.metadata["folded_blocks"] = {biggest_family: n_layers}
    mv_warn(
        f"Folded {n_layers} repeated blocks under {biggest_family!r}. "
        "Pass expand_groups=True to disable folding."
    )
    return graph


__all__ = ["HuggingFaceInspector"]
