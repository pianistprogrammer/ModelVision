"""PyTorch model inspector.

Extracts a :class:`~modelvision.core.ir.ModelGraph` from any
:class:`torch.nn.Module` **without running a forward pass**. Strategy:

1. **Unwrap** compilation / parallelism wrappers first — the order
   matters. ``torch.compile(DataParallel(m))`` requires unwrapping
   ``OptimizedModule._orig_mod`` *then* ``DataParallel.module``.
2. **Walk** ``named_modules()`` and treat every leaf (a module with no
   child modules) as a :class:`LayerNode`. Sequential/ModuleList
   containers are exploded into indexed leaves; pure structural
   containers with children are folded into a :class:`SegmentGroup`.
3. **Attributes** are pulled by a per-type dispatch table — safer than
   scraping ``__dict__``, which surfaces internal state (masks, buffers).
4. **Edges** are the sequential order within each parent scope. Full
   cross-scope edges require a symbolic trace and are opt-in via
   ``symbolic_shapes=True`` — off by default per PRD §13 Q1.
5. **Weight-tied modules** are detected by ``id()`` duplicates. Each
   duplicate site is rendered as its own node but wired to the first
   site with a dashed ``kind="shared"`` edge (PRD §6.1, §13 Q4).
"""

from __future__ import annotations

from typing import Any

from modelvision.core._optional import require
from modelvision.core.exceptions import mv_warn
from modelvision.core.ids import join, uniquify
from modelvision.core.ir import Edge, LayerNode, ModelGraph, SegmentGroup
from modelvision.inspectors.base import BaseInspector


class PyTorchInspector(BaseInspector):
    framework = "torch"

    def can_handle(self, model: Any) -> bool:
        # Prefix-match rather than isinstance to avoid importing torch here.
        return type(model).__module__.startswith(("torch.", "torchvision."))

    def inspect(
        self,
        model: Any,
        *,
        symbolic_shapes: bool = False,
        show_shared_weights: bool = True,
        **_: Any,
    ) -> ModelGraph:
        torch = require("torch")
        model = _unwrap(model, torch)

        # ---- Walk the module tree -----------------------------------
        # PyTorch's ``named_modules()`` deduplicates by identity — so a
        # weight-tied module only appears at its first assignment path.
        # We want to see every assignment so shared-weight edges can be
        # emitted, so walk ``_modules`` directly.
        leaves, containers = _walk_modules(model)
        if not leaves:
            return _empty_graph(model)

        # ---- Build nodes --------------------------------------------
        # Node ID mirrors the qualified name; sanitized + uniquified for
        # golden-file stability.
        proposed_ids = [join(*name.split(".")) for name, _ in leaves]
        node_ids = uniquify(proposed_ids)

        nodes: list[LayerNode] = []
        # Map ``id(module)`` -> canonical node ID for weight-tie detection.
        first_site: dict[int, str] = {}
        tied_edges: list[Edge] = []
        for node_id, (qual_name, module) in zip(node_ids, leaves, strict=True):
            module_id = id(module)
            attrs = _extract_attributes(module)
            params = _leaf_params(module)
            layer_type = type(module).__name__
            group_id = _parent_group_id(qual_name)
            node = LayerNode(
                id=node_id,
                name=qual_name.split(".")[-1] or layer_type,
                layer_type=layer_type,
                framework="torch",
                params=params,
                attributes=attrs,
                group_id=group_id,
            )
            nodes.append(node)

            if module_id in first_site:
                if show_shared_weights:
                    tied_edges.append(
                        Edge(
                            source_id=first_site[module_id],
                            target_id=node_id,
                            label="shared",
                            kind="shared",
                        )
                    )
            else:
                first_site[module_id] = node_id

        # ---- Sequential edges within each parent scope --------------
        # Two leaves whose IDs share the same parent-dotted-prefix and are
        # adjacent in ``named_modules`` order get connected. This works
        # perfectly for ``nn.Sequential`` / ``ModuleList`` and is the
        # documented best-effort for the general case.
        edges: list[Edge] = []
        for prev, curr in zip(node_ids, node_ids[1:], strict=False):
            if _parent_group_id(prev) == _parent_group_id(curr):
                edges.append(Edge(source_id=prev, target_id=curr))
        edges.extend(tied_edges)

        # ---- Segment groups from container modules ------------------
        groups = _build_groups(containers, node_ids, leaves)

        # ---- Optional torch.fx symbolic trace for cross-scope edges --
        metadata: dict[str, Any] = {
            "framework": "torch",
            "model_class": type(model).__name__,
            "total_params": sum(p.numel() for p in model.parameters()),
        }
        if symbolic_shapes:
            fx_edges = _try_symbolic_edges(model, node_ids, torch)
            if fx_edges is not None:
                # Preserve tied edges even when we swap the data edges.
                edges = [e for e in edges if e.kind != "data"] + fx_edges + tied_edges
                metadata["symbolic_trace"] = "torch.fx"
            else:
                metadata["symbolic_trace"] = "failed"

        graph = ModelGraph(nodes=nodes, edges=edges, groups=groups, metadata=metadata)
        # Attach dtype + quantization badges — no-op if the model doesn't
        # carry that information.
        from modelvision.core.postprocess import annotate_dtype_and_quantization

        return annotate_dtype_and_quantization(graph, model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_modules(root: Any) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    """Recursive walk of ``_modules`` that preserves duplicate assignments.

    Returns ``(leaves, containers)`` — each a list of ``(qualified_name, module)``
    pairs in traversal order. Unlike ``named_modules()``, this yields every
    attribute site of a weight-tied module.

    Known atomic types (:data:`_ATOMIC_TYPES`) — modules that logically
    represent a single operation but internally contain submodules — are
    treated as leaves. Without this, ``MultiheadAttention`` would decompose
    into an obscure ``out_proj`` linear that users don't recognize.
    """
    leaves: list[tuple[str, Any]] = []
    containers: list[tuple[str, Any]] = []

    def _recurse(module: Any, prefix: str) -> None:
        children = getattr(module, "_modules", None) or {}
        for name, child in children.items():
            if child is None:
                continue
            qual = f"{prefix}.{name}" if prefix else name
            grand = getattr(child, "_modules", None) or {}
            is_atomic = type(child).__name__ in _ATOMIC_TYPES
            if grand and not is_atomic:
                containers.append((qual, child))
                _recurse(child, qual)
            else:
                leaves.append((qual, child))

    _recurse(root, "")
    return leaves, containers


# Modules that logically own one operation, even though they contain
# submodules internally. Extend this if downstream users report
# opaque names showing up in their diagrams.
_ATOMIC_TYPES: frozenset[str] = frozenset({
    "MultiheadAttention",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "LSTM",
    "GRU",
    "RNN",
    "LSTMCell",
    "GRUCell",
    "RNNCell",
})


def _unwrap(model: Any, torch: Any) -> Any:
    """Peel back ``torch.compile``, ``DataParallel``, and ``DDP`` wrappers.

    Order matters — a ``torch.compile(DataParallel(m))`` needs the
    ``OptimizedModule`` unwrap first, then ``.module``.
    """
    for _ in range(8):  # bounded loop — real stacks are 1–3 deep.
        cls_name = type(model).__name__
        if cls_name == "OptimizedModule" and hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        if _is_data_parallel(model, torch) and hasattr(model, "module"):
            model = model.module
            continue
        break
    return model


def _is_data_parallel(model: Any, torch: Any) -> bool:
    try:
        return isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        )
    except AttributeError:  # pragma: no cover - older torch layouts
        return type(model).__name__ in {"DataParallel", "DistributedDataParallel"}


def _leaf_params(module: Any) -> int:
    """Direct parameter count for a leaf module (no recursion)."""
    return sum(p.numel() for p in module.parameters(recurse=False))


def _parent_group_id(node_id: str) -> str | None:
    """Return the dotted-prefix parent, or ``None`` for a top-level leaf."""
    if "." not in node_id:
        return None
    return node_id.rsplit(".", 1)[0]


def _build_groups(
    containers: list[tuple[str, Any]],
    node_ids: list[str],
    leaves: list[tuple[str, Any]],
) -> list[SegmentGroup]:
    """Wrap each container that owns ≥ 2 leaf nodes in a :class:`SegmentGroup`."""
    node_to_id = {qual: nid for nid, (qual, _) in zip(node_ids, leaves, strict=True)}
    groups: list[SegmentGroup] = []
    for qual_name, module in containers:
        prefix = qual_name + "."
        owned = [nid for qual, nid in node_to_id.items() if qual.startswith(prefix)]
        if len(owned) >= 2:
            groups.append(
                SegmentGroup(
                    id=join(*qual_name.split(".")),
                    name=f"{qual_name.split('.')[-1]} ({type(module).__name__})",
                    node_ids=owned,
                )
            )
    return groups


def _empty_graph(model: Any) -> ModelGraph:
    mv_warn(
        f"Model {type(model).__name__!r} has no child modules — rendering placeholder."
    )
    placeholder = LayerNode(
        id="empty",
        name=type(model).__name__,
        layer_type="EmptyModule",
        framework="torch",
    )
    return ModelGraph(
        nodes=[placeholder],
        metadata={"framework": "torch", "model_class": type(model).__name__, "empty": True},
    )


# ---------------------------------------------------------------------------
# Attribute extraction — type-dispatch table
# ---------------------------------------------------------------------------


def _extract_attributes(module: Any) -> dict[str, Any]:
    """Pull a small, semantically meaningful set of attributes per layer type.

    We avoid ``__dict__`` scraping — it exposes buffers, hooks, and
    quantization tensors that clutter the diagram.
    """
    layer_type = type(module).__name__
    extractor = _ATTR_TABLE.get(layer_type, _generic_attrs)
    try:
        return extractor(module)
    except Exception:  # pragma: no cover — inspection must never throw
        return {}


def _conv_attrs(m: Any) -> dict[str, Any]:
    return {
        "in_channels": getattr(m, "in_channels", None),
        "out_channels": getattr(m, "out_channels", None),
        "kernel_size": _tuple_attr(m, "kernel_size"),
        "stride": _tuple_attr(m, "stride"),
        "padding": _tuple_attr(m, "padding"),
        "dilation": _tuple_attr(m, "dilation"),
        "groups": getattr(m, "groups", None),
    }


def _linear_attrs(m: Any) -> dict[str, Any]:
    return {
        "in_features": getattr(m, "in_features", None),
        "out_features": getattr(m, "out_features", None),
        "bias": getattr(m, "bias", None) is not None,
    }


def _norm_attrs(m: Any) -> dict[str, Any]:
    return {
        "num_features": getattr(m, "num_features", getattr(m, "normalized_shape", None)),
        "eps": getattr(m, "eps", None),
        "affine": getattr(m, "affine", None),
    }


def _pool_attrs(m: Any) -> dict[str, Any]:
    return {
        "kernel_size": _tuple_attr(m, "kernel_size"),
        "stride": _tuple_attr(m, "stride"),
        "padding": _tuple_attr(m, "padding"),
    }


def _dropout_attrs(m: Any) -> dict[str, Any]:
    return {"p": getattr(m, "p", None)}


def _embedding_attrs(m: Any) -> dict[str, Any]:
    return {
        "num_embeddings": getattr(m, "num_embeddings", None),
        "embedding_dim": getattr(m, "embedding_dim", None),
    }


def _mha_attrs(m: Any) -> dict[str, Any]:
    return {
        "embed_dim": getattr(m, "embed_dim", None),
        "num_heads": getattr(m, "num_heads", None),
        "dropout": getattr(m, "dropout", None),
    }


def _generic_attrs(_m: Any) -> dict[str, Any]:
    return {}


def _tuple_attr(m: Any, name: str) -> Any:
    v = getattr(m, name, None)
    if v is None:
        return None
    return tuple(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v


_ATTR_TABLE: dict[str, Any] = {
    "Conv1d": _conv_attrs,
    "Conv2d": _conv_attrs,
    "Conv3d": _conv_attrs,
    "ConvTranspose1d": _conv_attrs,
    "ConvTranspose2d": _conv_attrs,
    "ConvTranspose3d": _conv_attrs,
    "Linear": _linear_attrs,
    "Bilinear": _linear_attrs,
    "BatchNorm1d": _norm_attrs,
    "BatchNorm2d": _norm_attrs,
    "BatchNorm3d": _norm_attrs,
    "LayerNorm": _norm_attrs,
    "GroupNorm": _norm_attrs,
    "InstanceNorm1d": _norm_attrs,
    "InstanceNorm2d": _norm_attrs,
    "MaxPool1d": _pool_attrs,
    "MaxPool2d": _pool_attrs,
    "MaxPool3d": _pool_attrs,
    "AvgPool1d": _pool_attrs,
    "AvgPool2d": _pool_attrs,
    "AvgPool3d": _pool_attrs,
    "AdaptiveAvgPool1d": _pool_attrs,
    "AdaptiveAvgPool2d": _pool_attrs,
    "AdaptiveMaxPool2d": _pool_attrs,
    "Dropout": _dropout_attrs,
    "Dropout2d": _dropout_attrs,
    "Embedding": _embedding_attrs,
    "MultiheadAttention": _mha_attrs,
}


# ---------------------------------------------------------------------------
# Optional torch.fx path
# ---------------------------------------------------------------------------


def _try_symbolic_edges(model: Any, node_ids: list[str], torch: Any) -> list[Edge] | None:
    """Attempt a ``torch.fx.symbolic_trace`` to recover cross-scope edges.

    Returns ``None`` (with a warning) on any failure. Never raises — the
    caller falls back to the sequential edges built during the module walk.
    """
    try:
        fx = torch.fx.symbolic_trace(model)
    except Exception as exc:  # dynamic control flow, unsupported ops, etc.
        mv_warn(
            f"symbolic_shapes=True but torch.fx.symbolic_trace failed "
            f"({type(exc).__name__}: {exc}). Falling back to sequential edges."
        )
        return None

    valid = set(node_ids)
    edges: list[Edge] = []
    for node in fx.graph.nodes:
        if node.op != "call_module":
            continue
        target = join(*node.target.split("."))
        for user in node.users:
            if user.op != "call_module":
                continue
            user_target = join(*user.target.split("."))
            if target in valid and user_target in valid:
                edges.append(Edge(source_id=target, target_id=user_target))
    return edges


__all__ = ["PyTorchInspector"]
