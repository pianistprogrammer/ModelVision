"""Keras / TensorFlow inspector.

Supports three Keras model flavors:

- **Sequential** — linear stack; edges are trivial.
- **Functional** — DAG reconstructed from ``layer._inbound_nodes`` /
  ``layer.input`` / ``layer.output``.
- **Subclassed** — the call graph is unknown statically. We render the
  declared ``model.layers`` list as a flat chain with a
  :class:`~modelvision.core.exceptions.ModelVisionWarning` and set
  ``ModelGraph.metadata["subclassed"] = True`` so the renderer can
  decorate the container.

If ``model.built is False`` and ``input_shape`` is supplied, the
inspector calls ``model.build(input_shape)`` first so shapes can be
recovered without a forward pass.
"""

from __future__ import annotations

from typing import Any

from modelvision.core._optional import require
from modelvision.core.exceptions import mv_warn
from modelvision.core.ids import sanitize, uniquify
from modelvision.core.ir import Edge, LayerNode, ModelGraph
from modelvision.inspectors.base import BaseInspector


class KerasInspector(BaseInspector):
    framework = "keras"

    def can_handle(self, model: Any) -> bool:
        mod = type(model).__module__
        return mod.startswith(("keras.", "tensorflow.", "tf_keras."))

    def inspect(
        self,
        model: Any,
        *,
        input_shape: tuple[int, ...] | None = None,
        **_: Any,
    ) -> ModelGraph:
        # Force ``require`` — this fails cleanly if neither TF nor Keras
        # is installed. We don't actually use the imported module below
        # (only isinstance checks against ``model``), but the require call
        # produces the "install extra" hint.
        _tf_or_keras()

        # Lazy build.
        if getattr(model, "built", True) is False:
            if input_shape is None:
                mv_warn(
                    f"Keras model {type(model).__name__!r} is not built and no "
                    "input_shape was provided — shapes will be missing."
                )
            else:
                model.build(input_shape)

        flavor = _flavor(model)
        if flavor == "sequential":
            return _inspect_sequential(model)
        if flavor == "functional":
            return _inspect_functional(model)
        return _inspect_subclassed(model)


# ---------------------------------------------------------------------------
# Flavor detection
# ---------------------------------------------------------------------------


def _tf_or_keras() -> Any:
    for candidate in ("keras", "tensorflow"):
        try:
            return require(candidate)
        except ImportError:
            continue
    # Force the friendly error via require if neither is installed.
    return require("keras")


def _flavor(model: Any) -> str:
    cls_name = type(model).__name__
    if cls_name == "Sequential":
        return "sequential"
    # Functional models have a well-defined ``inputs`` list.
    if getattr(model, "inputs", None) and getattr(model, "outputs", None):
        return "functional"
    if hasattr(model, "layers") and model.layers:
        return "subclassed"
    return "subclassed"


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------


def _inspect_sequential(model: Any) -> ModelGraph:
    layers = list(model.layers)
    ids = uniquify([sanitize(l.name) for l in layers])
    nodes = [_layer_node(l, nid) for nid, l in zip(ids, layers, strict=True)]
    edges = [Edge(source_id=a, target_id=b) for a, b in zip(ids, ids[1:], strict=False)]
    return ModelGraph(
        nodes=nodes,
        edges=edges,
        metadata={
            "framework": "keras",
            "model_class": type(model).__name__,
            "flavor": "sequential",
            "total_params": _count_params(model),
        },
    )


# ---------------------------------------------------------------------------
# Functional
# ---------------------------------------------------------------------------


def _inspect_functional(model: Any) -> ModelGraph:
    layers = list(model.layers)
    id_of: dict[int, str] = {}
    proposed = [sanitize(l.name) for l in layers]
    ids = uniquify(proposed)
    for l, nid in zip(layers, ids, strict=True):
        id_of[id(l)] = nid

    nodes = [_layer_node(l, id_of[id(l)]) for l in layers]

    # Rebuild edges from ``_inbound_nodes`` — each inbound node knows the
    # layers that fed into it. The API has drifted across Keras versions
    # (``inbound_layers`` vs ``keras_inputs`` etc.), so we probe both.
    edges: list[Edge] = []
    for layer in layers:
        target = id_of[id(layer)]
        for source_layer in _inbound_layers(layer):
            src_id = id_of.get(id(source_layer))
            if src_id and src_id != target:
                edges.append(Edge(source_id=src_id, target_id=target))

    return ModelGraph(
        nodes=nodes,
        edges=edges,
        metadata={
            "framework": "keras",
            "model_class": type(model).__name__,
            "flavor": "functional",
            "total_params": _count_params(model),
        },
    )


def _inbound_layers(layer: Any) -> list[Any]:
    """Return the layer instances that feed ``layer``.

    Keras versions differ in how this is exposed. We check the two
    common shapes: ``_inbound_nodes[*].inbound_layers`` (older Keras /
    TF Keras) and ``_inbound_nodes[*].keras_inputs`` (Keras 3).
    """
    sources: list[Any] = []
    for node in getattr(layer, "_inbound_nodes", []) or []:
        inbound = getattr(node, "inbound_layers", None)
        if inbound is not None:
            sources.extend(_as_iterable(inbound))
            continue
        inputs = getattr(node, "keras_inputs", None) or getattr(node, "input_tensors", None)
        if inputs is not None:
            for t in _as_iterable(inputs):
                src = getattr(t, "_keras_history", None)
                if src is not None:
                    sources.append(src[0] if isinstance(src, tuple) else src.layer)
    return sources


def _as_iterable(x: Any) -> list[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


# ---------------------------------------------------------------------------
# Subclassed
# ---------------------------------------------------------------------------


def _inspect_subclassed(model: Any) -> ModelGraph:
    layers = list(getattr(model, "layers", []) or [])
    mv_warn(
        f"Keras model {type(model).__name__!r} is subclassed — the call graph "
        "is not statically knowable. Rendering declared layers as a flat chain."
    )
    if not layers:
        placeholder = LayerNode(
            id="empty",
            name=type(model).__name__,
            layer_type="EmptyModel",
            framework="keras",
        )
        return ModelGraph(
            nodes=[placeholder],
            metadata={
                "framework": "keras",
                "model_class": type(model).__name__,
                "flavor": "subclassed",
                "empty": True,
            },
        )

    ids = uniquify([sanitize(l.name) for l in layers])
    nodes = [_layer_node(l, nid) for nid, l in zip(ids, layers, strict=True)]
    edges = [Edge(source_id=a, target_id=b) for a, b in zip(ids, ids[1:], strict=False)]
    return ModelGraph(
        nodes=nodes,
        edges=edges,
        metadata={
            "framework": "keras",
            "model_class": type(model).__name__,
            "flavor": "subclassed",
            "subclassed": True,
        },
    )


# ---------------------------------------------------------------------------
# Layer → LayerNode
# ---------------------------------------------------------------------------


def _layer_node(layer: Any, node_id: str) -> LayerNode:
    return LayerNode(
        id=node_id,
        name=getattr(layer, "name", type(layer).__name__),
        layer_type=type(layer).__name__,
        framework="keras",
        params=_layer_params(layer),
        input_shape=_shape(layer, "input_shape"),
        output_shape=_shape(layer, "output_shape"),
        attributes=_layer_attributes(layer),
    )


def _shape(layer: Any, name: str) -> tuple[Any, ...] | None:
    v = getattr(layer, name, None)
    if v is None:
        return None
    # Keras represents unknown dims as ``None`` — leave them symbolic.
    try:
        return tuple(v)
    except TypeError:
        return None


def _layer_params(layer: Any) -> int | None:
    try:
        return int(layer.count_params())
    except Exception:
        return None


def _count_params(model: Any) -> int | None:
    try:
        return int(model.count_params())
    except Exception:
        return None


def _layer_attributes(layer: Any) -> dict[str, Any]:
    """Pull a small, whitelisted config subset — avoids the huge get_config() dump."""
    try:
        cfg = layer.get_config()
    except Exception:
        return {}
    keep = {
        "units",
        "filters",
        "kernel_size",
        "strides",
        "padding",
        "activation",
        "rate",
        "axis",
        "epsilon",
        "num_heads",
        "key_dim",
    }
    return {k: cfg[k] for k in keep if k in cfg}


__all__ = ["KerasInspector"]
