"""JAX inspector — supports Flax (linen) and Haiku modules.

Zero-forward inspection is subtle in JAX because modules are lazily
bound to shapes. Strategy:

- **Flax linen:** call ``module.tabulate(jax.random.PRNGKey(0), *dummy_args)``
  and parse the returned string. The tabulate call does a shape-init
  pass but no data flow — no gradient / no user data. Fresh RNG created
  internally so callers don't need to thread keys.
- **Haiku:** call ``hk.experimental.tabulate(fn)(*dummy_args)``; same
  no-data property.

If neither an ``input_shape`` nor concrete ``inputs`` are provided we
fall back to structural walking without shapes.
"""

from __future__ import annotations

import re
from typing import Any

from modelvision.core._optional import require
from modelvision.core.exceptions import InspectionError, mv_warn
from modelvision.core.ids import sanitize, uniquify
from modelvision.core.ir import Edge, LayerNode, ModelGraph
from modelvision.inspectors.base import BaseInspector


class JAXInspector(BaseInspector):
    framework = "jax"

    def can_handle(self, model: Any) -> bool:
        mod = type(model).__module__
        return mod.startswith(("flax.", "haiku.", "jax."))

    def inspect(
        self,
        model: Any,
        *,
        input_shape: tuple[int, ...] | None = None,
        inputs: Any = None,
        **_: Any,
    ) -> ModelGraph:
        # Walk the MRO — a user-defined subclass of ``flax.linen.Module``
        # lives in ``__main__`` (or the user's file), not ``flax.*``, so
        # checking only ``type(model).__module__`` misses it.
        flavor = _detect_jax_flavor(model)
        if flavor == "flax":
            return _inspect_flax(model, input_shape=input_shape, inputs=inputs)
        if flavor == "haiku":
            return _inspect_haiku(model, input_shape=input_shape, inputs=inputs)
        raise InspectionError(
            f"Unrecognized JAX module type {type(model).__name__!r}. "
            "Expected a flax.linen.Module subclass or a haiku transform."
        )


def _detect_jax_flavor(model: Any) -> str | None:
    """Walk ``type(model).__mro__`` looking for a flax or haiku ancestor."""
    for cls in type(model).__mro__:
        mod = getattr(cls, "__module__", "") or ""
        if mod.startswith("flax."):
            return "flax"
        if mod.startswith("haiku."):
            return "haiku"
    return None


# ---------------------------------------------------------------------------
# Flax
# ---------------------------------------------------------------------------


def _inspect_flax(module: Any, *, input_shape: tuple[int, ...] | None, inputs: Any) -> ModelGraph:
    flax = require("flax")  # noqa: F841 - imported for the friendly error message
    jax = require("jax")
    jnp = jax.numpy

    if inputs is None:
        if input_shape is None:
            mv_warn(
                "Flax module inspected without input_shape or inputs — "
                "returning a shape-less structural graph."
            )
            return _flax_structural(module)
        inputs = jnp.zeros(input_shape)

    key = jax.random.PRNGKey(0)
    try:
        # ``tabulate`` is the canonical no-execution shape-introspection
        # helper in flax.linen. Its console output is deterministic when
        # given ``console_kwargs={"force_terminal": False}``.
        tabulate_fn = module.tabulate
    except AttributeError as exc:  # pragma: no cover
        raise InspectionError("This Flax module has no .tabulate method.") from exc

    try:
        table = tabulate_fn(
            key,
            inputs,
            console_kwargs={"force_terminal": False, "width": 200},
        )
    except Exception as exc:
        mv_warn(f"flax tabulate failed ({exc!r}) — falling back to structural walk.")
        return _flax_structural(module)

    return _parse_flax_tabulate(table, module_class=type(module).__name__)


def _flax_structural(module: Any) -> ModelGraph:
    """No-input fallback: enumerate submodules recorded on the class body."""
    submodules = _flax_submodules(module)
    if not submodules:
        placeholder = LayerNode(
            id="empty",
            name=type(module).__name__,
            layer_type=type(module).__name__,
            framework="jax",
        )
        return ModelGraph(
            nodes=[placeholder],
            metadata={"framework": "jax", "flavor": "flax", "empty": True},
        )

    ids = uniquify([sanitize(name) for name, _ in submodules])
    nodes = [
        LayerNode(
            id=nid,
            name=name,
            layer_type=type(child).__name__,
            framework="jax",
        )
        for nid, (name, child) in zip(ids, submodules, strict=True)
    ]
    edges = [Edge(source_id=a, target_id=b) for a, b in zip(ids, ids[1:], strict=False)]
    return ModelGraph(
        nodes=nodes,
        edges=edges,
        metadata={
            "framework": "jax",
            "flavor": "flax",
            "model_class": type(module).__name__,
        },
    )


def _flax_submodules(module: Any) -> list[tuple[str, Any]]:
    """Return declared submodules by inspecting the dataclass fields."""
    submodules: list[tuple[str, Any]] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            value = getattr(module, name)
        except AttributeError:
            continue
        cls_module = type(value).__module__
        if cls_module.startswith("flax.linen"):
            submodules.append((name, value))
    return submodules


# The tabulate string has header rows we skip, then per-layer rows shaped
# like ``│ 0 │ path.to.layer │ LayerType │ in_shape → out_shape │ params │``.
# Regex tuned for the current flax versions; parsing failures fall back
# to the structural walk.
# Flax's ``Module.tabulate`` output has columns:
#     | path | module | inputs | outputs | batch_stats | params |
# We only need the first two. Table borders use the │ box character, which
# we normalize to | before matching.
_TABULATE_ROW = re.compile(r"^\|\s*([^|]*?)\s*\|\s*([A-Za-z_][A-Za-z0-9_]*)\s*\|")


def _parse_flax_tabulate(table: str, *, module_class: str) -> ModelGraph:
    """Parse ``flax.linen.Module.tabulate`` output into a :class:`ModelGraph`.

    We keep only rows whose *path* is non-empty (skipping the root row that
    lists the whole module) and whose *type* isn't the literal header token.
    """
    lines = [line for line in table.splitlines() if line.strip().startswith(("|", "│"))]
    rows: list[tuple[str, str]] = []
    for line in lines:
        m = _TABULATE_ROW.match(line.replace("│", "|"))
        if not m:
            continue
        path, layer_type = m.group(1).strip(), m.group(2).strip()
        if not path or layer_type.lower() in {"module", "type", "path"}:
            continue
        # Path segments like ``Conv_0/kernel: float32[3,3,3,16]`` for the
        # nested "params" cell — skip anything that looks like a shape/detail.
        if ":" in path or "[" in path:
            continue
        rows.append((path, layer_type))

    if not rows:
        mv_warn("Could not parse flax tabulate output — falling back to structural walk.")
        return ModelGraph(
            nodes=[
                LayerNode(
                    id="root",
                    name=module_class,
                    layer_type=module_class,
                    framework="jax",
                )
            ],
            metadata={"framework": "jax", "flavor": "flax"},
        )

    ids = uniquify([sanitize(p.replace(".", "_")) for p, _ in rows])
    nodes = [
        LayerNode(id=nid, name=p, layer_type=lt, framework="jax")
        for nid, (p, lt) in zip(ids, rows, strict=True)
    ]
    edges = [Edge(source_id=a, target_id=b) for a, b in zip(ids, ids[1:], strict=False)]
    return ModelGraph(
        nodes=nodes,
        edges=edges,
        metadata={"framework": "jax", "flavor": "flax", "model_class": module_class},
    )


# ---------------------------------------------------------------------------
# Haiku
# ---------------------------------------------------------------------------


def _inspect_haiku(model: Any, *, input_shape, inputs) -> ModelGraph:  # type: ignore[no-untyped-def]
    """Haiku inspection.

    Haiku models are functions transformed by :func:`haiku.transform`.
    We accept either the transformed pair ``(init_fn, apply_fn)`` or a
    plain function; in both cases we use ``hk.experimental.tabulate`` to
    produce a shape-only summary, then reuse the flax-tabulate parser.
    """
    hk = require("haiku")
    jax = require("jax")
    jnp = jax.numpy

    if inputs is None:
        if input_shape is None:
            mv_warn(
                "Haiku module inspected without input_shape or inputs — "
                "returning a placeholder graph."
            )
            return _haiku_placeholder(model)
        inputs = jnp.zeros(input_shape)

    apply_fn = model
    if hasattr(model, "apply") and hasattr(model, "init"):
        # It's already a Transformed pair.
        apply_fn = model.apply

    tabulate = getattr(hk.experimental, "tabulate", None)
    if tabulate is None:
        return _haiku_placeholder(model)
    try:
        table = tabulate(apply_fn)(inputs)
    except Exception as exc:
        mv_warn(f"haiku tabulate failed ({exc!r}) — placeholder graph returned.")
        return _haiku_placeholder(model)

    graph = _parse_flax_tabulate(table, module_class=type(model).__name__)
    graph.metadata["flavor"] = "haiku"
    return graph


def _haiku_placeholder(model: Any) -> ModelGraph:
    return ModelGraph(
        nodes=[
            LayerNode(
                id="haiku_root",
                name=type(model).__name__,
                layer_type="HaikuModule",
                framework="jax",
            )
        ],
        metadata={"framework": "jax", "flavor": "haiku", "partial": True},
    )


__all__ = ["JAXInspector"]
