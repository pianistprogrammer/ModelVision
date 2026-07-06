"""scikit-learn inspector.

Walks ``Pipeline`` / ``FeatureUnion`` / ``ColumnTransformer`` /
``GridSearchCV`` / ``BaggingClassifier`` recursively. Each estimator
becomes a :class:`LayerNode`; each composite becomes a
:class:`SegmentGroup`.
"""

from __future__ import annotations

from typing import Any

from modelvision.core._optional import require
from modelvision.core.ids import sanitize
from modelvision.core.ir import Edge, LayerNode, ModelGraph, SegmentGroup
from modelvision.inspectors.base import BaseInspector


class SklearnInspector(BaseInspector):
    framework = "sklearn"

    def can_handle(self, model: Any) -> bool:
        return type(model).__module__.startswith("sklearn.")

    def inspect(self, model: Any, **_: Any) -> ModelGraph:
        require("sklearn")
        builder = _Builder()
        builder.walk(model, prefix="")
        return ModelGraph(
            nodes=builder.nodes,
            edges=builder.edges,
            groups=builder.groups,
            metadata={"framework": "sklearn", "model_class": type(model).__name__},
        )


class _Builder:
    def __init__(self) -> None:
        self.nodes: list[LayerNode] = []
        self.edges: list[Edge] = []
        self.groups: list[SegmentGroup] = []

    def walk(self, obj: Any, *, prefix: str) -> list[str]:
        """Recursively expand ``obj`` into leaf node IDs. Returns the IDs added."""
        cls_name = type(obj).__name__
        if cls_name == "Pipeline":
            return self._walk_pipeline(obj, prefix)
        if cls_name == "FeatureUnion":
            return self._walk_feature_union(obj, prefix)
        if cls_name == "ColumnTransformer":
            return self._walk_column_transformer(obj, prefix)
        if cls_name in {"GridSearchCV", "RandomizedSearchCV"}:
            best = getattr(obj, "best_estimator_", None)
            if best is not None:
                return self.walk(best, prefix=prefix)
        return [self._leaf(obj, prefix)]

    def _walk_pipeline(self, obj: Any, prefix: str) -> list[str]:
        all_ids: list[str] = []
        prev_tail: list[str] = []
        step_group_ids: list[str] = []
        for name, step in obj.steps:
            sub_prefix = f"{prefix}.{name}" if prefix else name
            step_ids = self.walk(step, prefix=sub_prefix)
            if not step_ids:
                continue
            # Chain: each step's *first* node receives edges from prev step's tail.
            head_ids = step_ids[:1] if len(step_ids) == 1 else _heads(self, step_ids)
            for prev in prev_tail:
                for head in head_ids:
                    self.edges.append(Edge(source_id=prev, target_id=head))
            step_group_ids.extend(step_ids)
            prev_tail = step_ids[-1:] if len(step_ids) == 1 else _tails(self, step_ids)
            all_ids.extend(step_ids)
        if len(step_group_ids) >= 2:
            gid = sanitize(f"{prefix}_pipeline") if prefix else "pipeline"
            self.groups.append(SegmentGroup(id=gid, name="Pipeline", node_ids=step_group_ids))
        return all_ids

    def _walk_feature_union(self, obj: Any, prefix: str) -> list[str]:
        all_ids: list[str] = []
        for name, transformer in obj.transformer_list:
            sub_prefix = f"{prefix}.{name}" if prefix else name
            all_ids.extend(self.walk(transformer, prefix=sub_prefix))
        if len(all_ids) >= 2:
            gid = sanitize(f"{prefix}_union") if prefix else "union"
            self.groups.append(SegmentGroup(id=gid, name="FeatureUnion", node_ids=all_ids))
        return all_ids

    def _walk_column_transformer(self, obj: Any, prefix: str) -> list[str]:
        all_ids: list[str] = []
        transformers = getattr(obj, "transformers_", None) or obj.transformers
        for entry in transformers:
            # (name, transformer, columns)
            name, transformer = entry[0], entry[1]
            if transformer in ("drop", "passthrough"):
                continue
            sub_prefix = f"{prefix}.{name}" if prefix else name
            all_ids.extend(self.walk(transformer, prefix=sub_prefix))
        if len(all_ids) >= 2:
            gid = sanitize(f"{prefix}_columns") if prefix else "columns"
            self.groups.append(SegmentGroup(id=gid, name="ColumnTransformer", node_ids=all_ids))
        return all_ids

    def _leaf(self, obj: Any, prefix: str) -> str:
        base = prefix or type(obj).__name__.lower()
        nid = _unique_id(self.nodes, sanitize(base))
        params = _param_count(obj)
        self.nodes.append(
            LayerNode(
                id=nid,
                name=prefix.rsplit(".", 1)[-1] or type(obj).__name__,
                layer_type=type(obj).__name__,
                framework="sklearn",
                params=params,
            )
        )
        return nid


def _unique_id(nodes: list[LayerNode], base: str) -> str:
    existing = {n.id for n in nodes}
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _param_count(estimator: Any) -> int | None:
    """Rough parameter count for sklearn estimators — coefficient count if fitted."""
    total = 0
    counted = False
    for attr in ("coef_", "intercept_", "components_", "feature_importances_"):
        v = getattr(estimator, attr, None)
        if v is None:
            continue
        try:
            total += int(v.size)
            counted = True
        except AttributeError:
            pass
    return total if counted else None


def _heads(_builder: _Builder, ids: list[str]) -> list[str]:
    # For a linear leaf list produced by a subwalk, first element is the head.
    return ids[:1]


def _tails(_builder: _Builder, ids: list[str]) -> list[str]:
    return ids[-1:]


__all__ = ["SklearnInspector"]
