"""Framework-agnostic Intermediate Representation (IR).

Every framework inspector emits a :class:`ModelGraph`. Every renderer
consumes one. The dataclasses are deliberately narrow — anything a
particular framework wants to surface (kernel size, activation, etc.)
lives inside :attr:`LayerNode.attributes`.

See PRD §5.2 for the full spec.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from modelvision.core.style import StyleSpec


@dataclass(slots=True, kw_only=True)
class LayerNode:
    """A single layer / operation in the model graph.

    ``id`` must be stable across runs of the same model — it is the
    primary key users pass to :attr:`node_styles` and :attr:`Group.nodes`.
    """

    id: str
    name: str
    layer_type: str
    framework: str
    params: int | None = None
    input_shape: tuple[Any, ...] | None = None
    output_shape: tuple[Any, ...] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    group_id: str | None = None
    style_override: StyleSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, kw_only=True)
class Edge:
    """A directed connection between two :class:`LayerNode`\\s.

    ``kind`` distinguishes data flow (``"data"``) from shared-weight
    references (``"shared"``, rendered as dashed lines) and skip
    connections (``"skip"``).
    """

    source_id: str
    target_id: str
    label: str | None = None
    kind: str = "data"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, kw_only=True)
class SegmentGroup:
    """A named block of nodes rendered as a shaded region.

    The user-facing :class:`~modelvision.core.style.Group` type resolves
    to this at render time — :class:`Group` supports glob/regex patterns,
    while ``SegmentGroup`` is always a concrete node ID list.
    """

    id: str
    name: str
    node_ids: list[str]
    style_override: StyleSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, kw_only=True)
class ModelGraph:
    """The full IR emitted by an inspector.

    ``metadata`` is a free-form dict — inspectors stash framework name,
    model class name, total parameter count, and any inspection
    warnings/limitations here.
    """

    nodes: list[LayerNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    groups: list[SegmentGroup] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "groups": [g.to_dict() for g in self.groups],
            "metadata": self.metadata,
        }

    # -- helpers ---------------------------------------------------------

    def node_ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def get_node(self, node_id: str) -> LayerNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def in_degree(self, node_id: str) -> int:
        return sum(1 for e in self.edges if e.target_id == node_id)

    def out_degree(self, node_id: str) -> int:
        return sum(1 for e in self.edges if e.source_id == node_id)
