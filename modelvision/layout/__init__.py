"""Layout engines — assign 2D coordinates to :class:`ModelGraph` nodes."""

from __future__ import annotations

from dataclasses import dataclass

from modelvision.core.ir import ModelGraph


@dataclass(slots=True, frozen=True)
class NodeBox:
    """Placed node — coordinates are the box's top-left corner."""

    node_id: str
    x: float
    y: float
    width: float
    height: float

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2


@dataclass(slots=True, frozen=True)
class LaidOutGraph:
    """A :class:`ModelGraph` plus a placement for every node."""

    graph: ModelGraph
    boxes: dict[str, NodeBox]
    width: float
    height: float
