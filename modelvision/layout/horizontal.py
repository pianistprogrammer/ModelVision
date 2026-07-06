"""Horizontal (left-to-right) layered layout.

Thin wrapper over :func:`modelvision.layout.vertical.layout_layered`
with ``axis="horizontal"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelvision.layout import LaidOutGraph
from modelvision.layout.vertical import layout_layered

if TYPE_CHECKING:
    from modelvision.core.ir import ModelGraph


def layout_horizontal(graph: ModelGraph, **kwargs) -> LaidOutGraph:  # type: ignore[no-untyped-def]
    return layout_layered(graph, axis="horizontal", **kwargs)


__all__ = ["layout_horizontal"]
