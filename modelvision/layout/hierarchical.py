"""Hierarchical layout — currently an alias for vertical.

Full collapsible-box hierarchical rendering is a future enhancement;
for M3 we route ``layout="hierarchical"`` to the vertical layout so
the API accepts the string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelvision.layout import LaidOutGraph
from modelvision.layout.vertical import layout_vertical

if TYPE_CHECKING:
    from modelvision.core.ir import ModelGraph


def layout_hierarchical(graph: ModelGraph, **kwargs) -> LaidOutGraph:  # type: ignore[no-untyped-def]
    return layout_vertical(graph, **kwargs)


__all__ = ["layout_hierarchical"]
