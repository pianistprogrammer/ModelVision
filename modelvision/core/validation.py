"""Validation + accessibility helpers used by the render pipeline.

These live in a small module so :func:`modelvision._api.render` can
import them without pulling in framework inspectors or renderers.
"""

from __future__ import annotations

from typing import Literal

from modelvision.core.color import adjust_for_contrast, meets_wcag_aa
from modelvision.core.exceptions import mv_warn
from modelvision.core.ir import ModelGraph
from modelvision.core.style import Group, NodeStyle, StyleSpec, Theme, resolve_style


def validate_node_styles(graph: ModelGraph, node_styles: dict[str, NodeStyle] | None) -> None:
    """Ensure every key in ``node_styles`` names a real node.

    Raises :class:`ValueError` with a sorted list of valid IDs on the
    first mismatch — matches PRD §6.3.
    """
    if not node_styles:
        return
    valid = set(graph.node_ids())
    for key in node_styles:
        if key not in valid:
            raise ValueError(
                f"node_styles key {key!r} does not match any node ID. Valid IDs: {sorted(valid)}"
            )


def validate_groups(graph: ModelGraph, groups: list[Group] | None, *, strict: bool) -> None:
    """Detect nodes claimed by more than one :class:`Group`.

    ``strict=True`` raises on overlap; ``strict=False`` warns and lets
    the first-matching-group-wins rule inside the resolver apply.
    """
    if not groups:
        return
    valid = set(graph.node_ids())
    owner_of: dict[str, str] = {}
    for group in groups:
        matched = [nid for nid in valid if group.matches(nid)]
        for nid in matched:
            if nid in owner_of and owner_of[nid] != group.id:
                msg = f"Node {nid!r} is claimed by groups {owner_of[nid]!r} and {group.id!r}."
                if strict:
                    raise ValueError(msg + " Pass strict=False to allow overlaps.")
                mv_warn(msg + " First matching group wins in the resolver.")
            owner_of[nid] = group.id


AccessibilityMode = bool | Literal["enforce"]


def apply_accessibility(
    graph: ModelGraph,
    *,
    mode: AccessibilityMode,
    theme: Theme,
    layer_palette: dict[str, str] | None,
    groups: list[Group] | None,
    node_styles: dict[str, NodeStyle] | None,
) -> dict[str, NodeStyle] | None:
    """Warn or auto-adjust per PRD §13 Q5.

    - ``mode=False`` — no-op.
    - ``mode=True`` — emit a :class:`ModelVisionWarning` for every node
      whose (font_color, fill) contrast falls below WCAG AA.
    - ``mode="enforce"`` — same, but return a modified ``node_styles``
      dict where the offending font_color is bumped until it passes.
    """
    if mode is False:
        return node_styles

    enforce = mode == "enforce"
    updated: dict[str, NodeStyle] = dict(node_styles or {})

    for node in graph.nodes:
        resolved = resolve_style(
            node,
            theme=theme,
            layer_palette=layer_palette,
            groups=groups,
            node_styles=updated,
        )
        fill = resolved.fill or theme.default_fill
        font = resolved.font_color or theme.font_color
        if meets_wcag_aa(font, fill):
            continue
        msg = f"Node {node.id!r}: font color {font!r} on fill {fill!r} fails WCAG AA."
        if enforce:
            new_font = adjust_for_contrast(font, fill)
            base = updated.get(node.id) or NodeStyle()
            updated[node.id] = _with_font_color(base, new_font)
        else:
            mv_warn(msg)

    return updated or node_styles


def _with_font_color(base: StyleSpec, font_color: str) -> NodeStyle:
    """Return ``base`` with ``font_color`` set, preserving the concrete type."""
    fields = {name: getattr(base, name) for name in base.__dataclass_fields__}  # type: ignore[attr-defined]
    fields["font_color"] = font_color
    return NodeStyle(**fields)


__all__ = [
    "AccessibilityMode",
    "apply_accessibility",
    "validate_groups",
    "validate_node_styles",
]
