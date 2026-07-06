"""Style dataclasses and the resolver.

Priority order (highest wins), per PRD §5.3::

    Per-Node Override  >  Per-Group Override  >  Layer-Type Palette
        >  Global Theme  >  Default

The resolver is a pure function — every renderer calls it, so the
priority order lives in exactly one place. See
:func:`resolve_style` below.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelvision.core.ir import LayerNode


# ---------------------------------------------------------------------------
# Value dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True, kw_only=True)
class StyleSpec:
    """Per-element visual override.

    Every field is optional; ``None`` means "inherit from the next
    level down in the priority chain."
    """

    fill: str | None = None
    stroke: str | None = None
    stroke_width: float | None = None
    opacity: float | None = None
    font_color: str | None = None
    font_size: int | None = None
    font_weight: str | None = None
    border_radius: float | None = None
    icon: str | None = None
    label: str | None = None
    caption: str | None = None
    shape: str | None = None
    dash: str | None = None
    glow: bool | None = None

    def merge(self, other: StyleSpec | None) -> StyleSpec:
        """Return a new :class:`StyleSpec` with non-None fields from ``other`` overriding ``self``."""
        if other is None:
            return self
        updates = {k: v for k, v in _iter_fields(other) if v is not None}
        return replace(self, **updates)


@dataclass(slots=True, kw_only=True)
class NodeStyle(StyleSpec):
    """Alias for :class:`StyleSpec` used at the per-node API level for readability."""


@dataclass(slots=True, kw_only=True)
class Theme:
    """A global theme. Renderers pull background/edge properties from here.

    Node defaults come from :attr:`layer_palette` (per layer type) with
    ``"*"`` wildcard fallback to :attr:`default_fill`/:attr:`default_stroke`.
    """

    name: str = "custom"
    background: str = "#ffffff"
    default_fill: str = "#f5f5f5"
    default_stroke: str = "#333333"
    default_stroke_width: float = 1.0
    font_color: str = "#111111"
    font_family: str = "Inter, system-ui, sans-serif"
    font_size: int = 12
    edge_color: str = "#666666"
    edge_width: float = 1.0
    group_fill: str = "#eeeeee"
    group_stroke: str = "#bbbbbb"
    layer_palette: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class Group:
    """User-facing group spec — supports node lists, glob patterns, or regex.

    Exactly one of ``nodes``, ``node_pattern``, or ``node_pattern_re``
    must be provided. This is validated in :meth:`__post_init__`.
    """

    id: str
    name: str | None = None
    nodes: list[str] | None = None
    node_pattern: str | None = None
    node_pattern_re: str | None = None
    fill: str | None = None
    stroke: str | None = None
    label_color: str | None = None
    style: StyleSpec | None = None

    def __post_init__(self) -> None:
        provided = sum(x is not None for x in (self.nodes, self.node_pattern, self.node_pattern_re))
        if provided != 1:
            raise ValueError(
                f"Group {self.id!r} must specify exactly one of "
                "'nodes', 'node_pattern', or 'node_pattern_re'."
            )

    def matches(self, node_id: str) -> bool:
        if self.nodes is not None:
            return node_id in self.nodes
        if self.node_pattern is not None:
            return fnmatch.fnmatchcase(node_id, self.node_pattern)
        if self.node_pattern_re is not None:
            return bool(re.match(self.node_pattern_re, node_id))
        return False  # pragma: no cover - guarded by __post_init__

    def as_style_spec(self) -> StyleSpec:
        """Fold the shorthand ``fill``/``stroke``/``label_color`` args into a :class:`StyleSpec`."""
        base = self.style or StyleSpec()
        overrides: dict[str, Any] = {}
        if self.fill is not None:
            overrides["fill"] = self.fill
        if self.stroke is not None:
            overrides["stroke"] = self.stroke
        if self.label_color is not None:
            overrides["font_color"] = self.label_color
        return base.merge(StyleSpec(**overrides)) if overrides else base


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def resolve_style(
    node: LayerNode,
    *,
    theme: Theme,
    layer_palette: dict[str, str] | None = None,
    groups: list[Group] | None = None,
    node_styles: dict[str, StyleSpec] | None = None,
) -> StyleSpec:
    """Compute the final :class:`StyleSpec` for a single node.

    Composes the five layers in priority order (lowest first, so each
    subsequent :meth:`StyleSpec.merge` overrides the previous):

    1. Theme defaults (background-independent fill/stroke).
    2. Layer-type palette (with ``"*"`` wildcard).
    3. Group override.
    4. Node's own :attr:`~LayerNode.style_override` (set by the inspector).
    5. User-provided ``node_styles[node.id]``.
    """
    # 1. Theme defaults.
    style = StyleSpec(
        fill=theme.default_fill,
        stroke=theme.default_stroke,
        stroke_width=theme.default_stroke_width,
        font_color=theme.font_color,
        font_size=theme.font_size,
    )

    # 2. Layer-type palette — theme's own, then user override.
    palette = {**theme.layer_palette, **(layer_palette or {})}
    if node.layer_type in palette:
        style = style.merge(StyleSpec(fill=palette[node.layer_type]))
    elif "*" in palette:
        style = style.merge(StyleSpec(fill=palette["*"]))

    # 3. Group override — first matching group wins (users get a warning
    #    at group-validation time for overlaps if strict=True).
    if groups:
        for group in groups:
            if group.matches(node.id):
                style = style.merge(group.as_style_spec())
                break

    # 4. Inspector-set node override (e.g., a quantized layer badge).
    if node.style_override is not None:
        style = style.merge(node.style_override)

    # 5. User-facing node_styles map — final word.
    if node_styles and node.id in node_styles:
        style = style.merge(node_styles[node.id])

    return style


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _iter_fields(spec: StyleSpec):  # type: ignore[no-untyped-def]
    """Yield ``(name, value)`` pairs — slots-friendly ``__dataclass_fields__`` walk."""
    for name in spec.__dataclass_fields__:  # type: ignore[attr-defined]
        yield name, getattr(spec, name)
