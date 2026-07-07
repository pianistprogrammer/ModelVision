"""First-class palette registry.

Palettes are ordered lists of hex colors. Themes pull from them to build
their per-layer-type mappings, and users can pass a palette name directly
to :func:`~modelvision.render` via the ``palette=`` argument.

The ``okabe_ito`` palette is the recommended default for scientific
visualization — 7 colors chosen to be distinguishable to viewers with
common forms of color vision deficiency. Introduced by Okabe & Ito
(2008) and adopted by visualtorch, matplotlib's ``tab:``-plus-cvd
guidance, and many stats packages.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Palette definitions
# ---------------------------------------------------------------------------


PALETTES: dict[str, list[str]] = {
    # Colorblind-safe, from Okabe & Ito (2008). Same set visualtorch defaults to.
    "okabe_ito": [
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
    ],
    # Paul Tol's bright — another CVD-safe favourite.
    "tol_bright": [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ],
    # Vivid saturated defaults for slides.
    "vivid": [
        "#4a90d9",
        "#9b59b6",
        "#e74c3c",
        "#f5a623",
        "#27ae60",
        "#1abc9c",
        "#95a5a6",
    ],
    # Pastel — the softer set used by our ``pastel`` theme.
    "pastel": [
        "#b5d0e6",
        "#d9c2e6",
        "#c8e6d0",
        "#fce0b8",
        "#fbc9a8",
        "#b8e0d2",
        "#f5b5b5",
    ],
    # High-contrast — max chroma for accessibility-first diagrams.
    "high_contrast": [
        "#0044aa",
        "#7700aa",
        "#aa0033",
        "#aa5500",
        "#007744",
        "#006677",
        "#333333",
    ],
    # Grayscale ramp — print-safe, mono-tone.
    "grayscale": [
        "#cccccc",
        "#b3b3b3",
        "#999999",
        "#808080",
        "#666666",
        "#4d4d4d",
        "#333333",
    ],
}


# The order in which layer types are assigned palette colors when the
# user doesn't specify one explicitly. Kept small on purpose — anything
# not in this list falls through the palette's tail with wrap-around.
_ASSIGNMENT_ORDER: list[str] = [
    "Conv2d",
    "Linear",
    "BatchNorm2d",
    "ReLU",
    "MaxPool2d",
    "Dropout",
    "Embedding",
    "MultiheadAttention",
    "LayerNorm",
    "AvgPool2d",
    "GELU",
    "AdaptiveAvgPool2d",
]


def resolve_palette(name_or_list: str | list[str]) -> list[str]:
    """Return the color list for ``name_or_list``.

    Accepts a registered palette name or a passthrough list. Raises
    :class:`ValueError` with the available names on typo.
    """
    if isinstance(name_or_list, list):
        return list(name_or_list)
    if name_or_list not in PALETTES:
        raise ValueError(f"Unknown palette {name_or_list!r}. Available: {sorted(PALETTES)}")
    return list(PALETTES[name_or_list])


def build_layer_palette(
    palette: str | list[str] = "okabe_ito",
    *,
    types: list[str] | None = None,
    wildcard: str | None = None,
) -> dict[str, str]:
    """Build a ``{layer_type: hex}`` dict from a palette.

    Layer types are assigned palette colors in the order given by
    :data:`_ASSIGNMENT_ORDER` (or ``types``), wrapping if the palette is
    shorter than the type list.

    Parameters
    ----------
    palette:
        Palette name (see :data:`PALETTES`) or an explicit color list.
    types:
        Override the default assignment order.
    wildcard:
        Extra ``"*"`` entry — the fallback fill for any unmapped type.
    """
    colors = resolve_palette(palette)
    assignment = types or _ASSIGNMENT_ORDER
    result: dict[str, str] = {t: colors[i % len(colors)] for i, t in enumerate(assignment)}
    if wildcard is not None:
        result["*"] = wildcard
    return result


__all__ = ["PALETTES", "build_layer_palette", "resolve_palette"]
