"""Built-in themes.

Each theme module defines a module-level ``THEME`` constant. The
:func:`get_theme` helper resolves either a name (``"light"``, ``"dark"``,
etc.) or a passthrough :class:`Theme` instance.

Layer palettes shipped with each theme aim at WCAG AA against the
theme's background on layer-name text. See ``core/color.py`` for the
accessibility utilities.
"""

from __future__ import annotations

from modelvision.core.style import Theme
from modelvision.themes.dark import THEME as DARK
from modelvision.themes.grayscale import THEME as GRAYSCALE
from modelvision.themes.high_contrast import THEME as HIGH_CONTRAST
from modelvision.themes.light import THEME as LIGHT
from modelvision.themes.pastel import THEME as PASTEL

_BUILTIN: dict[str, Theme] = {
    "light": LIGHT,
    "dark": DARK,
    "pastel": PASTEL,
    "grayscale": GRAYSCALE,
    "high_contrast": HIGH_CONTRAST,
}


def get_theme(name_or_theme: str | Theme) -> Theme:
    """Resolve ``name_or_theme`` to a :class:`Theme` instance.

    Raises :class:`ValueError` for unknown names.
    """
    if isinstance(name_or_theme, Theme):
        return name_or_theme
    key = name_or_theme.lower()
    if key not in _BUILTIN:
        raise ValueError(
            f"Unknown theme: {name_or_theme!r}. "
            f"Built-in themes: {sorted(_BUILTIN)}"
        )
    return _BUILTIN[key]


__all__ = ["DARK", "GRAYSCALE", "HIGH_CONTRAST", "LIGHT", "PASTEL", "get_theme"]
