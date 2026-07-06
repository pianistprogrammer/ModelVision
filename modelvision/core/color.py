"""Hex color parsing and WCAG contrast utilities.

Used by the accessibility check (:pep:`WCAG 2.1` AA requires a contrast
ratio of at least 4.5:1 for normal text) and by the ``"enforce"`` mode
which nudges lightness until the requirement is met.
"""

from __future__ import annotations

import colorsys
import re

_HEX3 = re.compile(r"^#([0-9a-fA-F]{3})$")
_HEX6 = re.compile(r"^#([0-9a-fA-F]{6})$")
_HEX8 = re.compile(r"^#([0-9a-fA-F]{8})$")

# Minimal CSS named-color set. Kept small on purpose — users targeting
# publication diagrams will pass hex; named colors are a convenience.
_CSS_NAMES: dict[str, str] = {
    "black": "#000000",
    "white": "#ffffff",
    "red": "#ff0000",
    "green": "#008000",
    "blue": "#0000ff",
    "gray": "#808080",
    "grey": "#808080",
    "transparent": "#00000000",
}


def parse_hex(color: str) -> tuple[int, int, int, int]:
    """Parse ``#rgb`` / ``#rrggbb`` / ``#rrggbbaa`` / CSS name → ``(r, g, b, a)``.

    Alpha defaults to 255. Raises :class:`ValueError` on malformed input.
    """
    key = color.strip().lower()
    if key in _CSS_NAMES:
        return parse_hex(_CSS_NAMES[key])

    if m := _HEX3.match(key):
        r, g, b = (int(c * 2, 16) for c in m.group(1))
        return r, g, b, 255
    if m := _HEX6.match(key):
        h = m.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255
    if m := _HEX8.match(key):
        h = m.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
    raise ValueError(f"Invalid color: {color!r}")


def _to_hex(r: int, g: int, b: int, a: int = 255) -> str:
    if a == 255:
        return f"#{r:02x}{g:02x}{b:02x}"
    return f"#{r:02x}{g:02x}{b:02x}{a:02x}"


def relative_luminance(color: str) -> float:
    """WCAG 2.1 relative luminance in the range [0, 1]."""
    r, g, b, _ = parse_hex(color)

    def _lin(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio between two colors. Always ≥ 1.0."""
    a = relative_luminance(fg)
    b = relative_luminance(bg)
    hi, lo = (a, b) if a > b else (b, a)
    return (hi + 0.05) / (lo + 0.05)


def meets_wcag_aa(fg: str, bg: str, *, large_text: bool = False) -> bool:
    """WCAG AA: 4.5:1 for normal text, 3:1 for large text."""
    threshold = 3.0 if large_text else 4.5
    return contrast_ratio(fg, bg) >= threshold


def adjust_for_contrast(fg: str, bg: str, *, large_text: bool = False) -> str:
    """Nudge ``fg`` toward black or white (in HLS space) until AA passes.

    Returns the original color unchanged if it already meets the target,
    or the closest lightness-adjusted color that does. Falls back to
    pure black/white if neither direction converges.
    """
    if meets_wcag_aa(fg, bg, large_text=large_text):
        return fg

    r, g, b, a = parse_hex(fg)
    h, _, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    bg_lum = relative_luminance(bg)
    # Push away from the background luminance: darker bg → lighter fg, and vice versa.
    directions = (1.0, 0.0) if bg_lum < 0.5 else (0.0, 1.0)

    for target_l in directions:
        for step in range(1, 21):
            new_l = _lerp(_current_l(fg), target_l, step / 20)
            nr, ng, nb = colorsys.hls_to_rgb(h, new_l, s)
            candidate = _to_hex(round(nr * 255), round(ng * 255), round(nb * 255), a)
            if meets_wcag_aa(candidate, bg, large_text=large_text):
                return candidate
    return "#000000" if bg_lum > 0.5 else "#ffffff"


def _current_l(color: str) -> float:
    r, g, b, _ = parse_hex(color)
    _, l, _ = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return l


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
