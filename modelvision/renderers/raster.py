"""PNG export via cairosvg.

Isolated so the ``pdf`` extra (which ships cairosvg) can be added
independently. Also used by the PDF renderer in M5.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from modelvision.core._optional import require


def svg_to_png(
    svg: str,
    output: str | os.PathLike[str],
    *,
    dpi: int = 300,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Rasterize ``svg`` to a PNG file at ``output``.

    ``dpi`` defaults to 300 — publication standard. ``width`` and
    ``height`` override the SVG's natural viewBox pixel size; passing
    just one preserves the aspect ratio.
    """
    cairosvg = require("cairosvg", extra="pdf")
    p = Path(os.fspath(output))
    p.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "bytestring": svg.encode("utf-8"),
        "write_to": str(p),
        "dpi": dpi,
    }
    if width is not None:
        kwargs["output_width"] = width
    if height is not None:
        kwargs["output_height"] = height
    cairosvg.svg2png(**kwargs)


def svg_to_pdf(svg: str, output: str | os.PathLike[str]) -> None:
    """Convert ``svg`` to a PDF file at ``output``."""
    cairosvg = require("cairosvg", extra="pdf")
    p = Path(os.fspath(output))
    p.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(p))


def svg_to_pil(svg: str) -> Any:
    """Return a :class:`PIL.Image.Image` — useful for Jupyter inline display."""
    import io

    cairosvg = require("cairosvg", extra="pdf")
    from PIL import Image

    buf = io.BytesIO()
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=buf)
    buf.seek(0)
    return Image.open(buf)


__all__ = ["svg_to_pdf", "svg_to_pil", "svg_to_png"]
