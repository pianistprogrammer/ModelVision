"""Exception and warning taxonomy.

Every "silent failure would be bad" branch calls :func:`mv_warn`, which
routes through :func:`warnings.warn` with a stable category so users can
filter or promote to errors via ``warnings.filterwarnings``.
"""

from __future__ import annotations

import warnings


class ModelVisionError(Exception):
    """Base class for all ModelVision errors."""


class InspectionError(ModelVisionError):
    """Raised when a model cannot be inspected into a :class:`ModelGraph`."""


class RenderError(ModelVisionError):
    """Raised when a :class:`ModelGraph` cannot be rendered."""


class AmbiguousFrameworkError(ModelVisionError):
    """Raised when framework auto-detection cannot pick a single winner."""


class ModelVisionWarning(UserWarning):
    """Base warning category for all non-fatal ModelVision issues."""


def mv_warn(message: str, *, stacklevel: int = 2) -> None:
    """Emit a :class:`ModelVisionWarning`. Prefer this over bare ``warnings.warn``."""
    warnings.warn(message, ModelVisionWarning, stacklevel=stacklevel + 1)
