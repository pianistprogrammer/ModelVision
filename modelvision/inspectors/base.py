"""Abstract base class for all inspectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from modelvision.core.ir import ModelGraph


class BaseInspector(ABC):
    """Every framework inspector implements this two-method contract."""

    #: Short string identifying the framework — ``"torch"``, ``"keras"``, etc.
    framework: str = "unknown"

    @abstractmethod
    def can_handle(self, model: Any) -> bool:
        """Return ``True`` if this inspector recognizes ``model``."""

    @abstractmethod
    def inspect(self, model: Any, **kwargs: Any) -> ModelGraph:
        """Extract the :class:`ModelGraph` IR from ``model``."""
