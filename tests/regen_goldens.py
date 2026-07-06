"""Regenerate every golden SVG. Usage: ``python -m tests.regen_goldens``."""

from tests.test_golden import _regen_all

if __name__ == "__main__":
    _regen_all()
