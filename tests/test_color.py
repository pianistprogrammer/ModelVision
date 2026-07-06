"""Tests for the WCAG color utilities."""

from __future__ import annotations

import pytest

from modelvision.core.color import (
    adjust_for_contrast,
    contrast_ratio,
    meets_wcag_aa,
    parse_hex,
    relative_luminance,
)


def test_parse_hex_forms() -> None:
    assert parse_hex("#fff") == (255, 255, 255, 255)
    assert parse_hex("#ffffff") == (255, 255, 255, 255)
    assert parse_hex("#ffffff80") == (255, 255, 255, 128)
    assert parse_hex("black") == (0, 0, 0, 255)


def test_parse_hex_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid color"):
        parse_hex("#xyz")


def test_relative_luminance_endpoints() -> None:
    assert relative_luminance("#000000") == 0
    assert relative_luminance("#ffffff") == pytest.approx(1.0)


def test_contrast_ratio_extremes() -> None:
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, rel=0.01)
    assert contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0)


def test_wcag_aa_thresholds() -> None:
    assert meets_wcag_aa("#000000", "#ffffff")
    assert not meets_wcag_aa("#cccccc", "#ffffff")


def test_adjust_for_contrast_converges() -> None:
    fixed = adjust_for_contrast("#cccccc", "#ffffff")
    assert meets_wcag_aa(fixed, "#ffffff")

    fixed_dark = adjust_for_contrast("#333333", "#000000")
    assert meets_wcag_aa(fixed_dark, "#000000")


def test_adjust_pass_through_when_already_ok() -> None:
    # Black on white already passes; should return unchanged.
    assert adjust_for_contrast("#000000", "#ffffff") == "#000000"
