"""Consistent display rules; mathematical values retain full precision."""

from typing import Optional


def format_decimal(value: float, signed: bool = False) -> str:
    return ("{:+.3f}" if signed else "{:.3f}").format(value)


def format_integer(value: int) -> str:
    return "{:,}".format(value)


def _format_error_percentage(value: float) -> str:
    """Relative-error percent with six places or scientific notation if tiny."""

    percent = value * 100
    if percent != 0 and abs(percent) < 0.0000005:
        return "{:.6e}%".format(percent)
    return "{:.6f}%".format(percent)


def format_error(error: float, relative: Optional[float]) -> str:
    """Relative improvement error, or signed absolute error near zero gain."""

    if relative is not None:
        return _format_error_percentage(relative) + " relative"
    return "{:+.6e} absolute".format(error)
