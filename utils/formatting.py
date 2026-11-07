"""Formatting helpers and app-wide settings."""
from __future__ import annotations

import math

CURRENCY = "€"
GYM_NAME = "GymMind 360"
GYM_CAPACITY = 70          # max people on the floor at once (used for occupancy %)
OPEN_HOUR, CLOSE_HOUR = 6, 23
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _bad(v) -> bool:
    return v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def money(v, decimals: int = 0) -> str:
    if _bad(v):
        return "—"
    sign, v = ("−" if v < 0 else ""), abs(v)
    if v >= 1_000_000:
        return f"{sign}{CURRENCY}{v / 1_000_000:.2f}M"
    if v >= 10_000:
        return f"{sign}{CURRENCY}{v / 1_000:.1f}k"
    return f"{sign}{CURRENCY}{v:,.{decimals}f}"


def num(v, decimals: int = 0) -> str:
    if _bad(v):
        return "—"
    return f"{v:,.{decimals}f}"


def pct(v, decimals: int = 1, signed: bool = False) -> str:
    """`v` is a ratio (0.12 -> 12.0%)."""
    if _bad(v):
        return "—"
    s = f"{v * 100:+.{decimals}f}%" if signed else f"{v * 100:.{decimals}f}%"
    return s


def change(cur, prev):
    """Relative change cur vs prev (ratio) or None."""
    if _bad(cur) or _bad(prev) or prev == 0:
        return None
    return (cur - prev) / abs(prev)


def hour_label(h: int) -> str:
    return f"{int(h):02d}:00"
