"""Premium KPI cards (HTML) with delta vs previous period and an inline sparkline."""
from __future__ import annotations

import math

import streamlit as st

from components.theme import GREEN, PURPLE, RED, TURQUOISE, icon
from utils.formatting import change


def sparkline(values, color: str = TURQUOISE, w: int = 92, h: int = 30) -> str:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    step = w / (len(vals) - 1)
    pts = [(i * step, h - 3 - (v - lo) / rng * (h - 6)) for i, v in enumerate(vals)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{h} " + line + f" {w},{h}"
    gid = f"g{abs(hash((tuple(vals), color))) % 10**8}"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity=".35"/><stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
            f'</linearGradient></defs><polygon points="{area}" fill="url(#{gid})"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="2.6" fill="{color}"/></svg>')


def kpi_html(label: str, value: str, cur=None, prev=None, *, icon_name: str = "activity",
             accent: str = PURPLE, higher_is_better: bool = True, spark=None,
             delta_mode: str = "relative", delta_text: str | None = None, vs: str = "vs last month") -> str:
    """delta_mode: 'relative' (% change) or 'points' (difference of two ratios in pp)."""
    delta_html = ""
    if delta_text is not None:
        delta_html = f'<div class="kpi-delta flat">{delta_text}</div>'
    elif cur is not None and prev is not None:
        if delta_mode == "points":
            d = None if any(isinstance(x, float) and math.isnan(x) for x in (cur, prev)) else cur - prev
            txt = None if d is None else f"{d * 100:+.1f} pp"
        else:
            d = change(cur, prev)
            txt = None if d is None else f"{d * 100:+.1f}%"
        if d is None:
            delta_html = '<div class="kpi-delta flat">— <span class="vs">no comparison</span></div>'
        else:
            good = (d >= 0) == higher_is_better
            cls = "flat" if abs(d) < 0.0005 else ("good" if good else "bad")
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "■")
            delta_html = f'<div class="kpi-delta {cls}">{arrow} {txt} <span class="vs">{vs}</span></div>'
    spark_html = f'<div class="kpi-spark">{sparkline(spark, accent)}</div>' if spark else ""
    return (f'<div class="kpi-card" style="--accent:{accent}">'
            f'<div class="kpi-top"><span class="kpi-icon">{icon(icon_name, 15)}</span>{label}</div>'
            f'<div class="kpi-value">{value}</div>{delta_html}{spark_html}</div>')


def kpi_row(cards: list[dict], cols: int | None = None) -> None:
    """Render a row of KPI cards. Each dict holds kpi_html kwargs."""
    n = cols or len(cards)
    for i in range(0, len(cards), n):
        columns = st.columns(n, gap="small")
        for c, card in zip(columns, cards[i:i + n]):
            with c:
                st.markdown(kpi_html(**card), unsafe_allow_html=True)


def mini_stat(label: str, value: str, sub: str = "") -> str:
    return f'<div class="mini-stat"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


__all__ = ["kpi_html", "kpi_row", "mini_stat", "sparkline", "GREEN", "RED"]
