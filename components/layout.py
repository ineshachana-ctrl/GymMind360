"""Page-level layout helpers: header, sections, notes, insight cards, sidebar brand."""
from __future__ import annotations

import html

import streamlit as st

from components.theme import AMBER, BLUE, GREEN, RED, icon

SEVERITY = {"critical": (RED, "Critical"), "warning": (AMBER, "Warning"),
            "opportunity": (BLUE, "Opportunity"), "positive": (GREEN, "Positive")}


def page_header(eyebrow: str, title: str, subtitle: str = "", chips: list[str] | None = None,
                predictive: bool = False) -> None:
    chip_html = "".join(f'<span class="chip"><span class="dot"></span>{c}</span> ' for c in (chips or []))
    if predictive:
        chip_html += '<span class="chip pred"><span class="dot"></span>Contains ML predictions</span>'
    st.markdown(f'<div class="page-header"><div><div class="page-eyebrow">{eyebrow}</div>'
                f'<h1 class="page-title">{title}</h1><div class="page-sub">{subtitle}</div></div>'
                f'<div style="display:flex;gap:.4rem;flex-wrap:wrap">{chip_html}</div></div>',
                unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section"><h3>{title}</h3><p>{caption}</p></div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def insight_card(ins) -> None:
    color, label = SEVERITY.get(ins.severity, (BLUE, ins.severity.title()))
    st.markdown(
        f'<div class="insight" style="--sev:{color}">'
        f'<div class="insight-head"><div class="insight-title">{html.escape(ins.title)}</div>'
        f'<span class="sev">{label} · {html.escape(ins.category)}</span></div>'
        f'<div class="insight-metric">{html.escape(ins.metric)}</div>'
        f'<div class="insight-body">{ins.evidence}</div>'
        f'<div class="insight-why"><b>Why this was generated:</b> {ins.why}</div>'
        f'<div class="insight-action">→ {ins.action}</div></div>', unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        f'<div class="brand"><div class="brand-logo">{icon("activity", 22, "#fff", 2.4)}</div>'
        f'<div><div class="brand-name">GymMind <span>360</span></div>'
        f'<div class="brand-tag">Smart Gym Intelligence</div></div></div>', unsafe_allow_html=True)
