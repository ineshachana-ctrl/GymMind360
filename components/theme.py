"""Visual identity: colour tokens, global CSS and the Plotly template."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY = "#0B1020"
NAVY_2 = "#11182E"
NAVY_3 = "#18213D"
BORDER = "rgba(148, 163, 209, 0.14)"
TEXT = "#E7EAF6"
MUTED = "#8E97B8"
PURPLE = "#8B5CF6"
PURPLE_SOFT = "#A78BFA"
TURQUOISE = "#22D3EE"
TEAL = "#2DD4BF"
PINK = "#F472B6"
AMBER = "#FBBF24"
GREEN = "#34D399"
RED = "#F87171"
BLUE = "#60A5FA"

PALETTE = [PURPLE, TURQUOISE, PINK, AMBER, GREEN, BLUE, "#C084FC", "#5EEAD4"]
SEQ = [[0, "#141B34"], [0.35, "#3B2A7A"], [0.7, PURPLE], [1, TURQUOISE]]
DIVERGING = [[0, RED], [0.5, "#1C2444"], [1, GREEN]]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
  --navy:{NAVY}; --navy2:{NAVY_2}; --navy3:{NAVY_3}; --border:{BORDER};
  --text:{TEXT}; --muted:{MUTED}; --purple:{PURPLE}; --turq:{TURQUOISE};
}}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(139,92,246,0.16), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(34,211,238,0.10), transparent 60%),
    {NAVY};
  color: var(--text);
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1480px; }}

/* sidebar */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0E1530 0%, #0A0F22 100%);
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {{ border-radius: 10px; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
  background: linear-gradient(90deg, rgba(139,92,246,0.28), rgba(34,211,238,0.08));
  box-shadow: inset 3px 0 0 {PURPLE};
}}
[data-testid="stNavSectionHeader"] {{
  color: {MUTED}; letter-spacing: .12em; font-size: .68rem; font-weight: 700; text-transform: uppercase;
}}

[data-testid="stSidebarLogo"], [data-testid="stHeaderLogo"] {{ height: 2.9rem; max-width: 15rem; margin-top: .3rem; }}
/* brand */
.brand {{ display:flex; align-items:center; gap:.7rem; padding:.2rem 0 .9rem 0; }}
.brand-logo {{ width:40px; height:40px; border-radius:12px; display:grid; place-items:center;
  background: linear-gradient(135deg, {PURPLE}, {TURQUOISE}); box-shadow: 0 6px 24px rgba(139,92,246,.45); }}
.brand-name {{ font-family:'Space Grotesk', 'Inter', sans-serif; font-weight:700; font-size:1.18rem; line-height:1.05; color:#fff; }}
.brand-name span {{ background: linear-gradient(90deg, {PURPLE_SOFT}, {TURQUOISE}); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.brand-tag {{ color:{MUTED}; font-size:.72rem; letter-spacing:.04em; }}

/* page header */
.page-header {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap; margin-bottom: 1.4rem; }}
.page-eyebrow {{ color:{TURQUOISE}; font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.35rem; }}
.page-title {{ font-family:'Space Grotesk','Inter',sans-serif; font-size:2.05rem; font-weight:700; color:#fff; margin:0; line-height:1.1; }}
.page-sub {{ color:{MUTED}; font-size:.95rem; margin-top:.45rem; max-width: 760px; }}
.chip {{ display:inline-flex; align-items:center; gap:.4rem; padding:.32rem .7rem; border-radius:999px;
  border:1px solid var(--border); background:rgba(17,24,46,.7); color:{TEXT}; font-size:.75rem; font-weight:500; }}
.chip .dot {{ width:7px; height:7px; border-radius:50%; background:{GREEN}; box-shadow:0 0 8px {GREEN}; }}
.chip.pred {{ border-color: rgba(139,92,246,.5); color:{PURPLE_SOFT}; }}
.chip.pred .dot {{ background:{PURPLE}; box-shadow:0 0 8px {PURPLE}; }}

/* section */
.section {{ margin: 1.6rem 0 .7rem 0; display:flex; align-items:baseline; gap:.75rem; }}
.section h3 {{ font-family:'Space Grotesk','Inter',sans-serif; font-size:1.12rem; font-weight:600; color:#fff; margin:0; }}
.section p {{ color:{MUTED}; font-size:.82rem; margin:0; }}

/* KPI cards */
.kpi-card {{ position:relative; overflow:hidden; border-radius:16px; padding:1rem 1.1rem .8rem 1.1rem;
  background: linear-gradient(160deg, rgba(24,33,61,.95), rgba(14,20,40,.95));
  border:1px solid var(--border); min-height: 138px; transition: transform .15s ease, border-color .15s ease; }}
.kpi-card:hover {{ transform: translateY(-2px); border-color: rgba(139,92,246,.45); }}
.kpi-card::before {{ content:""; position:absolute; inset:0 0 auto 0; height:2px; background: var(--accent); opacity:.9; }}
.kpi-top {{ display:flex; align-items:center; gap:.55rem; color:{MUTED}; font-size:.78rem; font-weight:500; letter-spacing:.02em; }}
.kpi-icon {{ width:28px; height:28px; border-radius:8px; display:grid; place-items:center;
  background: color-mix(in srgb, var(--accent) 16%, transparent); color: var(--accent); }}
.kpi-value {{ font-family:'Space Grotesk','Inter',sans-serif; font-size:1.75rem; font-weight:700; color:#fff; margin:.55rem 0 .15rem 0; letter-spacing:-.01em; }}
.kpi-delta {{ font-size:.76rem; font-weight:600; display:flex; align-items:center; gap:.35rem; }}
.kpi-delta .vs {{ color:{MUTED}; font-weight:400; }}
.kpi-delta.good {{ color:{GREEN}; }} .kpi-delta.bad {{ color:{RED}; }} .kpi-delta.flat {{ color:{MUTED}; }}
.kpi-spark {{ position:absolute; right:.8rem; bottom:.7rem; opacity:.95; }}

/* insight cards */
.insight {{ border-radius:14px; padding:1rem 1.15rem; margin-bottom:.8rem; border:1px solid var(--border);
  background: linear-gradient(160deg, rgba(24,33,61,.9), rgba(14,20,40,.9)); border-left:4px solid var(--sev); }}
.insight-head {{ display:flex; justify-content:space-between; gap:1rem; align-items:center; }}
.insight-title {{ font-weight:650; color:#fff; font-size:1rem; }}
.sev {{ font-size:.68rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; padding:.2rem .55rem; border-radius:999px;
  color: var(--sev); background: color-mix(in srgb, var(--sev) 14%, transparent); border:1px solid color-mix(in srgb, var(--sev) 40%, transparent); white-space:nowrap; }}
.insight-metric {{ font-family:'Space Grotesk','Inter',sans-serif; font-size:1.35rem; font-weight:700; color: var(--sev); margin:.35rem 0 .2rem 0; }}
.insight-body {{ color:#C5CAE0; font-size:.86rem; line-height:1.5; }}
.insight-body b {{ color:#fff; }}
.insight-why {{ margin-top:.55rem; padding:.55rem .7rem; border-radius:10px; background: rgba(11,16,32,.6); color:{MUTED}; font-size:.8rem; }}
.insight-action {{ margin-top:.5rem; color:{TURQUOISE}; font-size:.84rem; font-weight:600; }}

/* misc */
.note {{ border:1px dashed rgba(139,92,246,.45); background: rgba(139,92,246,.07); border-radius:12px; padding:.7rem .9rem; color:#CFC6F7; font-size:.84rem; }}
.mini-stat {{ border-radius:12px; padding:.7rem .9rem; background: rgba(17,24,46,.75); border:1px solid var(--border); }}
.mini-stat .l {{ color:{MUTED}; font-size:.74rem; }}
.mini-stat .v {{ color:#fff; font-family:'Space Grotesk','Inter',sans-serif; font-size:1.25rem; font-weight:700; }}
.mini-stat .s {{ color:{MUTED}; font-size:.72rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--border) !important; border-radius: 16px !important;
  background: rgba(14, 20, 40, .55); }}
div[data-testid="stTabs"] button[role="tab"] {{ font-weight:600; }}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: linear-gradient(90deg, {PURPLE}, {TURQUOISE}); }}
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 10px; font-weight: 600; }}
.stButton > button[kind="primary"] {{ background: linear-gradient(90deg, {PURPLE}, #6D5CF6); border: 0; }}
footer {{ visibility: hidden; }}
@media (max-width: 640px) {{
  .page-title {{ font-size: 1.55rem; }}
  .kpi-value {{ font-size: 1.45rem; }}
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
}}
</style>
"""


def register_plotly_template() -> None:
    axis = dict(gridcolor="rgba(148,163,209,0.09)", zeroline=False, linecolor="rgba(148,163,209,0.18)",
                tickfont=dict(color=MUTED, size=11), title=dict(font=dict(color=MUTED, size=12)))
    pio.templates["gymmind"] = go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#C9CEE6", size=12),
        colorway=PALETTE, xaxis=axis, yaxis=axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#C9CEE6", size=11)),
        margin=dict(l=8, r=8, t=48, b=8),
        hoverlabel=dict(bgcolor="#141C38", bordercolor=PURPLE, font=dict(color="#fff", family="Inter")),
        title=dict(font=dict(size=14, color=TEXT, family="Inter"), x=0, xanchor="left", y=0.97),
        colorscale=dict(sequential=SEQ),
        bargap=0.28,
    ))
    pio.templates.default = "plotly_dark+gymmind"


def inject() -> None:
    register_plotly_template()
    st.markdown(CSS, unsafe_allow_html=True)


ICONS = {
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "dollar": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "trend-up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "trend-down": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "refresh": '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    "user-x": '<path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><line x1="18" y1="8" x2="23" y2="13"/><line x1="23" y1="8" x2="18" y2="13"/>',
    "user-plus": '<path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><line x1="20" y1="8" x2="20" y2="14"/><line x1="23" y1="11" x2="17" y2="11"/>',
    "card": '<rect x="1" y="4" width="22" height="16" rx="2"/><line x1="1" y1="10" x2="23" y2="10"/>',
    "pie": '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "tool": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
}


def icon(name: str, size: int = 16, color: str = "currentColor", stroke: float = 2) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
            f'{ICONS.get(name, ICONS["activity"])}</svg>')
