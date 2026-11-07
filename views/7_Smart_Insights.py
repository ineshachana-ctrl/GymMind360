"""Module 7 — Smart Business Insights."""
import re

import pandas as pd
import streamlit as st

from analytics.insights import summary_counts
from components import layout
from components.kpi_cards import kpi_row
from components.theme import AMBER, BLUE, GREEN, RED
from utils import state

ins = state.get_insights()
ref = state.ref_date()
layout.page_header("Module 7 · Recommendation engine", "Smart Business Insights",
                   "Automated, explainable recommendations. Every card shows the evidence, the rule and threshold "
                   "that triggered it, and a suggested action.",
                   chips=[f"Analysed on {ref:%d %b %Y}", f"{len(ins)} insights"], predictive=True)

cnt = summary_counts(ins)
kpi_row([
    dict(label="Critical", value=str(cnt["critical"]), icon_name="alert", accent=RED, delta_text="Act this week"),
    dict(label="Warnings", value=str(cnt["warning"]), icon_name="zap", accent=AMBER, delta_text="Monitor closely"),
    dict(label="Opportunities", value=str(cnt["opportunity"]), icon_name="target", accent=BLUE, delta_text="Growth levers"),
    dict(label="Positive signals", value=str(cnt["positive"]), icon_name="check", accent=GREEN, delta_text="Keep doing"),
])
st.write("")

c1, c2 = st.columns([1, 1])
cats = sorted({i.category for i in ins})
with c1:
    sel_cat = st.multiselect("Category", cats, placeholder="All categories")
with c2:
    sel_sev = st.multiselect("Severity", ["critical", "warning", "opportunity", "positive"], placeholder="All severities",
                             format_func=str.title)
show = [i for i in ins if (not sel_cat or i.category in sel_cat) and (not sel_sev or i.severity in sel_sev)]

left, right = st.columns(2)
for n, i in enumerate(show):
    with (left if n % 2 == 0 else right):
        layout.insight_card(i)
if not show:
    st.info("No insight matches these filters.")


def strip(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


report = pd.DataFrame([{"severity": i.severity, "category": i.category, "title": i.title, "metric": i.metric,
                        "evidence": strip(i.evidence), "why": strip(i.why), "action": i.action} for i in ins])
md = "\n\n".join(f"### [{i.severity.upper()}] {i.title}\n**{i.metric}**\n\n{strip(i.evidence)}\n\n"
                 f"*Why:* {strip(i.why)}\n\n**Action:** {i.action}" for i in ins)
c1, c2, _ = st.columns([1, 1, 3])
c1.download_button("Download CSV", report.to_csv(index=False).encode(), "gymmind_insights.csv", "text/csv")
c2.download_button("Download report (.md)", f"# GymMind 360 — Business insights ({ref:%d %b %Y})\n\n{md}".encode(),
                   "gymmind_insights.md", "text/markdown")

with st.expander("How the recommendation engine works"):
    st.markdown("""
The engine (`analytics/insights.py`) runs a set of independent, transparent rules on the cleaned data and model outputs:

| Rule | Signal | Threshold |
|---|---|---|
| Churn risk | Share of active members in the *High* band of the churn model | ≥ 3% (critical ≥ 8%) |
| Dormant members | Active members with no check-in for 21+ days | ≥ 5% |
| Revenue trend | Month-over-month and rolling 3-month revenue change | ≤ −5% / ≤ −3% (positive ≥ +5%) |
| Registrations | New sign-ups vs trailing 3-month average | ≤ −25% |
| Renewals | Renewal rate by billing cycle; auto-renew churn ratio | −4 pp vs overall; ≥ 1.5× |
| Occupancy | Peak concurrent occupancy; weekday off-peak hours | ≥ 70% (warning ≥ 85%); < 15% |
| Equipment | Utilisation vs peers; overdue service; repeated repairs | < 45% of peers; > 115% interval; ≥ 4 repairs / 12 m |
| Costs & payments | Rolling 3-month expenses; failed payment rate; margin | ≥ +5%; ≥ 2%; ≥ 25% |

Insights are sorted by severity, then by estimated monthly financial impact.
""")
