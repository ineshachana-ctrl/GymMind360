"""Module 3 — Revenue Analytics."""
import plotly.graph_objects as go
import streamlit as st

from analytics import revenue as rev
from components import charts, layout
from components.kpi_cards import kpi_row
from components.theme import AMBER, BLUE, GREEN, PALETTE, PINK, PURPLE, RED, TURQUOISE
from utils import state
from utils.formatting import money, pct

F = state.get_filtered()
mo = state.get_monthly()
start, end, _ = state.get_filters()
P = F["payments"]

layout.page_header("Module 3 · Finance", "Revenue Analytics",
                   "Where the money comes from, how it evolves and what remains after operating costs.",
                   chips=[f"{start:%d %b %Y} → {end:%d %b %Y}"])
if mo.empty:
    st.warning("No data in the selected period.")
    st.stop()

total, profit = mo["revenue"].sum(), mo["profit"].sum()
best = mo.loc[mo["revenue"].idxmax()]
g_last = state.get_kpi_context()[1]["revenue_growth"].iloc[-1]  # vs the true previous month
kpi_row([
    dict(label="Total revenue", value=money(total), icon_name="dollar", accent=GREEN,
         delta_text=f"{money(total / len(mo))} per month"),
    dict(label="Estimated profit", value=money(profit), icon_name="pie", accent=PURPLE,
         delta_text=f"Margin {pct(profit / total if total else 0)}"),
    dict(label="Last month vs previous", value=pct(g_last, signed=True),
         icon_name="trend-up" if g_last >= 0 else "trend-down",
         accent=TURQUOISE if g_last >= 0 else RED, delta_text=f"{mo['month'].iloc[-1]:%B %Y}"),
    dict(label="Revenue per member", value=money(mo["arpu"].mean(), 2), icon_name="target", accent=BLUE,
         delta_text="Monthly average (ARPU)"),
    dict(label="Best month", value=money(best["revenue"]), icon_name="zap", accent=AMBER,
         delta_text=f"{best['month']:%B %Y}"),
])

layout.section("Revenue by month", "Stacked by revenue stream, with estimated profit")
if state.get_filters()[2]:
    st.caption("Membership-type filter active: gym-wide expenses are allocated pro rata to the selected plans' share of active members.")
by_type = rev.by_revenue_type(P)
fig = charts.stacked_bar(by_type, colors=[PURPLE, TURQUOISE, AMBER], money=True)
fig.add_scatter(x=mo["month"], y=mo["profit"], name="Profit", mode="lines+markers",
                line=dict(color=GREEN, width=3), marker=dict(size=6))
charts.show(fig, 380)

c1, c2 = st.columns(2)
with c1:
    g = mo["revenue_growth"].fillna(0) * 100
    charts.show(charts.bar(mo["month"], g, "Month-over-month revenue growth (%)",
                           colors=[GREEN if v >= 0 else RED for v in g], text_fmt="{:+.1f}%"), 330)
with c2:
    fig = go.Figure()
    fig.add_scatter(x=mo["month"], y=mo["profit_margin"] * 100, mode="lines+markers", name="Profit margin",
                    line=dict(color=TURQUOISE, width=2.6), fill="tozeroy", fillcolor="rgba(34,211,238,0.10)")
    fig.update_layout(title="Monthly profit margin (%)")
    charts.show(fig, 330)

layout.section("Revenue breakdown")
c1, c2, c3 = st.columns(3)
with c1:
    t = rev.by_membership_type(P)
    charts.show(charts.bar(t["membership_type"], t["revenue"], "By membership type", money=True,
                           colors=PALETTE[:len(t)], text_fmt="€{:,.0f}"), 330)
with c2:
    m = rev.by_payment_method(P)
    charts.show(charts.donut(m["payment_method"], m["revenue"], "By payment method"), 330)
with c3:
    seg = state.get_segments()
    s = rev.by_segment(P, seg.data.set_index("member_id")["segment"])
    charts.show(charts.bar(s["segment"], s["revenue"], "By member segment (K-Means)", horizontal=True, money=True,
                           colors=PALETTE[:len(s)]), 330)

layout.section("Costs & payment health")
c1, c2 = st.columns([1.5, 1])
with c1:
    ex = rev.expense_breakdown(F["expenses"])
    charts.show(charts.stacked_bar(ex, "Operating expenses by category", money=True,
                                   colors=[PURPLE, TURQUOISE, PINK, AMBER, GREEN, BLUE, "#C084FC"]), 360)
with c2:
    ph = rev.payment_health(P)
    fig = go.Figure(go.Scatter(x=ph.index, y=ph["failure_rate"] * 100, mode="lines+markers",
                               line=dict(color=RED, width=2.4), name="Failure rate"))
    fig.update_layout(title="Failed payment rate (%)")
    charts.show(fig, 360)

with st.expander("Monthly financial table"):
    tbl = mo[["month", "revenue", "membership_revenue", "expenses", "profit", "profit_margin", "revenue_growth", "arpu"]]
    st.dataframe(tbl.style.format({"month": "{:%b %Y}", "revenue": "€{:,.0f}", "membership_revenue": "€{:,.0f}",
                                   "expenses": "€{:,.0f}", "profit": "€{:,.0f}", "profit_margin": "{:.1%}",
                                   "revenue_growth": "{:+.1%}", "arpu": "€{:,.2f}"}, na_rep="—"),
                 hide_index=True, width="stretch")
    st.download_button("Download CSV", tbl.to_csv(index=False).encode(), "gymmind_monthly_finance.csv", "text/csv")
