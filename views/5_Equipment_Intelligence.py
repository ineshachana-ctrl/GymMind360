"""Module 5 — Equipment Intelligence."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import equipment as eqa
from components import charts, layout
from components.kpi_cards import kpi_row
from components.theme import AMBER, GREEN, PALETTE, PURPLE, RED, TURQUOISE
from utils import state
from utils.formatting import money, num, pct

T = state.get_data()
F = state.get_filtered()
S = state.get_equipment_summary()

layout.page_header("Module 5 · Assets", "Equipment Intelligence",
                   "Usage, utilisation and maintenance of every machine, with a health score and automatic alerts.",
                   chips=[f"{len(S)} units · {S['equipment_type'].nunique()} equipment types"])

STATUS_COLOR = {"Healthy": GREEN, "Watch": AMBER, "Critical": RED}
mt_period = F["maintenance"]
kpi_row([
    dict(label="Avg utilisation", value=pct(S["utilization"].mean()), icon_name="activity", accent=PURPLE,
         delta_text=f"Last 30 days: {pct(S['utilization_30d'].mean())}"),
    dict(label="Units needing attention", value=num(S["needs_attention"].sum()), icon_name="alert",
         accent=RED if S["needs_attention"].sum() else GREEN,
         delta_text=f"{int((S['status'] == 'Critical').sum())} critical"),
    dict(label="Maintenance cost", value=money(mt_period["cost"].sum()), icon_name="tool", accent=AMBER,
         delta_text=f"{len(mt_period)} interventions in period"),
    dict(label="Downtime", value=f"{mt_period['downtime_hours'].sum():,.0f} h", icon_name="clock", accent=TURQUOISE,
         delta_text=f"Availability {pct(S['availability'].mean(), 2)}"),
    dict(label="Avg health score", value=f"{S['health_score'].mean():.0f}/100", icon_name="check", accent=GREEN,
         delta_text=f"Lowest: {S.loc[S['health_score'].idxmin(), 'name']}"),
])

layout.section("Equipment requiring attention", "Automatically flagged by utilisation, repair and maintenance rules")
att = S[S["needs_attention"]].sort_values("health_score")
if att.empty:
    st.success("All equipment is healthy.")
else:
    view = att[["name", "zone", "status", "health_score", "utilization", "repairs_12m", "days_since_service", "flags"]].copy()
    view["flags"] = view["flags"].map(lambda f: " • ".join(f))
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "name": "Equipment", "zone": "Zone", "status": "Status",
        "health_score": st.column_config.ProgressColumn("Health", min_value=0, max_value=100, format="%d"),
        "utilization": st.column_config.NumberColumn("Utilisation", format="percent"),
        "repairs_12m": "Repairs 12m", "days_since_service": "Days since service", "flags": "Reasons"})

c1, c2 = st.columns([1.4, 1])
with c1:
    s = S.sort_values("utilization", ascending=False)
    charts.show(charts.bar(s["name"], s["utilization"] * 100, "Utilisation by unit (% of opening hours)", horizontal=True,
                           colors=[STATUS_COLOR[x] for x in s["status"]], text_fmt="{:.0f}%"), 760)
with c2:
    u = eqa.usage_by_type(T)
    fig = px.treemap(u, path=["category", "equipment_type"], values="usage_hours", color="usage_hours",
                     color_continuous_scale=[[0, "#1E1B4B"], [0.5, PURPLE], [1, TURQUOISE]])
    fig.update_traces(texttemplate="<b>%{label}</b><br>%{value:,.0f} h", marker=dict(line=dict(color="#0B1020", width=2)))
    fig.update_layout(title="Equipment popularity (total usage hours)", coloraxis_showscale=False,
                      margin=dict(l=4, r=4, t=48, b=4))
    charts.show(fig, 370)
    fig = px.scatter(S, x="utilization", y="maintenance_cost", size=S["repairs_12m"] + 1, color="status",
                     color_discrete_map=STATUS_COLOR, hover_name="name", size_max=26)
    fig.update_layout(title="Utilisation vs maintenance cost", xaxis_tickformat=".0%", yaxis_tickprefix="€",
                      legend_title_text="", xaxis_title="Utilisation", yaxis_title="Maintenance cost")
    charts.show(fig, 370)

layout.section("Maintenance history")
c1, c2 = st.columns([1.5, 1])
with c1:
    mm = eqa.maintenance_monthly(mt_period)
    charts.show(charts.stacked_bar(mm, "Maintenance cost by month", money=True, colors=[RED, TURQUOISE]), 340)
with c2:
    mu = eqa.monthly_usage(F)
    charts.show(charts.stacked_bar(mu, "Usage hours by category", colors=PALETTE), 340)

layout.section("Equipment performance indicators")
kpi_tbl = S[["equipment_id", "name", "category", "brand", "age_years", "daily_sessions", "usage_hours", "utilization",
             "peer_util_ratio", "maintenance_events", "maintenance_cost", "cost_per_usage_hour", "availability",
             "health_score", "status"]]
st.dataframe(kpi_tbl, hide_index=True, width="stretch", column_config={
    "equipment_id": "ID", "name": "Equipment", "age_years": st.column_config.NumberColumn("Age (y)", format="%.1f"),
    "daily_sessions": st.column_config.NumberColumn("Sessions/day", format="%.1f"),
    "usage_hours": st.column_config.NumberColumn("Usage h", format="%,.0f"),
    "utilization": st.column_config.NumberColumn("Utilisation", format="percent"),
    "peer_util_ratio": st.column_config.NumberColumn("vs peers", format="%.2f×"),
    "maintenance_events": "Interventions",
    "maintenance_cost": st.column_config.NumberColumn("Maint. cost", format="€%.0f"),
    "cost_per_usage_hour": st.column_config.NumberColumn("€/usage h", format="€%.2f"),
    "availability": st.column_config.NumberColumn("Availability", format="percent"),
    "health_score": st.column_config.ProgressColumn("Health", min_value=0, max_value=100, format="%d")})

layout.section("Unit drill-down")
unit = st.selectbox("Select equipment", S["equipment_id"], format_func=lambda e: f"{e} · {S.set_index('equipment_id').at[e, 'name']}")
r = S.set_index("equipment_id").loc[unit]
c1, c2 = st.columns([1.5, 1])
with c1:
    us = F["equipment_usage"][F["equipment_usage"]["equipment_id"] == unit].copy()
    us["rolling"] = us["usage_minutes"].rolling(14, min_periods=1).mean() / 60
    fig = go.Figure()
    fig.add_bar(x=us["date"], y=us["usage_minutes"] / 60, name="Daily usage (h)", marker_color="rgba(139,92,246,0.45)")
    fig.add_scatter(x=us["date"], y=us["rolling"], name="14-day avg", line=dict(color=TURQUOISE, width=2.4))
    for m in F["maintenance"][F["maintenance"]["equipment_id"] == unit].itertuples():
        fig.add_vline(x=m.date, line=dict(color=RED if m.maintenance_type == "Repair" else GREEN, width=1, dash="dot"))
    fig.update_layout(title=f"{r['name']} — usage (red = repair, green = preventive service)", hovermode="x unified")
    charts.show(fig, 330)
with c2:
    charts.show(charts.gauge(float(r["health_score"]), "Health score", suffix=""), 220)
    st.markdown(" ".join(f"`{f}`" for f in r["flags"]) or "No alerts for this unit.")
log = T["maintenance"][T["maintenance"]["equipment_id"] == unit].sort_values("date", ascending=False)
st.dataframe(log[["date", "maintenance_type", "description", "cost", "downtime_hours"]], hide_index=True, width="stretch",
             column_config={"date": st.column_config.DateColumn("Date"), "cost": st.column_config.NumberColumn("Cost", format="€%.2f")})
