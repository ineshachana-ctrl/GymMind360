"""Module 4 — Smart Gym Occupancy."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import occupancy as occ
from components import charts, layout
from components.kpi_cards import kpi_row
from components.theme import AMBER, BLUE, GREEN, MUTED, PINK, PURPLE, RED, TURQUOISE
from models import forecasting
from utils import state
from utils.formatting import GYM_CAPACITY, WEEKDAYS, hour_label, num, pct

T = state.get_data()
F = state.get_filtered()
A = F["attendance"]
start, end, _ = state.get_filters()

layout.page_header("Module 4 · Operations", "Smart Gym Occupancy",
                   "When members come, how crowded the floor gets and what the coming days will look like.",
                   chips=[f"{start:%d %b %Y} → {end:%d %b %Y}", f"Capacity {GYM_CAPACITY} people"], predictive=True)
if A.empty:
    st.warning("No attendance in the selected period.")
    st.stop()

pq = occ.peak_quiet(A)
prof = occ.hourly_profile(A)
wd = occ.weekday_profile(A)
daily = occ.daily_trend(A)
kpi_row([
    dict(label="Avg daily visits", value=num(daily["visits"].mean()), icon_name="users", accent=PURPLE,
         delta_text=f"{num(len(A))} visits in period"),
    dict(label="Peak hour", value=hour_label(pq["peak_hour"]), icon_name="zap", accent=PINK,
         delta_text=f"Busiest slot: {pq['peak_slots'][0][0]} {hour_label(pq['peak_slots'][0][1])}"),
    dict(label="Quietest hour", value=hour_label(pq["quiet_hour"]), icon_name="clock", accent=TURQUOISE,
         delta_text="Lowest average occupancy"),
    dict(label="Max avg occupancy", value=pct(pq["max_occupancy"], 0), icon_name="alert",
         accent=RED if pq["max_occupancy"] >= 0.85 else AMBER, delta_text=f"of {GYM_CAPACITY}-person capacity"),
    dict(label="Avg session length", value=f"{A['duration_min'].mean():.0f} min", icon_name="activity", accent=GREEN,
         delta_text=f"Busiest day: {wd.loc[wd['avg_visits'].idxmax(), 'weekday']}"),
])

layout.section("Weekly attendance heatmap", "Average per weekday and hour (real historical data)")
mode = st.segmented_control("Metric", ["People on the floor", "Check-ins", "Occupancy %"], default="People on the floor",
                            label_visibility="collapsed")
if mode == "Check-ins":
    z, fmt, suf = occ.arrivals_matrix(A), ".0f", " check-ins"
elif mode == "Occupancy %":
    z, fmt, suf = occ.concurrency_matrix(A) / GYM_CAPACITY * 100, ".0f", "%"
else:
    z, fmt, suf = occ.concurrency_matrix(A), ".0f", " people"
z.columns = [hour_label(h) for h in z.columns]
charts.show(charts.heatmap(z, fmt=fmt, hover_suffix=suf), 360)

c1, c2 = st.columns([1.6, 1])
with c1:
    cols = [RED if h == pq["peak_hour"] else (TURQUOISE if h == pq["quiet_hour"] else PURPLE) for h in prof["hour"]]
    fig = go.Figure()
    fig.add_bar(x=[hour_label(h) for h in prof["hour"]], y=prof["avg_present"], marker_color=cols, name="People present")
    fig.add_scatter(x=[hour_label(h) for h in prof["hour"]], y=prof["avg_arrivals"], name="Check-ins / hour",
                    mode="lines+markers", line=dict(color=AMBER, width=2))
    fig.add_hline(y=GYM_CAPACITY * 0.85, line=dict(color=RED, dash="dot", width=1),
                  annotation_text="85% capacity", annotation_font_color=RED)
    fig.update_layout(title="Attendance by hour (peak in red, quietest in turquoise)", hovermode="x unified")
    charts.show(fig, 360)
with c2:
    cols = [PINK if v == wd["avg_visits"].max() else (TURQUOISE if v == wd["avg_visits"].min() else PURPLE)
            for v in wd["avg_visits"]]
    charts.show(charts.bar(wd["weekday"], wd["avg_visits"], "Average visits by weekday", colors=cols,
                           text_fmt="{:.0f}"), 360)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Peak slots**")
    st.dataframe(pd.DataFrame([{"Day": d, "Hour": hour_label(h), "Avg people": round(v, 1),
                                "Occupancy": f"{v / GYM_CAPACITY:.0%}"} for d, h, v in pq["peak_slots"]]),
                 hide_index=True, width="stretch")
with c2:
    st.markdown("**Quiet slots**")
    st.dataframe(pd.DataFrame([{"Day": d, "Hour": hour_label(h), "Avg people": round(v, 1),
                                "Occupancy": f"{v / GYM_CAPACITY:.0%}"} for d, h, v in pq["quiet_slots"]]),
                 hide_index=True, width="stretch")

layout.section("Attendance trends")
fig = go.Figure()
fig.add_scatter(x=daily["date"], y=daily["visits"], name="Daily visits", mode="lines",
                line=dict(color="rgba(139,92,246,0.35)", width=1))
fig.add_scatter(x=daily["date"], y=daily["rolling_7"], name="7-day avg", line=dict(color=PURPLE, width=2))
fig.add_scatter(x=daily["date"], y=daily["rolling_28"], name="28-day avg", line=dict(color=TURQUOISE, width=2.6))
fig.update_layout(hovermode="x unified")
charts.show(fig, 330)

layout.section("Predicted attendance — next 7 days",
               "Model output (Gradient Boosting) — not observed data. See Predictive Analytics for evaluation.")
fc = state.get_attendance_forecast()
hf = forecasting.hourly_forecast(T["attendance"], fc.forecast.head(7))
if not hf.empty:
    piv = hf.pivot_table(index="date", columns="hour", values="expected_arrivals").fillna(0)
    piv.index = [f"{d:%a %d %b}" for d in piv.index]
    piv.columns = [hour_label(h) for h in piv.columns]
    charts.show(charts.heatmap(piv, title="Predicted check-ins per hour", fmt=".0f", hover_suffix=" predicted check-ins",
                               colorscale=[[0, "#141B34"], [0.5, "#5B3FC4"], [1, "#C4B5FD"]]), 330)
    nxt = fc.forecast.head(7)
    st.caption("Daily totals (predicted): " + " · ".join(f"{d:%a} {p:.0f}" for d, p in zip(nxt["date"], nxt["predicted"])))
