"""Module 8 — Member Fitness Analytics (voluntary, non-medical)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import fitness as fit
from components import charts, layout
from components.kpi_cards import kpi_row
from components.theme import AMBER, BLUE, GREEN, PINK, PURPLE, TURQUOISE
from utils import state
from utils.formatting import WEEKDAYS, num, pct

T = state.get_data()
layout.page_header("Module 8 · Personal performance", "Member Fitness Analytics",
                   "Training history, frequency, consistency and progress for members who chose to share their workout logs.")
layout.note("<b>Privacy & scope.</b> Only members who gave explicit consent (<code>data_consent = True</code>) and logged "
            "workouts are listed. Metrics such as estimated 1-rep max (Epley formula) are <b>training indicators, not "
            "medical or health assessments</b>. No diagnosis or health advice is provided.")
st.write("")

cm = fit.consenting_members(T["members"], T["workouts"])
if cm.empty:
    st.info("No consenting member with workout data.")
    st.stop()
c1, c2 = st.columns([2, 1])
with c1:
    mid = st.selectbox(f"Member ({len(cm)} consenting members)", cm["member_id"],
                       format_func=lambda m: cm.set_index("member_id").at[m, "label"])
with c2:
    st.caption(f"Consent rate: {pct(T['members']['data_consent'].mean())} of all members")

log = fit.member_log(T["workouts"], mid)
s = fit.summary(log)
kpi_row([
    dict(label="Sessions logged", value=num(s["sessions"]), icon_name="calendar", accent=PURPLE,
         delta_text=f"{s['first']:%d %b %Y} → {s['last']:%d %b %Y}"),
    dict(label="Training frequency", value=f"{s['per_week']:.1f}/wk", icon_name="activity", accent=TURQUOISE,
         delta_text=f"Current streak: {s['streak_weeks']} weeks"),
    dict(label="Consistency (12 wk)", value=pct(s["consistency"], 0), icon_name="check", accent=GREEN,
         delta_text="Weeks with ≥ 2 sessions"),
    dict(label="Strength progress", value=pct(s["avg_strength_gain"], 1, signed=True) if pd.notna(s["avg_strength_gain"]) else "—",
         icon_name="trend-up", accent=AMBER, delta_text="Avg est. 1RM change, start → now"),
    dict(label="Favourite exercise", value=s["favourite"], icon_name="target", accent=PINK,
         delta_text=f"Total volume {s['total_volume'] / 1000:,.0f} t"),
])

layout.section("Workout history")
cal = fit.calendar(log)
piv = cal.pivot_table(index="weekday", columns="week", values="sessions", aggfunc="sum").reindex(range(7)).fillna(0)
piv.index = WEEKDAYS
piv.columns = [f"{c:%d %b}" for c in piv.columns]
charts.show(charts.heatmap(piv, "Training calendar (sessions per day)", fmt=".0f", show_text=False,
                           colorscale=[[0, "#141B34"], [0.01, "#141B34"], [0.5, PURPLE], [1, TURQUOISE]], zmin=0, zmax=1), 260)

c1, c2 = st.columns(2)
with c1:
    w = fit.sessions_per_week(log)
    fig = go.Figure(go.Bar(x=w["week"], y=w["sessions"], marker_color=[TURQUOISE if v >= 2 else PURPLE for v in w["sessions"]]))
    fig.add_hline(y=2, line=dict(color=AMBER, dash="dot"), annotation_text="consistency target", annotation_font_color=AMBER)
    fig.update_layout(title="Training frequency (sessions / week)")
    charts.show(fig, 320)
with c2:
    v = fit.weekly_volume(log)
    charts.show(charts.area_line(v, "week", "volume", "Weekly strength volume (kg lifted)", color=PURPLE), 320)

layout.section("Exercise progress")
ex_list = log["exercise"].value_counts().index.tolist()
ex = st.selectbox("Exercise", ex_list)
p = fit.exercise_progress(log, ex)
is_strength = log.loc[log["exercise"] == ex, "exercise_type"].iloc[0] == "Strength"
fig = go.Figure()
fig.add_scatter(x=p["date"], y=p["metric"], mode="markers", name="Session", marker=dict(color=PURPLE, size=7, opacity=0.7))
fig.add_scatter(x=p["date"], y=p["trend"], mode="lines", name="5-session trend", line=dict(color=TURQUOISE, width=3))
fig.update_layout(title=f"{ex} — {'estimated 1RM (kg)' if is_strength else 'average pace (km/h)'}", hovermode="x unified")
charts.show(fig, 340)

c1, c2 = st.columns([1.4, 1])
with c1:
    sp = fit.strength_progress(log)
    if not sp.empty:
        st.markdown("**Strength evolution by exercise**")
        st.dataframe(sp, hide_index=True, width="stretch", column_config={
            "exercise": "Exercise", "start_1rm": st.column_config.NumberColumn("Start est. 1RM", format="%.1f kg"),
            "current_1rm": st.column_config.NumberColumn("Current est. 1RM", format="%.1f kg"),
            "best_1rm": st.column_config.NumberColumn("Best", format="%.1f kg"),
            "change_pct": st.column_config.NumberColumn("Change", format="percent")})
with c2:
    e = fit.effort_distribution(log)
    charts.show(charts.bar(e["rpe"].astype(int).astype(str), e["sets"], "Perceived effort (RPE 1–10)",
                           colors=[BLUE, BLUE, PURPLE, PURPLE, PINK, PINK][:len(e)]), 300)
