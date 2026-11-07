"""Module 2 — Member Intelligence."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import members as man
from components import charts, layout, mpl
from components.kpi_cards import kpi_row
from components.theme import AMBER, BLUE, GREEN, PALETTE, PINK, PURPLE, RED, TURQUOISE
from models.segmentation import SEG_LABELS
from utils import state
from utils.formatting import money, num, pct

T = state.get_data()
F = state.get_filtered()
mo = state.get_monthly()
start, end, plans = state.get_filters()
ref = state.ref_date()
M = F["members"]

layout.page_header("Module 2 · Members", "Member Intelligence",
                   "Acquisition, retention, churn and behavioural segmentation of your member base.",
                   chips=[f"{start:%d %b %Y} → {end:%d %b %Y}"])

ten = man.tenure(M, ref)
new = M[(M["join_date"] >= start) & (M["join_date"] <= end)]
coh = man.cohort_retention(M, ref, start=ref - pd.DateOffset(months=24))
freq = man.visit_frequency(T["attendance"], M, ref)
kpi_row([
    dict(label="New registrations", value=num(len(new)), icon_name="user-plus", accent=GREEN,
         delta_text=f"{len(new) / max(len(mo), 1):.0f} per month on average"),
    dict(label="Avg membership duration", value=f"{ten['tenure_months'].mean():.1f} mo", icon_name="calendar",
         accent=PURPLE, delta_text=f"Active: {ten.loc[ten.status == 'Active', 'tenure_months'].mean():.1f} mo"),
    dict(label="Avg monthly churn", value=pct(mo["churn_rate"].mean()), icon_name="trend-down", accent=PINK,
         delta_text=f"{int(mo['churned'].sum())} cancellations in period"),
    dict(label="6-month retention", value=pct(coh["M6"].mean() / 100), icon_name="refresh", accent=TURQUOISE,
         delta_text="Avg of cohorts that reached month 6"),
    dict(label="Visits / week (active)", value=f"{freq['visits_per_week'].mean():.2f}", icon_name="activity",
         accent=AMBER, delta_text="Last 90 days"),
])

tab1, tab2, tab3, tab4 = st.tabs(["Acquisition", "Retention & churn", "Behaviour", "Segmentation (K-Means)"])

with tab1:
    c1, c2 = st.columns([1.6, 1])
    with c1:
        reg = man.registrations(M, start, end)
        charts.show(charts.stacked_bar(reg, "New registrations by month & plan"), 360)
    with c2:
        mix = M["membership_type"].value_counts()
        charts.show(charts.donut(mix.index, mix.values, "Membership types (all members)"), 360)
    c1, c2 = st.columns(2)
    with c1:
        d = man.demographics(M)
        g = d.groupby(["age_band", "gender"], observed=False).size().unstack(fill_value=0)
        charts.show(charts.stacked_bar(g, "Age & gender profile", colors=[PINK, PURPLE, AMBER, BLUE]), 320)
    with c2:
        cyc = M.groupby(["billing_cycle", "membership_type"]).size().unstack(fill_value=0)
        charts.show(charts.stacked_bar(cyc, "Billing cycle by plan", horizontal=True), 320)

with tab2:
    c1, c2 = st.columns([1.25, 1])
    with c1:
        layout.section("Cohort retention", "% of each joining cohort still active N months later")
        f, ax = mpl.fig(8, 5.2)
        show = coh.tail(14)
        import seaborn as sns
        sns.heatmap(show, annot=True, fmt=".0f", cmap=mpl.CMAP, vmin=40, vmax=100, linewidths=1.5,
                    linecolor=mpl.NAVY_2, cbar_kws={"label": "% retained"}, ax=ax, annot_kws={"size": 7})
        ax.set_xlabel("Months since joining")
        ax.set_ylabel("Join cohort")
        st.pyplot(f, clear_figure=True)
    with c2:
        charts.show(charts.area_line(mo.assign(churn_pct=mo["churn_rate"] * 100), "month", "churn_pct",
                                     "Monthly churn rate (%)", color=PINK), 260)
        charts.show(charts.area_line(mo.assign(ren=mo["renewal_rate"] * 100), "month", "ren",
                                     "Renewal rate (%)", color=TURQUOISE), 260)
    c1, c2, c3 = st.columns(3)
    for col, by, title in [(c1, "membership_type", "Churn by plan"), (c2, "billing_cycle", "Churn by billing cycle"),
                           (c3, "auto_renew", "Churn by auto-renew")]:
        with col:
            g = man.churn_by(M, by)
            g[by] = g[by].astype(str)
            colors = [RED if v == g["churn_rate"].max() else PURPLE for v in g["churn_rate"]]
            charts.show(charts.bar(g[by], g["churn_rate"] * 100, title, colors=colors, text_fmt="{:.1f}%"), 280)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        band = freq["band"].value_counts().sort_index()
        charts.show(charts.bar(band.index.astype(str), band.values, "Attendance frequency (visits / week, last 90 days)",
                               colors=[RED, AMBER, BLUE, PURPLE, TURQUOISE, GREEN], text_fmt="{:,.0f}"), 340)
    with c2:
        fig = go.Figure(go.Histogram(x=ten["tenure_months"], nbinsx=30, marker_color=PURPLE))
        fig.update_layout(title="Membership duration distribution (months)", bargap=0.06)
        charts.show(fig, 340)
    by_plan = freq.merge(M[["member_id", "membership_type"]], on="member_id")
    fig = px.box(by_plan, x="membership_type", y="visits_per_week", color="membership_type",
                 color_discrete_sequence=PALETTE, points=False)
    fig.update_layout(title="Visit frequency by plan", showlegend=False, xaxis_title="", yaxis_title="Visits / week")
    charts.show(fig, 320)

with tab4:
    k_sel = st.select_slider("Number of segments", options=["Auto", 3, 4, 5, 6, 7], value="Auto",
                             help="Auto picks k using the silhouette score.")
    seg = state.get_segments(None if k_sel == "Auto" else int(k_sel))
    st.caption(f"K-Means on {len(seg.data)} active members · k = {seg.k} · silhouette = {seg.metrics['Silhouette']:.3f} · "
               "features standardised (visits/week, session length, usual hour, weekend share, tenure, monthly value, trend).")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = px.scatter(seg.data, x="pca_x", y="pca_y", color="segment", color_discrete_sequence=PALETTE, opacity=0.8,
                         hover_data={"member_id": True, "visits_per_week": ":.2f", "monthly_value": ":.0f",
                                     "pca_x": False, "pca_y": False})
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
        fig.update_layout(title=f"Segments in 2-D (PCA, {seg.explained_variance:.0%} variance explained)",
                          xaxis_title="PC 1", yaxis_title="PC 2")
        charts.show(fig, 420)
    with c2:
        p = seg.profiles
        charts.show(charts.donut(p.index, p["members"], "Segment sizes"), 420)
    prof = seg.profiles[["members", "share"] + list(SEG_LABELS)].rename(columns=SEG_LABELS)
    st.dataframe(prof.style.format({"share": "{:.0%}", "Visits / week": "{:.2f}", "Session (min)": "{:.0f}",
                                    "Usual hour": "{:.1f}h", "Weekend share": "{:.0%}", "Tenure (months)": "{:.1f}",
                                    "Monthly value": "€{:.0f}", "Attendance trend": "{:.2f}"})
                 .background_gradient(cmap=mpl.CMAP, subset=list(SEG_LABELS.values())), width="stretch")
    st.caption("Segment names are generated from each cluster's most distinctive standardised trait.")
