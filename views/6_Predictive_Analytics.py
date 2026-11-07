"""Module 6 — Predictive Analytics (Scikit-learn)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from components import charts, layout, mpl
from components.kpi_cards import mini_stat
from components.theme import AMBER, GREEN, MUTED, PALETTE, PINK, PURPLE, RED, TURQUOISE
from models.segmentation import SEG_FEATURES, SEG_LABELS
from utils import state
from utils.formatting import money, pct

layout.page_header("Module 6 · Machine learning", "Predictive Analytics",
                   "Churn prediction, revenue & attendance forecasting and K-Means segmentation — each model is "
                   "evaluated on held-out data and compared with a simple baseline.", predictive=True)
layout.note("<b>Reading the charts:</b> <span style='color:#22D3EE'>solid turquoise = real historical data</span> · "
            "<span style='color:#A78BFA'>dashed purple = model prediction</span> · "
            "<span style='color:#FBBF24'>dotted amber = back-test prediction on held-out history</span>. "
            "Shaded bands are 95% prediction intervals.")
st.write("")


def metric_row(m: dict, base: dict | None = None, fmt=None, lower_better=("MAE", "RMSE", "MAPE")):
    fmt = fmt or {}
    cols = st.columns(len(m))
    for c, (k, v) in zip(cols, m.items()):
        f = fmt.get(k, "{:.3f}")
        sub = ""
        if base and k in base and base[k] is not None and not np.isnan(base[k]):
            better = v < base[k] if k in lower_better else v > base[k]
            sub = f"baseline {f.format(base[k])} {'✓' if better else '✗'}"
        c.markdown(mini_stat(k, f.format(v), sub), unsafe_allow_html=True)


t1, t2, t3, t4 = st.tabs(["Churn prediction", "Revenue forecast", "Attendance forecast", "Member segmentation"])

# ------------------------------------------------------------------ churn
with t1:
    C = state.get_churn()
    st.markdown(f"**Model:** Random Forest classifier (300 trees, balanced classes) · **Target:** member cancels within "
                f"the next 90 days · **Training rows:** {C.n_train:,} member-snapshots from {len(C.snapshots)} historical "
                f"dates · **Test:** {C.n_test:,} members at the most recent labelled snapshot "
                f"({C.snapshots[-1]:%d %b %Y}) · base churn rate {pct(C.positive_rate)} · decision threshold {C.threshold:.2f}")
    metric_row(C.metrics, C.baseline_metrics)
    st.caption("Baseline = logistic regression on the same features. Temporal validation: the model never sees the test period.")
    c1, c2, c3 = st.columns([1.1, 0.9, 1.1])
    with c1:
        fpr, tpr = C.roc
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=PURPLE, width=3), name=f"ROC (AUC {C.metrics['ROC-AUC']:.3f})",
                        fill="tozeroy", fillcolor="rgba(139,92,246,0.12)")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot"), name="Random")
        fig.update_layout(title="ROC curve (test set)", xaxis_title="False positive rate", yaxis_title="True positive rate")
        charts.show(fig, 340)
    with c2:
        f, ax = mpl.fig(4.2, 3.6)
        sns.heatmap(C.confusion, annot=True, fmt="d", cmap=mpl.CMAP, cbar=False, ax=ax, linewidths=2, linecolor=mpl.NAVY_2,
                    xticklabels=["Stays", "Churns"], yticklabels=["Stays", "Churns"], annot_kws={"size": 13, "weight": "bold"})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix (test)")
        st.pyplot(f, clear_figure=True)
    with c3:
        imp = C.importance.head(10)
        charts.show(charts.bar(imp["label"], imp["importance"], "Feature importance", horizontal=True,
                               colors=[PURPLE] * 3 + [TURQUOISE] * 7), 340)

    layout.section("Current churn risk — active members", "Predicted probability of cancelling in the next 90 days")
    sc = C.scores
    c1, c2 = st.columns([1, 1.6])
    with c1:
        band = sc["risk_band"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        charts.show(charts.donut(band.index, band.values, "Members by risk band", colors=[RED, AMBER, GREEN],
                                 center=f"<b>{len(sc)}</b><br>scored"), 320)
    with c2:
        fig = go.Figure(go.Histogram(x=sc["churn_probability"], nbinsx=40, marker_color=PURPLE))
        fig.add_vline(x=C.threshold, line=dict(color=RED, dash="dash"), annotation_text="decision threshold",
                      annotation_font_color=RED)
        fig.update_layout(title="Distribution of predicted churn probability", xaxis_tickformat=".0%", bargap=0.05)
        charts.show(fig, 320)
    risk = st.multiselect("Risk bands", ["High", "Medium", "Low"], default=["High"])
    tbl = sc[sc["risk_band"].isin(risk)][["member_id", "first_name", "last_name", "email", "membership_type", "billing_cycle",
                                          "churn_probability", "risk_band", "visits_30", "recency_days", "days_to_renewal",
                                          "risk_drivers"]]
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "churn_probability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="percent"),
        "risk_band": "Risk", "visits_30": "Visits 30d", "recency_days": "Days since visit",
        "days_to_renewal": "Days to renewal", "risk_drivers": "Main drivers"})
    st.download_button("Export retention list (CSV)", tbl.to_csv(index=False).encode(), "gymmind_churn_risk.csv", "text/csv")

# ---------------------------------------------------------------- revenue
with t2:
    h = st.slider("Forecast horizon (months)", 3, 12, 6)
    R = state.get_revenue_forecast(h)
    st.markdown(f"**Model:** {R.model_name} · trained on {len(R.history)} months · back-tested on the last "
                f"{len(R.test)} months · baseline = same month last year (seasonal naive).")
    metric_row({k: v for k, v in R.metrics.items() if k != "R²"}, R.baseline_metrics,
               fmt={"MAE": "€{:,.0f}", "RMSE": "€{:,.0f}", "MAPE": "{:.1%}"})
    charts.show(charts.forecast_chart(R.history, R.forecast, R.test, "Monthly revenue — actual vs forecast", money=True), 420)
    tbl = R.forecast.copy()
    tbl["date"] = tbl["date"].dt.strftime("%b %Y")
    st.dataframe(tbl.rename(columns={"date": "Month", "predicted": "Predicted revenue", "lower": "Lower 95%",
                                     "upper": "Upper 95%"}).style.format({"Predicted revenue": "€{:,.0f}", "Lower 95%": "€{:,.0f}",
                                                                          "Upper 95%": "€{:,.0f}"}),
                 hide_index=True, width="stretch")
    st.caption("Monthly revenue is lumpy because quarterly and annual plans are paid upfront; the interval reflects this.")

# ------------------------------------------------------------- attendance
with t3:
    A = state.get_attendance_forecast()
    st.markdown(f"**Model:** {A.model_name} · back-tested over the last {len(A.test)} days with a fully recursive "
                f"multi-step forecast · baseline = repeat the last observed week.")
    metric_row(A.metrics, A.baseline_metrics, fmt={"MAE": "{:.1f}", "RMSE": "{:.1f}", "MAPE": "{:.1%}", "R²": "{:.3f}"})
    charts.show(charts.forecast_chart(A.history, A.forecast, A.test, "Daily visits — actual vs forecast",
                                      history_tail=150), 420)
    better = sum(A.metrics[k] < A.baseline_metrics[k] for k in ("MAE", "RMSE"))
    if better < 2:
        st.info("The seasonal-naive baseline is competitive here: attendance follows a very regular weekly rhythm. "
                "The ML model adds value by adapting to seasonality (January/September peaks, summer dip) and to "
                "changes in membership level over longer horizons.")
    from models import forecasting
    hf = forecasting.hourly_forecast(state.get_data()["attendance"], A.forecast.head(7))
    piv = hf.pivot_table(index="date", columns="hour", values="expected_arrivals").fillna(0)
    piv.index = [f"{d:%a %d %b}" for d in piv.index]
    piv.columns = [f"{c:02d}:00" for c in piv.columns]
    charts.show(charts.heatmap(piv, "Predicted check-ins per hour — next 7 days", fmt=".0f",
                               colorscale=[[0, "#141B34"], [0.5, "#5B3FC4"], [1, "#C4B5FD"]]), 330)

# ----------------------------------------------------------- segmentation
with t4:
    S = state.get_segments()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(mini_stat("Segments (k)", str(S.k), "chosen by silhouette"), unsafe_allow_html=True)
    c2.markdown(mini_stat("Silhouette", f"{S.metrics['Silhouette']:.3f}", "higher = better separated"), unsafe_allow_html=True)
    c3.markdown(mini_stat("Davies-Bouldin", f"{S.metrics['Davies-Bouldin']:.2f}", "lower = better"), unsafe_allow_html=True)
    c4.markdown(mini_stat("Calinski-Harabasz", f"{S.metrics['Calinski-Harabasz']:.0f}", "higher = better"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ks = S.k_scores
        fig = go.Figure()
        fig.add_scatter(x=ks["k"], y=ks["silhouette"], name="Silhouette", line=dict(color=TURQUOISE, width=2.6), mode="lines+markers")
        fig.add_scatter(x=ks["k"], y=ks["inertia"], name="Inertia (elbow)", yaxis="y2", line=dict(color=PURPLE, width=2, dash="dot"))
        fig.update_layout(title="Choosing k", xaxis_title="k", yaxis2=dict(overlaying="y", side="right", showgrid=False))
        charts.show(fig, 340)
    with c2:
        prof = S.profiles[SEG_FEATURES]
        z = (prof - S.data[SEG_FEATURES].mean()) / S.data[SEG_FEATURES].std()
        fig = go.Figure()
        for i, (name, row) in enumerate(z.iterrows()):
            fig.add_scatterpolar(r=list(row.values) + [row.values[0]], theta=[SEG_LABELS[c] for c in SEG_FEATURES] + [SEG_LABELS[SEG_FEATURES[0]]],
                                 name=name, line=dict(color=PALETTE[i % len(PALETTE)], width=2), fill="toself", opacity=0.55)
        fig.update_layout(title="Segment profiles (standardised)", polar=dict(bgcolor="rgba(0,0,0,0)",
                          radialaxis=dict(gridcolor="rgba(148,163,209,0.15)", tickfont=dict(color=MUTED, size=9)),
                          angularaxis=dict(gridcolor="rgba(148,163,209,0.15)")))
        charts.show(fig, 340)
    c1, c2 = st.columns([1, 1])
    with c1:
        f, ax = mpl.fig(6, 4.6)
        corr = S.data[SEG_FEATURES + ["churn_probability"] if "churn_probability" in S.data else SEG_FEATURES].corr()
        corr = corr.rename(index=SEG_LABELS, columns=SEG_LABELS)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=mpl.CMAP_DIV, vmin=-1, vmax=1, ax=ax, linewidths=1,
                    linecolor=mpl.NAVY_2, annot_kws={"size": 7}, cbar_kws={"shrink": .8})
        ax.set_title("Feature correlation matrix")
        st.pyplot(f, clear_figure=True)
    with c2:
        churn_scores = state.get_churn().scores.set_index("member_id")["churn_probability"]
        d = S.data.assign(churn=S.data["member_id"].map(churn_scores))
        g = d.groupby("segment")["churn"].mean().sort_values(ascending=False)
        charts.show(charts.bar(g.index, g.values * 100, "Avg predicted churn risk by segment (%)",
                               colors=[RED if v == g.max() else PURPLE for v in g.values], text_fmt="{:.1f}%"), 400)
