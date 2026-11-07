import pytest

from analytics import equipment, fitness, insights, kpis, members, occupancy
from models import churn, forecasting, segmentation


def test_monthly_table_consistency(T, monthly):
    pay = T["payments"]
    assert monthly["revenue"].sum() == pytest.approx(pay.loc[pay["status"] == "Completed", "amount"].sum())
    assert (monthly["active_members"] <= len(T["members"])).all()
    assert monthly["renewal_rate"].dropna().between(0, 1).all()
    assert (monthly["profit"] == monthly["revenue"] - monthly["expenses"]).all()


def test_executive_kpis(T, monthly):
    k = kpis.executive_kpis(T, monthly)
    assert k["active_members"]["value"] + k["inactive_members"]["value"] == k["total_members"]["value"]
    assert len(k["revenue"]["spark"]) == 12


def test_filters_restrict_period_and_plan(T):
    F = kpis.apply_filters(T, "2026-01-01", "2026-03-31", ["Premium"])
    assert F["attendance"]["check_in"].between("2026-01-01", "2026-04-01").all()
    assert set(F["members"]["membership_type"]) == {"Premium"}


def test_member_analytics(T, ref):
    coh = members.cohort_retention(T["members"], ref)
    assert (coh["M0"].dropna() <= 100).all()
    f = members.visit_frequency(T["attendance"], T["members"], ref)
    assert (f["visits_per_week"] >= 0).all()


def test_occupancy_peaks_are_planted_hours(T):
    pq = occupancy.peak_quiet(T["attendance"])
    assert pq["peak_hour"] in (7, 12, 17, 18, 19)
    assert occupancy.concurrency_matrix(T["attendance"]).shape == (7, 17)


def test_equipment_flags_planted_units(T, ref):
    s = equipment.equipment_summary(T, ref).set_index("equipment_id")
    assert s.at["EQ-RM-03", "needs_attention"]                 # under-used
    assert s.at["EQ-LC-01", "service_overdue"]                 # skipped maintenance
    assert s.at["EQ-TR-02", "repairs_12m"] >= 4                # failure-prone


def test_fitness_only_consenting(T):
    cm = fitness.consenting_members(T["members"], T["workouts"])
    assert cm["data_consent"].all()
    log = fitness.member_log(T["workouts"], cm["member_id"].iloc[0])
    s = fitness.summary(log)
    assert s["sessions"] > 0 and 0 <= s["consistency"] <= 1


def test_churn_model_quality(T, ref):
    r = churn.train(T, ref)
    assert r.metrics["ROC-AUC"] > 0.72
    assert r.scores["churn_probability"].between(0, 1).all()
    assert set(r.scores["risk_band"].dropna()) <= {"Low", "Medium", "High"}


def test_segmentation(T, ref):
    s = segmentation.segment(T, ref)
    assert 3 <= s.k <= 7
    assert s.data["segment"].nunique() == s.k
    assert -1 <= s.metrics["Silhouette"] <= 1


def test_forecasts(T, monthly):
    r = forecasting.revenue_forecast(monthly, horizon=6)
    assert len(r.forecast) == 6 and (r.forecast["upper"] >= r.forecast["lower"]).all()
    assert r.metrics["MAPE"] < r.baseline_metrics["MAPE"]       # beats seasonal naive
    a = forecasting.attendance_forecast(T["attendance"], horizon_days=14)
    assert len(a.forecast) == 14
    assert a.metrics["R²"] > 0.7 and a.metrics["MAPE"] < 0.12


def test_insights_detect_planted_signals(T, monthly, ref):
    eq = equipment.equipment_summary(T, ref)
    ins = insights.generate_insights(T, monthly, ref, churn_scores=churn.train(T, ref).scores, equipment=eq)
    ids = {i.id for i in ins}
    assert "revenue_drop" in ids
    assert "eq_fail_EQ-TR-02" in ids and "eq_overdue_EQ-LC-01" in ids and "eq_low_EQ-RM-03" in ids
    assert all(i.why and i.action for i in ins)
    assert not any(i.id.startswith("err_") for i in ins)


def test_fitness_functions_for_every_consenting_member(T):
    cm = fitness.consenting_members(T["members"], T["workouts"])
    for mid in cm["member_id"]:
        log = fitness.member_log(T["workouts"], mid)
        s = fitness.summary(log)
        assert s["sessions"] >= 1
        fitness.calendar(log)
        fitness.sessions_per_week(log)
        fitness.weekly_volume(log)
        fitness.strength_progress(log)
        fitness.effort_distribution(log)
        for ex in log["exercise"].unique():
            p = fitness.exercise_progress(log, ex)
            assert not p.empty and p["metric"].notna().any()


def test_equipment_drilldown_every_unit(T, ref):
    s = equipment.equipment_summary(T, ref)
    assert len(s) == len(T["equipment"])
    assert s["health_score"].between(0, 100).all()
    assert s["utilization"].between(0, 1).all()
    for e in s["equipment_id"]:
        assert (T["equipment_usage"]["equipment_id"] == e).any()


def test_monthly_table_single_month(T):
    mo = kpis.monthly_table(kpis.apply_filters(T, "2026-10-01", "2026-10-31"), "2026-10-01", "2026-10-31")
    assert len(mo) == 1
    k = kpis.executive_kpis(T, mo)
    assert k["revenue"]["value"] > 0


def test_plan_filter_allocates_expenses(T):
    full = kpis.apply_filters(T, "2025-11-01", "2026-10-31")
    plans = sorted(T["members"]["membership_type"].unique())
    parts = [kpis.apply_filters(T, "2025-11-01", "2026-10-31", [p])["expenses"]["amount"].sum() for p in plans]
    assert sum(parts) == pytest.approx(full["expenses"]["amount"].sum())   # allocations add up to total
    student = kpis.apply_filters(T, "2025-11-01", "2026-10-31", ["Student"])
    assert student["expenses"]["amount"].sum() < 0.3 * full["expenses"]["amount"].sum()
