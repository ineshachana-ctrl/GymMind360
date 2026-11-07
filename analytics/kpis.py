"""Executive KPIs and the monthly business backbone table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.features import active_at


def completed_payments(payments: pd.DataFrame) -> pd.DataFrame:
    return payments[payments["status"] == "Completed"]


def monthly_table(T: dict[str, pd.DataFrame], start=None, end=None) -> pd.DataFrame:
    """One row per month with members, revenue, expenses, profit, attendance, renewals."""
    members, pay = T["members"], completed_payments(T["payments"])
    att, subs, exp = T["attendance"], T["subscriptions"], T["expenses"]
    lo = pd.Timestamp(start) if start is not None else pay["payment_date"].min()
    hi = pd.Timestamp(end) if end is not None else pay["payment_date"].max()
    months = pd.period_range(lo.to_period("M"), hi.to_period("M"), freq="M")

    rev = pay.groupby(pay["payment_date"].dt.to_period("M"))["amount"].sum()
    mrev = pay[pay["revenue_type"] == "Membership"].groupby(
        pay["payment_date"].dt.to_period("M"))["amount"].sum()
    ex = exp.groupby(exp["month"].dt.to_period("M"))["amount"].sum()
    visits = att.groupby(att["check_in"].dt.to_period("M")).size()
    new = members.groupby(members["join_date"].dt.to_period("M")).size()
    churned = members.groupby(members["churn_date"].dt.to_period("M")).size()
    ended = subs[subs["status"].isin(["Renewed", "Cancelled"])]
    ended = ended.groupby([ended["end_date"].dt.to_period("M"), "status"]).size().unstack(fill_value=0)

    rows = []
    for p in months:
        m_end = p.to_timestamp(how="end").normalize()
        m_start = p.to_timestamp(how="start")
        active_end = int(active_at(members, m_end).sum())
        active_start = int(active_at(members, m_start - pd.Timedelta(days=1)).sum())
        renewed = int(ended.at[p, "Renewed"]) if p in ended.index and "Renewed" in ended else 0
        cancelled = int(ended.at[p, "Cancelled"]) if p in ended.index and "Cancelled" in ended else 0
        r = float(rev.get(p, 0.0))
        e = float(ex.get(p, 0.0))
        v = int(visits.get(p, 0))
        rows.append({
            "month": p.to_timestamp(), "period": p, "active_members": active_end,
            "active_start": active_start, "new_members": int(new.get(p, 0)),
            "churned": int(churned.get(p, 0)), "revenue": r,
            "membership_revenue": float(mrev.get(p, 0.0)), "expenses": e, "profit": r - e,
            "visits": v, "visits_per_member": v / active_end if active_end else np.nan,
            "renewed": renewed, "cancelled": cancelled,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["renewal_rate"] = df["renewed"] / (df["renewed"] + df["cancelled"]).replace(0, np.nan)
    df["churn_rate"] = df["churned"] / df["active_start"].replace(0, np.nan)
    df["revenue_growth"] = df["revenue"].pct_change()
    df["member_growth"] = df["active_members"].pct_change()
    df["profit_margin"] = df["profit"] / df["revenue"].replace(0, np.nan)
    df["arpu"] = df["revenue"] / df["active_members"].replace(0, np.nan)
    days = df["period"].map(lambda p: p.days_in_month)
    df["avg_daily_visits"] = df["visits"] / days
    return df


def executive_kpis(T: dict[str, pd.DataFrame], monthly: pd.DataFrame) -> dict[str, dict]:
    """Current vs previous month KPIs for the dashboard cards."""
    members = T["members"]
    if len(monthly) < 2:
        cur = prev = monthly.iloc[-1]
    else:
        cur, prev = monthly.iloc[-1], monthly.iloc[-2]
    total = len(members[members["join_date"] <= cur["period"].to_timestamp(how="end")])
    total_prev = len(members[members["join_date"] <= prev["period"].to_timestamp(how="end")])

    def k(c, p, spark):
        return {"value": c, "prev": p, "spark": list(monthly[spark].fillna(0).tail(12))}

    return {
        "total_members": k(total, total_prev, "new_members"),
        "active_members": k(cur["active_members"], prev["active_members"], "active_members"),
        "inactive_members": k(total - cur["active_members"], total_prev - prev["active_members"],
                              "churned"),
        "revenue": k(cur["revenue"], prev["revenue"], "revenue"),
        "renewal_rate": k(cur["renewal_rate"], prev["renewal_rate"], "renewal_rate"),
        "avg_daily_visits": k(cur["avg_daily_visits"], prev["avg_daily_visits"], "avg_daily_visits"),
        "member_growth": k(cur["member_growth"], prev["member_growth"], "member_growth"),
        "expenses": k(cur["expenses"], prev["expenses"], "expenses"),
        "profit": k(cur["profit"], prev["profit"], "profit"),
        "churn_rate": k(cur["churn_rate"], prev["churn_rate"], "churn_rate"),
        "arpu": k(cur["arpu"], prev["arpu"], "arpu"),
        "new_members": k(cur["new_members"], prev["new_members"], "new_members"),
    }


def apply_filters(T: dict[str, pd.DataFrame], start, end, plans=None) -> dict[str, pd.DataFrame]:
    """Filter tables by date window and membership types (members are kept regardless of date
    so that point-in-time active counts remain correct)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    out = dict(T)
    members = T["members"]
    if plans:
        members = members[members["membership_type"].isin(plans)]
    ids = set(members["member_id"])
    out["members"] = members

    def by(df, col, member=True):
        mask = (df[col] >= start) & (df[col] < end)
        if member and "member_id" in df.columns and plans:
            mask &= df["member_id"].isin(ids)
        return df[mask]

    out["attendance"] = by(T["attendance"], "check_in")
    out["payments"] = by(T["payments"], "payment_date")
    out["workouts"] = by(T["workouts"], "workout_date")
    out["equipment_usage"] = by(T["equipment_usage"], "date", member=False)
    out["maintenance"] = by(T["maintenance"], "date", member=False)
    out["expenses"] = by(T["expenses"], "month", member=False)
    if plans:
        out["expenses"] = allocate_expenses(out["expenses"], T["members"], members)
    subs = T["subscriptions"]
    smask = (subs["end_date"] >= start) & (subs["start_date"] < end)
    if plans:
        smask &= subs["member_id"].isin(ids)
    out["subscriptions"] = subs[smask]
    return out


def allocate_expenses(expenses: pd.DataFrame, all_members: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Allocate gym-wide operating costs to a subset of membership plans, pro rata to the
    selected plans' share of active members at the end of each month (cost-accounting allocation).
    Without this, plan-filtered views would compare one plan's revenue with the whole gym's costs."""
    ex = expenses.copy()
    if ex.empty:
        return ex
    factors = {}
    for m in ex["month"].unique():
        end = pd.Timestamp(m) + pd.offsets.MonthEnd(0)
        total = int(active_at(all_members, end).sum())
        factors[m] = active_at(selected, end).sum() / total if total else 0.0
    ex["amount"] = ex["amount"] * ex["month"].map(factors)
    return ex
