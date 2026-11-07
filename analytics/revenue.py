"""Revenue analytics."""
from __future__ import annotations

import pandas as pd

from analytics.kpis import completed_payments


def _monthly(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby([df["payment_date"].dt.to_period("M"), by])["amount"].sum().unstack(fill_value=0)
    g.index = g.index.to_timestamp()
    return g


def by_revenue_type(payments: pd.DataFrame) -> pd.DataFrame:
    return _monthly(completed_payments(payments), "revenue_type")


def by_membership_type(payments: pd.DataFrame) -> pd.DataFrame:
    p = completed_payments(payments)
    return p.groupby("membership_type")["amount"].agg(["sum", "count", "mean"]).reset_index() \
        .rename(columns={"sum": "revenue", "count": "transactions", "mean": "avg_ticket"}) \
        .sort_values("revenue", ascending=False)


def by_payment_method(payments: pd.DataFrame) -> pd.DataFrame:
    p = completed_payments(payments)
    return p.groupby("payment_method")["amount"].agg(["sum", "count"]).reset_index() \
        .rename(columns={"sum": "revenue", "count": "transactions"}).sort_values("revenue", ascending=False)


def by_segment(payments: pd.DataFrame, segments: pd.Series) -> pd.DataFrame:
    """`segments`: Series member_id -> segment name."""
    p = completed_payments(payments).copy()
    p["segment"] = p["member_id"].map(segments).fillna("Former / Unsegmented")
    g = p.groupby("segment").agg(revenue=("amount", "sum"), members=("member_id", "nunique")).reset_index()
    g["revenue_per_member"] = g["revenue"] / g["members"]
    return g.sort_values("revenue", ascending=False)


def payment_health(payments: pd.DataFrame) -> pd.DataFrame:
    g = payments.groupby([payments["payment_date"].dt.to_period("M"), "status"]).size().unstack(fill_value=0)
    g.index = g.index.to_timestamp()
    tot = g.sum(axis=1)
    g["failure_rate"] = g.get("Failed", 0) / tot
    return g


def expense_breakdown(expenses: pd.DataFrame) -> pd.DataFrame:
    g = expenses.groupby([expenses["month"].dt.to_period("M"), "category"])["amount"].sum().unstack(fill_value=0)
    g.index = g.index.to_timestamp()
    return g
