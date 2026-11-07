"""Point-in-time member feature engineering (shared by churn model & segmentation).

All features for a snapshot date `as_of` use only data observed *before* `as_of`,
which keeps the churn model free of look-ahead leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLAN_PRICE = {"Student": 25.0, "Basic": 35.0, "Standard": 55.0, "Premium": 85.0}
CYCLE_MONTHS = {"Monthly": 1, "Quarterly": 3, "Annual": 12}

NUMERIC_FEATURES = ["tenure_days", "visits_30", "visits_90", "visit_trend", "recency_days",
                    "avg_duration", "weekend_share", "avg_hour", "monthly_fee",
                    "extra_spend_90", "age", "days_to_renewal", "auto_renew", "visits_per_week"]
CATEGORICAL_FEATURES = ["membership_type", "billing_cycle"]


def active_at(members: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    return (members["join_date"] <= date) & (members["churn_date"].isna() | (members["churn_date"] > date))


def member_features(T: dict[str, pd.DataFrame], as_of: pd.Timestamp,
                    member_ids=None) -> pd.DataFrame:
    members = T["members"]
    if member_ids is None:
        member_ids = members.loc[active_at(members, as_of), "member_id"]
    m = members[members["member_id"].isin(member_ids)].set_index("member_id")
    if m.empty:
        return pd.DataFrame()

    att = T["attendance"]
    att = att[(att["check_in"] < as_of) & (att["check_in"] >= as_of - pd.Timedelta(days=180))
              & att["member_id"].isin(m.index)]
    a90 = att[att["check_in"] >= as_of - pd.Timedelta(days=90)]
    a30 = att[att["check_in"] >= as_of - pd.Timedelta(days=30)]

    f = pd.DataFrame(index=m.index)
    f["tenure_days"] = (as_of - m["join_date"]).dt.days.clip(lower=0)
    f["visits_30"] = a30.groupby("member_id").size().reindex(f.index, fill_value=0)
    f["visits_90"] = a90.groupby("member_id").size().reindex(f.index, fill_value=0)
    # recent 30-day pace relative to the 90-day pace (1 = stable, <1 = slowing down)
    f["visit_trend"] = (f["visits_30"] + 0.5) / (f["visits_90"] / 3 + 0.5)
    last = att.groupby("member_id")["check_in"].max().reindex(f.index)
    f["recency_days"] = ((as_of - last).dt.days).fillna(180).clip(upper=180)
    f["avg_duration"] = a90.groupby("member_id")["duration_min"].mean().reindex(f.index)
    f["weekend_share"] = (a90.assign(w=a90["check_in"].dt.day_of_week >= 5)
                          .groupby("member_id")["w"].mean().reindex(f.index))
    f["avg_hour"] = (a90["check_in"].dt.hour + a90["check_in"].dt.minute / 60).groupby(
        a90["member_id"]).mean().reindex(f.index)
    f["membership_type"] = m["membership_type"]
    f["billing_cycle"] = m["billing_cycle"]
    f["monthly_fee"] = m["membership_type"].map(PLAN_PRICE).fillna(45.0)
    f["auto_renew"] = m["auto_renew"].astype(int)
    f["age"] = m["age"]

    pay = T["payments"]
    pay = pay[(pay["payment_date"] < as_of) & (pay["payment_date"] >= as_of - pd.Timedelta(days=90))
              & (pay["revenue_type"] != "Membership") & (pay["status"] == "Completed")]
    f["extra_spend_90"] = pay.groupby("member_id")["amount"].sum().reindex(f.index, fill_value=0)

    subs = T["subscriptions"]
    cur = subs[(subs["start_date"] <= as_of) & (subs["end_date"] > as_of)
               & subs["member_id"].isin(f.index)]
    nxt = cur.groupby("member_id")["end_date"].min().reindex(f.index)
    f["days_to_renewal"] = (nxt - as_of).dt.days.fillna(30).clip(lower=0)

    f["avg_duration"] = f["avg_duration"].fillna(f["avg_duration"].median())
    f["weekend_share"] = f["weekend_share"].fillna(0)
    f["avg_hour"] = f["avg_hour"].fillna(f["avg_hour"].median())
    f["age"] = f["age"].fillna(f["age"].median())
    f["visits_per_week"] = f["visits_90"] / (np.minimum(f["tenure_days"], 90).clip(lower=7) / 7)
    return f
