"""Member intelligence: registrations, tenure, retention cohorts, churn, frequency."""
from __future__ import annotations

import numpy as np
import pandas as pd


def registrations(members: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    m = members
    if start is not None:
        m = m[(m["join_date"] >= pd.Timestamp(start)) & (m["join_date"] <= pd.Timestamp(end))]
    g = m.groupby([m["join_date"].dt.to_period("M"), "membership_type"]).size().unstack(fill_value=0)
    g.index = g.index.to_timestamp()
    return g


def tenure(members: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    end = members["churn_date"].fillna(ref)
    out = members[["member_id", "membership_type", "billing_cycle", "status"]].copy()
    out["tenure_months"] = ((end - members["join_date"]).dt.days / 30.44).round(1)
    return out


def cohort_retention(members: pd.DataFrame, ref: pd.Timestamp, start=None,
                     max_months: int = 12) -> pd.DataFrame:
    """Rows = join cohort (month), columns = months since joining, values = % still active."""
    m = members.copy()
    if start is not None:
        m = m[m["join_date"] >= pd.Timestamp(start)]
    m["cohort"] = m["join_date"].dt.to_period("M")
    ref_p = ref.to_period("M")
    rows = {}
    for c, grp in m.groupby("cohort"):
        size = len(grp)
        vals = []
        for k in range(max_months + 1):
            p = c + k
            if p > ref_p:
                vals.append(np.nan)
                continue
            cutoff = p.to_timestamp(how="end")
            alive = (grp["churn_date"].isna() | (grp["churn_date"] > cutoff)).sum()
            vals.append(alive / size * 100)
        rows[str(c)] = vals
    df = pd.DataFrame.from_dict(rows, orient="index", columns=[f"M{k}" for k in range(max_months + 1)])
    return df


def churn_by(members: pd.DataFrame, col: str) -> pd.DataFrame:
    g = members.groupby(col).agg(members=("member_id", "count"),
                                 churned=("churn_date", lambda s: s.notna().sum()))
    g["churn_rate"] = g["churned"] / g["members"]
    return g.reset_index()


def visit_frequency(attendance: pd.DataFrame, members: pd.DataFrame, ref: pd.Timestamp,
                    days: int = 90) -> pd.DataFrame:
    """Visits per week over the last `days` days for currently active members."""
    active = members[members["churn_date"].isna() | (members["churn_date"] > ref)]
    a = attendance[(attendance["check_in"] > ref - pd.Timedelta(days=days))
                   & attendance["member_id"].isin(active["member_id"])]
    v = a.groupby("member_id").size().reindex(active["member_id"], fill_value=0)
    exposure = np.minimum((ref - active.set_index("member_id")["join_date"]).dt.days, days).clip(lower=7)
    vpw = (v / (exposure / 7)).rename("visits_per_week")
    bins = [-0.01, 0.5, 1, 2, 3, 4, 100]
    labels = ["<0.5", "0.5–1", "1–2", "2–3", "3–4", "4+"]
    out = vpw.reset_index()
    out["band"] = pd.cut(out["visits_per_week"], bins=bins, labels=labels)
    return out


def demographics(members: pd.DataFrame) -> pd.DataFrame:
    bins = [17, 24, 34, 44, 54, 100]
    labels = ["18–24", "25–34", "35–44", "45–54", "55+"]
    out = members.copy()
    out["age_band"] = pd.cut(out["age"], bins=bins, labels=labels)
    return out
