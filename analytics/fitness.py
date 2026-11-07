"""Member fitness analytics (voluntary, non-medical training data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def consenting_members(members: pd.DataFrame, workouts: pd.DataFrame) -> pd.DataFrame:
    ids = set(workouts["member_id"].unique())
    m = members[members["data_consent"] & members["member_id"].isin(ids)].copy()
    m["label"] = m["member_id"] + " · " + m["first_name"] + " " + m["last_name"].str[0] + "."
    return m.sort_values("member_id")


def member_log(workouts: pd.DataFrame, member_id: str) -> pd.DataFrame:
    w = workouts[workouts["member_id"] == member_id].copy()
    w["date"] = w["workout_date"].dt.normalize()
    # Epley estimated one-rep max
    w["est_1rm"] = w["weight_kg"] * (1 + w["reps"] / 30)
    w["volume"] = w["sets"] * w["reps"] * w["weight_kg"]
    w["pace_kmh"] = w["distance_km"] / (w["duration_min"] / 60)
    return w


def summary(log: pd.DataFrame) -> dict:
    if log.empty:
        return {}
    sessions = log["visit_id"].nunique()
    days = (log["date"].max() - log["date"].min()).days + 1
    weeks = max(days / 7, 1)
    weekly = sessions_per_week(log)
    last12 = weekly.tail(12)
    strength = log[log["exercise_type"] == "Strength"]
    first_last = strength_progress(log)
    avg_gain = first_last["change_pct"].mean() if not first_last.empty else np.nan
    return {
        "sessions": sessions,
        "per_week": sessions / weeks,
        "consistency": float((last12["sessions"] >= 2).mean()) if len(last12) else np.nan,
        "streak_weeks": _streak(weekly),
        "total_volume": float(strength["volume"].sum()),
        "avg_strength_gain": avg_gain,
        "favourite": log["exercise"].value_counts().idxmax(),
        "first": log["date"].min(), "last": log["date"].max(),
    }


def _streak(weekly: pd.DataFrame) -> int:
    n = 0
    for v in weekly["sessions"][::-1]:
        if v >= 1:
            n += 1
        else:
            break
    return n


def sessions_per_week(log: pd.DataFrame) -> pd.DataFrame:
    s = log.groupby("date")["visit_id"].nunique()
    w = s.resample("W-MON", label="left", closed="left").sum()
    return w.rename("sessions").reset_index().rename(columns={"date": "week"})


def calendar(log: pd.DataFrame) -> pd.DataFrame:
    s = log.groupby("date")["visit_id"].nunique()
    full = s.reindex(pd.date_range(s.index.min() - pd.Timedelta(days=s.index.min().day_of_week),
                                   s.index.max()), fill_value=0)
    df = pd.DataFrame({"date": full.index, "sessions": full.to_numpy()})
    df["week"] = df["date"] - pd.to_timedelta(df["date"].dt.day_of_week, unit="D")
    df["weekday"] = df["date"].dt.day_of_week
    return df


def exercise_progress(log: pd.DataFrame, exercise: str) -> pd.DataFrame:
    e = log[log["exercise"] == exercise]
    if e.empty:
        return e
    if e["exercise_type"].iloc[0] == "Strength":
        g = e.groupby("date").agg(metric=("est_1rm", "max"), weight=("weight_kg", "max"),
                                  volume=("volume", "sum")).reset_index()
    else:
        g = e.groupby("date").agg(metric=("pace_kmh", "mean"), distance=("distance_km", "sum"),
                                  duration=("duration_min", "sum")).reset_index()
    g["trend"] = g["metric"].rolling(5, min_periods=1).mean()
    return g


def strength_progress(log: pd.DataFrame) -> pd.DataFrame:
    s = log[log["exercise_type"] == "Strength"].dropna(subset=["est_1rm"])
    rows = []
    for ex, g in s.groupby("exercise"):
        g = g.sort_values("date")
        if g["date"].nunique() < 4:
            continue
        n = max(len(g) // 5, 1)
        first, last = g["est_1rm"].head(n).mean(), g["est_1rm"].tail(n).mean()
        rows.append({"exercise": ex, "start_1rm": first, "current_1rm": last,
                     "best_1rm": g["est_1rm"].max(), "change_pct": (last - first) / first})
    return pd.DataFrame(rows).sort_values("change_pct", ascending=False) if rows else pd.DataFrame(
        columns=["exercise", "start_1rm", "current_1rm", "best_1rm", "change_pct"])


def weekly_volume(log: pd.DataFrame) -> pd.DataFrame:
    s = log[log["exercise_type"] == "Strength"].groupby("date")["volume"].sum()
    if s.empty:
        return pd.DataFrame(columns=["week", "volume"])
    w = s.resample("W-MON", label="left", closed="left").sum()
    return w.rename("volume").reset_index().rename(columns={"date": "week"})


def effort_distribution(log: pd.DataFrame) -> pd.DataFrame:
    return log.groupby("rpe").size().rename("sets").reset_index()
