"""Smart occupancy: arrivals, concurrent occupancy, peak / quiet hours, trends."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.formatting import CLOSE_HOUR, GYM_CAPACITY, OPEN_HOUR, WEEKDAYS

HOURS = list(range(OPEN_HOUR, CLOSE_HOUR))


def _n_days(att: pd.DataFrame) -> pd.Series:
    """Number of calendar days per weekday in the attendance window."""
    if att.empty:
        return pd.Series(1, index=range(7))
    days = pd.date_range(att["check_in"].min().normalize(), att["check_in"].max().normalize())
    return pd.Series(days.day_of_week).value_counts().reindex(range(7), fill_value=1)


def concurrency_matrix(att: pd.DataFrame) -> pd.DataFrame:
    """Average number of people present, weekday x hour (overlap of check-in/check-out)."""
    if att.empty:
        return pd.DataFrame(0.0, index=WEEKDAYS, columns=HOURS)
    start = (att["check_in"].dt.hour * 60 + att["check_in"].dt.minute).to_numpy()
    end_ts = att["check_out"].fillna(att["check_in"] + pd.to_timedelta(att["duration_min"].fillna(60), unit="m"))
    end = np.minimum((end_ts - att["check_in"].dt.normalize()).dt.total_seconds().to_numpy() / 60,
                     CLOSE_HOUR * 60)
    dow = att["check_in"].dt.day_of_week.to_numpy()
    out = np.zeros((7, len(HOURS)))
    for j, h in enumerate(HOURS):
        overlap = np.clip(np.minimum(end, (h + 1) * 60) - np.maximum(start, h * 60), 0, 60) / 60
        out[:, j] = np.bincount(dow, weights=overlap, minlength=7)
    out = out / _n_days(att).to_numpy()[:, None]
    return pd.DataFrame(out, index=WEEKDAYS, columns=HOURS)


def arrivals_matrix(att: pd.DataFrame) -> pd.DataFrame:
    """Average check-ins per weekday x hour."""
    if att.empty:
        return pd.DataFrame(0.0, index=WEEKDAYS, columns=HOURS)
    g = att.groupby([att["check_in"].dt.day_of_week, att["check_in"].dt.hour]).size().unstack(fill_value=0)
    g = g.reindex(index=range(7), columns=HOURS, fill_value=0)
    g = g.div(_n_days(att), axis=0)
    g.index = WEEKDAYS
    return g


def hourly_profile(att: pd.DataFrame) -> pd.DataFrame:
    conc = concurrency_matrix(att)
    arr = arrivals_matrix(att)
    w = _n_days(att).to_numpy()
    prof = pd.DataFrame({
        "hour": HOURS,
        "avg_arrivals": (arr.to_numpy() * w[:, None]).sum(0) / w.sum(),
        "avg_present": (conc.to_numpy() * w[:, None]).sum(0) / w.sum(),
    })
    prof["occupancy_pct"] = prof["avg_present"] / GYM_CAPACITY
    return prof


def weekday_profile(att: pd.DataFrame) -> pd.DataFrame:
    if att.empty:
        return pd.DataFrame({"weekday": WEEKDAYS, "avg_visits": 0.0})
    daily = att.groupby(att["check_in"].dt.normalize()).size()
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0)
    g = full.groupby(full.index.day_of_week).mean().reindex(range(7), fill_value=0)
    return pd.DataFrame({"weekday": WEEKDAYS, "avg_visits": g.to_numpy()})


def peak_quiet(att: pd.DataFrame, top: int = 3) -> dict:
    conc = concurrency_matrix(att)
    s = conc.stack()
    s.index.names = ["weekday", "hour"]
    s = s.sort_values()
    peaks = s.tail(top)[::-1]
    quiet = s[s > 0].head(top)
    prof = hourly_profile(att)
    return {
        "peak_slots": [(d, h, v) for (d, h), v in peaks.items()],
        "quiet_slots": [(d, h, v) for (d, h), v in quiet.items()],
        "peak_hour": int(prof.loc[prof["avg_present"].idxmax(), "hour"]),
        "quiet_hour": int(prof.loc[prof["avg_present"].idxmin(), "hour"]),
        "max_occupancy": float(s.max() / GYM_CAPACITY),
    }


def daily_trend(att: pd.DataFrame) -> pd.DataFrame:
    if att.empty:
        return pd.DataFrame(columns=["date", "visits", "rolling_7", "rolling_28"])
    d = att.groupby(att["check_in"].dt.normalize()).size()
    d = d.reindex(pd.date_range(d.index.min(), d.index.max()), fill_value=0)
    out = pd.DataFrame({"date": d.index, "visits": d.to_numpy()})
    out["rolling_7"] = out["visits"].rolling(7, min_periods=1).mean()
    out["rolling_28"] = out["visits"].rolling(28, min_periods=1).mean()
    return out


def duration_by_hour(att: pd.DataFrame) -> pd.DataFrame:
    return att.groupby(att["check_in"].dt.hour)["duration_min"].mean().reindex(HOURS).reset_index() \
        .rename(columns={"check_in": "hour", "duration_min": "avg_duration"})
