"""Equipment intelligence: utilisation, maintenance, health score and attention flags."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.formatting import CLOSE_HOUR, OPEN_HOUR

OPEN_MIN = (CLOSE_HOUR - OPEN_HOUR) * 60


def equipment_summary(T: dict[str, pd.DataFrame], ref: pd.Timestamp) -> pd.DataFrame:
    eq = T["equipment"].set_index("equipment_id")
    use, mt = T["equipment_usage"], T["maintenance"]
    n_days = max(use["date"].nunique(), 1)

    g = use.groupby("equipment_id").agg(usage_minutes=("usage_minutes", "sum"),
                                        sessions=("sessions", "sum"))
    recent = use[use["date"] > ref - pd.Timedelta(days=30)]
    r = recent.groupby("equipment_id")["usage_minutes"].sum() / (max(recent["date"].nunique(), 1) * OPEN_MIN)

    s = eq.join(g, how="left").fillna({"usage_minutes": 0, "sessions": 0})
    s["usage_hours"] = s["usage_minutes"] / 60
    s["utilization"] = s["usage_minutes"] / (n_days * OPEN_MIN)
    s["utilization_30d"] = r.reindex(s.index).fillna(0)
    s["daily_sessions"] = s["sessions"] / n_days

    last12 = mt[mt["date"] > ref - pd.Timedelta(days=365)]
    s["repairs_12m"] = last12[last12["maintenance_type"] == "Repair"].groupby("equipment_id").size() \
        .reindex(s.index, fill_value=0)
    s["maintenance_events"] = mt.groupby("equipment_id").size().reindex(s.index, fill_value=0)
    s["maintenance_cost"] = mt.groupby("equipment_id")["cost"].sum().reindex(s.index, fill_value=0)
    s["downtime_hours"] = mt.groupby("equipment_id")["downtime_hours"].sum().reindex(s.index, fill_value=0)
    prev = mt[mt["maintenance_type"] == "Preventive"]
    s["last_service"] = prev.groupby("equipment_id")["date"].max().reindex(s.index)
    s["days_since_service"] = (ref - s["last_service"]).dt.days
    s["service_overdue"] = s["days_since_service"].fillna(9999) > s["service_interval_days"] * 1.15
    s["cost_per_usage_hour"] = s["maintenance_cost"] / s["usage_hours"].replace(0, np.nan)
    s["age_years"] = ((ref - s["purchase_date"]).dt.days / 365.25).round(1)
    s["availability"] = 1 - s["downtime_hours"] / (n_days * (CLOSE_HOUR - OPEN_HOUR))

    # utilisation relative to peers of the same type (1 = typical)
    s["peer_util_ratio"] = s["utilization"] / s.groupby("equipment_type")["utilization"].transform("median")

    # health score (0-100): penalise repairs, overdue service, downtime, age
    score = 100 - np.minimum(s["repairs_12m"] * 9, 45) - np.where(s["service_overdue"], 20, 0) \
        - np.minimum((1 - s["availability"]) * 400, 20) - np.minimum(s["age_years"] * 1.5, 12)
    s["health_score"] = np.clip(score, 0, 100).round(0)
    s["flags"] = s.apply(_flags, axis=1)
    s["needs_attention"] = s["flags"].str.len() > 0
    s["status"] = np.select([s["health_score"] < 55, s["needs_attention"]],
                            ["Critical", "Watch"], "Healthy")
    return s.reset_index()


def _flags(r) -> list[str]:
    f = []
    if r["service_overdue"]:
        f.append(f"Preventive service overdue ({int(r['days_since_service'])} days, interval "
                 f"{int(r['service_interval_days'])})")
    if r["repairs_12m"] >= 4:
        f.append(f"{int(r['repairs_12m'])} repairs in the last 12 months")
    if r["peer_util_ratio"] < 0.45:
        f.append(f"Utilisation {r['peer_util_ratio']:.0%} of similar {r['equipment_type']} units")
    if r["utilization_30d"] > 0.6:
        f.append(f"Heavy load: {r['utilization_30d']:.0%} utilisation in the last 30 days")
    return f


def usage_by_type(T: dict[str, pd.DataFrame]) -> pd.DataFrame:
    u = T["equipment_usage"].merge(T["equipment"][["equipment_id", "equipment_type", "category"]],
                                   on="equipment_id")
    g = u.groupby(["category", "equipment_type"]).agg(usage_hours=("usage_minutes", lambda x: x.sum() / 60),
                                                      sessions=("sessions", "sum")).reset_index()
    return g.sort_values("usage_hours", ascending=False)


def monthly_usage(T: dict[str, pd.DataFrame]) -> pd.DataFrame:
    u = T["equipment_usage"].merge(T["equipment"][["equipment_id", "category"]], on="equipment_id")
    g = u.groupby([u["date"].dt.to_period("M"), "category"])["usage_minutes"].sum().unstack(fill_value=0) / 60
    g.index = g.index.to_timestamp()
    return g


def maintenance_monthly(mt: pd.DataFrame) -> pd.DataFrame:
    g = mt.groupby([mt["date"].dt.to_period("M"), "maintenance_type"])["cost"].sum().unstack(fill_value=0)
    g.index = g.index.to_timestamp()
    return g
