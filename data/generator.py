"""
Synthetic data generator for GymMind 360.

Generates 24 months of realistic, reproducible gym data with *planted signals*
so the analytics, ML models and recommendation engine have real patterns to find:

* Morning (06-09h) and evening (17-20h) attendance peaks, weekend dip.
* January / September seasonality in registrations and attendance.
* Churn linked to low engagement, monthly billing and disabled auto-renew;
  members visibly reduce attendance during the ~8 weeks before cancelling.
* A revenue dip in the most recent month (fewer sign-ups, more cancellations).
* One under-used rowing machine, one failure-prone treadmill and one machine
  with overdue preventive maintenance.
* A small share of dirty data (duplicates, missing values, invalid amounts)
  so the cleaning pipeline has visible work to do.

Run:  python -m data.generator
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def data_dir() -> Path:
    """CSV folder (override with GYMMIND_DATA_DIR, used by tests)."""
    return Path(os.environ.get("GYMMIND_DATA_DIR") or DATA_DIR)
SEED = 42
START = pd.Timestamp("2024-11-01")
END = pd.Timestamp("2026-10-31")
OPEN_HOUR, CLOSE_HOUR = 6, 23

PLANS = {"Student": 25.0, "Basic": 35.0, "Standard": 55.0, "Premium": 85.0}
PLAN_MIX = [0.15, 0.30, 0.35, 0.20]
CYCLES = {"Monthly": (1, 1.00), "Quarterly": (3, 0.95), "Annual": (12, 0.85)}
CYCLE_MIX = [0.60, 0.22, 0.18]
BASE_CHURN = {"Monthly": 0.07, "Quarterly": 0.14, "Annual": 0.24}
METHODS = ["Credit Card", "Direct Debit", "Cash", "Mobile Pay"]
METHOD_P = [0.46, 0.28, 0.10, 0.16]

JOIN_SEASON = {1: 1.8, 2: 1.3, 3: 1.1, 4: 1.0, 5: 1.05, 6: 0.8, 7: 0.7,
               8: 0.8, 9: 1.5, 10: 1.2, 11: 0.9, 12: 0.6}
VISIT_SEASON = {1: 1.25, 2: 1.15, 3: 1.05, 4: 1.0, 5: 1.0, 6: 0.9, 7: 0.8,
                8: 0.78, 9: 1.1, 10: 1.05, 11: 1.0, 12: 0.85}
WEEKDAY_W = np.array([1.25, 1.15, 1.15, 1.10, 0.95, 0.85, 0.55])  # Mon..Sun

FIRST = ["Adam", "Sara", "Yanis", "Lina", "Omar", "Ines", "Lucas", "Emma", "Karim",
         "Nour", "Hugo", "Chloe", "Mehdi", "Lea", "Rayan", "Jade", "Amine", "Manon",
         "Noah", "Yasmine", "Leo", "Camille", "Ilyes", "Sofia", "Nathan", "Maya",
         "Samir", "Alice", "Elias", "Rania", "Louis", "Zoe", "Bilal", "Laura",
         "Thomas", "Hana", "Gabriel", "Salma", "Arthur", "Julia", "Youssef", "Clara"]
LAST = ["Martin", "Benali", "Dubois", "Haddad", "Bernard", "Mansouri", "Petit",
        "Cherif", "Moreau", "Rahmani", "Laurent", "Bouzid", "Simon", "Kaci",
        "Michel", "Brahimi", "Garcia", "Saidi", "Roux", "Ait Ali", "Fournier",
        "Meziane", "Girard", "Amrani", "Bonnet", "Ziani", "Lambert", "Hamdi"]

# name, category, zone, units, popularity, purchase_cost, service_interval_days
EQUIPMENT = [
    ("Treadmill", "Cardio", "Cardio Zone", 6, 1.6, 6500, 90),
    ("Spin Bike", "Cardio", "Cycling Studio", 5, 1.1, 1800, 90),
    ("Rowing Machine", "Cardio", "Cardio Zone", 3, 0.6, 1400, 120),
    ("Elliptical", "Cardio", "Cardio Zone", 3, 0.8, 3200, 90),
    ("Stair Climber", "Cardio", "Cardio Zone", 2, 0.7, 4800, 90),
    ("Squat Rack", "Strength", "Free Weights", 3, 1.4, 1600, 180),
    ("Bench Press", "Strength", "Free Weights", 3, 1.3, 900, 180),
    ("Dumbbell Station", "Strength", "Free Weights", 2, 1.5, 3500, 180),
    ("Cable Crossover", "Strength", "Machines", 2, 1.0, 4200, 120),
    ("Leg Press", "Strength", "Machines", 2, 0.9, 3800, 120),
    ("Lat Pulldown", "Strength", "Machines", 2, 0.9, 2600, 120),
    ("Smith Machine", "Strength", "Machines", 1, 0.5, 3900, 120),
    ("Chest Press Machine", "Strength", "Machines", 1, 0.6, 2900, 120),
    ("Leg Curl Machine", "Strength", "Machines", 1, 0.45, 2700, 120),
    ("Kettlebell Set", "Functional", "Functional Area", 1, 0.7, 900, 365),
    ("Battle Ropes", "Functional", "Functional Area", 1, 0.3, 300, 365),
]
ABBR = {"Treadmill": "TR", "Spin Bike": "SB", "Rowing Machine": "RM", "Elliptical": "EL",
        "Stair Climber": "SC", "Squat Rack": "SQ", "Bench Press": "BP",
        "Dumbbell Station": "DB", "Cable Crossover": "CC", "Leg Press": "LP",
        "Lat Pulldown": "LT", "Smith Machine": "SM", "Chest Press Machine": "CP",
        "Leg Curl Machine": "LC", "Kettlebell Set": "KB", "Battle Ropes": "BR"}
UNDERUSED_UNIT = "EQ-RM-03"     # planted: very low utilisation
FAILURE_UNIT = "EQ-TR-02"       # planted: frequent breakdowns
OVERDUE_UNIT = "EQ-LC-01"       # planted: preventive maintenance skipped

# exercise -> (equipment type, base load kg or None for cardio)
STRENGTH_EX = {"Back Squat": ("Squat Rack", 60), "Bench Press": ("Bench Press", 50),
               "Deadlift": ("Squat Rack", 80), "Dumbbell Shoulder Press": ("Dumbbell Station", 14),
               "Lat Pulldown": ("Lat Pulldown", 45), "Leg Press": ("Leg Press", 110),
               "Cable Row": ("Cable Crossover", 40), "Dumbbell Curl": ("Dumbbell Station", 10)}
CARDIO_EX = {"Treadmill Run": ("Treadmill", 0.150), "Indoor Cycling": ("Spin Bike", 0.420),
             "Rowing": ("Rowing Machine", 0.200), "Elliptical Session": ("Elliptical", 0.180)}


def _add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=n)


# --------------------------------------------------------------------------- members
def _make_members(rng: np.random.Generator) -> pd.DataFrame:
    joins: list[pd.Timestamp] = []
    # existing member base, joined before the analysis window
    pre_days = (START - pd.Timestamp("2022-06-01")).days
    for d in rng.integers(0, pre_days, 380):
        joins.append(pd.Timestamp("2022-06-01") + pd.Timedelta(days=int(d)))
    # new members inside the window
    months = pd.date_range(START, END, freq="MS")
    for i, m in enumerate(months):
        lam = 38 * JOIN_SEASON[m.month] * (1 + 0.012 * i)
        if m == months[-1]:
            lam *= 0.55  # planted sign-up slowdown in the latest month
        n = rng.poisson(lam)
        days_in = m.days_in_month
        for d in rng.integers(0, days_in, n):
            joins.append(m + pd.Timedelta(days=int(d)))

    n = len(joins)
    plan = rng.choice(list(PLANS), n, p=PLAN_MIX)
    cycle = rng.choice(list(CYCLES), n, p=CYCLE_MIX)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.52, 0.46, 0.02])
    age = np.where(plan == "Student", rng.integers(18, 26, n),
                   np.clip(rng.normal(36, 11, n), 18, 72).round()).astype(float)
    engagement = np.clip(rng.gamma(2.0, 1.1, n), 0.25, 6.5)
    engagement *= np.select([plan == "Premium", plan == "Student"], [1.25, 1.10], 1.0)
    slot = rng.choice(["morning", "midday", "evening"], n, p=[0.30, 0.18, 0.52])
    auto_renew = rng.random(n) < 0.68
    consent = rng.random(n) < 0.28
    first = rng.choice(FIRST, n)
    last = rng.choice(LAST, n)
    df = pd.DataFrame({
        "join_date": pd.to_datetime(joins), "first_name": first, "last_name": last,
        "gender": gender, "age": age, "membership_type": plan, "billing_cycle": cycle,
        "auto_renew": auto_renew, "data_consent": consent,
        "h_engagement": engagement, "h_slot": slot,
    }).sort_values("join_date").reset_index(drop=True)
    df.insert(0, "member_id", [f"M{i:05d}" for i in range(1, len(df) + 1)])
    df["email"] = (df["first_name"].str.lower() + "." + df["last_name"].str.lower()
                   .str.replace(" ", "") + df.index.astype(str) + "@mail.com")
    return df


# --------------------------------------------------------------------- subscriptions
def _simulate_subscriptions(members: pd.DataFrame, rng: np.random.Generator):
    subs, churn_dates = [], []
    for row in members.itertuples(index=False):
        months, disc = CYCLES[row.billing_cycle]
        price = round(PLANS[row.membership_type] * months * disc, 2)
        t = row.join_date
        churn = pd.NaT
        while True:
            end = _add_months(t, months)
            if end > END:
                subs.append((row.member_id, row.membership_type, row.billing_cycle, t, end,
                             price, row.auto_renew, "Active"))
                break
            f_eng = np.clip((1.7 / (row.h_engagement + 0.3)) ** 1.6, 0.2, 5.0)
            f_auto = 0.7 if row.auto_renew else 1.45
            f_dip = 1.9 if end >= pd.Timestamp("2026-09-15") else 1.0
            p = min(BASE_CHURN[row.billing_cycle] * f_eng * f_auto * f_dip, 0.9)
            if rng.random() < p:
                subs.append((row.member_id, row.membership_type, row.billing_cycle, t, end,
                             price, row.auto_renew, "Cancelled"))
                churn = end
                break
            subs.append((row.member_id, row.membership_type, row.billing_cycle, t, end,
                         price, row.auto_renew, "Renewed"))
            t = end
        churn_dates.append(churn)
    members = members.copy()
    members["churn_date"] = pd.to_datetime(churn_dates)
    subs_df = pd.DataFrame(subs, columns=["member_id", "membership_type", "billing_cycle",
                                          "start_date", "end_date", "price", "auto_renew",
                                          "status"])
    # drop members who left before the analysis window
    keep = members["churn_date"].isna() | (members["churn_date"] >= START)
    members = members[keep].reset_index(drop=True)
    subs_df = subs_df[subs_df["member_id"].isin(members["member_id"])
                      & (subs_df["end_date"] >= START)].reset_index(drop=True)
    subs_df.insert(0, "subscription_id", [f"S{i:06d}" for i in range(1, len(subs_df) + 1)])
    members["status"] = np.where(members["churn_date"].isna(), "Active", "Inactive")
    return members, subs_df


# ------------------------------------------------------------------------ attendance
def _visit_hours(rng, slot: str, weekend: np.ndarray) -> np.ndarray:
    n = len(weekend)
    slots = np.array(["morning", "midday", "evening"])
    chosen = np.where(rng.random(n) < 0.75, slot, rng.choice(slots, n))
    mu = np.select([chosen == "morning", chosen == "midday"], [7.3, 12.6], 18.6)
    sd = np.select([chosen == "morning", chosen == "midday"], [0.9, 0.8], 1.3)
    h = rng.normal(mu, sd)
    wk = np.where(rng.random(n) < 0.68, rng.normal(10.6, 1.6, n), rng.normal(16.8, 1.8, n))
    h = np.where(weekend, wk, h)
    return np.clip(h, OPEN_HOUR, CLOSE_HOUR - 0.75)


def _make_attendance(members: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    frames = []
    season = np.array([VISIT_SEASON[m] for m in range(1, 13)])
    for row in members.itertuples(index=False):
        a = max(row.join_date, START)
        b = END if pd.isna(row.churn_date) else min(row.churn_date, END)
        if b <= a:
            continue
        days = pd.date_range(a.normalize(), b.normalize() - pd.Timedelta(days=1), freq="D")
        if len(days) == 0:
            continue
        dow = days.day_of_week.to_numpy()
        rate = row.h_engagement / 7 * WEEKDAY_W[dow] * season[days.month.to_numpy() - 1]
        since_join = (days - row.join_date).days.to_numpy()
        rate = rate * np.where(since_join < 28, 1.2, 1.0)
        if not pd.isna(row.churn_date):
            to_churn = (row.churn_date - days).days.to_numpy()
            rate = rate * np.where(to_churn < 75, np.clip(0.2 + 0.8 * to_churn / 75, 0.2, 1), 1)
        visited = rng.random(len(days)) < np.minimum(rate, 0.92)
        vd = days[visited]
        if len(vd) == 0:
            continue
        hours = _visit_hours(rng, row.h_slot, vd.day_of_week.to_numpy() >= 5)
        check_in = vd + pd.to_timedelta(np.round(hours * 60), unit="min")
        dur = np.clip(rng.normal(62, 18, len(vd)), 20, 150).round()
        close = vd + pd.Timedelta(hours=CLOSE_HOUR)
        check_out = pd.Series(check_in + pd.to_timedelta(dur, unit="min")).where(
            lambda s: s <= close, close)
        frames.append(pd.DataFrame({"member_id": row.member_id, "check_in": check_in,
                                    "check_out": check_out.values}))
    att = pd.concat(frames, ignore_index=True).sort_values("check_in").reset_index(drop=True)
    att["duration_min"] = ((att["check_out"] - att["check_in"]).dt.total_seconds() / 60).round()
    att.insert(0, "visit_id", [f"V{i:07d}" for i in range(1, len(att) + 1)])
    return att


# -------------------------------------------------------------------------- payments
def _make_payments(members, subs, rng) -> pd.DataFrame:
    rows = []
    plan_of = members.set_index("member_id")["membership_type"].to_dict()
    for s in subs.itertuples(index=False):
        if START <= s.start_date <= END:
            method = rng.choice(METHODS, p=[0.38, 0.40, 0.07, 0.15] if s.billing_cycle == "Monthly"
                                else METHOD_P)
            status = rng.choice(["Completed", "Failed", "Refunded"], p=[0.965, 0.022, 0.013])
            rows.append((s.member_id, s.subscription_id, s.start_date, s.price, method,
                         "Membership", s.membership_type, status))
    # ancillary revenue: personal training packs & shop
    months = pd.date_range(START, END, freq="MS")
    for m in members.itertuples(index=False):
        b = END if pd.isna(m.churn_date) else m.churn_date
        for mo in months:
            if mo < m.join_date.replace(day=1) or mo > b:
                continue
            p_pt = 0.16 if m.membership_type == "Premium" else 0.04
            if rng.random() < p_pt:
                d = mo + pd.Timedelta(days=int(rng.integers(0, 28)))
                rows.append((m.member_id, None, d, 160.0, rng.choice(METHODS, p=METHOD_P),
                             "Personal Training", plan_of[m.member_id], "Completed"))
            if rng.random() < 0.06:
                d = mo + pd.Timedelta(days=int(rng.integers(0, 28)))
                rows.append((m.member_id, None, d, float(rng.choice([15, 25, 35, 49, 60])),
                             rng.choice(METHODS, p=METHOD_P), "Shop & Supplements",
                             plan_of[m.member_id], "Completed"))
    pay = pd.DataFrame(rows, columns=["member_id", "subscription_id", "payment_date", "amount",
                                      "payment_method", "revenue_type", "membership_type",
                                      "status"])
    pay = pay[pay["payment_date"] <= END].sort_values("payment_date").reset_index(drop=True)
    pay.insert(0, "payment_id", [f"P{i:06d}" for i in range(1, len(pay) + 1)])
    return pay


# ------------------------------------------------------------------------- equipment
def _make_equipment(rng):
    rows = []
    for name, cat, zone, units, pop, cost, interval in EQUIPMENT:
        for u in range(1, units + 1):
            eid = f"EQ-{ABBR[name]}-{u:02d}"
            p = pop * rng.uniform(0.85, 1.15)
            if eid == UNDERUSED_UNIT:
                p = 0.07
            purchase = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 1500)))
            rows.append((eid, f"{name} #{u}", name, cat, zone,
                         rng.choice(["Technogym", "Life Fitness", "Matrix", "Precor", "Eleiko"]),
                         purchase, float(cost), interval, round(p, 3)))
    return pd.DataFrame(rows, columns=["equipment_id", "name", "equipment_type", "category", "zone",
                                       "brand", "purchase_date", "purchase_cost",
                                       "service_interval_days", "h_popularity"])


def _make_maintenance(equipment, rng):
    rows = []
    for e in equipment.itertuples(index=False):
        # preventive services
        t = START + pd.Timedelta(days=int(rng.integers(0, e.service_interval_days)))
        stop = pd.Timestamp("2026-01-20") if e.equipment_id == OVERDUE_UNIT else END
        while t <= stop:
            rows.append((e.equipment_id, t, "Preventive",
                         round(e.purchase_cost * rng.uniform(0.012, 0.025), 2),
                         round(rng.uniform(1, 3), 1), "Scheduled inspection & lubrication"))
            t += pd.Timedelta(days=int(e.service_interval_days + rng.integers(-10, 11)))
        # corrective repairs
        yearly = 6.5 if e.equipment_id == FAILURE_UNIT else (0.9 if e.category == "Cardio" else 0.35)
        n_rep = rng.poisson(yearly * 2)
        for d in rng.integers(0, (END - START).days, n_rep):
            rows.append((e.equipment_id, START + pd.Timedelta(days=int(d)), "Repair",
                         round(e.purchase_cost * rng.uniform(0.03, 0.09), 2),
                         round(rng.uniform(6, 72), 1),
                         rng.choice(["Belt replacement", "Motor fault", "Display failure",
                                     "Cable wear", "Bearing noise", "Sensor calibration"])))
    mt = pd.DataFrame(rows, columns=["equipment_id", "date", "maintenance_type", "cost",
                                     "downtime_hours", "description"])
    mt = mt.sort_values("date").reset_index(drop=True)
    mt.insert(0, "maintenance_id", [f"MT{i:05d}" for i in range(1, len(mt) + 1)])
    return mt


def _make_equipment_usage(equipment, maintenance, attendance, rng):
    daily_visits = attendance.groupby(attendance["check_in"].dt.normalize()).size()
    days = pd.date_range(START, END, freq="D")
    daily_visits = daily_visits.reindex(days, fill_value=0).to_numpy()
    pops = equipment["h_popularity"].to_numpy()
    share = pops / pops.sum()
    open_min = (CLOSE_HOUR - OPEN_HOUR) * 60
    minutes = daily_visits[:, None] * 3 * 12 * share[None, :]
    minutes = minutes * rng.lognormal(0, 0.22, minutes.shape)
    # unit offline during repairs
    rep = maintenance[maintenance["maintenance_type"] == "Repair"]
    idx = {e: i for i, e in enumerate(equipment["equipment_id"])}
    for r in rep.itertuples(index=False):
        d0 = (r.date.normalize() - START).days
        span = max(1, int(np.ceil(r.downtime_hours / 17)))
        minutes[d0:d0 + span, idx[r.equipment_id]] = 0
    minutes = np.clip(minutes, 0, open_min * 0.92).round()
    sessions = np.round(minutes / rng.uniform(10, 16, minutes.shape)).astype(int)
    usage = pd.DataFrame({
        "date": np.repeat(days, len(equipment)),
        "equipment_id": np.tile(equipment["equipment_id"].to_numpy(), len(days)),
        "sessions": sessions.ravel(), "usage_minutes": minutes.ravel(),
    })
    return usage


# -------------------------------------------------------------------------- workouts
def _make_workouts(members, attendance, equipment, rng):
    by_type = equipment.groupby("equipment_type")["equipment_id"].apply(list).to_dict()
    consenting = members[members["data_consent"]].set_index("member_id")
    since = END - pd.DateOffset(months=12)
    att = attendance[attendance["member_id"].isin(consenting.index)
                     & (attendance["check_in"] >= since)]
    rows = []
    s_names, c_names = list(STRENGTH_EX), list(CARDIO_EX)
    for mid, visits in att.groupby("member_id"):
        focus_strength = rng.random() < 0.6
        strength_f = rng.uniform(0.6, 1.4) * (0.8 if consenting.at[mid, "gender"] == "Female" else 1.0)
        gain = rng.uniform(0.08, 0.35)
        fitness_gain = rng.uniform(0.05, 0.25)
        fav_s = rng.choice(s_names, 5, replace=False)
        fav_c = rng.choice(c_names, 2, replace=False)
        first = visits["check_in"].min()
        for v in visits.itertuples(index=False):
            weeks = (v.check_in - first).days / 7
            prog = 1 + gain * (1 - np.exp(-weeks / 18))
            n_s = rng.integers(3, 5) if focus_strength else rng.integers(0, 2)
            n_c = rng.integers(0, 2) if focus_strength else rng.integers(1, 3)
            for ex in rng.choice(fav_s, n_s, replace=False):
                etype, base = STRENGTH_EX[ex]
                w = base * strength_f * prog * rng.normal(1, 0.04)
                rows.append((v.visit_id, mid, v.check_in, ex, "Strength",
                             rng.choice(by_type[etype]), int(rng.integers(3, 5)),
                             int(rng.integers(5, 13)), round(w / 2.5) * 2.5, None, None,
                             int(rng.integers(6, 10))))
            for ex in rng.choice(fav_c, min(n_c, 2), replace=False):
                etype, pace = CARDIO_EX[ex]
                dur = int(rng.integers(15, 41))
                dist = dur * pace * (1 + fitness_gain * (1 - np.exp(-weeks / 20))) * rng.normal(1, 0.05)
                rows.append((v.visit_id, mid, v.check_in, ex, "Cardio",
                             rng.choice(by_type[etype]), None, None, None, dur,
                             round(dist, 2), int(rng.integers(5, 9))))
    wk = pd.DataFrame(rows, columns=["visit_id", "member_id", "workout_date", "exercise",
                                     "exercise_type", "equipment_id", "sets", "reps",
                                     "weight_kg", "duration_min", "distance_km", "rpe"])
    wk.insert(0, "workout_id", [f"W{i:07d}" for i in range(1, len(wk) + 1)])
    return wk


# -------------------------------------------------------------------------- expenses
def _make_expenses(maintenance, members, rng):
    months = pd.date_range(START, END, freq="MS")
    mt = maintenance.groupby(maintenance["date"].dt.to_period("M"))["cost"].sum()
    rows = []
    for i, m in enumerate(months):
        winter = m.month in (11, 12, 1, 2)
        summer = m.month in (6, 7, 8)
        rows += [
            (m, "Rent", 7200.0),
            (m, "Staff Salaries", round(15800 * (1 + 0.004 * i) + rng.normal(0, 300), 2)),
            (m, "Utilities", round(2100 * (1.25 if winter else 1.1 if summer else 1.0)
                                   + rng.normal(0, 120), 2)),
            (m, "Marketing", round(rng.uniform(1400, 2600) * (1.5 if m.month in (1, 9) else 1), 2)),
            (m, "Supplies & Cleaning", round(rng.uniform(650, 950), 2)),
            (m, "Software & Insurance", 780.0),
            (m, "Equipment Maintenance", round(float(mt.get(m.to_period("M"), 0.0)), 2)),
        ]
    ex = pd.DataFrame(rows, columns=["month", "category", "amount"])
    ex.insert(0, "expense_id", [f"E{i:05d}" for i in range(1, len(ex) + 1)])
    return ex


# ------------------------------------------------------------------------ dirty data
def _inject_noise(tables: dict[str, pd.DataFrame], rng) -> None:
    m = tables["members"]
    m.loc[rng.choice(m.index, int(len(m) * 0.02), replace=False), "age"] = np.nan
    m.loc[rng.choice(m.index, int(len(m) * 0.01), replace=False), "gender"] = np.nan
    p = tables["payments"]
    p.loc[rng.choice(p.index, int(len(p) * 0.01), replace=False), "payment_method"] = np.nan
    bad = rng.choice(p.index, 4, replace=False)
    p.loc[bad, "amount"] = -p.loc[bad, "amount"]
    tables["payments"] = pd.concat([p, p.sample(frac=0.005, random_state=1)]).sort_index()
    a = tables["attendance"]
    tables["attendance"] = pd.concat([a, a.sample(frac=0.004, random_state=2)]).sort_index()


# ---------------------------------------------------------------------------- public
def generate(seed: int = SEED, out_dir: Path | None = None, write: bool = True) -> dict[str, pd.DataFrame]:
    """Generate all demo tables. Returns raw (dirty) DataFrames and optionally writes CSVs."""
    rng = np.random.default_rng(seed)
    members = _make_members(rng)
    members, subs = _simulate_subscriptions(members, rng)
    attendance = _make_attendance(members, rng)
    payments = _make_payments(members, subs, rng)
    equipment = _make_equipment(rng)
    maintenance = _make_maintenance(equipment, rng)
    usage = _make_equipment_usage(equipment, maintenance, attendance, rng)
    workouts = _make_workouts(members, attendance, equipment, rng)
    expenses = _make_expenses(maintenance, members, rng)

    members = members.drop(columns=["h_engagement", "h_slot"])
    members = members[["member_id", "first_name", "last_name", "gender", "age", "email",
                       "join_date", "membership_type", "billing_cycle", "auto_renew",
                       "data_consent", "status", "churn_date"]]
    equipment = equipment.drop(columns=["h_popularity"])

    tables = {"members": members, "subscriptions": subs, "attendance": attendance,
              "payments": payments, "equipment": equipment, "equipment_usage": usage,
              "maintenance": maintenance, "workouts": workouts, "expenses": expenses}
    _inject_noise(tables, rng)
    if write:
        out = Path(out_dir) if out_dir else data_dir()
        out.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(out / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    t = generate()
    for k, v in t.items():
        print(f"{k:16s} {len(v):>8,d} rows  -> data/{k}.csv")
