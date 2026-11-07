"""Data validation & cleaning pipeline (Pandas).

Each table has a schema describing required columns, dtypes, primary key and
fill rules. `clean_table` returns the cleaned frame plus a `QualityReport`
documenting everything that was changed - shown in the Data Management page.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SCHEMAS: dict[str, dict] = {
    "members": {
        "required": ["member_id", "join_date", "membership_type", "status"],
        "pk": "member_id",
        "dates": ["join_date", "churn_date"],
        "numeric": ["age"],
        "bool": ["auto_renew", "data_consent"],
        "fill": {"age": "median", "gender": "Unknown", "billing_cycle": "Monthly",
                 "auto_renew": False, "data_consent": False},
        "positive": [],
    },
    "subscriptions": {
        "required": ["subscription_id", "member_id", "start_date", "end_date", "price", "status"],
        "pk": "subscription_id",
        "dates": ["start_date", "end_date"],
        "numeric": ["price"],
        "bool": ["auto_renew"],
        "fill": {"auto_renew": False},
        "positive": ["price"],
    },
    "attendance": {
        "required": ["member_id", "check_in"],
        "pk": "visit_id",
        "dates": ["check_in", "check_out"],
        "numeric": ["duration_min"],
        "bool": [],
        "fill": {"duration_min": "median"},
        "positive": [],
    },
    "payments": {
        "required": ["member_id", "payment_date", "amount"],
        "pk": "payment_id",
        "dates": ["payment_date"],
        "numeric": ["amount"],
        "bool": [],
        "fill": {"payment_method": "Unknown", "revenue_type": "Membership", "status": "Completed",
                 "membership_type": "Unknown"},
        "positive": ["amount"],
    },
    "equipment": {
        "required": ["equipment_id", "name", "category"],
        "pk": "equipment_id",
        "dates": ["purchase_date"],
        "numeric": ["purchase_cost", "service_interval_days"],
        "bool": [],
        "fill": {"service_interval_days": 120, "zone": "General",
                 "equipment_type": "Other", "brand": "Unknown"},
        "positive": [],
    },
    "equipment_usage": {
        "required": ["date", "equipment_id", "usage_minutes"],
        "pk": None,
        "dates": ["date"],
        "numeric": ["sessions", "usage_minutes"],
        "bool": [],
        "fill": {"sessions": 0, "usage_minutes": 0},
        "positive": [],
    },
    "maintenance": {
        "required": ["equipment_id", "date", "cost"],
        "pk": "maintenance_id",
        "dates": ["date"],
        "numeric": ["cost", "downtime_hours"],
        "bool": [],
        "fill": {"downtime_hours": 0, "maintenance_type": "Repair", "description": ""},
        "positive": [],
    },
    "workouts": {
        "required": ["member_id", "workout_date", "exercise"],
        "pk": "workout_id",
        "dates": ["workout_date"],
        "numeric": ["sets", "reps", "weight_kg", "duration_min", "distance_km", "rpe"],
        "bool": [],
        "fill": {"exercise_type": "Other"},
        "positive": [],
    },
    "expenses": {
        "required": ["month", "category", "amount"],
        "pk": "expense_id",
        "dates": ["month"],
        "numeric": ["amount"],
        "bool": [],
        "fill": {},
        "positive": [],
    },
}

TABLE_LABELS = {
    "members": "Members", "subscriptions": "Subscriptions", "attendance": "Attendance",
    "payments": "Payments", "equipment": "Equipment", "equipment_usage": "Equipment Usage",
    "maintenance": "Maintenance", "workouts": "Workouts", "expenses": "Expenses",
}


@dataclass
class QualityReport:
    table: str
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    invalid_removed: int = 0
    missing_before: dict = field(default_factory=dict)
    filled: dict = field(default_factory=dict)
    type_errors: dict = field(default_factory=dict)
    missing_columns: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_columns

    @property
    def completeness(self) -> float:
        total = sum(self.missing_before.values())
        cells = max(self.rows_in * max(len(self.missing_before), 1), 1)
        return 1 - total / cells

    def as_row(self) -> dict:
        return {"Table": TABLE_LABELS.get(self.table, self.table), "Rows in": self.rows_in,
                "Rows out": self.rows_out, "Duplicates removed": self.duplicates_removed,
                "Invalid removed": self.invalid_removed,
                "Missing cells": int(sum(self.missing_before.values())),
                "Values imputed": int(sum(self.filled.values())),
                "Type errors": int(sum(self.type_errors.values())),
                "Completeness": round(self.completeness * 100, 2)}


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    mapping = {"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False,
               "no": False, "n": False}
    return s.map(lambda v: mapping.get(str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame, table: str) -> list[str]:
    """Return the list of required columns missing from `df`."""
    req = SCHEMAS[table]["required"]
    return [c for c in req if c not in df.columns]


def clean_table(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, QualityReport]:
    schema = SCHEMAS[table]
    rep = QualityReport(table=table, rows_in=len(df))
    df = normalize_columns(df)
    rep.missing_columns = validate_columns(df, table)
    if rep.missing_columns:
        rep.notes.append(f"Missing required columns: {', '.join(rep.missing_columns)}")
        rep.rows_out = len(df)
        return df, rep

    # completeness is measured on fields that are expected to be filled
    # (structurally optional fields such as churn_date or cardio-only metrics are excluded)
    expected = [c for c in dict.fromkeys(schema["required"] + list(schema["fill"])) if c in df.columns]
    rep.missing_before = {c: int(df[c].isna().sum()) for c in expected}

    # trim strings
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            if c in schema["bool"]:
                continue
            s = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[c] = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    # dtype coercion
    for c in schema["dates"]:
        if c in df.columns:
            before = df[c].notna().sum()
            df[c] = pd.to_datetime(df[c], errors="coerce")
            bad = int(before - df[c].notna().sum())
            if bad:
                rep.type_errors[c] = bad
    for c in schema["numeric"]:
        if c in df.columns:
            before = df[c].notna().sum()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            bad = int(before - df[c].notna().sum())
            if bad:
                rep.type_errors[c] = bad
    for c in schema["bool"]:
        if c in df.columns:
            df[c] = _to_bool(df[c])

    # exact duplicates, then primary-key duplicates
    n = len(df)
    df = df.drop_duplicates()
    pk = schema["pk"]
    if pk and pk in df.columns:
        df = df.drop_duplicates(subset=[pk], keep="first")
    if table == "attendance":
        df = df.drop_duplicates(subset=["member_id", "check_in"], keep="first")
    rep.duplicates_removed = n - len(df)

    # invalid rows: missing required values / non-positive monetary amounts
    n = len(df)
    df = df.dropna(subset=schema["required"])
    for c in schema["positive"]:
        df = df[df[c] > 0]
    if table == "attendance" and "check_out" in df.columns:
        df = df[~(df["check_out"] < df["check_in"])]
    rep.invalid_removed = n - len(df)

    # imputation
    for c, rule in schema["fill"].items():
        if c not in df.columns:
            if not isinstance(rule, str) or rule != "median":
                df[c] = rule
            continue
        k = int(df[c].isna().sum())
        if not k:
            continue
        value = df[c].median() if rule == "median" else rule
        df[c] = df[c].fillna(value)
        rep.filled[c] = k
    for c in schema["bool"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    if table == "attendance" and "duration_min" in df.columns and "check_out" in df.columns:
        miss = df["duration_min"].isna() & df["check_out"].notna()
        df.loc[miss, "duration_min"] = (df["check_out"] - df["check_in"]).dt.total_seconds() / 60

    df = df.reset_index(drop=True)
    rep.rows_out = len(df)
    return df, rep


def clean_all(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[QualityReport]]:
    out, reports = {}, []
    for name, df in tables.items():
        if name not in SCHEMAS:
            continue
        cleaned, rep = clean_table(df, name)
        out[name] = cleaned
        reports.append(rep)
    return out, reports


def read_upload(file) -> pd.DataFrame:
    """Read an uploaded CSV / Excel file object into a DataFrame."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


def guess_table(df: pd.DataFrame) -> str | None:
    """Guess which schema an imported frame matches best."""
    cols = set(normalize_columns(df).columns)
    best, score = None, 0.0
    for name, s in SCHEMAS.items():
        req = set(s["required"])
        hit = len(req & cols) / len(req)
        if hit > score:
            best, score = name, hit
    return best if score >= 0.6 else None
