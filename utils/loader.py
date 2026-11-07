"""Central data access: bootstrap demo data, load cleaned tables, apply imports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data import generator
from database import db
from utils.cleaning import SCHEMAS, QualityReport, clean_all, clean_table


def bootstrap(force: bool = False) -> list[QualityReport]:
    """Ensure the SQLite database exists. Generates demo CSVs if missing.

    Returns the quality reports from the cleaning step (empty if nothing was done).
    """
    if db.is_ready() and not force:
        return []
    csvs = {n: generator.data_dir() / f"{n}.csv" for n in SCHEMAS}
    if force or not all(p.exists() for p in csvs.values()):
        raw = generator.generate()
    else:
        raw = {n: pd.read_csv(p) for n, p in csvs.items()}
    cleaned, reports = clean_all(raw)
    for name, df in cleaned.items():
        db.write_table(name, df, source="demo")
    return reports


def load_all() -> dict[str, pd.DataFrame]:
    bootstrap()
    return {name: db.read_table(name) for name in SCHEMAS}


def quality_reports_from_csv() -> list[QualityReport]:
    """Re-run the cleaning pipeline on the raw CSV files (for the data-quality view)."""
    reports = []
    for name in SCHEMAS:
        p = generator.data_dir() / f"{name}.csv"
        if p.exists():
            _, rep = clean_table(pd.read_csv(p), name)
            reports.append(rep)
    return reports


def import_table(name: str, df: pd.DataFrame, filename: str) -> QualityReport:
    cleaned, rep = clean_table(df, name)
    if rep.ok and len(cleaned):
        db.write_table(name, cleaned, source=f"import:{filename}")
    return rep


def data_window(tables: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    att = tables["attendance"]["check_in"]
    pay = tables["payments"]["payment_date"]
    lo = min(att.min(), pay.min())
    hi = max(att.max(), pay.max())
    return pd.Timestamp(lo).normalize(), pd.Timestamp(hi).normalize()
