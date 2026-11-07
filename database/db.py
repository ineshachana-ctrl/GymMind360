"""SQLite persistence layer.

The database location defaults to database/gymmind.db and can be overridden with the
GYMMIND_DB environment variable (used by the test-suite to work on an isolated copy).
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.cleaning import SCHEMAS

DB_PATH = Path(__file__).resolve().parent / "gymmind.db"
SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"


def db_path(path: Path | str | None = None) -> Path:
    return Path(path or os.environ.get("GYMMIND_DB") or DB_PATH)


def connect(path: Path | str | None = None, readonly: bool = False) -> sqlite3.Connection:
    p = db_path(path)
    if readonly:
        return sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro", uri=True)
    return sqlite3.connect(str(p))


def init_metadata(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))


def write_table(name: str, df: pd.DataFrame, source: str = "demo", path: Path | str | None = None) -> None:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S").where(out[c].notna(), None)
    with closing(connect(path)) as conn:
        init_metadata(conn)
        out.to_sql(name, conn, if_exists="replace", index=False, chunksize=20000)
        conn.execute(
            "INSERT OR REPLACE INTO _metadata(table_name, row_count, source, updated_at) VALUES (?,?,?,?)",
            (name, len(out), source, datetime.now().isoformat(timespec="microseconds")))
        _index(conn, name)
        conn.commit()


def _index(conn: sqlite3.Connection, name: str) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info('{name}')")}
    for c in ("member_id", "equipment_id"):
        if c in cols:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{name}_{c} ON '{name}'({c})")


def read_table(name: str, path: Path | str | None = None) -> pd.DataFrame:
    with closing(connect(path)) as conn:
        df = pd.read_sql(f"SELECT * FROM '{name}'", conn)
    schema = SCHEMAS.get(name, {})
    for c in schema.get("dates", []):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in schema.get("bool", []):
        if c in df.columns:
            df[c] = df[c].astype(bool)
    return df


def list_tables(path: Path | str | None = None) -> list[str]:
    if not db_path(path).exists():
        return []
    with closing(connect(path)) as conn:
        return [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '\\_%' ESCAPE '\\'")]


def metadata(path: Path | str | None = None) -> pd.DataFrame:
    if not db_path(path).exists():
        return pd.DataFrame(columns=["table_name", "row_count", "source", "updated_at"])
    with closing(connect(path)) as conn:
        init_metadata(conn)
        return pd.read_sql("SELECT * FROM _metadata ORDER BY table_name", conn)


def query(sql: str, path: Path | str | None = None) -> pd.DataFrame:
    """Run a read-only SQL query (SQL explorer). The connection itself is opened read-only,
    so statements such as `WITH ... DELETE` are rejected by SQLite."""
    if not sql.strip().lower().startswith(("select", "with")):
        raise ValueError("Only SELECT / WITH queries are allowed.")
    with closing(connect(path, readonly=True)) as conn:
        return pd.read_sql(sql, conn)


def is_ready(path: Path | str | None = None) -> bool:
    return set(SCHEMAS).issubset(set(list_tables(path)))


def columns(name: str, path: Path | str | None = None) -> list[str]:
    with closing(connect(path)) as conn:
        return [r[1] for r in conn.execute(f"PRAGMA table_info('{name}')")]
