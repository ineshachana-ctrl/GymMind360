import numpy as np
import pandas as pd

from data import generator
from database import db
from utils.cleaning import SCHEMAS, clean_table, guess_table, validate_columns


def test_generator_tables_and_reproducibility(raw):
    assert set(raw) == set(SCHEMAS)
    assert 1000 < len(raw["members"]) < 2000
    assert len(raw["attendance"]) > 100_000
    again = generator.generate(seed=42, write=False)
    pd.testing.assert_frame_equal(raw["members"], again["members"])


def test_cleaning_removes_planted_noise(raw, T):
    # duplicates were injected in payments & attendance
    assert raw["attendance"].duplicated().sum() > 0
    assert T["attendance"].duplicated(subset=["member_id", "check_in"]).sum() == 0
    assert T["payments"]["payment_id"].is_unique
    assert (T["payments"]["amount"] > 0).all()            # negative amounts removed
    assert T["members"]["age"].isna().sum() == 0          # imputed with median
    assert T["payments"]["payment_method"].isna().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(T["attendance"]["check_in"])


def test_clean_table_report_and_bool_parsing():
    df = pd.DataFrame({"member_id": ["M1", "M1", "M2", "M3"],
                       "join_date": ["2025-01-01", "2025-01-01", "not a date", "2025-03-01"],
                       "membership_type": ["Basic"] * 4, "status": ["Active"] * 4,
                       "age": [30, 30, np.nan, 40], "auto_renew": ["yes", "yes", "0", "TRUE"]})
    out, rep = clean_table(df, "members")
    assert rep.duplicates_removed == 1
    assert rep.type_errors == {"join_date": 1}
    assert rep.invalid_removed == 1                       # row with unparseable required date
    assert list(out["auto_renew"]) == [True, True]
    assert rep.ok


def test_missing_columns_detected():
    df = pd.DataFrame({"member_id": ["M1"]})
    assert "check_in" in validate_columns(df, "attendance")
    _, rep = clean_table(df, "attendance")
    assert not rep.ok


def test_guess_table(T):
    assert guess_table(T["payments"].head()) == "payments"
    assert guess_table(T["equipment"].head()) == "equipment"
    assert guess_table(pd.DataFrame({"foo": [1]})) is None


def test_sqlite_roundtrip(tmp_path, T):
    path = tmp_path / "t.db"
    db.write_table("members", T["members"], path=path)
    back = db.read_table("members", path=path)
    assert len(back) == len(T["members"])
    assert pd.api.types.is_datetime64_any_dtype(back["join_date"])
    assert back["auto_renew"].dtype == bool
    assert db.metadata(path).iloc[0]["row_count"] == len(T["members"])
    assert db.query("SELECT COUNT(*) AS n FROM members", path)["n"][0] == len(T["members"])
