"""Feature tests: drive every interactive control of the Streamlit app through AppTest.

Runs against an isolated copy of the database and CSV files (GYMMIND_DB / GYMMIND_DATA_DIR),
so imports and demo-data regeneration never touch the real data.
"""
import io
import os
import re
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
V = "views/"


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gymmind")
    from utils import loader
    loader.bootstrap()                                   # make sure the real DB exists to copy
    shutil.copy(ROOT / "database" / "gymmind.db", tmp / "gymmind.db")
    (tmp / "data").mkdir()
    for f in (ROOT / "data").glob("*.csv"):
        shutil.copy(f, tmp / "data" / f.name)
    old = {k: os.environ.get(k) for k in ("GYMMIND_DB", "GYMMIND_DATA_DIR")}
    os.environ["GYMMIND_DB"] = str(tmp / "gymmind.db")
    os.environ["GYMMIND_DATA_DIR"] = str(tmp / "data")
    yield tmp
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def app(sandbox):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def ok(at):
    assert not at.exception, "\n".join(e.value for e in at.exception)


def goto(at, page):
    at.switch_page(V + page)
    at.run()
    ok(at)


def by_label(widgets, label):
    for w in widgets:
        if w.label == label:
            return w
    raise AssertionError(f"widget {label!r} not found: {[w.label for w in widgets]}")


def kpi_values(at) -> dict[str, str]:
    out = {}
    for m in at.markdown:
        for lbl, val in re.findall(r'</span>([^<]+)</div><div class="kpi-value">([^<]+)</div>', m.value):
            out[lbl.strip()] = val
    return out


PAGES = sorted(p.name for p in (ROOT / "views").glob("*.py"))


# ----------------------------------------------------------------------------- sidebar filters
@pytest.mark.parametrize("period,plans", [
    ((date(2026, 10, 1), date(2026, 10, 31)), []),            # single month
    ((date(2024, 11, 1), date(2026, 10, 31)), []),            # full history
    ((date(2026, 1, 1), date(2026, 6, 30)), ["Student"]),     # one plan
    ((date(2025, 6, 1), date(2026, 10, 31)), ["Premium", "Basic"]),
])
def test_filters_on_every_page(app, period, plans):
    app.sidebar.date_input[0].set_value(period)
    app.sidebar.multiselect[0].set_value(plans)
    app.run()
    ok(app)
    for p in PAGES:
        goto(app, p)


def test_plan_filter_changes_kpis(app):
    goto(app, "1_Executive_Dashboard.py")
    all_active = kpi_values(app)["Active members"]
    app.sidebar.multiselect[0].set_value(["Student"]).run()
    ok(app)
    from utils import state  # noqa: F401  (ensures same process cache)
    from utils.loader import load_all
    m = load_all()["members"]
    students_active = ((m["membership_type"] == "Student") & m["churn_date"].isna()).sum()
    assert kpi_values(app)["Active members"] == f"{students_active:,}"
    assert kpi_values(app)["Active members"] != all_active


# ----------------------------------------------------------------------------- page controls
def test_occupancy_heatmap_modes(app):
    goto(app, "4_Smart_Occupancy.py")
    for mode in ["Check-ins", "Occupancy %", "People on the floor"]:
        app.segmented_control[0].set_value(mode).run()
        ok(app)


def test_segmentation_k_values(app):
    goto(app, "2_Member_Intelligence.py")
    for k in [3, 4, 5, 6, 7, "Auto"]:
        by_label(app.select_slider, "Number of segments").set_value(k).run()
        ok(app)


def test_predictive_controls(app):
    goto(app, "6_Predictive_Analytics.py")
    for h in (3, 12, 6):
        by_label(app.slider, "Forecast horizon (months)").set_value(h).run()
        ok(app)
    by_label(app.multiselect, "Risk bands").set_value(["High", "Medium", "Low"]).run()
    ok(app)
    by_label(app.multiselect, "Risk bands").set_value([]).run()
    ok(app)
    assert any(b.label == "Export retention list (CSV)" for b in app.get("download_button"))


def test_insight_filters(app):
    goto(app, "7_Smart_Insights.py")
    cat = by_label(app.multiselect, "Category")
    for c in cat.options:
        cat.set_value([c]).run()
        ok(app)
        cat = by_label(app.multiselect, "Category")
    cat.set_value([]).run()
    sev = by_label(app.multiselect, "Severity")
    for s in ["critical", "warning", "opportunity", "positive"]:
        sev.set_value([s]).run()
        ok(app)
        sev = by_label(app.multiselect, "Severity")
    labels = [b.label for b in app.get("download_button")]
    assert "Download CSV" in labels and "Download report (.md)" in labels


def test_equipment_drilldown_planted_units(app):
    goto(app, "5_Equipment_Intelligence.py")
    for unit in ["EQ-TR-02", "EQ-RM-03", "EQ-LC-01", "EQ-KB-01"]:
        by_label(app.selectbox, "Select equipment").set_value(unit).run()
        ok(app)


def test_fitness_members_and_exercises(app):
    goto(app, "8_Member_Fitness.py")
    sel = app.selectbox[0]
    opts = sel.options
    for mid in [opts[0], opts[len(opts) // 2], opts[-1]]:
        # options are rendered labels; select by index position
        app.selectbox[0].select_index(opts.index(mid)).run()
        ok(app)
        ex = by_label(app.selectbox, "Exercise")
        for i in range(len(ex.options)):
            by_label(app.selectbox, "Exercise").select_index(i).run()
            ok(app)


# ----------------------------------------------------------------------------- data management
def test_sql_explorer(app):
    goto(app, "9_Data_Management.py")
    ex = by_label(app.selectbox, "Example query")
    for i in range(len(ex.options)):
        by_label(app.selectbox, "Example query").select_index(i).run()
        by_label(app.button, "Run query").click().run()
        ok(app)
        assert not app.error
    for bad in ["DELETE FROM members", "WITH x AS (SELECT 1) DELETE FROM members"]:
        by_label(app.text_area, "SQL").set_value(bad).run()
        by_label(app.button, "Run query").click().run()
        ok(app)
        assert app.error, f"write query was not rejected: {bad}"
    from database import db
    assert db.query("SELECT COUNT(*) n FROM members")["n"][0] > 0     # data intact


def test_import_csv_xlsx_and_rejection(app):
    from database import db
    goto(app, "9_Data_Management.py")
    pay = db.read_table("payments")
    n = len(pay)
    # CSV with title-case headers + duplicates -> cleaned & imported
    df = pd.concat([pay, pay.head(7)])
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    app.get("file_uploader")[0].set_value(("payments.csv", df.to_csv(index=False).encode(), "text/csv")).run()
    ok(app)
    assert by_label(app.selectbox, "Target table").value == "payments"
    by_label(app.button, "Validate & import into Payments").click().run()
    ok(app)
    assert app.success and "7 duplicates" in app.success[0].value
    assert len(db.read_table("payments")) == n
    # Excel equipment file
    eq = db.read_table("equipment")
    buf = io.BytesIO()
    eq.to_excel(buf, index=False)
    app.get("file_uploader")[0].set_value(("equipment.xlsx", buf.getvalue(),
                                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")).run()
    ok(app)
    by_label(app.button, "Validate & import into Equipment").click().run()
    ok(app)
    assert app.success
    # missing required column -> clear error, no import button
    bad = db.read_table("members").drop(columns=["join_date"])
    app.get("file_uploader")[0].set_value(("members.csv", bad.to_csv(index=False).encode(), "text/csv")).run()
    ok(app)
    by_label(app.selectbox, "Target table").set_value("members").run()
    ok(app)
    assert any("join_date" in e.value for e in app.error)
    # every page still works after imports
    for p in PAGES:
        goto(app, p)


def test_regenerate_demo_data(app, sandbox):
    goto(app, "9_Data_Management.py")
    by_label(app.number_input, "Random seed").set_value(7).run()
    by_label(app.button, "Regenerate demo data").click().run()
    ok(app)
    assert app.success and "regenerated" in app.success[0].value
    from database import db
    assert "seed=7" in db.metadata()["source"].iloc[0]
    assert (sandbox / "data" / "members.csv").exists()
    for p in PAGES:
        goto(app, p)


def test_dashboard_link_to_insights(app):
    goto(app, "1_Executive_Dashboard.py")
    links = [l.proto.page for l in app.get("page_link")]
    assert "Smart_Insights" in links
