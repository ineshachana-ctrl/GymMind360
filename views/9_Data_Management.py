"""Data Management — import CSV/Excel, data quality, SQLite explorer, demo data."""
import io

import pandas as pd
import streamlit as st

from components import layout
from database import db
from utils import loader, state
from utils.cleaning import SCHEMAS, TABLE_LABELS, guess_table, normalize_columns, read_upload

layout.page_header("Data", "Data Management",
                   "Import your own data, review data quality and explore the SQLite database.",
                   chips=[f"SQLite · {db.DB_PATH.name}"])

t1, t2, t3, t4 = st.tabs(["Database overview", "Import data", "SQL explorer", "Demo data & templates"])

with t1:
    md = db.metadata()
    md["table"] = md["table_name"].map(TABLE_LABELS).fillna(md["table_name"])
    st.dataframe(md[["table", "table_name", "row_count", "source", "updated_at"]], hide_index=True, width="stretch",
                 column_config={"table": "Table", "table_name": "SQL name", "row_count": st.column_config.NumberColumn("Rows", format="%,d"),
                                "source": "Source", "updated_at": "Last updated"})
    layout.section("Data quality report", "Cleaning pipeline applied to the raw CSV files (duplicates, types, missing values)")
    with st.spinner("Profiling raw files…"):
        reps = loader.quality_reports_from_csv()
    if reps:
        q = pd.DataFrame([r.as_row() for r in reps])
        st.dataframe(q, hide_index=True, width="stretch", column_config={
            "Completeness": st.column_config.ProgressColumn("Completeness", min_value=0, max_value=100, format="%.1f%%")})
        with st.expander("Details per table"):
            for r in reps:
                if r.filled or r.type_errors or r.duplicates_removed or r.invalid_removed:
                    st.markdown(f"**{TABLE_LABELS[r.table]}** — duplicates removed: {r.duplicates_removed}, "
                                f"invalid rows removed: {r.invalid_removed}, imputed: {r.filled or '—'}, "
                                f"type errors: {r.type_errors or '—'}")
    st.markdown("**Cleaning rules:** exact & primary-key duplicates dropped · dates/numbers coerced (invalid → missing) · "
                "rows missing required fields dropped · non-positive monetary amounts removed · numeric gaps imputed "
                "with the median, categorical gaps with an explicit *Unknown* · booleans normalised (yes/no/1/0/true/false).")

with t2:
    st.markdown("Upload a **CSV or Excel** file. GymMind detects the target table from its columns, validates and cleans it, "
                "then replaces that table in the database — every module recomputes automatically.")
    up = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    if up is not None:
        try:
            raw = read_upload(up)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            st.stop()
        g = guess_table(raw)
        names = list(SCHEMAS)
        target = st.selectbox("Target table", names, index=names.index(g) if g in names else 0,
                              format_func=lambda n: TABLE_LABELS[n])
        st.caption(f"{len(raw):,} rows · {raw.shape[1]} columns" + (f" · auto-detected: {TABLE_LABELS[g]}" if g else ""))
        st.dataframe(raw.head(20), width="stretch")
        cols = set(normalize_columns(raw).columns)
        req = SCHEMAS[target]["required"]
        miss = [c for c in req if c not in cols]
        expected = db.columns(target)
        c1, c2 = st.columns(2)
        c1.markdown("**Required columns:** " + ", ".join(f"`{c}`" + (" ❌" if c in miss else " ✅") for c in req))
        c2.markdown("**All supported columns:** " + ", ".join(f"`{c}`" for c in expected))
        if miss:
            st.error(f"Missing required columns: {', '.join(miss)}. Rename your columns to match the template.")
        elif st.button(f"Validate & import into {TABLE_LABELS[target]}", type="primary"):
            rep = loader.import_table(target, raw, up.name)
            if rep.ok:
                state.clear_all()
                st.success(f"Imported {rep.rows_out:,} clean rows into **{TABLE_LABELS[target]}** "
                           f"({rep.duplicates_removed} duplicates and {rep.invalid_removed} invalid rows removed, "
                           f"{sum(rep.filled.values())} values imputed).")
                st.dataframe(pd.DataFrame([rep.as_row()]), hide_index=True, width="stretch")
            else:
                st.error("; ".join(rep.notes))

with t3:
    st.markdown("Read-only SQL (`SELECT` / `WITH`) against the GymMind SQLite database.")
    examples = {
        "Revenue by plan": "SELECT membership_type, ROUND(SUM(amount),2) AS revenue, COUNT(*) AS payments\nFROM payments WHERE status='Completed'\nGROUP BY membership_type ORDER BY revenue DESC;",
        "Busiest hours": "SELECT CAST(strftime('%H', check_in) AS INT) AS hour, COUNT(*) AS visits\nFROM attendance GROUP BY hour ORDER BY visits DESC LIMIT 10;",
        "Most repaired equipment": "SELECT e.name, COUNT(*) AS repairs, ROUND(SUM(m.cost),0) AS cost\nFROM maintenance m JOIN equipment e USING(equipment_id)\nWHERE m.maintenance_type='Repair' GROUP BY e.name ORDER BY repairs DESC LIMIT 10;",
        "Members per status": "SELECT status, membership_type, COUNT(*) AS n FROM members GROUP BY 1,2 ORDER BY 1,3 DESC;",
    }
    ex = st.selectbox("Example query", list(examples))
    sql = st.text_area("SQL", examples[ex], height=140)
    if st.button("Run query"):
        try:
            res = db.query(sql)
            st.dataframe(res, hide_index=True, width="stretch")
            st.caption(f"{len(res):,} rows")
        except Exception as exc:
            st.error(str(exc))

with t4:
    st.markdown("Regenerate the synthetic demonstration dataset (24 months, ~1,300 members). "
                "This **replaces all tables**, including imported data.")
    c1, c2 = st.columns([1, 3])
    seed = c1.number_input("Random seed", 1, 9999, 42)
    if c1.button("Regenerate demo data", type="primary"):
        with st.spinner("Generating…"):
            from data import generator
            raw = generator.generate(seed=int(seed))
            from utils.cleaning import clean_all
            cleaned, _ = clean_all(raw)
            for name, df in cleaned.items():
                db.write_table(name, df, source=f"demo(seed={seed})")
            state.clear_all()
        st.success("Demo data regenerated.")
    layout.section("CSV templates", "Download a template (first rows of the current table) to prepare your own import")
    T = state.get_data()
    cols = st.columns(3)
    for i, name in enumerate(SCHEMAS):
        buf = io.StringIO()
        T[name].head(25).to_csv(buf, index=False)
        cols[i % 3].download_button(f"{TABLE_LABELS[name]} template", buf.getvalue().encode(), f"{name}_template.csv",
                                    "text/csv", key=f"tpl_{name}", width="stretch")
