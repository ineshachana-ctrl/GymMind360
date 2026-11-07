"""Streamlit-aware data & model access with caching keyed on the database version."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics import equipment as eq_an
from analytics import insights as ins_an
from analytics import kpis
from database import db
from models import churn, forecasting, segmentation
from utils import loader


def data_version() -> str:
    md = db.metadata()
    if md.empty:
        return "empty"
    return f"{md['updated_at'].max()}|{int(md['row_count'].sum())}"


@st.cache_data(show_spinner=False)
def _load(version: str) -> dict[str, pd.DataFrame]:
    return loader.load_all()


def get_data() -> dict[str, pd.DataFrame]:
    return _load(data_version())


def ref_date() -> pd.Timestamp:
    return loader.data_window(get_data())[1]


def window() -> tuple[pd.Timestamp, pd.Timestamp]:
    return loader.data_window(get_data())


def default_start() -> pd.Timestamp:
    """Start of the last 12 complete months."""
    lo, hi = window()
    return max(lo, (hi - pd.DateOffset(months=12)).replace(day=1) + pd.DateOffset(months=1))


def get_filters():
    lo, hi = window()
    start = st.session_state.get("f_start", default_start())
    end = st.session_state.get("f_end", hi)
    plans = st.session_state.get("f_plans", [])
    return pd.Timestamp(start), pd.Timestamp(end), tuple(plans)


@st.cache_data(show_spinner=False)
def _filtered(version: str, start, end, plans):
    return kpis.apply_filters(_load(version), start, end, list(plans) or None)


def get_filtered() -> dict[str, pd.DataFrame]:
    s, e, p = get_filters()
    return _filtered(data_version(), s, e, p)


@st.cache_data(show_spinner=False)
def _monthly(version: str, start, end, plans):
    T = _filtered(version, start, end, plans) if (start is not None) else _load(version)
    return kpis.monthly_table(T, start, end)


def get_monthly(filtered: bool = True) -> pd.DataFrame:
    if filtered:
        s, e, p = get_filters()
        return _monthly(data_version(), s, e, p)
    return _monthly(data_version(), None, None, ())


@st.cache_resource(show_spinner="Training churn model…")
def _churn(version: str):
    T = _load(version)
    return churn.train(T, loader.data_window(T)[1])


def get_churn():
    return _churn(data_version())


@st.cache_resource(show_spinner="Clustering members…")
def _segments(version: str, k):
    T = _load(version)
    return segmentation.segment(T, loader.data_window(T)[1], k=k)


def get_segments(k: int | None = None):
    return _segments(data_version(), k)


@st.cache_resource(show_spinner="Forecasting revenue…")
def _rev_fc(version: str, horizon: int):
    return forecasting.revenue_forecast(_monthly(version, None, None, ()), horizon=horizon)


def get_revenue_forecast(horizon: int = 6):
    return _rev_fc(data_version(), horizon)


@st.cache_resource(show_spinner="Forecasting attendance…")
def _att_fc(version: str, horizon: int):
    return forecasting.attendance_forecast(_load(version)["attendance"], horizon_days=horizon)


def get_attendance_forecast(horizon: int = 28):
    return _att_fc(data_version(), horizon)


@st.cache_data(show_spinner=False)
def _equipment(version: str):
    T = _load(version)
    return eq_an.equipment_summary(T, loader.data_window(T)[1])


def get_equipment_summary() -> pd.DataFrame:
    return _equipment(data_version())


@st.cache_resource(show_spinner="Generating insights…")
def _insights(version: str):
    T = _load(version)
    ref = loader.data_window(T)[1]
    return ins_an.generate_insights(T, _monthly(version, None, None, ()), ref,
                                    churn_scores=_churn(version).scores, equipment=_equipment(version))


def get_insights():
    return _insights(data_version())


def clear_all() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


def get_kpi_context():
    """Data for the KPI cards: history from the first available month up to the selected end date,
    so the reporting month is always compared with the real previous month even when the selected
    period covers a single month."""
    _, e, p = get_filters()
    lo, _ = window()
    v = data_version()
    return _filtered(v, lo, e, p), _monthly(v, lo, e, p)
