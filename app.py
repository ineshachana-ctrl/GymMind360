"""GymMind 360 — Smart Gym Intelligence. Entry point: `streamlit run app.py`."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import layout, theme  # noqa: E402
from utils import loader, state  # noqa: E402

st.set_page_config(page_title="GymMind 360 · Smart Gym Intelligence", page_icon=str(ROOT / "assets" / "logo.svg"),
                   layout="wide", initial_sidebar_state="expanded")
theme.inject()
st.logo(str(ROOT / "assets" / "logo_full.svg"), size="large", icon_image=str(ROOT / "assets" / "logo.svg"))

if "booted" not in st.session_state:
    with st.spinner("Preparing GymMind 360 — generating demo data and building the database…"):
        loader.bootstrap()
    st.session_state["booted"] = True

P = ROOT / "views"
pages = {
    "Overview": [
        st.Page(P / "1_Executive_Dashboard.py", title="Executive Dashboard", icon=":material/space_dashboard:", default=True),
        st.Page(P / "7_Smart_Insights.py", title="Smart Insights", icon=":material/tips_and_updates:"),
    ],
    "Business Analytics": [
        st.Page(P / "2_Member_Intelligence.py", title="Member Intelligence", icon=":material/groups:"),
        st.Page(P / "3_Revenue_Analytics.py", title="Revenue Analytics", icon=":material/payments:"),
        st.Page(P / "4_Smart_Occupancy.py", title="Smart Occupancy", icon=":material/schedule:"),
        st.Page(P / "5_Equipment_Intelligence.py", title="Equipment Intelligence", icon=":material/fitness_center:"),
    ],
    "AI & Performance": [
        st.Page(P / "6_Predictive_Analytics.py", title="Predictive Analytics", icon=":material/insights:"),
        st.Page(P / "8_Member_Fitness.py", title="Member Fitness", icon=":material/monitor_heart:"),
    ],
    "Data": [
        st.Page(P / "9_Data_Management.py", title="Data Management", icon=":material/database:"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    lo, hi = state.window()
    default_start = state.default_start()
    st.markdown("**Analysis filters**")
    rng = st.date_input("Period", value=(st.session_state.get("f_start", default_start).date(),
                                         st.session_state.get("f_end", hi).date()),
                        min_value=lo.date(), max_value=hi.date(), format="DD/MM/YYYY")
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        st.session_state["f_start"], st.session_state["f_end"] = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    plans = sorted(state.get_data()["members"]["membership_type"].dropna().unique())
    st.session_state["f_plans"] = st.multiselect("Membership types", plans, default=st.session_state.get("f_plans", []),
                                                 placeholder="All membership types")
    st.caption("Filters apply to descriptive modules. Predictive models and insights always use the full history.")
    st.divider()
    st.caption(f"Data through **{hi:%d %b %Y}** · v1.0")

nav.run()
