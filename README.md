# GymMind 360 — Smart Gym Intelligence

A Business Intelligence platform for gyms and fitness centres, built with **Python, Pandas, Streamlit, Plotly, Scikit-learn and SQLite**.
It helps managers understand membership, revenue, occupancy and equipment, predict churn, revenue and attendance, and act on automatically generated, explainable recommendations.

---

## Quick start (Windows / PowerShell)

```powershell
cd C:\Users\hacha\gymmind360
py -3.14 -m venv .venv            # any Python ≥ 3.11 works
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8501**. On first launch it generates the demo dataset and builds the SQLite database automatically (≈ 10 s).

Optional commands:

```powershell
python -m data.generator          # regenerate the demo CSV files in data/
python -m pytest tests -q         # run the 45 unit, feature and end-to-end tests (~4 min)
```

> macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate`, then the same commands.

---

## Modules

| # | Module | What it shows |
|---|---|---|
| 1 | **Executive Dashboard** | 12 KPI cards (total / active / inactive members, monthly revenue, renewal rate, average attendance, member growth, expenses, estimated profit, churn, ARPU, registrations) with % change vs previous month and sparklines; revenue vs expenses; member flow; top priorities |
| 2 | **Member Intelligence** | Registrations by plan, demographics, cohort retention heatmap (Seaborn), churn by plan / billing cycle / auto-renew, attendance frequency, membership duration, **K-Means segmentation** (PCA view, profiles) |
| 3 | **Revenue Analytics** | Revenue by month & stream, MoM growth, profit & margin, revenue by membership type, payment method and member segment, expense breakdown, failed-payment rate, exportable financial table |
| 4 | **Smart Occupancy** | Weekday × hour heatmap (people on the floor, check-ins, occupancy %), peak / quiet hours, weekday profile, trends, **predicted hourly attendance for the next 7 days** |
| 5 | **Equipment Intelligence** | Utilisation per unit, popularity treemap, maintenance cost & history, downtime, health score (0–100), performance indicators, automatic *needs attention* flags, unit drill-down |
| 6 | **Predictive Analytics** | Churn prediction (Random Forest), revenue forecast (Ridge), attendance forecast (Gradient Boosting, recursive), K-Means segmentation — all with evaluation metrics and baselines |
| 7 | **Smart Insights** | Rule-based recommendation engine: each card shows evidence, *why it was generated* (rule + threshold + observed value) and a recommended action; CSV / Markdown export |
| 8 | **Member Fitness** | Opt-in workout analytics: training calendar, frequency, consistency, estimated 1RM progression, weekly volume, perceived effort — explicitly non-medical |
| 9 | **Data Management** | CSV / Excel import with validation & cleaning, data-quality report, read-only SQL explorer, demo-data regeneration, CSV templates |

Sidebar filters (period, membership types) apply to descriptive modules; predictive models and insights always use the full history. KPI cards always compare the reporting month with the true previous month. When a membership-type filter is active, gym-wide expenses are allocated pro rata to the selected plans' share of active members, so profit stays meaningful.

---

## Architecture

```
gymmind360/
├── app.py                    # entry point: theme, navigation, sidebar filters, data bootstrap
├── views/                    # one file per module (1–9), routed by st.navigation in app.py
├── components/               # theme (CSS + Plotly template), KPI cards, chart builders, layout, Matplotlib style
├── analytics/                # pure Pandas analytics: features, kpis, members, revenue, occupancy, equipment, fitness, insights
├── models/                   # Scikit-learn: churn, segmentation, forecasting
├── data/                     # generator.py + generated CSV files
├── database/                 # SQLite layer (db.py, schema.sql) → gymmind.db
├── utils/                    # cleaning/validation, loader, Streamlit cache/state, formatting
├── assets/                   # logos
├── tests/                    # pytest: data pipeline, analytics, models, insights, page smoke tests
└── .streamlit/config.toml    # dark theme
```

**Data flow:** `generator → CSV → cleaning (validation, dedupe, imputation) → SQLite → loader → analytics / models → pages`.
Streamlit caches are keyed on the database version, so an import automatically refreshes every module.

---

## Data

Demo data is **synthetic and reproducible** (seed 42): 24 months (Nov 2024 – Oct 2026), ~1,350 members, ~160k visits, ~11k payments, 38 equipment units, ~80k workout sets.

| File | Content |
|---|---|
| `members.csv` | profile, plan, billing cycle, auto-renew, consent, status, churn date |
| `subscriptions.csv` | every billing period with price and outcome (Renewed / Cancelled / Active) |
| `attendance.csv` | check-in / check-out per visit |
| `payments.csv` | membership, personal-training and shop payments, method, status |
| `equipment.csv` | units, type, zone, purchase cost, service interval |
| `equipment_usage.csv` | daily usage minutes and sessions per unit |
| `maintenance.csv` | preventive and corrective interventions with cost and downtime |
| `workouts.csv` | exercise logs of consenting members (sets, reps, load, cardio distance, RPE) |
| `expenses.csv` | monthly operating costs by category |

The generator **plants realistic signals** for the analytics to discover: morning, lunch and evening peaks; January and September seasonality; churn linked to falling attendance, monthly billing and disabled auto-renew; a revenue dip in the latest month; one under-used rowing machine, one failure-prone treadmill and one machine whose maintenance was skipped. It also injects **dirty data** (duplicates, missing values, negative amounts) that the cleaning pipeline removes and reports.

### Importing your own data
In **Data Management → Import data**, upload a CSV or Excel file. The target table is detected from its columns (headers such as `Payment Date` are normalised to `payment_date`). Required columns are checked, types are coerced, duplicates and invalid rows are removed and gaps are imputed, then the table is replaced in SQLite. Templates for every table can be downloaded from the same page.

---

## Machine-learning methodology

| Model | Approach | Validation | Typical result on demo data |
|---|---|---|---|
| Churn | Random Forest on point-in-time features (visits 30/90 d, trend, recency, tenure, plan, billing cycle, auto-renew, days to renewal…); label = cancels within 90 days; 12 historical snapshots | Temporal: train on earlier snapshots, test on the most recent one; baseline = logistic regression | ROC-AUC ≈ 0.79 (baseline 0.75) |
| Segmentation | StandardScaler + K-Means; k chosen by silhouette; segments named from their most distinctive standardised trait | Silhouette, Davies-Bouldin, Calinski-Harabasz | 5–6 segments |
| Revenue forecast | Ridge regression on trend + Fourier seasonality + peak-month flags | Last 6 months held out; baseline = same month last year | MAPE ≈ 14% (baseline 31%) |
| Attendance forecast | Gradient Boosting on calendar + lag/level features, recursive multi-step | Last 28 days held out, recursive; baseline = repeat last week | R² ≈ 0.91, MAPE ≈ 6% (baseline 5.6%) |

The weekly attendance rhythm is very regular, so the seasonal-naive baseline is competitive over short horizons, and the app says so. Predictions are always visually distinct from real data: solid turquoise = actual, dashed purple = forecast with a shaded 95% interval, dotted amber = back-test on held-out history.

---

## Privacy & ethics
* The fitness module only lists members with `data_consent = True`.
* Fitness metrics (estimated 1RM, pace, consistency) are training indicators. They are **not medical or health assessments**.
* All data is local (CSV + SQLite). No external API or cloud service is used.

## Tech stack
Python · Pandas · NumPy · Streamlit · Plotly · Scikit-learn · SQLite · Matplotlib · Seaborn · openpyxl · pytest
