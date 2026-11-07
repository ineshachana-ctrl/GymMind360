"""Revenue and attendance forecasting.

* Revenue: Ridge regression on trend + annual seasonality (Fourier terms) + peak-season flags,
  validated on the last 4 months (time-ordered holdout), then refit on everything.
* Attendance: Gradient Boosting on calendar features (weekday, month, seasonality, trend),
  validated on the last 8 weeks and compared with a seasonal-naive baseline
  ("same weekday last week"). Hourly forecasts distribute the daily forecast with the
  historical weekday x hour profile.

Prediction intervals are residual-based (+/-1.96 x holdout RMSE).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


@dataclass
class ForecastResult:
    history: pd.DataFrame       # date, actual, fitted
    test: pd.DataFrame          # date, actual, predicted (holdout)
    forecast: pd.DataFrame      # date, predicted, lower, upper
    metrics: dict
    baseline_metrics: dict
    model_name: str


def _metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE": mean_absolute_percentage_error(y, p), "R²": r2_score(y, p) if len(y) > 1 else np.nan}


# ------------------------------------------------------------------------- revenue
def _rev_X(dates: pd.DatetimeIndex, t0: pd.Timestamp) -> pd.DataFrame:
    t = (dates.year - t0.year) * 12 + (dates.month - t0.month)
    m = dates.month
    return pd.DataFrame({"t": t, "sin1": np.sin(2 * np.pi * m / 12), "cos1": np.cos(2 * np.pi * m / 12),
                         "sin2": np.sin(4 * np.pi * m / 12), "cos2": np.cos(4 * np.pi * m / 12),
                         "jan": (m == 1).astype(int), "sep": (m == 9).astype(int),
                         "summer": m.isin([7, 8]).astype(int)}, index=dates)


def revenue_forecast(monthly: pd.DataFrame, horizon: int = 6, test_months: int = 6) -> ForecastResult:
    s = monthly.set_index("month")["revenue"].astype(float)
    dates = pd.DatetimeIndex(s.index)
    t0 = dates[0]
    X = _rev_X(dates, t0)
    tr, te = slice(0, len(s) - test_months), slice(len(s) - test_months, len(s))
    model = Ridge(alpha=1.0).fit(X.iloc[tr], s.iloc[tr])
    p_te = model.predict(X.iloc[te])
    metrics = _metrics(s.iloc[te], p_te)
    # baseline: same month last year if available, else last value
    naive = s.shift(12).iloc[te].fillna(s.iloc[tr].iloc[-1])
    base_metrics = _metrics(s.iloc[te], naive)

    final = Ridge(alpha=1.0).fit(X, s)
    fitted = final.predict(X)
    fut = pd.date_range(dates[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    pf = final.predict(_rev_X(fut, t0))
    band = 1.96 * max(metrics["RMSE"], np.std(s - fitted))
    return ForecastResult(
        history=pd.DataFrame({"date": dates, "actual": s.to_numpy(), "fitted": fitted}),
        test=pd.DataFrame({"date": dates[te], "actual": s.iloc[te].to_numpy(), "predicted": p_te}),
        forecast=pd.DataFrame({"date": fut, "predicted": pf, "lower": pf - band, "upper": pf + band}),
        metrics=metrics, baseline_metrics=base_metrics, model_name="Ridge regression (trend + seasonality)")


# ---------------------------------------------------------------------- attendance
ATT_FEATURES = ["dow", "weekend", "sin_y", "cos_y", "dom", "lag_7", "lag_14", "level_28", "level_7"]
GB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42)


def _calendar(d: pd.Timestamp) -> dict:
    doy = d.day_of_year
    return {"dow": d.day_of_week, "weekend": int(d.day_of_week >= 5), "dom": d.day,
            "sin_y": np.sin(2 * np.pi * doy / 365.25), "cos_y": np.cos(2 * np.pi * doy / 365.25)}


def _row(d: pd.Timestamp, hist: np.ndarray) -> dict:
    """Features for day `d` given the history array of previous daily counts."""
    r = _calendar(d)
    r["lag_7"], r["lag_14"] = hist[-7], hist[-14]
    r["level_28"], r["level_7"] = hist[-28:].mean(), hist[-7:].mean()
    return r


def _att_frame(s: pd.Series) -> pd.DataFrame:
    vals = s.to_numpy()
    rows = [_row(d, vals[:i]) for i, d in enumerate(s.index) if i >= 28]
    return pd.DataFrame(rows, index=s.index[28:])


def _recursive(model, hist: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Multi-step forecast: each prediction is fed back as history for the next day."""
    h = list(hist)
    out = []
    for d in dates:
        x = pd.DataFrame([_row(d, np.asarray(h))])[ATT_FEATURES]
        p = float(model.predict(x)[0])
        out.append(p)
        h.append(p)
    return np.array(out)


def daily_visits(attendance: pd.DataFrame) -> pd.Series:
    d = attendance.groupby(attendance["check_in"].dt.normalize()).size()
    return d.reindex(pd.date_range(d.index.min(), d.index.max()), fill_value=0).astype(float)


def attendance_forecast(attendance: pd.DataFrame, horizon_days: int = 28, test_days: int = 28) -> ForecastResult:
    s = daily_visits(attendance)
    dates = pd.DatetimeIndex(s.index)
    X = _att_frame(s)[ATT_FEATURES]
    y = s.loc[X.index]
    cut = len(s) - test_days
    train_idx = X.index[X.index < dates[cut]]
    model = GradientBoostingRegressor(**GB_PARAMS).fit(X.loc[train_idx], y.loc[train_idx])
    # honest multi-step evaluation: forecast the whole holdout recursively from the cut-off
    p_te = _recursive(model, s.to_numpy()[:cut], dates[cut:])
    metrics = _metrics(s.iloc[cut:], p_te)
    # baseline: repeat the last observed week (seasonal naive, multi-step)
    last_week = s.to_numpy()[cut - 7:cut]
    naive = np.resize(last_week, test_days)
    base_metrics = _metrics(s.iloc[cut:], naive)

    final = GradientBoostingRegressor(**GB_PARAMS).fit(X, y)
    fitted = pd.Series(final.predict(X), index=X.index).reindex(dates).to_numpy()
    fut = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    pf = _recursive(final, s.to_numpy(), fut)
    band = 1.96 * metrics["RMSE"]
    return ForecastResult(
        history=pd.DataFrame({"date": dates, "actual": s.to_numpy(), "fitted": fitted}),
        test=pd.DataFrame({"date": dates[cut:], "actual": s.iloc[cut:].to_numpy(), "predicted": p_te}),
        forecast=pd.DataFrame({"date": fut, "predicted": pf, "lower": np.maximum(pf - band, 0), "upper": pf + band}),
        metrics=metrics, baseline_metrics=base_metrics,
        model_name="Gradient Boosting (calendar + lag features, recursive)")


def hourly_forecast(attendance: pd.DataFrame, daily_fc: pd.DataFrame, weeks_profile: int = 12) -> pd.DataFrame:
    """Expected arrivals per hour for each forecast day (daily forecast x weekday-hour share)."""
    recent = attendance[attendance["check_in"] > attendance["check_in"].max() - pd.Timedelta(weeks=weeks_profile)]
    g = recent.groupby([recent["check_in"].dt.day_of_week, recent["check_in"].dt.hour]).size()
    share = g / g.groupby(level=0).transform("sum")
    rows = []
    for r in daily_fc.itertuples(index=False):
        dow = r.date.day_of_week
        if dow not in share.index.get_level_values(0):
            continue
        for h, sh in share.loc[dow].items():
            rows.append({"date": r.date, "hour": int(h), "expected_arrivals": r.predicted * sh})
    return pd.DataFrame(rows)
