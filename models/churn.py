"""Membership churn prediction.

Training design (no look-ahead leakage):
* Several historical snapshot dates are taken every 45 days.
* For each snapshot, features use only data *before* that date
  (see analytics.features.member_features) and the label is
  "member cancels within the next 90 days".
* Temporal validation: earlier snapshots train, the most recent labelled snapshot tests.
* The final model is refit on all snapshots and scores today's active members.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analytics.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, member_features

HORIZON_DAYS = 90
FEATURE_LABELS = {
    "tenure_days": "Tenure (days)", "visits_30": "Visits last 30d", "visits_90": "Visits last 90d",
    "visit_trend": "Attendance trend", "recency_days": "Days since last visit",
    "avg_duration": "Avg session length", "weekend_share": "Weekend share", "avg_hour": "Usual hour",
    "monthly_fee": "Monthly fee", "extra_spend_90": "Extra spend 90d", "age": "Age",
    "days_to_renewal": "Days to renewal", "auto_renew": "Auto-renew on",
    "membership_type": "Plan", "billing_cycle": "Billing cycle", "visits_per_week": "Visits per week",
}


@dataclass
class ChurnResult:
    model: Pipeline
    metrics: dict
    baseline_metrics: dict
    confusion: np.ndarray
    roc: tuple
    importance: pd.DataFrame
    scores: pd.DataFrame          # current active members with churn probability
    n_train: int
    n_test: int
    positive_rate: float
    threshold: float
    snapshots: list = field(default_factory=list)


def _label(T, as_of: pd.Timestamp, ids: pd.Index) -> pd.Series:
    m = T["members"].set_index("member_id").loc[ids]
    end = as_of + pd.Timedelta(days=HORIZON_DAYS)
    return ((m["churn_date"] > as_of) & (m["churn_date"] <= end)).astype(int)


def build_training_set(T: dict[str, pd.DataFrame], ref: pd.Timestamp, step_days: int = 45):
    first = T["attendance"]["check_in"].min() + pd.Timedelta(days=120)
    last = ref - pd.Timedelta(days=HORIZON_DAYS)
    snaps = []
    d = last
    while d >= first:
        snaps.append(d)
        d -= pd.Timedelta(days=step_days)
    snaps = sorted(snaps)
    frames = []
    for s in snaps:
        f = member_features(T, s)
        if f.empty:
            continue
        f = f[f["tenure_days"] >= 14]
        f["churn"] = _label(T, s, f.index)
        f["snapshot"] = s
        frames.append(f)
    return pd.concat(frames), snaps


def _pipeline(kind: str) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    if kind == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                     class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def _metrics(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    return {"ROC-AUC": roc_auc_score(y, proba), "PR-AUC": average_precision_score(y, proba),
            "Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0)}


def _best_threshold(y, proba) -> float:
    grid = np.linspace(0.15, 0.85, 57)
    f1s = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def train(T: dict[str, pd.DataFrame], ref: pd.Timestamp) -> ChurnResult:
    data, snaps = build_training_set(T, ref)
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    test_snap = data["snapshot"].max()
    tr, te = data[data["snapshot"] < test_snap], data[data["snapshot"] == test_snap]

    model = _pipeline("forest").fit(tr[cols], tr["churn"])
    base = _pipeline("logistic").fit(tr[cols], tr["churn"])
    p_tr = model.predict_proba(tr[cols])[:, 1]
    thr = _best_threshold(tr["churn"], p_tr)
    p = model.predict_proba(te[cols])[:, 1]
    pb = base.predict_proba(te[cols])[:, 1]
    metrics = _metrics(te["churn"], p, thr)
    base_metrics = _metrics(te["churn"], pb, _best_threshold(tr["churn"], base.predict_proba(tr[cols])[:, 1]))
    cm = confusion_matrix(te["churn"], (p >= thr).astype(int), labels=[0, 1])
    fpr, tpr, _ = roc_curve(te["churn"], p)

    # final model on all labelled snapshots
    final = _pipeline("forest").fit(data[cols], data["churn"])
    names = final.named_steps["pre"].get_feature_names_out()
    imp = pd.DataFrame({"feature": names, "importance": final.named_steps["clf"].feature_importances_})
    imp["group"] = imp["feature"].str.replace(r"^(num|cat)__", "", regex=True)
    for c in CATEGORICAL_FEATURES:
        imp.loc[imp["group"].str.startswith(c + "_"), "group"] = c
    imp = imp.groupby("group")["importance"].sum().sort_values(ascending=False).reset_index()
    imp["label"] = imp["group"].map(FEATURE_LABELS).fillna(imp["group"])

    cur = member_features(T, ref)
    cur["churn_probability"] = final.predict_proba(cur[cols])[:, 1]
    cur["risk_band"] = pd.cut(cur["churn_probability"], [-0.01, 0.35, thr if thr > 0.35 else 0.5, 1.0],
                              labels=["Low", "Medium", "High"])
    cur["risk_drivers"] = cur.apply(_drivers, axis=1)
    m = T["members"].set_index("member_id")[["first_name", "last_name", "email"]]
    scores = cur.join(m).reset_index().sort_values("churn_probability", ascending=False)

    return ChurnResult(model=final, metrics=metrics, baseline_metrics=base_metrics, confusion=cm,
                       roc=(fpr, tpr), importance=imp, scores=scores, n_train=len(tr), n_test=len(te),
                       positive_rate=float(data["churn"].mean()), threshold=thr, snapshots=snaps)


def _drivers(r) -> str:
    d = []
    if r["visit_trend"] < 0.75:
        d.append("attendance falling")
    if r["recency_days"] > 14:
        d.append(f"no visit for {int(r['recency_days'])}d")
    if r["visits_per_week"] < 1:
        d.append("<1 visit/week")
    if not r["auto_renew"]:
        d.append("auto-renew off")
    if r["billing_cycle"] == "Monthly" and r["days_to_renewal"] <= 14:
        d.append("renewal due soon")
    return ", ".join(d) if d else "—"
