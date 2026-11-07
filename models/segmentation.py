"""Behavioural member segmentation with K-Means."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from analytics.features import member_features

SEG_FEATURES = ["visits_per_week", "avg_duration", "avg_hour", "weekend_share", "tenure_months",
                "monthly_value", "visit_trend"]
SEG_LABELS = {"visits_per_week": "Visits / week", "avg_duration": "Session (min)",
              "avg_hour": "Usual hour", "weekend_share": "Weekend share",
              "tenure_months": "Tenure (months)", "monthly_value": "Monthly value",
              "visit_trend": "Attendance trend"}


@dataclass
class SegmentationResult:
    data: pd.DataFrame            # per member: features, cluster, segment, pca_x, pca_y
    profiles: pd.DataFrame        # mean feature values per segment
    k: int
    k_scores: pd.DataFrame        # silhouette / inertia per k
    metrics: dict
    explained_variance: float


def _features(T, ref) -> pd.DataFrame:
    f = member_features(T, ref)
    f = f[f["tenure_days"] >= 21].copy()
    f["tenure_months"] = f["tenure_days"] / 30.44
    f["monthly_value"] = f["monthly_fee"] + f["extra_spend_90"] / 3
    return f


NAME_MAP = {
    ("visits_per_week", 1): "Power Users", ("visits_per_week", -1): "Low Engagers",
    ("avg_hour", -1): "Early Birds", ("avg_hour", 1): "Evening Regulars",
    ("weekend_share", 1): "Weekend Warriors", ("weekend_share", -1): "Weekday Regulars",
    ("tenure_months", 1): "Loyal Veterans", ("tenure_months", -1): "New Joiners",
    ("monthly_value", 1): "Premium Spenders", ("monthly_value", -1): "Budget Members",
    ("visit_trend", 1): "Rising Engagement", ("visit_trend", -1): "Fading Members",
    ("avg_duration", 1): "Long-Session Trainers", ("avg_duration", -1): "Quick Workouts",
}


def _name_segments(centroids_z: pd.DataFrame) -> dict[int, str]:
    """Name each cluster after its most distinctive standardised trait (unique names).

    Clusters are processed by the strength of their strongest trait so the most
    clear-cut segments claim names first.
    """
    names: dict[int, str] = {}
    used: set[str] = set()
    order = centroids_z.abs().max(axis=1).sort_values(ascending=False).index
    for c in order:
        z = centroids_z.loc[c]
        for feat in z.abs().sort_values(ascending=False).index:
            name = NAME_MAP.get((feat, int(np.sign(z[feat]) or 1)))
            if name and name not in used:
                names[c] = name
                used.add(name)
                break
        else:
            names[c] = f"Segment {c + 1}"
    return names


def segment(T: dict[str, pd.DataFrame], ref: pd.Timestamp, k: int | None = None,
            k_range=range(3, 8)) -> SegmentationResult:
    f = _features(T, ref)
    X = StandardScaler().fit_transform(f[SEG_FEATURES])
    scores = []
    for kk in k_range:
        km = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(X)
        scores.append({"k": kk, "silhouette": silhouette_score(X, km.labels_), "inertia": km.inertia_})
    k_scores = pd.DataFrame(scores)
    if k is None:
        # best silhouette, preferring 4+ segments when the score is within 5% (more actionable)
        best = k_scores["silhouette"].max()
        cand = k_scores[(k_scores["silhouette"] >= best * 0.95) & (k_scores["k"] >= 4)]
        k = int(cand["k"].min()) if not cand.empty else int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
    f["cluster"] = km.labels_
    zc = pd.DataFrame(X, columns=SEG_FEATURES, index=f.index).groupby(f["cluster"]).mean()
    names = _name_segments(zc)
    f["segment"] = f["cluster"].map(names)
    pca = PCA(n_components=2, random_state=42).fit(X)
    xy = pca.transform(X)
    f["pca_x"], f["pca_y"] = xy[:, 0], xy[:, 1]
    profiles = f.groupby("segment")[SEG_FEATURES].mean()
    profiles["members"] = f.groupby("segment").size()
    profiles["share"] = profiles["members"] / len(f)
    metrics = {"Silhouette": silhouette_score(X, km.labels_),
               "Davies-Bouldin": davies_bouldin_score(X, km.labels_),
               "Calinski-Harabasz": calinski_harabasz_score(X, km.labels_)}
    return SegmentationResult(data=f.reset_index(), profiles=profiles.sort_values("members", ascending=False),
                              k=k, k_scores=k_scores, metrics=metrics,
                              explained_variance=float(np.sum(pca.explained_variance_ratio_)))
