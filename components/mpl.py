"""Matplotlib / Seaborn styling consistent with the dark navy theme."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from components.theme import MUTED, NAVY_2, PURPLE, TEXT, TURQUOISE  # noqa: E402

CMAP = LinearSegmentedColormap.from_list("gymmind", ["#141B34", "#3B2A7A", PURPLE, TURQUOISE])
CMAP_DIV = LinearSegmentedColormap.from_list("gymmind_div", ["#F87171", "#1C2444", "#34D399"])


def style() -> None:
    sns.set_theme(style="dark", rc={
        "figure.facecolor": NAVY_2, "axes.facecolor": NAVY_2, "savefig.facecolor": NAVY_2,
        "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#243056", "grid.color": "#1E2846", "font.family": "sans-serif",
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": TEXT,
    })


def fig(w: float = 8, h: float = 4):
    style()
    f, ax = plt.subplots(figsize=(w, h), dpi=130)
    return f, ax
