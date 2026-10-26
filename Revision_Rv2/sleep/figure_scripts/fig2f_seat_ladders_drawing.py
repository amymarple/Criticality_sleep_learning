"""Drawing functions from analysis_scripts/revision/ (fig2f_seat_ladders.py) used by the figure notebooks.
FIGDIR: where the figures drawn with them would be saved (the notebooks capture the figures instead).
Importing this module makes the matplotlib settings fig2f_seat_ladders.py makes when it is imported.
"""
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = "."


# --- the matplotlib settings fig2f_seat_ladders.py makes when it is imported
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


# --- fig2f_seat_ladders.py
def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# --- fig2f_seat_ladders.py
def box_panel(ax, data, labels, colors, ylabel=None, title=None, rng_seed=0):
    bp = ax.boxplot(data, positions=range(len(data)), widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#333333", lw=1.6), whiskerprops=dict(color="k"), capprops=dict(color="k"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_edgecolor("k"); patch.set_alpha(0.9)
    rng = np.random.default_rng(rng_seed)
    yr = max(np.max(np.concatenate(data)) - np.min(np.concatenate(data)), 1e-3)
    for i, b in enumerate(data):
        ax.scatter(rng.uniform(-0.15, 0.15, len(b)) + i, b, s=10, color="#444444", zorder=3)
        p = stats.wilcoxon(b).pvalue
        ax.text(i, b.max() + 0.04 * yr, f"{int((b > 0).sum())}/{len(b)}\n{stars(p)}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", linespacing=1.1)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(data))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.12 * yr)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")
