"""m22_dcc_threshold_box.py -- (a) DCC and (b) the KS distance of the avalanche-size power-law fit
in low vs high SWR-rate NREM at the 20th / 30th (submitted) / 40th activity percentile, 1x IEI only.

At 1x IEI the three percentiles are all 0 and detect the same avalanches (counts differ
negligibly between thresholds), so the columns are one comparison drawn three times and differ
only through the xmin estimate -- not three tests of the threshold.

Source: the avalanche pipeline of `ripple_criticality_batch.ipynb` (its batch run at PCT = 20, 30
and 40; Extended Data Fig. 5f, g draw the 30th only): bin = 1x the session's mean population
inter-event interval, detection inside NREM only (count > threshold), median split of each
avalanche's local SWR rate (+-5 s), one common xmin for sizes and one for durations, beta over
durations >= that xmin, 16 sessions.
    ripple_criticality_all_sessions.csv        (30th)
    ripple_criticality_all_sessions_pct20.csv  (20th)
    ripple_criticality_all_sessions_pct40.csv  (40th)
Panel b: `ks_xmin_convention.py` (same 16 sessions, `avalanche_ripple.detect`, 1x IEI, NREM only,
median split of local SWR rate), KS distance D of the discrete power-law fit to avalanche sizes with
xmin chosen separately for each group by the Clauset KS search (the `free` rule). Panel a keeps one
xmin per session on the pooled avalanches, because per-group xmin makes the DCC unstable at 1x IEI
(m22_dcc_automin.csv):
    ks_xmin_convention.csv  (bin_mult == 1, column ks_free)

    -> reports/revision_figs/m22_dcc_threshold_box.png / .svg
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                         # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, ".."))
LOG = os.path.join(REPO, "data", "revision", "log")
SRC = {20: "ripple_criticality_all_sessions_pct20.csv", 30: "ripple_criticality_all_sessions.csv",
       40: "ripple_criticality_all_sessions_pct40.csv"}
KS = "ks_xmin_convention.csv"
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
C = {"LOW": "#4575b4", "HIGH": "#d73027"}
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def panel(ax, tables, ylab, log, label):
    for k, pct in enumerate((20, 30, 40)):
        P, x0 = tables[pct], float(k)
        n = len(P)
        for j, grp in enumerate(("LOW", "HIGH")):
            ax.boxplot(P[grp], positions=[x0 + (j - 0.5) * 0.36], widths=0.28, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=C[grp], alpha=0.35, edgecolor=C[grp]),
                       medianprops=dict(color=C[grp], lw=2), whiskerprops=dict(color=C[grp]),
                       capprops=dict(color=C[grp]))
        for _, r in P.iterrows():
            ax.plot([x0 - 0.18, x0 + 0.18], [r.LOW, r.HIGH], "-", color="#999999", lw=0.7, zorder=2)
            ax.scatter([x0 - 0.18, x0 + 0.18], [r.LOW, r.HIGH], s=9, color=[C["LOW"], C["HIGH"]], zorder=3)
        p = stats.wilcoxon(P.HIGH, P.LOW).pvalue
        up = int((P.HIGH > P.LOW).sum())
        ax.text(x0, 1.02, f"{up}/{n}\n{stars(p)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
        print(f"{label} pct{pct}: n = {n}, LOW {P.LOW.median():.4f}, HIGH {P.HIGH.median():.4f}, "
              f"HIGH > LOW {up}/{n}, p = {p:.4f}")
    if log:
        ax.set_yscale("log")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["20th", "30th\n(submitted)", "40th"])
    ax.set_xlabel("activity-percentile threshold")
    ax.set_ylabel(ylab)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    dcc, bins = {}, []
    for pct in (20, 30, 40):
        d = pd.read_csv(os.path.join(LOG, SRC[pct]))
        bins.append(d.groupby("session").bin_ms.first())
        dcc[pct] = d[d.group.isin(["LOW", "HIGH"])].pivot_table(index="session", columns="group", values="DCC").dropna()
    k = pd.read_csv(os.path.join(LOG, KS)); k = k[k.bin_mult == 1]
    ks = {pct: k[k.pct == pct].pivot_table(index="session", columns="group", values="ks_free").dropna()
          for pct in (20, 30, 40)}
    panel(axes[0], dcc, "DCC (deviation from criticality)", True, "DCC")
    panel(axes[1], ks, "KS distance D, avalanche sizes\n(departure from a power law)", False, "KS")
    b = pd.concat(bins).groupby(level=0).first()
    axes[0].set_title("a", loc="left", fontweight="bold", pad=30)
    axes[1].set_title("b", loc="left", fontweight="bold", pad=30)
    fig.suptitle(f"1× IEI (median {b.median():.1f} ms), NREM only, {len(b)} sessions", fontsize=10, y=1.02)
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=C[g], alpha=0.35, edgecolor=C[g]) for g in ("LOW", "HIGH")]
    axes[1].legend(h, ["low SWR rate", "high SWR rate"], frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    fig.text(0.01, -0.09, "At 1× IEI all three percentiles equal 0 and detect the same avalanches: the columns repeat one "
             "comparison and differ only in the estimate of xmin.\nMedian split of each avalanche's local SWR rate "
             "(±5 s). a, one xmin per session on the pooled avalanches; b, xmin fitted to each group. Dots = sessions, "
             "lines pair low and high; numbers = sessions with high > low, signed-rank.",
             fontsize=7.5, color="#555555")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "m22_dcc_threshold_box")
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(f"{out}.{ext}"), dpi=300, bbox_inches="tight")
    print(f"-> {out}.png/.svg")


if __name__ == "__main__":
    main()
