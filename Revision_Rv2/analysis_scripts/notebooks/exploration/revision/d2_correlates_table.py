"""
d2_correlates_table.py -- per-session correlation table of d2 with window metrics,
extended with E/I balance and synchrony.

Per-session Pearson r between d2 (2024 Euclidean, p=20; published and ripple-excised
arms) and each window metric, Fisher-averaged over sessions, with the number of sessions
positive and a signed-rank p, on one common window set so every row is comparable:

    windows   Fig 1's non-overlapping NREM tiles (fig1_metrics pickles), as d2_composition_glm.py
    metrics   MUA_mean, MUA_std, MUA_fano, EI_logratio, synchrony (raw / 81-unit-subsampled,
              mean / SD, 50 ms), isi_entropy_pooled           -- from the pickles
              ripple_rate, ripple_amp, n_cells, replay_prob    -- r35_swr_property_control.build_events,
                                                                  8 decoded sessions only
    arms      published (18 sessions), excised (10 sessions with *_rippleablate.mat)

Reads the shares read-only; writes only to D:.
    -> data/revision/log/d2_correlates_table.csv      (per session x arm x metric, r)
    -> data/revision/log/d2_correlates_table.md       (the table, ready to paste)
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path          # noqa: E402
from r35_glm_ablation import LOGDIR                                     # noqa: E402
from r35_swr_property_control import SESSION_TEMPLATE                   # noqa: E402
from d2_composition_glm import load_session, add_swr, fig2_basepaths                    # noqa: E402

ROWS = [
    ("MUA_mean", "MUA (mean rate)"),
    ("MUA_std", "MUA std (rate variability)"),
    ("MUA_fano", "MUA Fano factor"),
    ("EI_logratio", "E/I balance (log E/I rate)"),
    ("synchrony_raw_mean", "synchrony, mean (raw)"),
    ("synchrony_raw_std", "synchrony, SD (raw)"),
    ("synchrony_sub_mean", "synchrony, mean (81-unit subsampled)"),
    ("synchrony_sub_std", "synchrony, SD (81-unit subsampled)"),
    ("isi_entropy_pooled", "ISI entropy"),
    ("ripple_rate", "SWR rate"),
    ("ripple_amp", "ripple amplitude"),
    ("ripple_dur", "ripple duration"),
    ("n_cells", "participating cells"),
    ("replay_prob", "replay probability"),
]
ARMS = [("d2_published", "Euclidean published"), ("d2_excised", "Euclidean excised")]


def fisher(v):
    z = np.arctanh(np.clip(np.asarray(v, float), -0.999999, 0.999999))
    return float(np.tanh(z.mean()))


def main():
    rows = []
    for bp in fig2_basepaths():
        sid, W = load_session(bp)
        if W is None:
            continue
        if sid in SESSION_TEMPLATE:
            W = add_swr(W, sid, bp)
        for arm, _ in ARMS:
            if W[arm].notna().mean() < 0.5:
                continue
            for col, _ in ROWS:
                if col not in W:
                    continue
                sub = W[[arm, col]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(sub) < 40:
                    continue
                rows.append(dict(session=sid, arm=arm, metric=col,
                                 r=float(np.corrcoef(sub[arm], sub[col])[0, 1]), n=len(sub)))
    R = pd.DataFrame(rows)
    assert_local_path(LOGDIR)
    R.to_csv(assert_local_path(os.path.join(LOGDIR, "d2_correlates_table.csv")), index=False)

    lines = ["| predictor | " + " | ".join(lab for _, lab in ARMS) + " |",
             "|---|" + "---|" * len(ARMS)]
    for col, lab in ROWS:
        cells = []
        for arm, _ in ARMS:
            v = R[(R.arm == arm) & (R.metric == col)].r.to_numpy()
            if v.size < 3:
                cells.append("—")
                continue
            p = stats.wilcoxon(v).pvalue
            star = "**" if p < 0.01 else ""
            cells.append(f"{star}{fisher(v):+.3f}{star} ({int((v > 0).sum())}/{v.size}, p = {p:.3f})")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    with open(assert_local_path(os.path.join(LOGDIR, "d2_correlates_table.md")), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print("\nPer-session Pearson r with d2, Fisher-averaged (sessions positive / N, signed-rank p)\n")
    print(md)
    # four-row subset: MUA, replay probability, amplitude, duration
    keep = {lab for c, lab in ROWS if c in ("MUA_mean", "replay_prob", "ripple_amp", "ripple_dur")}
    print("\nFour-row table (8 decoded sessions only, same windows):\n")
    print("\n".join(lines[:2] + [l for l in lines[2:] if l.split("|")[1].strip() in keep]))


def plot(arm="d2_excised", rows=("MUA_mean", "replay_prob", "ripple_amp", "ripple_dur")):
    """Fig 2F-style panel of the per-session r: one box per predictor, session points,
    signed-rank stars. Standardised slope == r for a single z-scored predictor."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from r35_glm_ablation import FIGDIR
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    R = pd.read_csv(os.path.join(LOGDIR, "d2_correlates_table.csv"))
    labels = {"MUA_mean": "MUA", "ripple_rate": "SWR rate", "replay_prob": "replay prob.",
              "ripple_amp": "ripple amp.", "ripple_dur": "ripple dur.", "n_cells": "cells / SWR",
              "EI_logratio": "E/I", "MUA_std": "MUA std", "MUA_fano": "MUA Fano",
              "synchrony_raw_mean": "sync. mean", "synchrony_raw_std": "sync. SD",
              "isi_entropy_pooled": "ISI entropy"}
    order = [labels[c] for c in rows]
    P = R[(R.arm == arm) & (R.metric.isin(rows))].copy()
    P["epoch"] = pd.Categorical(P.metric.map(labels), categories=order, ordered=True)
    col = ["#c03d3e", "#d4595a", "#e57373", "#f3a6a6"] if arm == "d2_excised" else ["#4d4d4d", "#6b6b6b", "#8a8a8a", "#a8a8a8"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.boxplot(data=P, x="epoch", y="r", order=order, ax=ax, width=0.6, showfliers=False,
                palette=dict(zip(order, col[:len(order)])), linewidth=2, hue="epoch", legend=False)
    rng = np.random.default_rng(0)
    yr = P.r.max() - P.r.min()
    for i, lab in enumerate(order):
        v = P[P.epoch == lab].r.to_numpy()
        ax.plot(i + rng.uniform(-0.12, 0.12, v.size), v, "o", ms=3.5, color="k", alpha=0.6, mew=0, zorder=3)
        pw = stats.wilcoxon(v).pvalue
        star = "***" if pw < .001 else "**" if pw < .01 else "*" if pw < .05 else "n.s."
        ax.text(i, v.max() + 0.04 * yr, star, ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.set_ylabel("r with d2 (ripple-excised)\nper session" if arm == "d2_excised"
                  else "r with d2 (published)\nper session")
    ax.set_xlabel("")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    tag = arm.replace("d2_", "")
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"d2_correlates_{tag}.{ext}")), dpi=300)
    plt.close(fig)
    print(f"  -> {FIGDIR}\d2_correlates_{tag}.png/.svg")


if __name__ == "__main__":
    if not os.environ.get("PLOT_ONLY"):
        main()
    plot("d2_excised")
    plot("d2_published")
