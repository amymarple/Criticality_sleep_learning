"""m25_tmaze_box.py -- T-maze alternation performance of every Figs 1-3 session that has scored
trials, against chance.

Reads m25_task_performance.csv (m25_task_performance.py: fraction of behavior.trialcorrect == 1 per task block).
Sessions: every row with scored trials -- 17 PPP sessions, 6 mice. Not included: PPP7_day7 (forced T-maze, no
choice), PPP7_day10, NN16_day13 and NN8_day7 (T-maze epochs without trial scoring), and the box / sleep-only NN
sessions. Per session, all T-maze trials of the day pooled (both blocks; two-maze days included).
Chance = 0.5 (two-arm choice). Tests: one-sample Wilcoxon signed-rank of the session fractions against 0.5
(session = unit), and a one-sided binomial test of k correct out of n per session.

    -> data/revision/log/m25_tmaze_sessions.csv
    -> reports/revision_figs/m25_tmaze_performance.png / .svg
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
from run_fig1_metrics import assert_local_path                                     # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOG = os.path.join(REPO, "data", "revision", "log")
OUT_CSV = os.path.join(LOG, "m25_tmaze_sessions.csv")
OUT = os.path.join(REPO, "reports", "revision_figs", "m25_tmaze_performance")
MOUSE_COL = {"PPP4": "#8c564b", "PPP7": "#d62728", "PPP8": "#9467bd", "PPP20": "#1f77b4", "PPP21": "#ff7f0e",
             "PPP23": "#2ca02c"}
# Animals are labelled Mouse 1-6 rather than by ID, numbered in ID order.
MOUSE_LABEL = {"PPP4": "Mouse 1", "PPP7": "Mouse 2", "PPP8": "Mouse 3", "PPP20": "Mouse 4", "PPP21": "Mouse 5",
               "PPP23": "Mouse 6"}
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})


def main():
    D = pd.read_csv(os.path.join(LOG, "m25_task_performance.csv"))
    D = D[D.n_trials.fillna(0) > 0].copy()
    D["n_correct"] = np.round(D.frac_correct * D.n_trials).astype(int)
    S = D.groupby(["session", "animal"], as_index=False).agg(n_trials=("n_trials", "sum"), n_correct=("n_correct", "sum"),
                                                             two_mazes=("two_mazes", "first"), figures=("figures", "first"))
    S["frac_correct"] = S.n_correct / S.n_trials
    S["p_binom"] = [stats.binomtest(int(k), int(n), 0.5, alternative="greater").pvalue
                    for k, n in zip(S.n_correct, S.n_trials)]
    S.to_csv(assert_local_path(OUT_CSV), index=False)
    w = stats.wilcoxon(S.frac_correct - 0.5, alternative="greater")
    n_sig = int((S.p_binom < 0.05).sum())
    print(S.round(3).to_string(index=False))
    print(f"\n{len(S)} sessions, {S.animal.nunique()} mice; fraction correct median {S.frac_correct.median():.3f} "
          f"(range {S.frac_correct.min():.2f}-{S.frac_correct.max():.2f}); above chance in {int((S.frac_correct > 0.5).sum())}/{len(S)}; "
          f"signed-rank vs 0.5 p = {w.pvalue:.2g}; binomial p < 0.05 in {n_sig}/{len(S)} sessions; "
          f"trials per session {int(S.n_trials.min())}-{int(S.n_trials.max())}")
    M = S.groupby("animal").agg(frac=("frac_correct", "mean"), n=("session", "size"))
    M.index = [MOUSE_LABEL[a] for a in M.index]
    M = M.sort_index()
    print("per mouse:", ", ".join(f"{a} {r.frac:.2f} ({int(r.n)} sessions)" for a, r in M.iterrows()))

    fig, ax = plt.subplots(figsize=(3.0, 4.2))
    ax.boxplot(S.frac_correct, positions=[0], widths=0.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="0.9", edgecolor="0.3"), medianprops=dict(color="k", lw=2))
    rng = np.random.default_rng(1)
    for a in sorted(S.animal.unique(), key=lambda x: MOUSE_LABEL[x]):
        g = S[S.animal == a]
        ax.scatter(rng.uniform(-0.15, 0.15, len(g)), g.frac_correct, s=28, color=MOUSE_COL.get(a, "0.5"),
                   edgecolor="white", lw=0.5, zorder=3, label=MOUSE_LABEL[a])
    ax.axhline(0.5, color="0.4", ls="--", lw=1)
    ax.text(0.38, 0.505, "chance", fontsize=8, color="0.4", va="bottom", ha="right")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(0.4, 1.0); ax.set_xticks([])
    ax.set_ylabel("T-maze alternation, fraction correct")
    star = "***" if w.pvalue < 0.001 else "**" if w.pvalue < 0.01 else "*" if w.pvalue < 0.05 else "n.s."
    ax.set_title(f"{len(S)} sessions, {S.animal.nunique()} mice\nvs chance: {star} (p = {w.pvalue:.1g})", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2, handletextpad=0.2, columnspacing=0.6)
    fig.text(0.02, -0.04, f"Dots = sessions (all T-maze trials of the day, {int(S.n_trials.min())}-{int(S.n_trials.max())} "
             f"per session); signed-rank vs 0.5;\nbinomial p < 0.05 in {n_sig} of {len(S)} sessions.", fontsize=7, color="0.35")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(f"{OUT}.{ext}"), dpi=250, bbox_inches="tight")
    print(f"-> {OUT}.png/.svg")


if __name__ == "__main__":
    main()
