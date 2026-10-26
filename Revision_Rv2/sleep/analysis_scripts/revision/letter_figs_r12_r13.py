"""
letter_figs_r12_r13.py -- two figures drawn from stored tables.

r12_order_selection.png -- AR order against residual whiteness: the fraction of non-overlapping 30 s windows
    whose AR(p) residuals are white (Ljung-Box, h = 50 lags, df = h - p, p > 0.05), by state and order
    (ljungbox_df_audit.csv, 12 345 windows x 6 orders). Extended Data Fig. 2g.

d2_arms_mua_coupling.png -- per-session Pearson r(d2, MUA) by state, full-window vs ripple-excised d2
    (2024 Euclidean, p = 20), both arms from d2_arms_raw_windows.csv (16 sessions, non-overlapping tiles);
    correlations are scale-free, so nothing is normalised. Dots = sessions, bar = Fisher-z mean +- SEM.

Reads only local CSVs; writes only to D:.
    -> reports/revision_figs/r12_order_selection.png / .svg
    -> reports/revision_figs/d2_arms_mua_coupling.png / .svg
    -> data/revision/log/d2_arms_mua_coupling.csv
"""
from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                       # noqa: E402
from r35_glm_ablation import LOGDIR, FIGDIR                          # noqa: E402

STATES = ["NREM", "nonTHETA", "REM", "THETA"]
LAB = {"NREM": "NREM", "nonTHETA": "non-theta wake", "REM": "REM", "THETA": "theta wake"}
COL = {"NREM": "#7b3294", "REM": "#d7b5d8", "THETA": "#d95f0e", "nonTHETA": "#fdb863"}   # Fig-1 state colours
ORDERS = [5, 10, 15, 20, 30, 40]
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    out = os.path.join(FIGDIR, name)
    for _ in range(4):
        try:
            fig.savefig(assert_local_path(out), dpi=200, bbox_inches="tight"); fig.savefig(out[:-4] + ".svg", bbox_inches="tight"); break
        except OSError:
            import time; time.sleep(3)
    print("  ->", out)


def r12():
    """AR order vs residual whiteness by state: Ljung-Box with df = h - p (white_df of
    ljungbox_df_audit.csv)."""
    LB = pd.read_csv(os.path.join(LOGDIR, "ljungbox_df_audit.csv"))
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    # y axis: the fraction only (the test, the peaks and the session / window counts are given in the figure
    # legend)
    W = LB.pivot_table(index="state", columns="order", values="white_df", aggfunc="mean")
    for s in STATES:
        ax.plot(W.columns, W.loc[s], "o-", color=COL[s], lw=2, ms=4, label=LAB[s])
    ax.axvline(20, color="#bbbbbb", lw=6, alpha=0.4, zorder=0)
    ax.set_xlabel("AR order p (40 ms bins)"); ax.set_ylabel("Fraction of windows with white residuals")
    ax.set_xticks(ORDERS)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.set_ylim(0.6, 1.0)
    print(f"whiteness (df-corrected), %: 18 sessions, {LB[LB.order == 20].shape[0]} windows "
          f"({LB[LB.order == 20].groupby('state').size().to_dict()})"); print((100 * W).round(1).to_string())
    fig.tight_layout(); save(fig, "r12_order_selection.png")


def r13():
    W = pd.read_csv(os.path.join(LOGDIR, "d2_arms_raw_windows.csv"))
    rows = []
    for arm in ("d2_full", "d2_excised"):
        for s in STATES:
            for sess, g in W[W.state == s].groupby("session"):
                v = g[[arm, "MUA_mean"]].dropna()
                if len(v) >= 8 and v.MUA_mean.nunique() > 2:
                    rows.append(dict(arm=arm, state=s, session=sess, r=stats.pearsonr(v[arm], v.MUA_mean).statistic, n=len(v)))
    R = pd.DataFrame(rows); R.to_csv(assert_local_path(os.path.join(LOGDIR, "d2_arms_mua_coupling.csv")), index=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    print("\nr(d2, MUA) per session, Fisher-z mean +- SEM [95% CI], sessions positive, signed-rank:")
    for i, s in enumerate(STATES):
        for k, (arm, c, lab) in enumerate((("d2_full", "#7f7f7f", "full window"), ("d2_excised", "#d62728", "ripple-excised"))):
            v = R[(R.arm == arm) & (R.state == s)].r.to_numpy()
            z = np.arctanh(np.clip(v, -.999, .999)); m, se = z.mean(), z.std(ddof=1) / np.sqrt(len(z))
            x = i + (k - 0.5) * 0.36
            ax.bar(x, np.tanh(m), width=0.32, color=c, alpha=0.35, edgecolor=c, label=lab if i == 0 else None)
            ax.errorbar(x, np.tanh(m), yerr=[[np.tanh(m) - np.tanh(m - se)], [np.tanh(m + se) - np.tanh(m)]], color=c, capsize=3, lw=1.5)
            ax.scatter(x + np.random.default_rng(i * 2 + k).uniform(-0.09, 0.09, len(v)), v, s=9, color=c, zorder=3)
            p = stats.wilcoxon(v).pvalue if len(v) > 2 else np.nan
            print(f"  {LAB[s]:15s} {arm:10s} n={len(v):2d}  {np.tanh(m):+.3f} ± {se:.3f} [{np.tanh(m - 1.96 * se):+.2f}, {np.tanh(m + 1.96 * se):+.2f}]  {int((v > 0).sum())}/{len(v)} positive  p={p:.4f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([LAB[s] for s in STATES])
    ax.set_ylabel("per-session Pearson r(d2, population rate)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(f"d2 (2024 Euclidean, p = 20) vs population firing rate, {W.session.nunique()} sessions, raw values", fontsize=9)
    fig.tight_layout(); save(fig, "d2_arms_mua_coupling.png")


if __name__ == "__main__":
    r12(); r13()
