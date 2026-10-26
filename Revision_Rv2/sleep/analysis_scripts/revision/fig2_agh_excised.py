"""
fig2_agh_excised.py -- Fig 2a / 2g / 2h on the replay-decoded nine sessions, non-overlapping windows,
full-window and ripple-excised -- d2, branching ratio and ISI entropy against SWR-rate sextile.

All Fig 2 panels use the decoded nine sessions here, and the ripple-excised arm is drawn
alongside the full-window one; windows are non-overlapping and the metrics are raw units, not
z-scores.

Reads `excised_br_entropy_windows.csv` (excised_br_entropy.py): per Fig-1 NREM
non-overlapping 30 s window, 9 sessions: d2_published / d2_excised (2024 Euclidean,
p = 20, the .mat values), BR_pyr_k200 / BR_excised (MR estimator, pyramidal, 4 ms,
k = 200), isi_entropy_pooled / isi_excised (pooled log10-ISI entropy, 20 bins), and
SWR rate (NREM ripples in the window / 30 s).

Per session, windows are split into SWR-rate sextiles; the metric's mean per
sextile is one thin line per session; the bold line is mean +- SEM over the 9 sessions.
Stats in the corner: per-session Spearman rho(sextile, metric), signed-rank across
sessions, sessions in the direction of the median. Full window in grey, excised in
colour.

    -> reports/revision_figs/fig2_agh_nine_sessions.png / .svg
    -> data/revision/log/fig2_agh_nine_sessions.csv   (per session x arm x metric: sextile means, rho)
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path              # noqa: E402
from r35_glm_ablation import LOGDIR, FIGDIR                  # noqa: E402

WIN_CSV = os.path.join(LOGDIR, "excised_br_entropy_windows.csv")
OUT = os.path.join(LOGDIR, "fig2_agh_nine_sessions.csv")
PANELS = [  # (title, full column, excised column, y label, colour)
    ("a  d2", "d2_published", "d2_excised", "d2 (2024 Euclidean, p = 20)", "#c03d3e"),
    ("g  branching ratio", "BR_pyr_k200", "BR_excised", "branching ratio m (pyr, k = 200)", "#2c7fb8"),
    ("h  ISI entropy", "isi_entropy_pooled", "isi_excised", "ISI entropy (bits)", "#238b45"),
]
NB = 6


def sextile_means(D, col):
    rows = []
    for s, g in D.groupby("session"):
        g = g.dropna(subset=[col, "ripple_rate"])
        if len(g) < 60:
            continue
        sex = pd.qcut(g["ripple_rate"].rank(method="first"), NB, labels=False) + 1
        m = g.groupby(sex)[col].mean()
        rho = stats.spearmanr(sex, g[col])[0]
        rows.append(dict(session=s, metric=col, rho=rho, n=len(g),
                         **{f"s{k}": m.get(k, np.nan) for k in range(1, NB + 1)}))
    return pd.DataFrame(rows)


def main():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5, "legend.fontsize": 10})
    D = pd.read_csv(WIN_CSV)
    print(f"{D.session.nunique()} sessions: {sorted(D.session.unique())}; {len(D)} NREM non-overlapping windows")
    allrows = []
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    x = np.arange(1, NB + 1)
    for ax, (title, cfull, cexc, ylab, colour) in zip(axes, PANELS):
        for col, arm, c, z in ((cfull, "full window", "#7f7f7f", 2), (cexc, "ripple-excised", colour, 3)):
            S = sextile_means(D, col)
            S["arm"] = arm
            allrows.append(S)
            Y = S[[f"s{k}" for k in x]].to_numpy()
            for row in Y:
                ax.plot(x, row, "-", color=c, alpha=0.18, lw=1, zorder=1)
            m, se = np.nanmean(Y, 0), np.nanstd(Y, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Y), 0))
            ax.errorbar(x, m, yerr=se, marker="o", ms=6, lw=2.2, capsize=3, color=c, zorder=z, label=arm)
            v = S.rho.to_numpy()
            up = np.median(v) > 0
            ndir = int((v > 0).sum()) if up else int((v < 0).sum())
            p = stats.wilcoxon(v).pvalue
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
            ax.text(0.03, 0.96 - 0.09 * (0 if arm == "full window" else 1),
                    f"{arm}: {ndir}/{len(v)} {'↑' if up else '↓'} {star}",
                    transform=ax.transAxes, va="top", color=c, fontsize=10)
            print(f"  {title:20s} {arm:15s} rho median {np.median(v):+.3f}  {ndir}/{len(v)} {'up' if up else 'down'}  p = {p:.3f}"
                  f"  | top-bottom {np.nanmedian(Y[:, -1] - Y[:, 0]):+.4f} ({ylab.split(' (')[0]})")
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlabel("SWR rate sextile (within session)")
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    plt.tight_layout()
    assert_local_path(FIGDIR)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"fig2_agh_nine_sessions.{ext}")), dpi=300)
    plt.close(fig)
    pd.concat(allrows, ignore_index=True).to_csv(assert_local_path(OUT), index=False)
    print(f"  -> {FIGDIR}\\fig2_agh_nine_sessions.png/.svg\n  -> {OUT}")


if __name__ == "__main__":
    main()
