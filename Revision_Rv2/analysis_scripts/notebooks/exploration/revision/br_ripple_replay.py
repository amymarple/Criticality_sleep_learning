"""
br_ripple_replay.py -- Fig. 2g/2h analogue for the branching ratio: BR vs ripple-rate
sextile and BR vs replay-probability sextile, on non-overlapping windows.

Fig. 2g/2h put BR and ISI entropy beside d2 against ripples on the original overlapping
grid; here the same contrasts are computed for BR on independent (non-overlapping)
windows, alongside d2 under the same design. Same windows and machinery as
d2_correlates_table.py / d2_composition_glm.py: Fig. 1's non-overlapping NREM tiles
(fig1_metrics pickles), SWR / replay per window from build_events (8 decoded sessions).

BR = BR_pyr_k200 (MR estimator, pyramidal cells, 4 ms bins, k_max 200, the same 800 ms
kernel as d2). The ripple-excised Euclidean d2 is drawn alongside on the same windows.

Panels (reports/revision_figs/br_ripple_replay.png):
  A  metric (within-session z) by ripple-rate sextile, mean +- SEM over sessions
  B  metric by replay-probability sextile (windows with >= 1 SWR)
  stats: per-session Spearman rho(sextile, metric), signed-rank across sessions
Table: per-session Pearson r of BR with every window metric, Fisher mean, beside the
excised d2 column.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/br_ripple_replay_sextiles.csv
    -> data/revision/log/br_correlates_table.md
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
from r35_glm_ablation import LOGDIR, FIGDIR                             # noqa: E402
from r35_swr_property_control import SESSION_TEMPLATE                   # noqa: E402
from d2_composition_glm import load_session, add_swr, fig2_basepaths                    # noqa: E402
from d2_correlates_table import ROWS, fisher                            # noqa: E402

METRICS = [("BR_pyr_k200", "branching ratio (pyr, k=200)"), ("d2_excised", "d2, ripple-excised"),
           ("d2_published", "d2, published"), ("isi_entropy_pooled", "ISI entropy (pooled)"),
           ("isi_entropy_perunit_mean", "ISI entropy (per-unit mean)")]
COLORS = {"BR_pyr_k200": "#2c7fb8", "d2_excised": "#c03d3e", "d2_published": "#7f7f7f",
          "isi_entropy_pooled": "#238b45", "isi_entropy_perunit_mean": "#74c476"}
# which metrics go on which figure
PANELS = {"br_ripple_replay": ["BR_pyr_k200", "d2_excised"],
          "isi_ripple_replay": ["isi_entropy_pooled", "isi_entropy_perunit_mean", "d2_excised"]}


def zs(x):
    x = np.asarray(x, float)
    s = np.nanstd(x, ddof=1)
    return (x - np.nanmean(x)) / (s if s > 0 else 1.0)


def load_all():
    parts = []
    for bp in fig2_basepaths():
        sid, W = load_session(bp)
        if W is None or sid not in SESSION_TEMPLATE:
            continue
        W = add_swr(W, sid, bp)
        W["session"] = sid
        parts.append(W)
    return pd.concat(parts, ignore_index=True)


def sextile_curves(D, by, metric, nb=6, min_swr=None):
    rows = []
    for s, g in D.groupby("session"):
        g = g.dropna(subset=[by, metric])
        if min_swr is not None:
            g = g[g["ripple_rate"] * 30 >= min_swr]
        if len(g) < 60:
            continue
        g = g.copy()
        g["z"] = zs(g[metric])
        g["sex"] = pd.qcut(g[by].rank(method="first"), nb, labels=False) + 1
        m = g.groupby("sex")["z"].mean()
        rho = stats.spearmanr(g["sex"], g["z"])[0]
        rows.append(dict(session=s, by=by, metric=metric, rho=rho, n=len(g),
                         **{f"s{k}": m.get(k, np.nan) for k in range(1, nb + 1)}))
    return pd.DataFrame(rows)


def main():
    D = load_all()
    print(f"{D.session.nunique()} decoded sessions, {len(D)} NREM non-overlapping windows; "
          f"excised d2 in {D.groupby('session').d2_excised.apply(lambda v: v.notna().mean() > .5).sum()} of them")
    # ---- sextile curves
    C = []
    for by, min_swr in (("ripple_rate", None), ("replay_prob", 1)):
        for metric, _ in METRICS:
            C.append(sextile_curves(D, by, metric, min_swr=min_swr))
    C = pd.concat(C, ignore_index=True)
    assert_local_path(LOGDIR)
    C.to_csv(assert_local_path(os.path.join(LOGDIR, "br_ripple_replay_sextiles.csv")), index=False)
    print("\nper-session Spearman rho(sextile, metric z), signed-rank across sessions:")
    for by in ("ripple_rate", "replay_prob"):
        for metric, lab in METRICS:
            v = C[(C.by == by) & (C.metric == metric)].rho.to_numpy()
            if len(v) < 3:
                continue
            top_bottom = (C[(C.by == by) & (C.metric == metric)].s6 - C[(C.by == by) & (C.metric == metric)].s1)
            print(f"  by {by:12s} {lab:30s} rho median {np.median(v):+.3f}  pos {int((v > 0).sum())}/{len(v)}  "
                  f"p = {stats.wilcoxon(v).pvalue:.3f} | top-bottom sextile {top_bottom.median():+.2f} SD")
    # ---- correlate table for BR beside excised d2
    lines = ["| predictor | BR (pyr, k=200) | ISI entropy (pooled) | d2 ripple-excised |", "|---|---|---|---|"]
    for col, lab in ROWS:
        cells = []
        for target in ("BR_pyr_k200", "isi_entropy_pooled", "d2_excised"):
            rs = []
            for s, g in D.groupby("session"):
                sub = g[[target, col]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(sub) >= 40:
                    rs.append(np.corrcoef(sub[target], sub[col])[0, 1])
            rs = np.asarray(rs)
            if rs.size < 3:
                cells.append("—")
                continue
            p = stats.wilcoxon(rs).pvalue
            b = "**" if p < 0.01 else ""
            cells.append(f"{b}{fisher(rs):+.3f}{b} ({int((rs > 0).sum())}/{rs.size}, p = {p:.3f})")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    with open(assert_local_path(os.path.join(LOGDIR, "br_correlates_table.md")), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print("\nPer-session Pearson r, Fisher mean (8 decoded sessions):\n" + md)
    # BR vs d2 themselves
    r_bd = [np.corrcoef(g[["BR_pyr_k200", "d2_excised"]].dropna().T.to_numpy())[0, 1]
            for _, g in D.groupby("session") if g.d2_excised.notna().sum() > 40]
    r_bp = [np.corrcoef(g[["BR_pyr_k200", "d2_published"]].dropna().T.to_numpy())[0, 1] for _, g in D.groupby("session")]
    print(f"\nr(BR, d2 excised) median {np.median(r_bd):+.3f} ({int((np.array(r_bd) > 0).sum())}/{len(r_bd)}); "
          f"r(BR, d2 published) median {np.median(r_bp):+.3f} ({int((np.array(r_bp) > 0).sum())}/{len(r_bp)})")
    for name, mets in PANELS.items():
        plot(C, mets, name)


def plot(C, mets=("BR_pyr_k200", "d2_excised"), name="br_ripple_replay"):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5, "legend.fontsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, by, xlabel in zip(axes, ("ripple_rate", "replay_prob"),
                              ("SWR rate sextile", "replay probability sextile")):
        for metric, lab in [(m, l) for m, l in METRICS if m in mets]:
            P = C[(C.by == by) & (C.metric == metric)]
            if not len(P):
                continue
            Y = P[[f"s{k}" for k in range(1, 7)]].to_numpy()
            m, se = np.nanmean(Y, 0), np.nanstd(Y, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Y), 0))
            x = np.arange(1, 7)
            ax.errorbar(x, m, yerr=se, marker="o", ms=6, lw=2, capsize=3, color=COLORS[metric], label=lab)
            v = P.rho.to_numpy()
            pw = stats.wilcoxon(v).pvalue
            star = "***" if pw < .001 else "**" if pw < .01 else "*" if pw < .05 else "n.s."
            up = np.median(v) > 0
            n_dir = int((v > 0).sum()) if up else int((v < 0).sum())
            ax.text(0.03, 0.95 - 0.09 * list(mets).index(metric),
                    f"{lab}: {n_dir}/{len(v)} {'↑' if up else '↓'} {star}",
                    transform=ax.transAxes, va="top", color=COLORS[metric], fontsize=11)
        ax.axhline(0, ls="--", lw=1, color="k")
        ax.set_xlabel(xlabel)
        ax.set_xticks(range(1, 7))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("metric (within-session z)\nmean ± SEM over sessions")
    axes[1].legend(frameon=False, loc="lower right")
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"{name}.{ext}")), dpi=300)
    plt.close(fig)
    print(f"  -> {FIGDIR}\\{name}.png/.svg")


if __name__ == "__main__":
    main()
