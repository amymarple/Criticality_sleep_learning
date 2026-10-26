"""fig2f_d2_replay_content.py -- Extended Data Fig. 7: the ripple-excised d2 against the SWRs'
cell recruitment and the geometry of the decoded replay events.

(a) per-session correlations with cells per SWR, net displacement and decoder fit; (b, c)
d2-sextile curves for cells per SWR and net displacement. The trajectory span is computed and
written to the CSV, not drawn; ALL=1 adds speed, replay duration, sequence r and a speed panel.

Ripple amplitude and duration are LFP properties of the SWR, measured without the spike-based
decoder, so they can enter the replay model as covariates (fig2f_swr_property_ladder.py).
Cells per SWR and the trajectory measures are computed from the spikes the decoder calls
replay with and are defined by it: the cell count is the decoder's input (cellcountthresh),
and the geometry exists only for events already called replay. Conditioning on them would
condition on components of the outcome (mediator / collider), so they are not covariates of
the replay model; their relation to d2 is described here instead. Only the ripple-excised d2
is used: it is fitted on spikes disjoint from the decoder's, whereas the full-window d2 shares
the in-ripple spikes with the decoder.

Tiles: those of the replay ladder (r35_replay_geometry_excised.csv: per window, cells per SWR
and the means over the significant replay events of net displacement, |slope|, duration,
span, r-value and goodness of fit). (a) per-session Spearman rho, one point per session,
signed-rank test across sessions; (b, c) sextiles of the excised d2 within session, each
measure as its session-centred mean per sextile in native units, mean +- SEM over sessions,
per-session Spearman of sextile vs measure annotated.

Reads only local CSVs; writes only to D:.
    -> data/revision/log/fig2f_d2_replay_content.csv        (per-session rho, per-sextile means)
    -> reports/revision_figs/fig2f_d2_replay_content.png
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
from r35_glm_ablation import LOGDIR                                   # noqa: E402
from fig2f_seat_ladders import box_panel, FIGDIR, stars               # noqa: E402

GEOM_CSV = os.path.join(LOGDIR, "r35_replay_geometry_excised.csv")
OUT_CSV = os.path.join(LOGDIR, "fig2f_d2_replay_content.csv")
D2 = "d2_excised"
# cells / SWR, net displacement, trajectory span and decoder fit; ALL=1 adds speed, replay
# duration and sequence r.
MEAS = [("n_cells", "cells / SWR"), ("traj_net", "net displacement"), ("traj_span", "trajectory span"), ("gof", "decoder fit")]
SEXT = [("n_cells", "cells / SWR", "Δ cells per SWR"), ("traj_net", "net displacement", "Δ net displacement (position bins)")]
RED, BOX = "#d62728", ["#c03d3e", "#e57373", "#f3a6a6"]
# trajectory span: computed and written to the CSV, not drawn.
PLOT_DROP = {"traj_span"}
if os.environ.get("ALL"):
    MEAS = [("n_cells", "cells / SWR"), ("traj_net", "net displacement"), ("speed", "trajectory speed"),
            ("dur_s", "replay duration"), ("traj_span", "trajectory span"), ("rval", "sequence r"), ("gof", "decoder fit")]
    SEXT = SEXT + [("speed", "trajectory speed", "Δ |slope|")]
    BOX = ["#c03d3e"] + ["#e57373"] * 3 + ["#f3a6a6"] * 3
    PLOT_DROP = set()
X6 = np.arange(1, 7)


def sextile_rows(W, col):
    rows = []
    for s, g in W.groupby("session"):
        g = g.dropna(subset=[D2, col])
        if len(g) < 12 or g[col].nunique() <= 2:      # e.g. speed is undefined in one session
            continue
        sex = pd.qcut(g[D2].rank(method="first"), 6, labels=False) + 1
        m = g.groupby(sex)[col].mean() - g[col].mean()
        rows.append(dict(session=s, measure=col, rho=stats.spearmanr(sex, g[col])[0],
                         **{f"s{k}": m.get(k, np.nan) for k in X6}))
    return pd.DataFrame(rows)


def main():
    W = pd.read_csv(GEOM_CSV)
    rows, per = [], {}
    for col, lab in MEAS:
        v = []
        for s, g in W.groupby("session"):
            x = g[[D2, col]].dropna()
            if len(x) >= 10 and x[col].nunique() > 2:
                r = stats.spearmanr(x[D2], x[col]).statistic
                v.append(r); rows.append(dict(part="corr", measure=col, label=lab, session=s, rho=r, n=len(x)))
        per[col] = np.array(v)
    S = {col: sextile_rows(W, col) for col, _, _ in SEXT}
    for col, _, _ in SEXT:
        for _, r in S[col].iterrows():
            rows.append(dict(part="sextile", measure=col, session=r.session, rho=r.rho, **{f"s{k}": r[f"s{k}"] for k in X6}))
    pd.DataFrame(rows).to_csv(assert_local_path(OUT_CSV), index=False)

    fig, axes = plt.subplots(1, 1 + len(SEXT), figsize=(3.4 + 3.1 * len(SEXT), 3.4),
                             gridspec_kw=dict(width_ratios=[1.35] + [1] * len(SEXT)))
    shown = [(c, l) for c, l in MEAS if c not in PLOT_DROP]
    box_panel(axes[0], [per[c] for c, _ in shown], [l for _, l in shown], BOX,
              ylabel="Spearman ρ with ripple-excised d2\n(per session)", title="a  correlation")
    print("a  per-session Spearman rho with the ripple-excised d2 (all measures; not drawn: "
          f"{', '.join(sorted(PLOT_DROP)) or 'none'}):")
    for (c, l), v in zip(MEAS, [per[c] for c, _ in MEAS]):
        print(f"  {l:18s} {np.median(v):+.3f}  {int((v > 0).sum())}/{len(v)}  p = {stats.wilcoxon(v).pvalue:.4f}")
    for ax, (col, lab, ylab), letter in zip(axes[1:], SEXT, "bcd"):
        D = S[col]
        Y = D[[f"s{k}" for k in X6]].to_numpy()
        m, se = np.nanmean(Y, 0), np.nanstd(Y, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Y), 0))
        ax.errorbar(X6, m, yerr=se, fmt="o-", capsize=4, lw=2, ms=5, color=RED)
        ax.axhline(0, ls="--", lw=0.8, color="k")
        v = D.rho.to_numpy()
        ax.text(0.04, 0.95, f"{int((v > 0).sum())}/{len(v)} ↑ {stars(stats.wilcoxon(v).pvalue)}", transform=ax.transAxes,
                va="top", color=RED, fontsize=9)
        ax.set_xticks(X6); ax.set_xlabel("ripple-excised d2 sextile (within session)")
        ax.set_ylabel(f"{ylab}\nrelative to session mean")
        ax.set_title(f"{letter}  {lab}", fontsize=9.5, fontweight="bold", loc="left")
        print(f"{letter}  {lab}: sextile means (session-centred) " + " ".join(f"{x:+.3f}" for x in m) +
              f"; rho(sextile) {np.median(v):+.3f}, {int((v > 0).sum())}/{len(v)}, p = {stats.wilcoxon(v).pvalue:.4f}")
    fig.suptitle(f"ripple-excised d2 (2024 Euclidean, p = 20) vs SWR recruitment and decoded-replay geometry -- "
                 f"{W.session.nunique()} sessions, decoded NREM tiles", fontsize=8.5, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2f_d2_replay_content.png")
    fig.savefig(assert_local_path(out), dpi=200, bbox_inches="tight"); fig.savefig(out[:-4] + ".svg", bbox_inches="tight")
    print("  ->", out)


if __name__ == "__main__":
    main()
