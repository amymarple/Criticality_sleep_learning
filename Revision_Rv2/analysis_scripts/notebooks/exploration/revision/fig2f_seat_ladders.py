"""fig2f_seat_ladders.py -- Figure 2f: rotating the metric seat of the replay ladder among
d2, MUA and SWR rate.

The ladder's covariate set is {MUA, SWR rate}. Each of d2, MUA and SWR rate is rotated
through the metric seat in turn: the last rung of every seat is the same joint model
(replay ~ d2 + MUA + SWR rate), so its three coefficients are the summary -- d2 keeps its
coefficient, while MUA and SWR rate (correlated within session) trade theirs. The middle
rung's order does not affect the joint model.

Windows: Fig 1's non-overlapping 30 s NREM tiles, decoded time only, 9 sessions
(metric_replay_ladder_windows.csv, written by metric_replay_ladder.py). Binomial GLM of
replay_count / n_swr per window, predictors z-scored within session, session as the unit,
two-sided signed-rank across sessions; d2 is the ripple-excised d2 (2024 Euclidean, p = 20).
Same box style and colours as the Figure 2f composite.

    -> data/revision/log/fig2f_seat_ladders.csv
    -> reports/revision_figs/fig2f_seat_ladders.png      (3 panels, one per seat)
    -> reports/revision_figs/fig2f_seat_joint.png        (the joint model, three coefficients)
    -> reports/revision_figs/fig2f_ladder_d2.png         (d2 seat alone, same as the composite's panel f)
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
from r35_binomial_replay import fit_binomial, pooled_fit, zs         # noqa: E402
from r35_glm_ablation import LOGDIR                                   # noqa: E402

WIN_CSV = os.path.join(LOGDIR, "metric_replay_ladder_windows.csv")
OUT_CSV = os.path.join(LOGDIR, "fig2f_seat_ladders.csv")
FIGDIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "reports", "revision_figs"))
D2 = "d2_excised" if not os.environ.get("D2_FULL") else "d2_published"
VARS = {D2: "d2", "MUA": "MUA", "rate": "SWR rate"}
BOX = ["#c03d3e", "#d4595a", "#e57373"]                 # ladder rungs, as the composite's panel f
JOINT = {D2: "#d62728", "MUA": "#7f7f7f", "rate": "#bfbfbf"}
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def ladder_for(seat):
    others = [v for v in VARS if v != seat]
    return [(VARS[seat], [seat]), (f"+ {VARS[others[0]]}", [seat, others[0]]),
            (f"+ {VARS[others[1]]}", [seat] + others)]


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


def main():
    W = pd.read_csv(WIN_CSV)
    for v in VARS:
        W[f"{v}_z"] = W.groupby("session")[v].transform(zs)
    rows, per = [], {}
    for seat in VARS:
        for rung, (lab, cols) in enumerate(ladder_for(seat)):
            zc = [f"{c}_z" for c in cols]
            b = []
            for s, sub in W.groupby("session"):
                r = fit_binomial(sub, zc)
                if r is not None:
                    rows.append(dict(seat=seat, rung=rung, model=lab, session=s, **r)); b.append(r["b1"])
            pf = pooled_fit(W, zc)
            rows.append(dict(seat=seat, rung=rung, model=lab, session="__pooled__", **pf))
            per[(seat, rung)] = (lab, np.array(b), pf)
    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(OUT_CSV), index=False)

    n_sess, n_win = W.session.nunique(), len(W)
    print(f"{n_sess} sessions, {n_win} windows, {int(W.n_swr.sum())} SWRs, {W.n_replay.sum() / W.n_swr.sum():.1%} replay"
          f"  ({'ripple-excised' if D2 == 'd2_excised' else 'full-window'} 2024 Euclidean d2, p = 20)")
    for seat in VARS:
        print(f"\nseat: {VARS[seat]}")
        for rung in range(3):
            lab, b, pf = per[(seat, rung)]
            print(f"  {lab:14s} {np.median(b):+.3f} ({int((b > 0).sum())}/{len(b)}, p={stats.wilcoxon(b).pvalue:.3f})"
                  f"   pooled b={pf['b1']:+.3f} p={pf['p']:.2g}")

    # --- figure 1: three seats -------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), sharey=True)
    for k, (ax, seat) in enumerate(zip(axes, VARS)):
        data = [per[(seat, r)][1] for r in range(3)]
        labels = [per[(seat, r)][0] for r in range(3)]
        box_panel(ax, data, labels, BOX, ylabel="log-odds of replay per SD\n(binomial, per session)" if k == 0 else None,
                  title=f"{'abc'[k]}  {VARS[seat]} in the seat")
    fig.suptitle(f"Seat rotation -- {n_sess} sessions, decoded NREM tiles; d2 = ripple-excised 2024 Euclidean (p = 20)",
                 fontsize=8.5, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2f_seat_ladders.png")
    fig.savefig(assert_local_path(out), dpi=200, bbox_inches="tight"); fig.savefig(out[:-4] + ".svg", bbox_inches="tight")
    print("  ->", out)

    # --- figure 2: the joint model ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.0, 3.2))
    data = [per[(seat, 2)][1] for seat in VARS]
    box_panel(ax, data, [VARS[s] for s in VARS], [JOINT[s] for s in VARS],
              ylabel="log-odds of replay per SD\n(joint model, per session)",
              title="replay ~ d2 + MUA + SWR rate")
    ax.set_xticklabels([VARS[s] for s in VARS], rotation=0, ha="center")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2f_seat_joint.png")
    fig.savefig(assert_local_path(out), dpi=200, bbox_inches="tight"); fig.savefig(out[:-4] + ".svg", bbox_inches="tight")
    print("  ->", out)

    # --- figure 3: the d2 seat alone, as in the Figure 2f composite's panel f -----------
    fig, ax = plt.subplots(figsize=(3.0, 3.2))
    data = [per[(D2, r)][1] for r in range(3)]
    box_panel(ax, data, [per[(D2, r)][0] for r in range(3)], BOX,
              ylabel="log-odds of replay per SD d2\n(ripple-excised; binomial, per session)")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2f_ladder_d2.png")
    fig.savefig(assert_local_path(out), dpi=200, bbox_inches="tight"); fig.savefig(out[:-4] + ".svg", bbox_inches="tight")
    print("  ->", out)


if __name__ == "__main__":
    main()
