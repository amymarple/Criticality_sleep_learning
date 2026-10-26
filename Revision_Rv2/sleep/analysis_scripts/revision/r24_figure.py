"""
r24_figure.py -- d2 (2024 Euclidean) against the branching ratio m on the same windows, and both
against SWR rate (Extended Data Fig. 4a-d).

PANELS (TOP=corr, default).
  a  m against d2 on the same non-overlapping 30 s NREM windows of Fig 1 (the per-session pickles,
     `aggregate_fig1_metrics.load_all`), one colour and one least-squares line per session.
  b  per-session Pearson r of d2 with m within each state, from `fig1_d2_vs_br.csv` (written by
     `aggregate_fig1_metrics.py`); Fisher-z mean +- SEM. Wake theta / non-theta are the revision
     tiles (whole window in state).
  CORR_POP=all (default) uses m on all CA1 units, the population d2 is fitted on; CORR_POP=pyr the
  pyramidal m.
  c, d  `r24_br_ripple_arms.sextile_panel`: NREM, 16 sessions, full window vs ripple-excised,
     pyramidal m, POP as in r24_br_ripple_arms.
TOP=states draws state box plots (`r24_dissociation_figure.state_box_panel`) in place of a and b.
Nothing is recomputed.

Writes only to D:.
    -> reports/revision_figs/r24_d2_vs_br.png / .svg          (TOP=corr, CORR_POP=all, POP=pyr)
    -> reports/revision_figs/backup/r24_d2_vs_br_corr_pyr.png  (CORR_POP=pyr)
    -> reports/revision_figs/backup/r24_d2_vs_br_states*.png   (TOP=states)
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

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                                  # noqa: E402
from r35_glm_ablation import FIGDIR, LOGDIR                                     # noqa: E402
from r24_dissociation_figure import SRC, STATES, PANELS, POP, LABEL, _SFX, LAB_SFX, LABELS, XPOS, contrast_table, state_box_panel  # noqa: E402
from r24_br_ripple_arms import load_table, arm_stats, sextile_panel, D2_YLAB, M_YLAB  # noqa: E402
from d2_arms_raw import fisher                                                  # noqa: E402

TOP = os.environ.get("TOP", "corr")
CORR_POP = os.environ.get("CORR_POP", "all")
CORR_COL = f"BR_{CORR_POP}_k200"
CORR_YLAB = f"m (MR, {'all CA1 units' if CORR_POP == 'all' else 'pyramidal'}, 4 ms, k$_{{max}}$ = 200)"
# box colours kept apart from the bottom row's grey (full window) / red (ripple-excised)
D2_BOX, M_BOX = "#756bb1", "#3182bd"


def corr_scatter_panel(ax, W, sessions):
    """m against d2 on the same NREM windows; one colour and one OLS line per session"""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    N = W[(W.state == "NREM") & W.session.isin(sessions)].dropna(subset=["d2", CORR_COL])
    for i, (s, g) in enumerate(N.groupby("session")):
        c = cmap(i % 20)
        ax.scatter(g.d2, g[CORR_COL], s=3, color=c, alpha=0.35, lw=0, rasterized=True)
        b1, b0 = np.polyfit(g.d2, g[CORR_COL], 1)
        x = np.array([g.d2.quantile(.02), g.d2.quantile(.98)])
        ax.plot(x, b0 + b1 * x, color=c, lw=1.4)
    ax.set_xlabel("d2 (2024 Euclidean, p = 20, 40 ms)"); ax.set_ylabel(CORR_YLAB)
    return len(N), N.session.nunique()


def corr_state_panel(ax, B):
    """per-session Pearson r(d2, m) by state from fig1_d2_vs_br.csv, Fisher-z mean +- SEM"""
    B = B[B.br_col == CORR_COL]
    x = np.array(XPOS, float)
    stats_out = {}
    for i, st in enumerate(STATES):
        v = B[B.state == st].pearson_r.to_numpy(float)
        if not v.size:
            continue
        m, se, n = fisher(v)
        stats_out[st] = (m, se, n, int((v < 0).sum()))
        ax.scatter(x[i] + np.random.default_rng(i).uniform(-0.13, 0.13, v.size), v, s=14, color="0.45", lw=0, alpha=0.8)
        ax.errorbar(x[i], m, yerr=se, fmt="s", color="k", ms=6, capsize=4, zorder=4)
        ax.text(x[i], 0.14, f"{int((v < 0).sum())}/{n} < 0", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="k", lw=0.8, ls="--"); ax.axhline(-1, color="0.6", lw=0.8, ls=":")
    ax.text(x[-1] + 0.45, -1, "r = −1: m and d2\ninterchangeable", fontsize=7, color="0.4", va="center", ha="right")
    ax.axvline(np.mean(XPOS[1:3]), color="0.6", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in STATES]); ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(-1.08, 0.3); ax.set_ylabel("Pearson r (d2, m) within session\nsame 30 s windows")
    return stats_out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    D = load_table()
    A = {r["arm"]: r for r in arm_stats(D, verbose=False)}
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.0))
    extra = []
    if TOP == "states":
        T = pd.read_csv(SRC); T = T[T.state.isin(STATES)]
        C = contrast_table(T)
        for ax, (col, ylab, c, title, arrow) in zip(axes[0], [
                ("d2", D2_YLAB, D2_BOX, "a  d2 by state", "larger = further from criticality"),
                (PANELS[1][0], M_YLAB, M_BOX, "b  branching ratio m by state", "larger = closer to criticality")]):
            extra.append(state_box_panel(ax, T, C, col, "", arrow, c, stats_offset=-28, fontsize=8))
            ax.set_title(title, loc="left", fontweight="bold", pad=18)
            ax.set_ylabel(ylab + "\nper-session median")
        out = os.path.join(FIGDIR, "backup", f"r24_d2_vs_br_states{_SFX}{LAB_SFX}.png")
    else:
        from aggregate_fig1_metrics import load_all
        B = pd.read_csv(os.path.join(LOGDIR, "fig1_d2_vs_br.csv"))
        W = load_all()
        n_win, n_s = corr_scatter_panel(axes[0, 0], W, set(B.session))
        axes[0, 0].set_title(f"a  m vs d2, same windows (NREM, {n_s} sessions)", loc="left", fontweight="bold")
        S = corr_state_panel(axes[0, 1], B)
        axes[0, 1].set_title("b  within-session correlation by state", loc="left", fontweight="bold")
        m, se, n, k = S["NREM"]
        axes[0, 0].text(0.97, 0.97, f"per-session r = {m:+.2f} ± {se:.2f}\n{k}/{n} sessions negative", transform=axes[0, 0].transAxes,
                        ha="right", va="top", fontsize=8.5)
        for st, (m, se, n, k) in S.items():
            print(f"  r(d2, {CORR_COL}) {st:8s} {m:+.3f} +- {se:.3f}  {k}/{n} negative")
        print(f"  scatter: {n_win} NREM windows")
        out = (os.path.join(FIGDIR, "r24_d2_vs_br.png") if CORR_POP == "all" and POP == "pyr"
               else os.path.join(FIGDIR, "backup", f"r24_d2_vs_br_corr_{CORR_POP}{_SFX}.png"))
    sextile_panel(axes[1, 0], D, A, "d2_full", "d2_excised", D2_YLAB, "c  d2 vs SWR rate (NREM)")
    sextile_panel(axes[1, 1], D, A, "m_full", "m_excised", M_YLAB, "d  branching ratio m vs SWR rate (NREM)",
                  legend_loc="upper right", count_y=0.14)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.07,
                        hspace=0.56 if TOP == "states" else 0.32, wspace=0.34)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(out[:-3] + ext), dpi=200, bbox_inches="tight", bbox_extra_artists=extra or None)
    print(f"  {D.session.nunique()} sessions (SWR rate) ->", out)


if __name__ == "__main__":
    main()
