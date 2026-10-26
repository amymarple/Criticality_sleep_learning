"""Drawing functions from analysis_scripts/revision/ (r24_figure.py, r24_dissociation_figure.py, run_fig1_metrics.py)
used by the figure notebooks; the SWR-rate panels and their statistics come from r24_br_ripple_arms_drawing.
LOGDIR, FIGDIR: the folder of the notebook's tables, set by the notebook.
"""
import numpy as np
import os
import pandas as pd

from r24_br_ripple_arms_drawing import load_table, arm_stats, sextile_panel, fisher, D2_YLAB, M_YLAB

LOGDIR = None
FIGDIR = "."


from final_figs import assert_local_path   # writes go to ../figure_output only


# --- r24_dissociation_figure.py
STATES = ["NREM", "REM", "THETA", "nonTHETA"]      # sleep pair, then wake pair


# --- r24_dissociation_figure.py
XPOS = [0, 1, 2.5, 3.5]                             # gap between the pairs; contrasts stay within a pair


# --- r24_dissociation_figure.py
LABELS = {"THETA": "Wake-θ", "REM": "REM",
          "nonTHETA": "Wake-nonθ", "NREM": "NREM"}


# --- r24_dissociation_figure.py
POP = os.environ.get("POP", "pyr")


# --- r24_dissociation_figure.py
_SFX = "" if POP == "pyr" else f"_{POP}"


# --- r24_figure.py
CORR_POP = os.environ.get("CORR_POP", "all")


# --- r24_figure.py
CORR_COL = f"BR_{CORR_POP}_k200"


# --- r24_figure.py
CORR_YLAB = f"m (MR, {'all CA1 units' if CORR_POP == 'all' else 'pyramidal'}, 4 ms, k$_{{max}}$ = 200)"


# --- r24_figure.py
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


# --- r24_figure.py
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


# --- r24_figure.py (a drawing section, as a function: main with TOP=corr)
def draw_main(W):
    """Draw r24_figure's four panels: m against d2 on the same NREM windows of table W (a), the per-session
    r(d2, m) by state from fig1_d2_vs_br.csv (b), and d2 (c) and m (d) by SWR-rate sextile, full window and
    ripple-excised (load_table)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    D = load_table()
    A = {r["arm"]: r for r in arm_stats(D, verbose=False)}
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.0))
    B = pd.read_csv(os.path.join(LOGDIR, "fig1_d2_vs_br.csv"), float_precision="round_trip")
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
    fig.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.07, hspace=0.32, wspace=0.34)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(out[:-3] + ext), dpi=200, bbox_inches="tight", bbox_extra_artists=None)
    print(f"  {D.session.nunique()} sessions (SWR rate) ->", out)
