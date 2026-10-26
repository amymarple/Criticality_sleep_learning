"""Drawing functions from analysis_scripts/revision/ (fig2f_d2_replay_content.py, run_fig1_metrics.py) used by the
figure notebooks. draw_main takes the per-window table the producer reads (r35_replay_geometry_excised.csv);
LOGDIR: where it writes its per-session table (skipped while a notebook captures the figure).
"""
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

from fig2f_seat_ladders_drawing import box_panel, FIGDIR, stars

warnings.filterwarnings("ignore")

LOGDIR = "."


# --- fig2f_d2_replay_content.py
OUT_CSV = os.path.join(LOGDIR, "fig2f_d2_replay_content.csv")


# --- fig2f_d2_replay_content.py
D2 = "d2_excised"


# --- fig2f_d2_replay_content.py
MEAS = [("n_cells", "cells / SWR"), ("traj_net", "net displacement"), ("traj_span", "trajectory span"), ("gof", "decoder fit")]


# --- fig2f_d2_replay_content.py
SEXT = [("n_cells", "cells / SWR", "Δ cells per SWR"), ("traj_net", "net displacement", "Δ net displacement (position bins)")]


# --- fig2f_d2_replay_content.py
RED, BOX = "#d62728", ["#c03d3e", "#e57373", "#f3a6a6"]


# --- fig2f_d2_replay_content.py (trajectory span: computed and printed, not drawn)
PLOT_DROP = {"traj_span"}


# --- fig2f_d2_replay_content.py
X6 = np.arange(1, 7)


from final_figs import assert_local_path   # writes go to ../figure_output only


# --- fig2f_d2_replay_content.py
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


# --- fig2f_d2_replay_content.py (main, with its table W as the argument)
def draw_main(W):
    """main() of fig2f_d2_replay_content.py on the per-window table W: (a) per-session Spearman rho of the
    ripple-excised d2 with each measure, (b, c) session-centred means by within-session d2 sextile."""
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
