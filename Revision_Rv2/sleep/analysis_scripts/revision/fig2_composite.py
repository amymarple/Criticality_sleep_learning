"""
fig2_composite.py -- Figure 2, revision composite: the quantitative panels rebuilt on the nine decoded
sessions, non-overlapping windows, in the manuscript's own colour code.

Colour code (from the Fig 2 notebooks / the submitted figure): d2 and SWR curves
tab:red / #d62728; replay dark red #691c1b (solid, circles) vs non-replay red (dashed,
squares); shuffles pink #e9afc6; boxes the Fig 2F red family (#c03d3e, #e57373,
#f3a6a6); greys for paired lines and the full-window reference arm.

Panels (b, c are the example panels of the submitted figure and are not reproduced):
  a  d2 vs SWR-rate sextile, raw units, full-window (grey) and ripple-excised (red)   [fig2_agh_nine_sessions.csv]
  d  replay probability vs ripple-excised-d2 sextile, mean +- SEM over sessions       [excised_br_entropy_windows.csv]
  e  d2 (ripple-excised) vs sextile of replay-SWR rate (dark red, solid) and of
     non-replay-SWR rate (red, dashed) -- the submitted panel's form, nine sessions;
     rate_in + rate_out == rate_all, so this is descriptive                              [excised_br_entropy_windows.csv]
  e' the matched-rate panel on common support (5.4's replacement, embedded)           [fig2e_nonoverlap_panel.png]
  f  binomial ladder: log-odds of replay per SD of ripple-excised d2, 4 rungs           [metric_replay_ladder.csv]
  g  peri-ripple branching-ratio PSTH, ripple-excised, EXAMPLE session PPP21_day5
     (shade = SEM over its ripples), inset = per-session change (9)       [fig2g_psth_*_PPP21_day5.csv, fig2g_psth_nine_sessions.csv]
  h  ISI entropy vs SWR-rate sextile, raw units, both arms                             [fig2_agh_nine_sessions.csv]

    -> reports/revision_figs/fig2_composite_nine_sessions.png / .svg
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

RED, DARK, PINK, GREY = "#d62728", "#691c1b", "#e9afc6", "#7f7f7f"
BOX = ["#c03d3e", "#d4595a", "#e57373", "#ee8f8f", "#f3a6a6"]
X6 = np.arange(1, 7)


def _stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def sextile_panel(ax, S, metric, ylab, arms=(("full window", GREY), ("ripple-excised", RED)), corner="left"):
    for arm, c in arms:
        P = S[(S.metric == metric) & (S.arm == arm)]
        Y = P[[f"s{k}" for k in X6]].to_numpy()
        m, se = np.nanmean(Y, 0), np.nanstd(Y, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Y), 0))
        ax.errorbar(X6, m, yerr=se, fmt="o-", capsize=4, lw=2, ms=5, color=c, label=arm, zorder=3 if c == RED else 2)
        v = P.rho.to_numpy()
        up = np.median(v) > 0
        nd = int((v > 0).sum()) if up else int((v < 0).sum())
        ax.text(0.03 if corner == "left" else 0.97, 0.96 - (0.09 if c == RED else 0),
                f"{nd}/{len(v)} {'↑' if up else '↓'} {_stars(stats.wilcoxon(v).pvalue)}",
                transform=ax.transAxes, va="top", ha=corner, color=c, fontsize=9)
    ax.set_xticks(X6)
    ax.set_xlabel("SWR rate sextile")
    ax.set_ylabel(ylab)
    _despine(ax)


def main():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "axes.linewidth": 1.2, "legend.fontsize": 9,
                         "xtick.major.width": 1.2, "ytick.major.width": 1.2})
    S = pd.read_csv(os.path.join(LOGDIR, "fig2_agh_nine_sessions.csv"))
    W = pd.read_csv(os.path.join(LOGDIR, "excised_br_entropy_windows.csv"))
    L = pd.read_csv(os.path.join(LOGDIR, "metric_replay_ladder.csv"))
    PC = pd.read_csv(os.path.join(LOGDIR, "fig2g_psth_curves.csv"))
    PS = pd.read_csv(os.path.join(LOGDIR, "fig2g_psth_nine_sessions.csv"))
    n_sess = W.session.nunique()

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.6))
    (ax_a, ax_d, ax_e, ax_e2), (ax_f, ax_g, ax_h, ax_x) = axes

    # f' -- the final Fig 2f panel: replay-rate ablation of d2 / MUA / FR std | MUA / E/I
    # with the mixed model's stars and brackets, drawn by fig2f_standalone.draw
    from fig2f_standalone import compute as fig2f_compute, draw as fig2f_draw
    fig2f_draw(ax_x, fig2f_compute(), compact=True)
    ax_x.set_title("f′  ablation, mixed-model test", loc="left", fontweight="bold")

    # a -----------------------------------------------------------------
    sextile_panel(ax_a, S, "d2_published", "d2 (2024 Euclidean, p = 20)",
                  arms=(("full window", GREY),))
    sextile_panel(ax_a, S, "d2_excised", "d2 (2024 Euclidean, p = 20)",
                  arms=(("ripple-excised", RED),))
    ax_a.legend(frameon=False, loc="lower right")
    ax_a.set_title("a", loc="left", fontweight="bold")

    # d -----------------------------------------------------------------
    # both d2 arms: sextiles are within session, so every tile holds the same number of
    # windows in the two arms and the curves are directly comparable
    for d2col, c, lab, dy in (("d2_published", GREY, "full-window d2", 0.0),
                              ("d2_excised", DARK, "ripple-excised d2", 0.09)):
        rows = []
        for s, g in W.groupby("session"):
            g = g.dropna(subset=[d2col, "replay_prob"])
            g = g[g.ripple_rate * 30 >= 1]
            sex = pd.qcut(g[d2col].rank(method="first"), 6, labels=False) + 1
            # session-centred, in probability units: removes the between-session level
            # from the SEM, as the submitted panel's within-session z-score did, without
            # leaving probability units
            m = (g.groupby(sex)["replay_prob"].mean() - g["replay_prob"].mean()) * 100
            rows.append(dict(session=s, rho=stats.spearmanr(sex, g["replay_prob"])[0],
                             **{f"s{k}": m.get(k, np.nan) for k in X6}))
        D = pd.DataFrame(rows)
        Y = D[[f"s{k}" for k in X6]].to_numpy()
        m, se = np.nanmean(Y, 0), np.nanstd(Y, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Y), 0))
        ax_d.errorbar(X6, m, yerr=se, fmt="o-", capsize=4, lw=2, ms=5, color=c, label=lab,
                      zorder=3 if c == DARK else 2)
        v = D.rho.to_numpy()
        ax_d.text(0.03, 0.96 - dy, f"{lab}: {int((v > 0).sum())}/{len(v)} ↑ {_stars(stats.wilcoxon(v).pvalue)}",
                  transform=ax_d.transAxes, va="top", color=c, fontsize=9)
        if c == DARK:
            v_ex, Y_ex = v, Y
    v, Y = v_ex, Y_ex
    ax_d.legend(frameon=False, loc="lower right")
    ax_d.set_xticks(X6)
    ax_d.set_xlabel("d2 sextile (within session)")
    ax_d.axhline(0, ls="--", lw=1, color="k")
    ax_d.set_ylabel("Δ replay probability (percentage points)\nrelative to session mean")
    ax_d.set_title("d", loc="left", fontweight="bold")
    _despine(ax_d)

    # e (native) --------------------------------------
    for col, c, fmt, lab in (("rate_in", DARK, "o-", "replay SWRs"), ("rate_out", RED, "s--", "non-replay SWRs")):
        rows = []
        for s_, g in W.groupby("session"):
            g = g.dropna(subset=["d2_excised", "ripple_rate", "replay_prob"]).copy()
            g["rate_in"] = g.ripple_rate * g.replay_prob
            g["rate_out"] = g.ripple_rate * (1 - g.replay_prob)
            sex = pd.qcut(g[col].rank(method="first"), 6, labels=False) + 1
            m_ = g.groupby(sex)["d2_excised"].mean()
            rows.append(dict(session=s_, rho=stats.spearmanr(sex, g["d2_excised"])[0],
                             **{f"s{k}": m_.get(k, np.nan) for k in X6}))
        E = pd.DataFrame(rows)
        Ye = E[[f"s{k}" for k in X6]].to_numpy()
        me, see = np.nanmean(Ye, 0), np.nanstd(Ye, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Ye), 0))
        ax_e.errorbar(X6, me, yerr=see, fmt=fmt, capsize=4, lw=2, ms=5, color=c, label=lab)
        ve = E.rho.to_numpy()
        ax_e.text(0.03, 0.96 - (0.09 if col == "rate_out" else 0),
                  f"{lab}: {int((ve > 0).sum())}/{len(ve)} ↑ {_stars(stats.wilcoxon(ve).pvalue)}",
                  transform=ax_e.transAxes, va="top", color=c, fontsize=9)
    ax_e.set_xticks(X6)
    ax_e.set_xlabel("SWR rate sextile (replay / non-replay SWRs)")
    ax_e.set_ylabel("d2 (ripple-excised)")
    ax_e.legend(frameon=False, loc="lower right")
    ax_e.set_title("e", loc="left", fontweight="bold")
    _despine(ax_e)

    # e' (matched-rate panel, embedded) ------------------------------------
    # ripple-EXCISED d2 (D2_SOURCE=ablate run of r35_fig2e_pipeline_nonoverlap.py); the
    # published-d2 version is fig2e_nonoverlap_panel.png
    png = os.path.join(FIGDIR, "fig2e_nonoverlap_panel_ablate.png")
    if os.path.exists(png):
        ax_e2.imshow(imread(png))
    ax_e2.axis("off")
    ax_e2.set_title("e′  matched-rate, excised d2", loc="left", fontweight="bold")

    # f -----------------------------------------------------------------
    # the submitted Fig 2F covariate set (MUA, FR_std) plus E/I, cumulative (metric_replay_ladder.py)
    ladder = [("M1", "d2"), ("M2", "+ MUA"), ("M5", "+ SWR rate")]
    Lx = L[(L.metric == "d2_excised") & (L.session != "__pooled__")]
    data = [Lx[Lx.model == t].b1.to_numpy() for t, _ in ladder]
    bp = ax_f.boxplot(data, positions=range(len(ladder)), widths=0.6, patch_artist=True, showfliers=False,
                      medianprops=dict(color="0.25", lw=2), whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5),
                      boxprops=dict(lw=1.5))
    for patch, c in zip(bp["boxes"], BOX):
        patch.set_facecolor(c)
    rng = np.random.default_rng(0)
    yr = max(np.max(d) for d in data) - min(np.min(d) for d in data)
    for i, d in enumerate(data):
        ax_f.plot(i + rng.uniform(-0.12, 0.12, d.size), d, "o", ms=3, color="k", alpha=0.6, mew=0, zorder=3)
        ax_f.text(i, d.max() + 0.04 * yr, _stars(stats.wilcoxon(d).pvalue), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_f.axhline(0, ls="--", lw=1, color="k")
    ax_f.set_xticks(range(len(ladder))); ax_f.set_xticklabels([l for _, l in ladder], rotation=30, ha="right")
    ax_f.set_ylabel("log-odds of replay per SD d2\n(ripple-excised; binomial, per session)")
    ax_f.set_title("f", loc="left", fontweight="bold")
    _despine(ax_f)

    # g (example session) ------------------------------------
    EX = "PPP21_day5"
    Ce = pd.read_csv(os.path.join(LOGDIR, f"fig2g_psth_curves_{EX}.csv"))
    Ce = Ce[Ce.arm == "excised"].sort_values("lag_s")
    for key, ek, c, lab in (("br_ripple", "sem_ripple", RED, "SWRs"), ("br_shuffle", "sem_shuffle", PINK, "shuffle")):
        ax_g.plot(Ce.lag_s, Ce[key], "-o", ms=3.5, lw=2, color=c, label=lab)
        ax_g.fill_between(Ce.lag_s, Ce[key] - Ce[ek], Ce[key] + Ce[ek], color=c, alpha=0.25, lw=0)
    ax_g.text(0.02, 0.02, f"example {EX}; shade = SEM over ripples", transform=ax_g.transAxes,
              va="bottom", fontsize=8, color="0.3")
    ax_g.axvline(0, ls="--", lw=1, color="red")
    lo, hi = ax_g.get_ylim()
    ax_g.set_ylim(lo, hi + 0.9 * (hi - lo))          # headroom for the inset
    ax_g.set_xlabel("time from SWR peak (s)")
    ax_g.set_ylabel("branching ratio (ripple-excised, example session)")
    ax_g.legend(frameon=False, loc="lower right")
    ax_g.set_title("g", loc="left", fontweight="bold")
    _despine(ax_g)
    ins = ax_g.inset_axes([0.60, 0.60, 0.38, 0.36])
    P = PS[PS.arm == "excised"].set_index("session")
    for _, r in P.iterrows():
        ins.plot([0, 1], [r.change_ripple, r.change_shuffle], "-", color="0.6", lw=0.8)
    ins.plot([0] * len(P), P.change_ripple, "o", color=RED, ms=4, zorder=3)
    ins.plot([1] * len(P), P.change_shuffle, "o", color=PINK, ms=4, zorder=3)
    dd = P.change_ripple - P.change_shuffle
    ins.set_title(f"ΔBR  {int((dd < 0).sum())}/{len(dd)} ↓ {_stars(stats.wilcoxon(dd).pvalue)}", fontsize=8)
    ins.axhline(0, ls="--", lw=0.8, color="k")
    ins.set_xticks([0, 1]); ins.set_xticklabels(["SWRs", "shuffle"], fontsize=7); ins.set_xlim(-0.5, 1.5)
    ins.tick_params(axis="y", labelsize=7)
    _despine(ins)

    # h -----------------------------------------------------------------
    sextile_panel(ax_h, S, "isi_entropy_pooled", "ISI entropy (bits)", arms=(("full window", GREY),), corner="right")
    sextile_panel(ax_h, S, "isi_excised", "ISI entropy (bits)", arms=(("ripple-excised", RED),), corner="right")
    ax_h.legend(frameon=False, loc="lower left")
    ax_h.set_title("h", loc="left", fontweight="bold")

    fig.suptitle(f"Figure 2, revision — {n_sess} sessions, non-overlapping 30 s NREM windows; red = ripple-excised, grey = full window",
                 fontsize=10, y=0.995)
    plt.tight_layout()
    assert_local_path(FIGDIR)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"fig2_composite_nine_sessions.{ext}")), dpi=300)
    plt.close(fig)
    print(f"  -> {FIGDIR}\\fig2_composite_nine_sessions.png/.svg")
    print("  panel d: replay probability by excised-d2 sextile -- rho median "
          f"{np.median(v):+.3f}, {int((v > 0).sum())}/{len(v)} up, p = {stats.wilcoxon(v).pvalue:.3f}; "
          f"top - bottom sextile {np.nanmedian(Y[:, -1] - Y[:, 0]):+.3f}")


if __name__ == "__main__":
    main()
