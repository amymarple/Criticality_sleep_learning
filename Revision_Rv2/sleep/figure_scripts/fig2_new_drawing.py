"""Drawing functions from analysis_scripts/revision/ (fig2_composite.py, fig2_new.py, fig2f_seat_ladders.py, fig2f_swr_property_ladder.py) used by the
figure notebooks. LOGDIR: the folder of the notebook's tables, set by the notebook.
"""
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

LOGDIR = None


# --- the matplotlib settings fig2f_seat_ladders.py makes when it is imported (once)
_DONE_fig2f_seat_ladders = []


def _imported_fig2f_seat_ladders():
    if _DONE_fig2f_seat_ladders:
        return
    _DONE_fig2f_seat_ladders.append(1)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


# --- fig2_composite.py
RED, DARK, PINK, GREY = "#d62728", "#691c1b", "#e9afc6", "#7f7f7f"


# --- fig2_composite.py
def _stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


# --- fig2_composite.py
def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# --- fig2_new.py
def panel_psth(ax):
    EX = "PPP21_day5"
    Ce = pd.read_csv(os.path.join(LOGDIR, f"fig2g_psth_curves_{EX}.csv"))
    Ce = Ce[Ce.arm == "excised"].sort_values("lag_s")
    PS = pd.read_csv(os.path.join(LOGDIR, "fig2g_psth_nine_sessions.csv"))
    for key, ek, c, lab in (("br_ripple", "sem_ripple", RED, "SWRs"), ("br_shuffle", "sem_shuffle", PINK, "shuffle")):
        ax.plot(Ce.lag_s, Ce[key], "-o", ms=3.5, lw=2, color=c, label=lab)
        ax.fill_between(Ce.lag_s, Ce[key] - Ce[ek], Ce[key] + Ce[ek], color=c, alpha=0.25, lw=0)
    ax.text(0.02, 0.02, f"example {EX}; shade = SEM over ripples", transform=ax.transAxes, va="bottom", fontsize=8, color="0.3")
    ax.axvline(0, ls="--", lw=1, color="red")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.9 * (hi - lo))
    ax.set_xlabel("time from SWR peak (s)")
    ax.set_ylabel("branching ratio\n(ripple-excised, example session)")
    ax.legend(frameon=False, loc="lower right")
    _despine(ax)
    ins = ax.inset_axes([0.60, 0.60, 0.38, 0.36])
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


# --- fig2_composite.py
X6 = np.arange(1, 7)


# --- fig2_new.py
def panel_replay_prob(ax, W):
    for d2col, c, lab, dy in (("d2_published", GREY, "full-window d2", 0.0), ("d2_excised", DARK, "ripple-excised d2", 0.09)):
        rows = []
        for s, g in W.groupby("session"):
            g = g.dropna(subset=[d2col, "replay_prob"])
            g = g[g.ripple_rate * 30 >= 1]
            sex = pd.qcut(g[d2col].rank(method="first"), 6, labels=False) + 1
            m = (g.groupby(sex)["replay_prob"].mean() - g["replay_prob"].mean()) * 100
            rows.append(dict(session=s, rho=stats.spearmanr(sex, g["replay_prob"])[0], **{f"s{k}": m.get(k, np.nan) for k in X6}))
        D = pd.DataFrame(rows)
        Y = D[[f"s{k}" for k in X6]].to_numpy()
        m, se = np.nanmean(Y, 0), np.nanstd(Y, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Y), 0))
        ax.errorbar(X6, m, yerr=se, fmt="o-", capsize=4, lw=2, ms=5, color=c, label=lab, zorder=3 if c == DARK else 2)
        v = D.rho.to_numpy()
        ax.text(0.03, 0.96 - dy, f"{lab}: {int((v > 0).sum())}/{len(v)} ↑ {_stars(stats.wilcoxon(v).pvalue)}",
                transform=ax.transAxes, va="top", color=c, fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xticks(X6); ax.set_xlabel("d2 sextile (within session)")
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.set_ylabel("Δ replay probability (percentage points)\nrelative to session mean")
    _despine(ax)


# --- fig2_new.py
def panel_replay_rates(ax, W):
    for col, c, fmt, lab in (("rate_in", DARK, "o-", "replay SWRs"), ("rate_out", RED, "s--", "non-replay SWRs")):
        rows = []
        for s_, g in W.groupby("session"):
            g = g.dropna(subset=["d2_excised", "ripple_rate", "replay_prob"]).copy()
            g["rate_in"] = g.ripple_rate * g.replay_prob
            g["rate_out"] = g.ripple_rate * (1 - g.replay_prob)
            sex = pd.qcut(g[col].rank(method="first"), 6, labels=False) + 1
            m_ = g.groupby(sex)["d2_excised"].mean()
            rows.append(dict(session=s_, rho=stats.spearmanr(sex, g["d2_excised"])[0], **{f"s{k}": m_.get(k, np.nan) for k in X6}))
        E = pd.DataFrame(rows)
        Ye = E[[f"s{k}" for k in X6]].to_numpy()
        me, see = np.nanmean(Ye, 0), np.nanstd(Ye, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Ye), 0))
        ax.errorbar(X6, me, yerr=see, fmt=fmt, capsize=4, lw=2, ms=5, color=c, label=lab)
        ve = E.rho.to_numpy()
        ax.text(0.03, 0.96 - (0.09 if col == "rate_out" else 0), f"{lab}: {int((ve > 0).sum())}/{len(ve)} ↑ {_stars(stats.wilcoxon(ve).pvalue)}",
                transform=ax.transAxes, va="top", color=c, fontsize=9)
    ax.set_xticks(X6); ax.set_xlabel("SWR rate sextile (replay / non-replay SWRs)")
    ax.set_ylabel("d2 (ripple-excised)")
    ax.legend(frameon=False, loc="lower right")
    _despine(ax)


# --- fig2f_seat_ladders.py
def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# --- fig2f_seat_ladders.py
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


# --- fig2f_swr_property_ladder.py
LADDER_DISPLAY = {"d2": "alone", "+ SWR rate": "+ rate", "+ amplitude": "+ rate, amplitude",
                  "+ duration": "+ rate, amplitude, duration", "+ cells / SWR": "+ ... cells / SWR",
                  "+ MUA": "+ ... MUA"}


# --- fig2_new.py
def panel_ladder(ax):
    _imported_fig2f_seat_ladders()
    L = pd.read_csv(os.path.join(LOGDIR, "fig2f_swr_property_ladder.csv"))
    L = L[(L.part == "ladder") & (L.session != "__pooled__")]
    rungs = ["d2", "+ SWR rate", "+ amplitude", "+ duration"]
    data = [L[L.model == r].sort_values("session").b1.to_numpy() for r in rungs]
    pass
    box_panel(ax, data, [LADDER_DISPLAY[r] for r in rungs], ["#c03d3e", "#d4595a", "#e57373", "#ee8f8f"],
              ylabel="d2 coefficient, log-odds per SD\n(ripple-excised d2; binomial, per session)")
    _despine(ax)


# --- fig2_composite.py
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
