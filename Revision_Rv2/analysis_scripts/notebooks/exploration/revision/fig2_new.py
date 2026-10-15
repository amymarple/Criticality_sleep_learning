"""
fig2_new.py -- Figure 2 panels: those drawn inline by `fig2_composite.py` redrawn from the same
CSVs, and those with their own producers imported or embedded.

Panels:
  a  pre- vs post-sleep d2 (the submitted Fig 2A), with the RIPPLE-EXCISED d2. Same selection as the
     published panel (`Fig2_A_FigS2_B_FigS3_A.ipynb` cell 1: the 9 sessions of
     data/ripple_barr/manipulation/pre_post, NREM windows, presleep vs midsleep -- postsleep where a
     session has no midsleep -- per-session mean), with d2 replaced by the published ripple-ablated d2
     (`{day}_CA1_D2_time_resolved_rippleablate.mat`, joined on window centre); paired signed-rank.
  b  avalanche DCC by SWR-rate tercile:
     `ripple_criticality_all_sessions.csv`, 1x IEI, NREM, 16 sessions.
  c  peri-SWR branching ratio, ripple-excised, example PPP21_day5 + 9-session inset (composite g).
  d  the submitted replay example (Fig 2B, PPP21_day4: replay event 1731 and non-replay event 934, right
     template), the original SVGs rasterised with Inkscape into data/revision/cache_fig2_new/.
  e  replay probability by within-session d2 sextile, both arms (composite d).
  f  ripple-excised d2 by sextile of replay-SWR and non-replay-SWR rate (composite e).
  g  per-SWR replay model, d2 with SWR rate, amplitude, duration added
     (`fig2f_swr_property_ladder.csv`).
  h  ablation, mixed-model test (composite f' = `fig2f_standalone.draw`, the final Fig 2f).
  i  ISI entropy by SWR-rate sextile, both arms (composite h).

    -> data/revision/log/fig2_new_prepost.csv
    -> reports/revision_figs/fig2_new.png / .svg
"""
from __future__ import annotations

import os
import sys
import subprocess
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
from run_fig1_metrics import assert_local_path                                    # noqa: E402
from r35_glm_ablation import LOGDIR, FIGDIR                                        # noqa: E402
from fig2_composite import sextile_panel, _stars, _despine, RED, DARK, PINK, GREY, X6   # noqa: E402
from fig1_d2_vs_mua_2x2 import nrem_mat                                            # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
PREPOST = os.path.join(REPO, "data", "ripple_barr", "manipulation", "pre_post")
CACHE = os.path.join(REPO, "data", "revision", "cache_fig2_new")
EXAMPLES = os.path.join(REPO, "reports", "new_planc", "individual_fig_main")
EX_SVG = ["fig2_B_Example_PPP21_day4_replay_event_1731_right", "fig2_B_Example_PPP21_day4_non_replay_event_934_right"]
INKSCAPE = r"C:\Program Files\Inkscape\bin\inkscape.exe"
PRE_POST_COL = {"Pre": "#f6d5ff", "Post": "#dd55ff"}                   # the submitted panel's palette
C_DCC = "#6a3d9a"


# --------------------------------------------------------------------------- a: data
def prepost():
    import pickle
    rows = []
    for f in sorted(os.listdir(PREPOST)):
        if not f.endswith(".pkl"):
            continue
        sess = f[:-4]; a, d = sess.split("_")
        df = pickle.load(open(os.path.join(PREPOST, f), "rb"))["ca1_complete_df"].copy()
        df["sleep_epoch"] = df.sleep_epoch.astype(str).str.lower()
        g = df[df.state.astype(str).str.upper() == "NREM"]
        post = "midsleep" if (g.sleep_epoch == "midsleep").any() else "postsleep"
        g = g[g.sleep_epoch.isin(["presleep", post])].copy()
        g["epoch"] = np.where(g.sleep_epoch == "presleep", "Pre", "Post")
        M = nrem_mat(rf"W:\data\PPP\{a}\{d}", "_rippleablate")
        g["key"] = g.center_s.astype(float).round(3)
        if M is not None:
            M = M.assign(key=M.center_s.round(3)).drop_duplicates("key").set_index("key").v
            g["d2_excised"] = g.key.map(M)
        else:
            g["d2_excised"] = np.nan
        for ep, h in g.groupby("epoch"):
            rows.append(dict(session=sess, epoch=ep, post_epoch=post, n=len(h), joined=int(h.d2_excised.notna().sum()),
                             d2_full=h.d2.mean(), d2_excised=h.d2_excised.mean(), ripple_rate=h.ripple_rate_30s.mean()))
    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(os.path.join(LOGDIR, "fig2_new_prepost.csv")), index=False)
    for col in ("d2_full", "d2_excised"):
        P = R.pivot(index="session", columns="epoch", values=col).dropna()
        dlt = P.Post - P.Pre
        print(f"a  {col}: Pre {P.Pre.mean():.4f}, Post {P.Post.mean():.4f}, Post - Pre {dlt.mean():+.4f} ± "
              f"{dlt.std(ddof=1) / np.sqrt(len(dlt)):.4f}, {int((dlt > 0).sum())}/{len(dlt)} up, "
              f"signed-rank p = {stats.wilcoxon(P.Post, P.Pre).pvalue:.4f}")
    print("   join rate (excised d2 found / NREM windows): "
          + ", ".join(f"{s} {100 * g.joined.sum() / g.n.sum():.0f} %" for s, g in R.groupby("session")))
    return R


def panel_prepost(ax, R):
    P = R.pivot(index="session", columns="epoch", values="d2_excised").dropna()[["Pre", "Post"]]
    bp = ax.boxplot([P.Pre, P.Post], positions=[0, 1], widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=2), whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5),
                    boxprops=dict(lw=1.5))
    for patch, k in zip(bp["boxes"], ("Pre", "Post")):
        patch.set_facecolor(PRE_POST_COL[k])
    for _, r in P.iterrows():
        ax.plot([0, 1], [r.Pre, r.Post], "-", color="0.55", lw=1.1, zorder=2)
    ax.plot(np.zeros(len(P)), P.Pre, "o", color="k", ms=5, zorder=3)
    ax.plot(np.ones(len(P)), P.Post, "o", color="k", ms=5, zorder=3)
    dlt = P.Post - P.Pre
    up = int((dlt > 0).sum()) if dlt.mean() > 0 else int((dlt < 0).sum())
    ax.text(0.5, 1.0, f"{up}/{len(P)} {'↑' if dlt.mean() > 0 else '↓'} {_stars(stats.wilcoxon(P.Post, P.Pre).pvalue)}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pre-sleep", "post-sleep"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("d2, ripple-excised\n(2024 Euclidean, p = 20; NREM)")
    _despine(ax)


# --------------------------------------------------------------------------- b
def panel_dcc(ax):
    T = pd.read_csv(os.path.join(LOGDIR, "ripple_criticality_all_sessions.csv"))
    T = T[(T.pct == 30) & T.group.isin(["T1_low", "T2_mid", "T3_high"])].assign(group=lambda t: t.group.str[:2])
    P = T.pivot_table(index="session", columns="group", values="DCC").dropna()[["T1", "T2", "T3"]]
    R = T.pivot_table(index="session", columns="group", values="ripple_rate_mean").loc[P.index]
    x = np.array([1, 2, 3])
    for _, row in P.iterrows():
        ax.plot(x, row.to_numpy(), "-", color="#bbbbbb", lw=0.9, zorder=1)
    med, q1, q3 = P.median().to_numpy(), P.quantile(.25).to_numpy(), P.quantile(.75).to_numpy()
    ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="o-", color=C_DCC, lw=2.2, ms=7, capsize=4, zorder=3)
    ax.set_yscale("log")
    t31, p31 = int((P.T3 > P.T1).sum()), stats.wilcoxon(P.T3, P.T1).pvalue
    ax.text(0.5, 1.0, f"high > low {t31}/{len(P)} {_stars(p31)}", transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"low\n{R.T1.median():.2f} Hz", f"mid\n{R.T2.median():.2f} Hz", f"high\n{R.T3.median():.2f} Hz"])
    ax.set_xlabel("SWR-rate tercile (NREM)")
    ax.set_ylabel("avalanche DCC\n(1× IEI; median, IQR)")
    _despine(ax)
    print(f"b  DCC terciles: " + " / ".join(f"{v:.3f}" for v in med) + f"; high > low {t31}/{len(P)}, p = {p31:.4f}")


# --------------------------------------------------------------------------- c (composite g)
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


# --------------------------------------------------------------------------- d
def example_pngs():
    os.makedirs(assert_local_path(CACHE), exist_ok=True)
    out = []
    for b in EX_SVG:
        png = os.path.join(CACHE, f"{b}.png")
        if not os.path.exists(png):
            subprocess.run([INKSCAPE, os.path.join(EXAMPLES, f"{b}.svg"), "--export-type=png", "--export-dpi=200",
                            f"--export-filename={assert_local_path(png)}"], check=True, capture_output=True)
        out.append(png)
    return out


# --------------------------------------------------------------------------- e, f (composite d, e)
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


# --------------------------------------------------------------------------- g
def panel_ladder(ax):
    from fig2f_seat_ladders import box_panel
    L = pd.read_csv(os.path.join(LOGDIR, "fig2f_swr_property_ladder.csv"))
    L = L[(L.part == "ladder") & (L.session != "__pooled__")]
    rungs = ["d2", "+ SWR rate", "+ amplitude", "+ duration"]
    data = [L[L.model == r].sort_values("session").b1.to_numpy() for r in rungs]
    from fig2f_swr_property_ladder import LADDER_DISPLAY   # every box is d2's coefficient
    box_panel(ax, data, [LADDER_DISPLAY[r] for r in rungs], ["#c03d3e", "#d4595a", "#e57373", "#ee8f8f"],
              ylabel="d2 coefficient, log-odds per SD\n(ripple-excised d2; binomial, per session)")
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
    R = prepost()
    W = pd.read_csv(os.path.join(LOGDIR, "excised_br_entropy_windows.csv"))
    S = pd.read_csv(os.path.join(LOGDIR, "fig2_agh_nine_sessions.csv"))

    fig = plt.figure(figsize=(16.5, 15.5))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38, top=0.93)
    ax = {k: fig.add_subplot(gs[i // 3, i % 3]) for i, k in enumerate("abc")}
    gd = gs[1, 0].subgridspec(1, 2, wspace=0.02)
    ax["d1"], ax["d2"] = fig.add_subplot(gd[0]), fig.add_subplot(gd[1])
    for i, k in enumerate("efghi", start=4):
        ax[k] = fig.add_subplot(gs[i // 3, i % 3])

    panel_prepost(ax["a"], R)
    panel_dcc(ax["b"])
    panel_psth(ax["c"])
    for a_, png, t in zip((ax["d1"], ax["d2"]), example_pngs(), ("replay SWR", "non-replay SWR")):
        a_.imshow(imread(png)); a_.axis("off")
        a_.text(0.5, -0.02, t, transform=a_.transAxes, ha="center", va="top", fontsize=9.5, fontweight="bold")
    panel_replay_prob(ax["e"], W)
    panel_replay_rates(ax["f"], W)
    panel_ladder(ax["g"])
    from fig2f_standalone import compute as fig2f_compute, draw as fig2f_draw
    fig2f_draw(ax["h"], fig2f_compute(), compact=True)
    sextile_panel(ax["i"], S, "isi_entropy_pooled", "ISI entropy (bits)", arms=(("full window", GREY),), corner="right")
    sextile_panel(ax["i"], S, "isi_excised", "ISI entropy (bits)", arms=(("ripple-excised", RED),), corner="right")
    ax["i"].legend(frameon=False, loc="lower left")

    titles = {"a": "a  pre- vs post-sleep", "b": "b  avalanche DCC", "c": "c  branching ratio around SWRs",
              "d1": "d  replay example (PPP21_day4)", "e": "e  replay probability", "f": "f  replay vs non-replay SWRs",
              "g": "g  per-SWR replay model", "h": "h  ablation, mixed-model test", "i": "i  ISI entropy"}
    for k, t in titles.items():
        ax[k].set_title(t, loc="left", fontweight="bold", pad=20 if k in ("a", "b", "c", "d1") else 6)
    fig.suptitle("Figure 2 (reorganised) — NREM; red = ripple-excised, grey = full window", fontsize=10, y=0.985)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"fig2_new.{ext}")), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIGDIR}\\fig2_new.png/.svg")


if __name__ == "__main__":
    main()
