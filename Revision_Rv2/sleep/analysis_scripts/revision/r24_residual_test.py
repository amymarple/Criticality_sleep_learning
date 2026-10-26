"""
r24_residual_test.py -- does d2 carry information beyond the branching ratio m, and m beyond d2?
(Extended Data Fig. 4e-g)

Run symmetrically, so that the result states what each measure carries that the other does not.

DESIGN. Per session, on the same windows: OLS of d2 on m -> residual d2|m (d2 units); OLS of m on d2
-> m|d2. Then per-session Spearman of d2, d2|m, m and m|d2 with the event rate, full window and
excised arms. Fisher-z mean +- SEM, sessions in the direction, signed-rank. Cohorts and tables (all
existing, nothing refitted):
  replay   9 decoded sessions, NREM decoded time (`excised_br_entropy_windows.csv`): replay-SWR rate
           (replay_prob x ripple_rate), partial on SWR rate; d2_published / d2_excised;
           m BR_recomp / BR_excised
  SWR      16 Fig-1 sessions, NREM (`d2_arms_raw_windows.csv` + `r24_br_ripple_windows{,_all}.csv`):
           SWR rate; d2_full / d2_excised; m_full / m_excised
  barrage  12 barrage-cohort sessions, NREM (`r16_event_excised_windows.csv` + `_br.csv`): barrage
           rate, partial on ripple rate; d2 / d2_bg and BR / BR_bg, where "bg" excises
           ripples AND barrages
POP=pyr (default): pyramidal m in every cohort (the published population; the only one stored for
all three). POP=all: m on all CA1 units, the population d2 is fitted on -- SWR cohort only (stored
there); the stricter residualisation, since all-unit m shares more variance with d2.

Writes only to D:.
    -> data/revision/log/r24_residual_test{,_all}.csv
    -> reports/revision_figs/r24_residual_test.png / .svg   (POP=pyr; Extended Data Fig. 4e-g)
    -> reports/revision_figs/r24_residual_test_4col.png     (POP=pyr FIG_ALL=1; adds m and m | d2)
       reports/revision_figs/backup/r24_residual_test_all.png (POP=all)
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
from run_fig1_metrics import assert_local_path                    # noqa: E402
from r35_glm_ablation import LOGDIR, FIGDIR                       # noqa: E402
from d2_arms_raw import fisher, stars, GREY, RED, BLUE            # noqa: E402
from r35_fig2d_replot import partial_spearman                     # noqa: E402

POP = os.environ.get("POP", "pyr")
_SFX = "" if POP == "pyr" else f"_{POP}"
OUT = os.path.join(LOGDIR, f"r24_residual_test{_SFX}.csv")
MEASURES = [("d2", "d2"), ("d2_res", "d2 | m"), ("m", "m"), ("m_res", "m | d2")]
# the figure shows d2 and d2 | m only; FIG_ALL=1 draws all four
FIG_ALL = os.environ.get("FIG_ALL") == "1"
FIG_MEASURES = MEASURES if FIG_ALL else MEASURES[:2]


def cohorts():
    """-> {name: (DataFrame with session, rate, covariate, and per arm d2_<arm>, m_<arm>; rate label)}"""
    C = {}
    R = pd.read_csv(os.path.join(LOGDIR, "excised_br_entropy_windows.csv")).dropna(subset=["replay_prob"])
    R["rate"] = R.replay_prob * R.ripple_rate; R["cov"] = R.ripple_rate
    R = R.rename(columns={"d2_published": "d2_full"})
    if POP == "pyr":
        R = R.rename(columns={"BR_recomp": "m_full", "BR_excised": "m_excised"})
        C["replay"] = (R, "replay-SWR rate, given SWR rate (NREM, decoded time)")
    else:   # all-unit m exists for the replay windows of the 8 decoded sessions that are also in Fig 1
        MA = pd.read_csv(os.path.join(LOGDIR, f"r24_br_ripple_windows{_SFX}.csv"))[["session", "center_s", "m_full", "m_excised"]]
        C["replay"] = (R.merge(MA, on=["session", "center_s"]), "replay-SWR rate, given SWR rate (NREM, decoded time)")
    W = pd.read_csv(os.path.join(LOGDIR, "d2_arms_raw_windows.csv"))
    W = W[W.state == "NREM"][["session", "center_s", "d2_full", "d2_excised", "ripple_rate"]]
    M = pd.read_csv(os.path.join(LOGDIR, f"r24_br_ripple_windows{_SFX}.csv"))[["session", "center_s", "m_full", "m_excised"]]
    S = W.merge(M, on=["session", "center_s"]); S["rate"] = S.ripple_rate; S["cov"] = np.nan
    C["SWR"] = (S, "SWR rate (NREM)")
    if POP == "pyr":
        A = pd.read_csv(os.path.join(LOGDIR, "r16_event_excised_windows.csv"))
        B = pd.read_csv(os.path.join(LOGDIR, "r16_event_excised_windows_br.csv"))
        X = A.merge(B[["session", "w", "BR", "BR_bg"]], on=["session", "w"])
        X = X.rename(columns={"d2": "d2_full", "d2_bg": "d2_excised", "BR": "m_full", "BR_bg": "m_excised"})
        X["rate"] = X.barrage_rate; X["cov"] = X.ripple_rate
        C["barrage"] = (X, "barrage rate, given SWR rate (NREM; excised = ripples and barrages)")
    return C


def resid(y, x):
    b1, b0 = np.polyfit(x, y, 1)
    return y - (b0 + b1 * x)


def main():
    rows = []
    for name, (D, lab) in cohorts().items():
        for arm in ("full", "excised"):
            for s, g in D.groupby("session"):
                g = g.dropna(subset=[f"d2_{arm}", f"m_{arm}", "rate"])
                if len(g) < 20 or g.rate.nunique() <= 2:
                    continue
                d2, m = g[f"d2_{arm}"].to_numpy(float), g[f"m_{arm}"].to_numpy(float)
                v = dict(d2=d2, d2_res=resid(d2, m), m=m, m_res=resid(m, d2))
                r_dm = np.corrcoef(d2, m)[0, 1]
                for key, x in v.items():
                    if g["cov"].notna().all():
                        rho = partial_spearman(x, g.rate.to_numpy(float), g["cov"].to_numpy(float))
                    else:
                        rho = stats.spearmanr(x, g.rate).statistic
                    rows.append(dict(cohort=name, arm=arm, session=s, measure=key, rho=rho, r_d2_m=r_dm, n=len(g)))
    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(OUT), index=False)

    summ = {}
    print(f"m population: {POP}")
    for (c, arm, meas), g in R.groupby(["cohort", "arm", "measure"], sort=False):
        v = g.rho.dropna().to_numpy()
        mu, se, n = fisher(v)
        k = int((v > 0).sum()) if mu > 0 else int((v < 0).sum())
        p = stats.wilcoxon(v).pvalue
        summ[(c, arm, meas)] = (mu, se, n, k, p, v)
        print(f"  {c:8s} {arm:8s} {meas:7s} {mu:+.3f} +- {se:.3f}  {k}/{n} {'+' if mu > 0 else '-'}  p = {p:.4f}")
    for (c, arm), g in R[R.measure == "d2"].groupby(["cohort", "arm"], sort=False):
        print(f"  {c:8s} {arm:8s} r(d2, m) within session: median {g.r_d2_m.median():+.2f} (shared variance {100 * g.r_d2_m.median() ** 2:.0f} %)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    names = [c for c in ("replay", "SWR", "barrage") if c in set(R.cohort)]
    labs = dict((c, lab) for c, (_, lab) in cohorts().items())
    fig, axes = plt.subplots(1, len(names), figsize=((5.0 if FIG_ALL else 3.4) * len(names), 4.2), squeeze=False)
    for ax, (letter, c) in zip(axes[0], zip("abc", names)):
        xs = np.arange(len(FIG_MEASURES))
        # barrage column: excised arm in blue
        for j, (arm, col, off) in enumerate((("full", GREY, -0.17), ("excised", BLUE if c == "barrage" else RED, 0.17))):
            for i, (key, _) in enumerate(FIG_MEASURES):
                if (c, arm, key) not in summ:
                    continue
                mu, se, n, k, p, v = summ[(c, arm, key)]
                ax.scatter(xs[i] + off + np.random.default_rng(10 * i + j).uniform(-0.05, 0.05, v.size), v, s=10,
                           color=col, alpha=0.5, lw=0)
                ax.errorbar(xs[i] + off, mu, yerr=se, fmt="s", color=col, mec="k", ms=6, capsize=3, zorder=4,
                            label=("full window" if arm == "full" else "excised") if i == 0 else None)
                ax.text(xs[i] + off, 0.62 - 0.1 * j, f"{k}/{n}{stars(p) if p < .05 else ''}", ha="center", fontsize=7.5, color=col,
                        fontweight="bold" if p < .05 else "normal")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        if FIG_ALL:
            ax.axvline(1.5, color="0.7", lw=0.8, ls=":")
        ax.set_xticks(xs); ax.set_xticklabels([m for _, m in FIG_MEASURES]); ax.set_xlim(-0.6, len(FIG_MEASURES) - 0.4); ax.set_ylim(-0.7, 0.7)
        ax.set_ylabel("per-session Spearman ρ" + (" (partial)" if "given" in labs[c] else ""))
        n_s = R[R.cohort == c].session.nunique()
        ax.set_title(f"{letter}  {labs[c].split(' (')[0]}\n{n_s} sessions; {labs[c].split(' (', 1)[1][:-1] if ' (' in labs[c] else ''}",
                     loc="left", fontweight="bold", fontsize=9)
        if letter == "a":
            ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"d2 | m = d2 (2024 Euclidean, 40 ms, p = 20) with its within-session linear dependence on the branching ratio m removed"
                 f"{'; m | d2 likewise' if FIG_ALL else ''} (m: {'pyramidal' if POP == 'pyr' else 'all CA1 units'}, 4 ms, k_max = 200); "
                 "same 30 s windows", fontsize=8.5, y=1.03)
    fig.tight_layout()
    # POP=pyr: 2-column (d2, d2 | m) or, with FIG_ALL=1, 4-column (adds m and m | d2: m's own information
    # beyond d2); POP=all: backup/
    out = (os.path.join(FIGDIR, f"r24_residual_test{'_4col' if FIG_ALL else ''}.png") if POP == "pyr"
           else os.path.join(FIGDIR, "backup", f"r24_residual_test{_SFX}{'_4col' if FIG_ALL else ''}.png"))
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(out[:-3] + ext), dpi=200, bbox_inches="tight")
    print("  ->", out)


if __name__ == "__main__":
    main()
