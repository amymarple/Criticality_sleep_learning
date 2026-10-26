"""Drawing functions from analysis_scripts/revision/ (r24_residual_test.py, d2_arms_raw.py, r35_fig2d_replot.py,
run_fig1_metrics.py) used by the figure notebooks. LOGDIR, FIGDIR: the folder of the notebook's tables, set by the
notebook.
"""
import numpy as np
import os
import pandas as pd
from scipy import stats

LOGDIR = None
FIGDIR = "."


from final_figs import assert_local_path   # writes go to ../figure_output only


# --- d2_arms_raw.py
GREY, RED, BLUE = "#7f7f7f", "#d62728", "#1f77b4"   # BLUE: excised arm of barrage-rate panels


# --- d2_arms_raw.py
def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


# --- d2_arms_raw.py
def fisher(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    z = np.arctanh(np.clip(v, -0.999, 0.999))
    return float(np.tanh(z.mean())), float(np.tanh(z.std(ddof=1) / np.sqrt(len(z)))) if len(z) > 1 else np.nan, len(v)


# --- r35_fig2d_replot.py
def partial_spearman(y, x, z):
    """Spearman partial correlation of y and x given covariate z."""
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if ok.sum() < 10:
        return np.nan
    ry, rx, rz = (stats.rankdata(v[ok]) for v in (y, x, z))
    A = np.column_stack([np.ones(rz.size), rz])
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    if ey.std() == 0 or ex.std() == 0:
        return np.nan
    return float(np.corrcoef(ey, ex)[0, 1])


# --- r24_residual_test.py
POP = os.environ.get("POP", "pyr")


# --- r24_residual_test.py
_SFX = "" if POP == "pyr" else f"_{POP}"


# --- r24_residual_test.py
MEASURES = [("d2", "d2"), ("d2_res", "d2 | m"), ("m", "m"), ("m_res", "m | d2")]


# --- r24_residual_test.py
# d2 and d2 | m only; FIG_ALL=1 draws all four
FIG_ALL = os.environ.get("FIG_ALL") == "1"


# --- r24_residual_test.py
FIG_MEASURES = MEASURES if FIG_ALL else MEASURES[:2]


# --- r24_residual_test.py (its tables read from LOGDIR)
def cohorts():
    """-> {name: (DataFrame with session, rate, covariate, and per arm d2_<arm>, m_<arm>; rate label)}"""
    C = {}
    R = pd.read_csv(os.path.join(LOGDIR, "excised_br_entropy_windows.csv"), float_precision="round_trip").dropna(subset=["replay_prob"])
    R["rate"] = R.replay_prob * R.ripple_rate; R["cov"] = R.ripple_rate
    R = R.rename(columns={"d2_published": "d2_full"})
    if POP == "pyr":
        R = R.rename(columns={"BR_recomp": "m_full", "BR_excised": "m_excised"})
        C["replay"] = (R, "replay-SWR rate, given SWR rate (NREM, decoded time)")
    else:   # all-unit m exists for the replay windows of the 8 decoded sessions that are also in Fig 1
        MA = pd.read_csv(os.path.join(LOGDIR, f"r24_br_ripple_windows{_SFX}.csv"), float_precision="round_trip")[["session", "center_s", "m_full", "m_excised"]]
        C["replay"] = (R.merge(MA, on=["session", "center_s"]), "replay-SWR rate, given SWR rate (NREM, decoded time)")
    W = pd.read_csv(os.path.join(LOGDIR, "d2_arms_raw_windows.csv"), float_precision="round_trip")
    W = W[W.state == "NREM"][["session", "center_s", "d2_full", "d2_excised", "ripple_rate"]]
    M = pd.read_csv(os.path.join(LOGDIR, f"r24_br_ripple_windows{_SFX}.csv"), float_precision="round_trip")[["session", "center_s", "m_full", "m_excised"]]
    S = W.merge(M, on=["session", "center_s"]); S["rate"] = S.ripple_rate; S["cov"] = np.nan
    C["SWR"] = (S, "SWR rate (NREM)")
    if POP == "pyr":
        A = pd.read_csv(os.path.join(LOGDIR, "r16_event_excised_windows.csv"), float_precision="round_trip")
        B = pd.read_csv(os.path.join(LOGDIR, "r16_event_excised_windows_br.csv"), float_precision="round_trip")
        X = A.merge(B[["session", "w", "BR", "BR_bg"]], on=["session", "w"])
        X = X.rename(columns={"d2": "d2_full", "d2_bg": "d2_excised", "BR": "m_full", "BR_bg": "m_excised"})
        X["rate"] = X.barrage_rate; X["cov"] = X.ripple_rate
        C["barrage"] = (X, "barrage rate, given SWR rate (NREM; excised = ripples and barrages)")
    return C


# --- r24_residual_test.py
def resid(y, x):
    b1, b0 = np.polyfit(x, y, 1)
    return y - (b0 + b1 * x)


# --- r24_residual_test.py (a drawing section, as a function: main without its table write)
def draw_main():
    """Per session, d2 regressed on m (and m on d2) on the same windows; Spearman of each with the event rate per
    cohort and arm (partial on SWR rate where a covariate is given); Fisher-z mean +- SEM, signed-rank; drawn."""
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
    # POP=pyr: 2-column (d2, d2 | m) or, with FIG_ALL=1, 4-column (adds m and m | d2); POP=all: backup/
    out = (os.path.join(FIGDIR, f"r24_residual_test{'_4col' if FIG_ALL else ''}.png") if POP == "pyr"
           else os.path.join(FIGDIR, "backup", f"r24_residual_test{_SFX}{'_4col' if FIG_ALL else ''}.png"))
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(out[:-3] + ext), dpi=200, bbox_inches="tight")
    print("  ->", out)
