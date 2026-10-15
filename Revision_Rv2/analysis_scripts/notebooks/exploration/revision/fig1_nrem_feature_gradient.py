"""
fig1_nrem_feature_gradient.py -- organisation of d2 (2024 Euclidean, 40 ms, AR p = 20) WITHIN NREM in
the window-feature space, beyond its own time structure.

The state-space figures (fig1_state_space_umap.py, fig1_state_space_umap_session.py,
fig1_lfp_manifold.py) colour the embedding by d2 and quote a structure index (SI) against a
label-permutation null. Across states this SI largely re-states the state partition; d2 and the
features are both autocorrelated in time, so consecutive windows are not independent samples for a
label-permutation null. This script instead locks the cohort and the missing-value rule in advance,
uses one geometry for the picture and the statistic, and tests against a time-preserving null (circular
shift within each NREM bout). The feature table holds more sessions than the Fig-1 set (e.g. PPP7
day8), so the cohort is restricted below.

Parameters (MIN_BOUT, OFFSET) are fixed from the observed NREM bout-length distribution and the
within-bout d2 autocorrelation (written to the _bouts table) before the test is run:
  cohort    Fig-1 sessions (run_fig1_metrics.FIG1_SESSIONS) that have SWR detection: 16 of 18. PPP20 day9
            and PPP21 day8 have no ripple file and are excluded.
  missing   ripple_amp_mean is undefined in a zero-ripple window; inside a cohort session it is set to the
            session mean, which places those windows at z = 0 on that axis; the 9- and 8-feature sets
            omit ripple_amp_mean (and ripple_rate) to assess its effect.
  geometry  features z-scored per session over all of the session's windows (the figure's geometry);
            Euclidean distance for SI, for the neighbour statistic and for the UMAP of
            fig1_state_space_umap_session.py.
  sets      10 features (fig1_state_space_umap.FEATURES); 9 (minus ripple_amp_mean); 8 (minus ripple_rate
            and ripple_amp_mean). Same windows for all three.
  sample    NREM windows in bouts of >= MIN_BOUT consecutive windows (a bout = a run of window centres
            30 s apart). Observed statistic and null use the same windows.
  null      d2 circularly shifted WITHIN each bout by OFFSET..L-OFFSET windows: keeps each bout's own d2
            values, their autocorrelation and any slow drift between bouts; breaks only the window-level
            pairing of d2 with the features. N_NULL surrogates.
  stats     per session, both on the same sample:
            SI      PridaLab structure_index, n_bins 10, n_neighbors 15 (the pupil paper's NREM setting), in
                    the z-scored feature space; excess = SI - median null SI.
            smooth  mean |d2_i - d2_j| over each window's K nearest feature-space neighbours, neighbours
                    within EXCL_S of it in time excluded; effect = 1 - observed / median null (> 0: windows
                    that look alike have more similar d2 than the time-preserving null gives).
            p = (1 + #null at least as extreme) / (1 + N_NULL), one-sided.
  across    16 sessions: signed-rank of the effects, sessions with p < 0.05; 6 animals: mean effect per
            animal, sign count.
  sensitivity  MIN_BOUT 10 / OFFSET 3, the one alternative, reported beside the primary.

SI needs the PridaLab `structure_index` package (GPL-3), not vendored here: SI_PATH (env) points at a
checkout, as in fig1_lfp_manifold.py.

Writes only to D:.
    -> data/revision/log/fig1_nrem_feature_gradient.csv          (session x feature set x variant)
    -> data/revision/log/fig1_nrem_feature_gradient_summary.csv
    -> data/revision/log/fig1_nrem_feature_gradient_bouts.csv    (bout lengths, within-bout autocorrelation)
    -> reports/revision_figs/fig1_nrem_feature_gradient.png / .svg
"""
from __future__ import annotations

import os
import sys
import zlib

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path, FIG1_SESSIONS         # noqa: E402
from fig1_state_space_umap import FEATURES, LOGDIR, FIGDIR, zscore_per_session   # noqa: E402
from fig1_lfp_manifold import SI_PATH                                 # noqa: E402
from m25_tmaze_box import MOUSE_COL, MOUSE_LABEL      # noqa: E402  the manuscript's Mouse 1-6 (ID order) and colours

SETS = {"10": FEATURES,
        "9": [f for f in FEATURES if f != "ripple_amp_mean"],
        "8": [f for f in FEATURES if f not in ("ripple_rate", "ripple_amp_mean")]}
VARIANTS = {"primary": (6, 2), "long_bouts": (10, 3)}     # (MIN_BOUT windows, OFFSET windows)
STEP_S = 30.0
K = 15
EXCL_S = 60.0
SI_BINS, SI_K = 10, 15
N_NULL = int(os.environ.get("N_NULL", "200"))
OUT = os.path.join(LOGDIR, "fig1_nrem_feature_gradient.csv")


def cohort_sessions(F):
    """Fig-1 sessions with SWR detection; prints what is left out and why."""
    fig1 = {"_".join(p.replace("\\", "/").rstrip("/").split("/")[-2:]) + "_CA1" for p in FIG1_SESSIONS}
    has_rip = {s for s, g in F.groupby("session") if g.ripple_rate.notna().any()}
    print(f"feature table: {F.session.nunique()} sessions; not Fig-1 (excluded): {sorted(set(F.session) - fig1)}")
    print(f"Fig-1 sessions without SWR detection (excluded): {sorted(fig1 - has_rip)}")
    return sorted(fig1 & has_rip)


def prepare(F, sessions):
    """The one missing-value rule: restrict to `sessions`, impute ripple_amp_mean in zero-ripple windows to the
    session mean, drop any remaining incomplete window, z-score per session over all its windows."""
    D = F[F.session.isin(sessions)].copy()
    m = D.ripple_rate.notna() & D.ripple_amp_mean.isna()
    D.loc[m, "ripple_amp_mean"] = D.groupby("session").ripple_amp_mean.transform("mean")[m]
    n0 = len(D)
    D = D.dropna(subset=FEATURES + ["d2"]).reset_index(drop=True)
    print(f"ripple_amp_mean imputed in {int(m.sum())} zero-ripple windows; {n0 - len(D)} incomplete windows dropped")
    return zscore_per_session(D, FEATURES)


def bouts(t):
    """bout id per window (t sorted): a new bout wherever consecutive centres are not STEP_S apart"""
    return np.concatenate([[0], np.cumsum(np.abs(np.diff(t) - STEP_S) > 1.0)])


def bout_table(Z):
    rows = []
    for s, g in Z[Z.state == "NREM"].sort_values(["session", "center_s"]).groupby("session"):
        b = bouts(g.center_s.to_numpy())
        L = pd.Series(b).value_counts()
        r = dict(session=s, n_nrem=len(g), n_bouts=len(L), median_bout=L.median(), max_bout=L.max(),
                 frac_in_bout6=L[L >= 6].sum() / len(g), frac_in_bout10=L[L >= 10].sum() / len(g))
        for lag in (1, 2, 3, 5):
            x = g.groupby(b).d2.shift(-lag).to_numpy(); ok = np.isfinite(x)
            r[f"d2_r_lag{lag}"] = np.corrcoef(g.d2.to_numpy()[ok], x[ok])[0, 1]
        rows.append(r)
    return pd.DataFrame(rows)


def si(X, y):
    if SI_PATH not in sys.path:
        sys.path.insert(0, SI_PATH)
    from structure_index import compute_structure_index
    r = compute_structure_index(X, y.copy(), n_bins=SI_BINS, n_neighbors=SI_K, num_shuffles=0, verbose=False)
    if not np.isfinite(r[0]):
        return np.nan, 0, 0, 0.0
    lab = np.asarray(r[1][0], float)
    cnt = pd.Series(lab[np.isfinite(lab)]).value_counts()
    return float(r[0]), len(cnt), (int(cnt.min()) if len(cnt) else 0), (float(cnt.median()) if len(cnt) else 0.0)


def neighbours(X, t):
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    d[np.abs(t[:, None] - t[None, :]) <= EXCL_S] = np.inf      # includes the diagonal
    return np.argsort(d, axis=1)[:, :K]


def run_session(args):
    s, g = args
    rng = np.random.default_rng(zlib.crc32(s.encode()))   # reproducible per session
    g = g.sort_values("center_s").reset_index(drop=True)
    t = g.center_s.to_numpy(float)
    b = bouts(t)
    rows = []
    for var, (min_bout, offset) in VARIANTS.items():
        L = pd.Series(b).map(pd.Series(b).value_counts()).to_numpy()
        keep = L >= min_bout
        gs, ts, bs = g[keep].reset_index(drop=True), t[keep], b[keep]
        y = gs.d2.to_numpy(float)
        groups = [np.where(bs == u)[0] for u in np.unique(bs)]
        nulls = []
        for _ in range(N_NULL):
            yn = y.copy()
            for idx in groups:
                yn[idx] = np.roll(y[idx], rng.integers(offset, len(idx) - offset + 1))
            nulls.append(yn)
        for name, feats in SETS.items():
            X = gs[feats].to_numpy(float)
            nb = neighbours(X, ts)
            S_obs = np.abs(y[:, None] - y[nb]).mean()
            S_null = np.array([np.abs(yn[:, None] - yn[nb]).mean() for yn in nulls])
            si_obs, nbins, pmin, pmed = si(X, y)
            si_null = np.array([si(X, yn)[0] for yn in nulls], float)
            rows.append(dict(session=s, animal=s.split("_")[0], feature_set=name, variant=var,
                             min_bout=min_bout, offset=offset, n_windows=len(gs), n_bouts=len(groups),
                             frac_nrem_used=len(gs) / len(g),
                             SI=si_obs, SI_null_median=np.nanmedian(si_null), SI_null_p95=np.nanpercentile(si_null, 95),
                             SI_excess=si_obs - np.nanmedian(si_null),
                             p_SI=(1 + np.sum(si_null >= si_obs)) / (1 + N_NULL),
                             SI_bins_kept=nbins, SI_min_pts_per_bin=pmin, SI_median_pts_per_bin=pmed,
                             nbr_absdiff=S_obs, nbr_absdiff_null_median=np.median(S_null),
                             smooth_effect=1 - S_obs / np.median(S_null),
                             p_smooth=(1 + np.sum(S_null <= S_obs)) / (1 + N_NULL)))
    print(f"  {s}: done", flush=True)
    return rows


def summarise(R):
    from scipy import stats
    out = []
    for (fs, var), g in R.groupby(["feature_set", "variant"], sort=False):
        r = dict(feature_set=fs, variant=var, n_sessions=len(g), n_windows=int(g.n_windows.sum()))
        for eff, p in (("SI_excess", "p_SI"), ("smooth_effect", "p_smooth")):
            v = g[eff].to_numpy(float)
            a = g.groupby("animal")[eff].mean()
            r.update({f"{eff}_median": np.median(v), f"{eff}_n_pos": int((v > 0).sum()),
                      f"{eff}_n_p05": int((g[p] < 0.05).sum()),
                      f"{eff}_wilcoxon_p": stats.wilcoxon(v).pvalue if np.any(v != 0) else np.nan,
                      f"{eff}_animals_pos": int((a > 0).sum()), f"{eff}_n_animals": len(a)})
        out.append(r)
    return pd.DataFrame(out)


def plot(R, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    P = R[R.variant == "primary"]
    animals = sorted(P.animal.unique(), key=lambda a: int(a[3:]))
    cols = MOUSE_COL
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, (eff, p, lab) in zip(axes, (("SI_excess", "p_SI", "SI of d2 minus median null SI"),
                                        ("smooth_effect", "p_smooth", "1 - neighbour |Δd2| / null"))):
        for i, fs in enumerate(SETS):
            g = P[P.feature_set == fs]
            jit = np.random.default_rng(i).uniform(-0.12, 0.12, len(g))
            for (_, r), j in zip(g.iterrows(), jit):
                sig = r[p] < 0.05
                ax.scatter(i + j, r[eff], s=16, color=cols[r.animal], edgecolor=cols[r.animal],
                           facecolor=cols[r.animal] if sig else "none", lw=0.8, zorder=3)
            am = g.groupby("animal")[eff].mean()
            ax.scatter(np.full(len(am), i + 0.3), am, marker="D", s=14, color="k", zorder=4)
            s = S[(S.feature_set == fs) & (S.variant == "primary")].iloc[0]
            ax.text(i, np.nanmax(P[eff]) * 1.12 + 0.005,
                    f"{s[eff + '_n_p05']}/{s.n_sessions} p<.05\nWilcoxon p={s[eff + '_wilcoxon_p']:.3g}\n"
                    f"animals {s[eff + '_animals_pos']}/{s[eff + '_n_animals']} > 0",
                    ha="center", va="bottom", fontsize=6.5)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_xticks(range(len(SETS))); ax.set_xticklabels([f"{k} features" for k in SETS]); ax.set_xlim(-0.65, len(SETS) - 0.35)
        ax.set_ylabel(lab)
        top = np.nanmax(P[eff]) * 1.12 + 0.005
        ax.set_ylim(min(np.nanmin(P[eff]) * 1.2, -0.02), top + 0.45 * (top - min(np.nanmin(P[eff]), 0)))
    axes[0].set_title("a  structure index, NREM only", loc="left", fontweight="bold")
    axes[1].set_title("b  local smoothness, NREM only", loc="left", fontweight="bold")
    h = [plt.Line2D([], [], marker="o", ls="", color=cols[a], label=MOUSE_LABEL[a]) for a in animals]
    h.append(plt.Line2D([], [], marker="D", ls="", color="k", label="animal mean"))
    axes[1].legend(handles=h, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("d2 (2024 Euclidean, 40 ms, p = 20) within NREM, in the z-scored window-feature space "
                 f"(Euclidean), against d2 circularly shifted within each NREM bout ({N_NULL} surrogates)",
                 fontsize=8)
    fig.text(0.01, -0.06,
             f"{P.session.nunique()} Fig-1 sessions with SWR detection (PPP20 day9, PPP21 day8 have none); NREM windows "
             f"in bouts of >= {VARIANTS['primary'][0]} consecutive 30 s windows, shift {VARIANTS['primary'][1]}..L-"
             f"{VARIANTS['primary'][1]} windows. Filled: session p < 0.05.\n"
             f"SI: PridaLab structure index, {SI_BINS} bins, {SI_K} neighbours. Smoothness: mean |d2 difference| to the "
             f"{K} nearest windows in feature space, excluding windows within {EXCL_S:.0f} s in time.\n"
             "8 features = without ripple rate and amplitude; 9 = without ripple amplitude (imputed in zero-ripple windows).",
             fontsize=6.5, va="top")
    fig.tight_layout()
    o = os.path.join(FIGDIR, "fig1_nrem_feature_gradient")
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(f"{o}.{ext}"), dpi=200, bbox_inches="tight")
    print("->", o + ".png")


def main():
    if os.environ.get("PLOT_ONLY"):
        R = pd.read_csv(OUT, dtype={"feature_set": str})
        return plot(R, pd.read_csv(OUT.replace(".csv", "_summary.csv"), dtype={"feature_set": str}))
    F = pd.read_csv(os.path.join(LOGDIR, "fig1_state_space_features.csv"))
    Z = prepare(F, cohort_sessions(F))
    B = bout_table(Z)
    B.to_csv(assert_local_path(OUT.replace(".csv", "_bouts.csv")), index=False)
    print(B.drop(columns="session").median().round(2).to_dict())
    N = Z[Z.state == "NREM"]
    print(f"{N.session.nunique()} sessions, {len(N)} NREM windows; {N_NULL} surrogates")
    from multiprocessing import Pool
    with Pool(min(16, os.cpu_count() or 1)) as pool:
        R = pd.DataFrame([r for rows in pool.map(run_session, list(N.groupby("session"))) for r in rows])
    R.to_csv(assert_local_path(OUT), index=False)
    S = summarise(R)
    S.to_csv(assert_local_path(OUT.replace(".csv", "_summary.csv")), index=False)
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False))
    plot(R, S)


if __name__ == "__main__":
    main()
