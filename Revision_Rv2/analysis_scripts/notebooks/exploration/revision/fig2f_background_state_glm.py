"""fig2f_background_state_glm.py -- Figure 2f replay GLM with background-state covariates,
measurement-error correction and temporal precedence.

Three variants of the ripple-excised d2 ladder, same windows:

  A. Background covariates: MUA_bg and MUA_std_bg are the mean and SD of the 40 ms population
     rate over the ripple-excised part of each 30 s window (raw ripple start/stop, as the
     published d2 excises), so every predictor is a property of the state between events.
     The in-ripple versions (MUA and MUA_std computed with ripples included) are recomputed
     the same way as a check against the pickle's MUA / MUA_std.
  B. Measurement-error correction (regression calibration): per session,
     lambda* = 1 - sigma_err^2 / var(d2 | covariates), with sigma_err from the delta-method
     error bar (excised arm: a fixed ratio to d2; published arm: d2_err from the error-bar
     .mat). The per-SD-of-true-d2 coefficient is b1 / sqrt(lambda*).
  C. Temporal precedence: the state of window i (all covariates, excised) predicts replay in
     the SWRs of window i+1 (consecutive non-overlapping windows, 30 s apart).

Model: binomial GLM per session, replay_count of n_swr, z-scored predictors, session as the
unit of analysis (paired across sessions), matching r35_binomial_replay / fig2fvars.

    -> data/revision/log/fig2f_background_state_glm.csv      (per model x arm x session)
    -> data/revision/log/fig2f_background_state_windows.csv  (the augmented window table)
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
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                           # noqa: E402
from r35_binomial_fig2fvars import load, LOGDIR, WIN_S, zs               # noqa: E402
from r35_swr_property_control import SESSIONS                            # noqa: E402
from r35_background_vs_ripple import load_d2_mat                         # noqa: E402

BIN = 0.04
ERR_RATIO_EXCISED = 0.216        # delta-method sd / d2 ratio, validated against the published arm's d2_err
OUT_CSV = os.path.join(LOGDIR, "fig2f_background_state_glm.csv")
OUT_WIN = os.path.join(LOGDIR, "fig2f_background_state_windows.csv")

# (tag, predictors, label). First predictor is the one whose b is reported as b1.
MODELS = [
    # the ladder as it was (in-ripple covariates), for reference on the same table
    ("M1",  ["d2_z"],                                        "d2"),
    ("M3",  ["d2_z", "MUA_z", "MUA_std_z"],                  "d2 + MUA + MUA_std   (in-ripple)"),
    ("M5",  ["d2_z", "MUA_z", "MUA_std_z", "rate_z"],        "d2 + MUA + MUA_std + rate (in-ripple)"),
    # A: background-state covariates
    ("B2",  ["d2_z", "MUA_bg_z"],                            "d2 + MUA_bg"),
    ("B3",  ["d2_z", "MUA_bg_z", "MUA_std_bg_z"],            "d2 + MUA_bg + MUA_std_bg"),
    ("B4",  ["d2_z", "MUA_bg_z", "MUA_std_bg_z", "rate_z"],  "d2 + MUA_bg + MUA_std_bg + rate"),
    # C: temporal precedence (state of window i -> replay in window i+1)
    ("L1",  ["d2_lag_z"],                                    "d2(i-1)"),
    ("L3",  ["d2_lag_z", "MUA_bg_lag_z", "MUA_std_bg_lag_z"], "d2(i-1) + MUA_bg(i-1) + MUA_std_bg(i-1)"),
    ("L4",  ["d2_lag_z", "MUA_bg_lag_z", "MUA_std_bg_lag_z", "rate_lag_z"],
                                                             "d2(i-1) + MUA_bg(i-1) + MUA_std_bg(i-1) + rate(i-1)"),
]
ARMS = ("excised", "published")


def window_fr_std(units, ripples, centers):
    """Across-neuron ("spatial") SD of the per-neuron rate in each window, with and without
    the bins that overlap a ripple. Returns (fr_std, fr_std_bg, fr_mean_bg).

    This is Fig 2F's third feature (generate_GLM.ipynb cell 4): per-neuron rate over
    [c-15, c+15) with searchsorted 'left' at both ends, SD across neurons with ddof=0 --
    not the 30 s rolling temporal SD that the same notebook stores as MUA_std in cells 5-7.
    The two are different variables.

    The full-window value is computed on the same 40 ms grid as the excised one rather
    than by direct searchsorted on [c-15, c+15), so the two differ only by the mask; the
    grid spans the window exactly (750 x 40 ms = 30 s), reproducing cell 4. fr_mean_bg is
    the across-neuron mean on the excised bins, the per-neuron analogue of MUA_bg, kept so
    the covariate set matches Fig 2F.
    """
    r0 = np.sort(ripples[:, 0])
    r1 = ripples[:, 1][np.argsort(ripples[:, 0])]
    cummax_stop = np.maximum.accumulate(r1) if len(r1) else np.array([])
    nb = int(round(WIN_S / BIN))
    sorted_units = [np.sort(np.asarray(u, float).ravel()) for u in units]
    sorted_units = [u[np.isfinite(u)] for u in sorted_units]
    out = np.full((len(centers), 3), np.nan)
    for i, c in enumerate(centers):
        edges = c - WIN_S / 2 + BIN * np.arange(nb + 1)
        j = np.searchsorted(r0, edges[1:], "left")
        hit = np.array([(j_ > 0) and (cummax_stop[j_ - 1] > e0)
                        for j_, e0 in zip(j, edges[:-1])])
        keep = ~hit
        nk = int(keep.sum())
        rates = np.empty(len(sorted_units))
        rates_bg = np.empty(len(sorted_units))
        for k, s in enumerate(sorted_units):
            cnt = (np.searchsorted(s, edges[1:], "left")
                   - np.searchsorted(s, edges[:-1], "left"))
            rates[k] = cnt.sum() / WIN_S
            rates_bg[k] = cnt[keep].sum() / (nk * BIN) if nk else np.nan
        out[i] = (rates.std(ddof=0),
                  rates_bg.std(ddof=0) if nk > 10 else np.nan,
                  rates_bg.mean() if nk > 10 else np.nan)
    return out


# ------------------------------------------------------------------ A: background covariates
def window_pop_stats(units, ripples, centers):
    """Mean and SD of the 40 ms population rate inside each window, with and without the
    bins that overlap a ripple. Returns (mua, mua_std, mua_bg, mua_std_bg, frac_kept)."""
    allsp = np.sort(np.concatenate(units)) if units else np.array([])
    r0, r1 = np.sort(ripples[:, 0]), ripples[:, 1][np.argsort(ripples[:, 0])]
    out = np.full((len(centers), 5), np.nan)
    nb = int(round(WIN_S / BIN))
    for i, c in enumerate(centers):
        edges = c - WIN_S / 2 + BIN * np.arange(nb + 1)
        cnt = np.searchsorted(allsp, edges[1:]) - np.searchsorted(allsp, edges[:-1])
        rate = cnt / BIN
        # bin overlaps a ripple if some ripple starts before bin end and ends after bin start
        j = np.searchsorted(r0, edges[1:], "left")           # ripples starting before bin end
        # any of those ending after bin start? check the max stop among ripples[:j]
        cummax_stop = np.maximum.accumulate(r1) if len(r1) else np.array([])
        hit = np.array([(j_ > 0) and (cummax_stop[j_ - 1] > e0) for j_, e0 in zip(j, edges[:-1])])
        keep = ~hit
        out[i] = (rate.mean(), rate.std(ddof=1),
                  rate[keep].mean() if keep.sum() > 10 else np.nan,
                  rate[keep].std(ddof=1) if keep.sum() > 10 else np.nan,
                  keep.mean())
    return out


def augment(D):
    import nelpy as nel                                             # noqa: F401
    from neuro_py.io import loading
    parts = []
    for s, W in D.groupby("session", sort=False):
        bp = SESSIONS[s]
        st, _ = loading.load_spikes(bp, brainRegion="CA1")
        units = [np.asarray(u, float) for u in np.asarray(st.data, dtype=object)]
        rip = loading.load_ripples_events(bp)
        riv = np.array(rip[["start", "stop"]], float)               # RAW, as the d2 producer
        W = W.sort_values("center_s").copy()
        stats_ = window_pop_stats(units, riv, W["center_s"].to_numpy(float))
        W["MUA_rc"], W["MUA_std_rc"], W["MUA_bg"], W["MUA_std_bg"], W["frac_kept"] = stats_.T
        # published-arm per-window error from the error_bar file; excised: ratio
        m = load_d2_mat(bp, "_error_bar")
        c = W["center_s"].to_numpy(float)
        mc = m["center"].to_numpy(float)
        idx = np.clip(np.searchsorted(mc, c), 0, len(mc) - 1)
        W["d2_published_err"] = m["d2_err_x"].to_numpy(float)[idx]
        W["d2_excised_err"] = ERR_RATIO_EXCISED * W["d2_excised"]
        r_mua = np.corrcoef(W["MUA"], W["MUA_rc"])[0, 1]
        r_std = np.corrcoef(W["MUA_std"], W["MUA_std_rc"])[0, 1]
        print(f"  {s:12s} {len(W):4d} windows | recomputed vs pickle: r(MUA) {r_mua:.3f} "
              f"r(MUA_std) {r_std:.3f} | ripple-excised keeps {W.frac_kept.median():.1%} of bins | "
              f"r(MUA_std_bg, MUA_std) {np.corrcoef(W.MUA_std_bg, W.MUA_std)[0, 1]:.3f}", flush=True)
        # lagged (window i-1 -> i), only where consecutive windows are exactly one step apart
        step = np.median(np.diff(c))
        for k in ("MUA_bg", "MUA_std_bg", "rate", "d2_excised", "d2_published"):
            v = W[k].to_numpy(float)
            lag = np.full(len(v), np.nan)
            ok = np.isclose(np.diff(c), step, rtol=0.05)
            lag[1:][ok] = v[:-1][ok]
            W[f"{k}_lag"] = lag
        parts.append(W)
    A = pd.concat(parts, ignore_index=True)
    for k in ("MUA_bg", "MUA_std_bg", "MUA_bg_lag", "MUA_std_bg_lag", "rate_lag"):
        A[f"{k}_z"] = A.groupby("session")[k].transform(zs)
    return A


# ------------------------------------------------------------------ fitting
def fit_all(sub, cols):
    import statsmodels.api as sm
    s = sub[["n_replay", "n_swr"] + cols].dropna()
    if len(s) < 25 or s["n_replay"].sum() == 0:
        return None
    endog = np.column_stack([s["n_replay"], s["n_swr"] - s["n_replay"]])
    X = sm.add_constant(s[cols].to_numpy(float), has_constant="add")
    try:
        r = sm.GLM(endog, X, family=sm.families.Binomial()).fit()
    except Exception:                                            # noqa: BLE001
        return None
    return dict(n=len(s), **{f"b_{c}": float(v) for c, v in zip(cols, r.params[1:])},
                b1=float(r.params[1]), se1=float(r.bse[1]), p1=float(r.pvalues[1]))


def reliability(sub, d2col, errcol, covs):
    """lambda* = 1 - sigma_err^2 / var(d2 | covs): the reliability of the part of d2 not
    explained by the covariates, which is what its regression coefficient uses."""
    s = sub[[d2col, errcol] + covs].dropna()
    if len(s) < 25:
        return np.nan
    x = s[d2col].to_numpy(float)
    if covs:
        A = np.column_stack([np.ones(len(s)), s[covs].to_numpy(float)])
        x = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    v = x.var(ddof=1)
    e2 = float(np.mean(s[errcol].to_numpy(float) ** 2))
    return float(np.clip(1 - e2 / v, 0.05, 1.0)) if v > 0 else np.nan


def main():
    D = load()
    A = augment(D)
    assert_local_path(LOGDIR)
    A.to_csv(assert_local_path(OUT_WIN), index=False)
    rows = []
    for arm in ARMS:
        Da = A.copy()
        Da["d2_z"] = Da[f"d2_{arm}_z"]
        Da["d2_lag_z"] = Da.groupby("session")[f"d2_{arm}_lag"].transform(zs)
        for tag, cols, lbl in MODELS:
            for s, sub in Da.groupby("session", sort=False):
                r = fit_all(sub, cols)
                if r is None:
                    continue
                covs = [c for c in cols[1:]]
                d2col = f"d2_{arm}_lag" if cols[0] == "d2_lag_z" else f"d2_{arm}"
                errcol = f"d2_{arm}_err"
                lam = reliability(sub.assign(**{errcol: (ERR_RATIO_EXCISED * sub[d2col]
                                                          if arm == "excised" else sub[errcol])}),
                                  d2col, errcol, covs)
                rows.append(dict(arm=arm, model=tag, label=lbl, session=s, lam=lam,
                                 b1_true_sd=r["b1"] / np.sqrt(lam) if np.isfinite(lam) else np.nan, **r))
    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(OUT_CSV), index=False)
    print(f"\nwrote {OUT_CSV}")
    report(R)


def _line(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    p = stats.wilcoxon(v).pvalue if v.size > 1 else np.nan
    return f"{np.median(v):+8.4f} {int((v > 0).sum()):2d}/{v.size:<2d} p={p:.3f}"


def report(R):
    for arm in ARMS:
        print(f"\n{'=' * 110}\nd2 = {arm.upper()} -- b1 per SD of d2 (observed), and per SD of TRUE d2 "
              f"(regression calibration); session = unit\n{'=' * 110}")
        print(f"  {'model':52s}{'b1 observed':>27s}{'lambda*':>9s}{'b1 / sqrt(lambda*)':>27s}")
        for tag, cols, lbl in MODELS:
            P = R[(R.arm == arm) & (R.model == tag)]
            if not len(P):
                continue
            print(f"  {tag:3s} {lbl:48s}{_line(P.b1):>27s}{np.nanmedian(P.lam):9.2f}{_line(P.b1_true_sd):>27s}")
        # the covariate coefficients, for scale
        for tag in ("M5", "B4", "L4"):
            P = R[(R.arm == arm) & (R.model == tag)]
            if not len(P):
                continue
            cols = [c for c in P.columns if c.startswith("b_") and P[c].notna().any()]
            print(f"  {tag} covariate coefficients per SD: " +
                  "  ".join(f"{c[2:]} {np.median(P[c]):+.3f}" for c in cols))


if __name__ == "__main__":
    if os.environ.get("REPORT_ONLY"):
        report(pd.read_csv(OUT_CSV))
    else:
        main()
