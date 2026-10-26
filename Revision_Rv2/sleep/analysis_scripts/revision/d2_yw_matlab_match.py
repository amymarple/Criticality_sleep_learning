"""
d2_yw_matlab_match.py -- Yule-Walker vs OLS AR-estimator comparison between the Python
ripple-excised d2 and the MATLAB `_rippleablate.mat` reference.

The MATLAB producers for the published full-window d2 (getDbetas_error_bar.m) and for
the ripple-excised d2 (time_resolved_d2_across_state_ripple_ablate.m -> getDbetas.m ->
myYuleWalker.m -> myAutocorr.m) both fit phi by Yule-Walker, not OLS; only the delta-
method error bar for the published arm uses an OLS design. This script reimplements the
MATLAB recipe in Python (`d2_yw_segments`: binning, pooled z-scoring, within-segment ACF,
Yule-Walker solve, Euclidean distance to the beta=2 hyperplane) and an OLS-across-the-
glued-segments variant (`d2_ols_glued`), both over the same excised sub-windows used
elsewhere in this directory (`excised_br_entropy.free_segments`).

Reports, per session: r(YW excised, _rippleablate.mat), r(YW full, OLS full) as the
estimator difference alone on the same series, r(YW full, pickle d2), r(OLS full, pickle
d2), and the consequence for the replay / SWR-rate correlations of the excised arm.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/d2_yw_matlab_match.csv
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
from scipy.linalg import toeplitz

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                       # noqa: E402
from fig1_metrics import MIN_FIRING_RATE                              # noqa: E402
from euclid_d2_err_orders import d2_and_sd                            # noqa: E402
from r35_glm_ablation import LOGDIR, WIN_S                            # noqa: E402
from r35_swr_property_control import SESSIONS                         # noqa: E402
from excised_br_entropy import free_segments, MIN_FREE_S              # noqa: E402
from d2_correlates_table import fisher                                # noqa: E402

WIN_CSV = os.path.join(LOGDIR, "excised_br_entropy_windows.csv")
OUT = os.path.join(LOGDIR, "d2_yw_matlab_match.csv")
DT, P = 0.04, 20


def getFR(sp, t0, t1, dt=DT):
    """MATLAB getFR: histc on t0:dt:t1, drop the last (edge) element."""
    edges = t0 + dt * np.arange(int(np.floor((t1 - t0) / dt + 1e-9)) + 1)
    if edges.size < 2:
        return np.zeros(0)
    lo, hi = np.searchsorted(sp, (t0, t1))
    return np.histogram(sp[lo:hi], bins=edges)[0].astype(float)


def d2_yw_segments(sp, segs, p=P, dt=DT):
    """myAutocorr + myYuleWalker + Euclidean distance, on a list of sub-windows."""
    sigs = [getFR(sp, a, b, dt) for a, b in segs]
    sigs = [s for s in sigs if s.size]
    if not sigs:
        return np.nan
    allsig = np.concatenate(sigs)
    mu, sd = allsig.mean(), allsig.std(ddof=1)          # MATLAB std: ddof=1
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    acf = np.zeros(p + 1)      # lags 1..p+1, matching MATLAB's convention
    cnt = np.zeros(p + 1)
    for s in sigs:
        z = (s - mu) / sd
        n = z.size
        for j in range(1, min(p, n - 2) + 1):
            acf[j - 1] += float(np.dot(z[:n - j], z[j:]))
            cnt[j - 1] += n - j
    ok = cnt[:p] > 0
    if not ok.all():
        return np.nan
    r = acf[:p] / cnt[:p]                                 # acf lags 1..p
    # myYuleWalker: R Toeplitz from [1, r1..r_{p-1}], rhs = [r1..rp]
    R = toeplitz(np.r_[1.0, r[:p - 1]])
    try:
        phi = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        return np.nan
    return float(abs(1.0 - phi.sum()) / np.sqrt(p))


def d2_ols_glued(sp, segs, p=P, dt=DT):
    """Glue the sub-windows and run OLS across the glue (rather than within each
    sub-window)."""
    from fig1_metrics import binned_counts, slice_sorted
    parts = [binned_counts(slice_sorted(sp, a, b), a, b, dt, presliced=True).astype(float) for a, b in segs]
    parts = [x for x in parts if x.size]
    if not parts:
        return np.nan
    x = np.concatenate(parts)
    if x.std() == 0:
        return np.nan
    out = d2_and_sd((x - x.mean()) / x.std(), p)
    return out[0] if out is not None else np.nan


def main():
    from neuro_py.io import loading
    W = pd.read_csv(WIN_CSV)
    rows = []
    for sess, g in W.groupby("session"):
        bp = SESSIONS[sess]
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        allu = [np.asarray(u, float) for u in np.asarray(st.data, dtype=object)]
        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        sp_all = np.sort(np.concatenate(allu))                              # MATLAB: every CA1 cell
        sp_thr = np.sort(np.concatenate([u for u, k in zip(allu, keep) if k]))
        rip = loading.load_ripples_events(bp)
        r0 = np.asarray(rip["start"], float); o = np.argsort(r0)
        r0, r1 = r0[o], np.asarray(rip["stop"], float)[o]
        for r in g.itertuples(index=False):
            t0, t1 = r.center_s - WIN_S / 2, r.center_s + WIN_S / 2
            segs = free_segments(t0, t1, r0, r1)
            if sum(b - a for a, b in segs) < MIN_FREE_S:
                segs = []
            rows.append(dict(
                session=sess, center_s=r.center_s, replay_prob=r.replay_prob, ripple_rate=r.ripple_rate,
                d2_published=r.d2_published, d2_excised_mat=r.d2_excised,
                yw_full_all=d2_yw_segments(sp_all, [(t0, t1)]),
                yw_full_thr=d2_yw_segments(sp_thr, [(t0, t1)]),
                ols_full_thr=d2_ols_glued(sp_thr, [(t0, t1)]),
                yw_exc_all=d2_yw_segments(sp_all, segs) if segs else np.nan,
                yw_exc_thr=d2_yw_segments(sp_thr, segs) if segs else np.nan,
                ols_exc_glued_thr=d2_ols_glued(sp_thr, segs) if segs else np.nan))
        D = pd.DataFrame([x for x in rows if x["session"] == sess])
        def rr(a, b):
            s = D[[a, b]].dropna(); return np.corrcoef(s.T.to_numpy())[0, 1]
        print(f"  {sess:12s} {len(D):4d} | vs _rippleablate.mat: YW-all {rr('yw_exc_all','d2_excised_mat'):.3f}  "
              f"YW-thr {rr('yw_exc_thr','d2_excised_mat'):.3f}  OLS-glued {rr('ols_exc_glued_thr','d2_excised_mat'):.3f} | "
              f"vs pickle: YW-full {rr('yw_full_thr','d2_published'):.3f} OLS-full {rr('ols_full_thr','d2_published'):.3f} | "
              f"r(YW full, OLS full) {rr('yw_full_thr','ols_full_thr'):.3f}", flush=True)
    D = pd.DataFrame(rows)
    assert_local_path(LOGDIR)
    D.to_csv(assert_local_path(OUT), index=False)
    print(f"\nwrote {OUT}")
    report(D)


def report(D):
    print("\nwithin-session Spearman rho with replay probability (windows >= 1 SWR) and SWR rate; Fisher mean, sessions positive / N, p")
    for col, lab in (("d2_excised_mat", "excised, MATLAB .mat (YW)"), ("yw_exc_all", "excised, Python YW, all cells"),
                     ("yw_exc_thr", "excised, Python YW, thresholded"), ("ols_exc_glued_thr", "excised, Python OLS glued (5.5(m))"),
                     ("d2_published", "full, pickle (OLS)"), ("yw_full_thr", "full, Python YW"), ("ols_full_thr", "full, Python OLS")):
        for x in ("replay_prob", "ripple_rate"):
            rs = []
            for _, g in D.groupby("session"):
                sub = g[list(dict.fromkeys([col, x, "ripple_rate"]))].dropna()
                if x == "replay_prob":
                    sub = sub[sub.ripple_rate * WIN_S >= 1]
                if len(sub) >= 40:
                    rs.append(stats.spearmanr(sub[col], sub[x])[0])
            rs = np.asarray(rs)
            print(f"  {lab:38s} vs {x:12s} {fisher(rs):+.3f} ({int((rs > 0).sum())}/{rs.size}, p = {stats.wilcoxon(rs).pvalue:.3f})")


if __name__ == "__main__":
    if os.environ.get("REPORT_ONLY"):
        report(pd.read_csv(OUT))
    else:
        main()
