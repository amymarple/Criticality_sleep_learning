"""
euclid_d2_err_orders.py -- d2 (2024 Euclidean) AND its delta-method error bar across
AR orders, for every NREM window.

The Euclidean d2 is the distance from phi-hat to the hyperplane
{phi : sum(phi) = 1}:

    d2 = |1 - sum(phi)| / sqrt(p)

which is LINEAR in sum(phi). So the delta-method gradient is analytic -- and in fact
constant. This is exactly what the MATLAB reference computes.
`helper/getDbetas_error_bar.m` does

    g = (varphi - closest_phi) / d_beta        % Euclidean unit normal
    d_beta_error = sqrt( g' * Cov(phi) * g )   % Cov = sigma^2 (X'X)^-1, cov_mode 'ols'

and for the beta=2 hyperplane the unit normal IS the constant vector 1/sqrt(p), so

    sd(d2) = sqrt( 1' Cov(phi) 1 / p ) = sigma_hat * sqrt( 1'(X'X)^-1 1 ) / sqrt(p)

One Cholesky solve per (window, order).

Estimator: OLS on the lagged design matrix, as in the MATLAB code, so the error bar is the
quantity stored in the .mat files. At p = 20 the result is compared with the stored MATLAB
`d2_err` (printed alongside the results for the other orders).

    -> data/revision/log/euclid_d2_err_orders.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig1_metrics import (  # noqa: E402
    GUARD_S, MIN_FIRING_RATE, STATES, binned_counts, build_state_intervals,
    make_windows, slice_sorted,
)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path, session_id  # noqa: E402

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
OUTDIR = os.path.join(REPO, "data", "revision", "log")
BIN = 0.040
ORDERS = (5, 10, 15, 20, 25, 30, 40)


def d2_and_sd(xz, p):
    """Euclidean d2 and its delta-method sd, OLS form, matching getDbetas_error_bar.m.

    Returns (d2, sd, sumphi, sigma2) or None if the window is too short."""
    n = xz.size
    if n < p + 20:
        return None
    X = sliding_window_view(xz, p)[:-1][:, ::-1]      # rows = x[t-1..t-p]
    y = xz[p:]
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        L = np.linalg.cholesky(XtX)
    except np.linalg.LinAlgError:
        return None
    phi = np.linalg.solve(XtX, Xty)
    resid = y - X @ phi
    dof = max(y.size - p, 1)
    sig2 = float(resid @ resid) / dof
    s = float(phi.sum())
    d2 = abs(1.0 - s) / np.sqrt(p)
    # 1' (X'X)^-1 1  via the Cholesky factor: ||L^-1 1||^2
    one = np.ones(p)
    w = np.linalg.solve(L, one)
    quad = float(w @ w)
    sd = np.sqrt(max(sig2 * quad, 0.0) / p)
    return d2, sd, s, sig2


def one_session(bp):
    from neuro_py.io import loading
    animal, day = session_id(bp)
    try:
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        sd_ = loading.load_SleepState_states(bp)
    except Exception as e:                                        # noqa: BLE001
        return animal, day, None, f"{type(e).__name__}: {e}"
    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    sp = np.sort(np.concatenate(
        [np.asarray(u, float) for u in np.asarray(st.data, dtype=object)[keep]]))
    iv = build_state_intervals(sd_, GUARD_S)

    rows = []
    for state in STATES:
        if state not in iv or not len(iv[state]):
            continue
        win = make_windows(iv[state], state)
        if not len(win) or "nonoverlap" not in win.columns:
            continue
        for r in win[win.nonoverlap].reset_index(drop=True).itertuples(index=False):
            seg = slice_sorted(sp, r.win_start, r.win_stop)
            x = binned_counts(seg, r.win_start, r.win_stop, BIN,
                              presliced=True).astype(float)
            if not x.size or x.std() == 0:
                continue
            xz = (x - x.mean()) / x.std()
            rec = {"animal": animal, "day": day, "state": state,
                   "w": int(round(r.win_start)), "N": int(xz.size)}
            for p in ORDERS:
                out = d2_and_sd(xz, p)
                if out is None:
                    continue
                d2, sd, s, sig2 = out
                rec[f"d2_p{p}"] = d2
                rec[f"sd_p{p}"] = sd
                rec[f"sumphi_p{p}"] = s
            rows.append(rec)
    return animal, day, pd.DataFrame(rows), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14,
                    help="default leaves ~10 cores free for other work")
    args = ap.parse_args()
    t0 = time.time()
    print(f"Euclidean d2 + delta-method sd | orders {ORDERS} | "
          f"{args.workers} workers\n", flush=True)
    from multiprocessing import Pool
    out = []
    with Pool(processes=args.workers) as pool:
        for animal, day, df, err in pool.imap_unordered(one_session, FIG1_SESSIONS):
            if err:
                print(f"  [FAIL] {animal}_{day}: {err}", flush=True)
                continue
            print(f"  {animal}_{day:9s} {len(df):5d} windows", flush=True)
            out.append(df)
    D = pd.concat(out, ignore_index=True)
    assert_local_path(OUTDIR); os.makedirs(OUTDIR, exist_ok=True)
    D.to_csv(assert_local_path(os.path.join(OUTDIR, "euclid_d2_err_orders.csv")),
             index=False)
    print(f"\nwrote {len(D)} windows ({time.time()-t0:.0f} s)\n")
    report(D)


def report(D):
    n = D[D.state == "NREM"]
    n = n.assign(sess=n.animal.astype(str) + "_" + n.day.astype(str))
    print("=" * 74)
    print("d2 (2024 Euclidean) and its delta-method sd, NREM, per-session medians")
    print("=" * 74)
    print(f"{'p':>4}{'d2':>11}{'sd':>11}{'sd/d2':>10}{'sessions':>10}{'windows':>10}")
    prev = None
    for p in ORDERS:
        dc, sc = f"d2_p{p}", f"sd_p{p}"
        if dc not in n.columns:
            continue
        per = n.groupby("sess")[[dc, sc]].median().dropna()
        if not len(per):
            continue
        ratio = (n[sc] / n[dc]).median()
        inc = "" if prev is None else f"  ({ratio-prev:+.4f})"
        print(f"{p:>4}{per[dc].median():>11.4f}{per[sc].median():>11.4f}"
              f"{ratio:>10.1%}{len(per):>10}{n[dc].notna().sum():>10}{inc}")
        prev = ratio
    print("\n  VALIDATE at p=20 against the stored MATLAB d2_err before trusting the")
    print("  other orders -- REVISION_STATUS 1.7 quotes d2_err median 0.0123 on d2 0.055")
    print("  (~22 %). A large mismatch means the estimand differs, not the order.")


if __name__ == "__main__":
    main()
