"""
ar_order_sweep.py -- AR order sweep for d2 (2025 KL, d2_fast.d2_from_timeseries) and residual whiteness.

d2 uses an AR(p) fit with p = 20 at 40 ms bins. Two questions:

  1. ROBUSTNESS -- does the STATE ORDERING survive the choice of p? Absolute d2 must
     move with p (more parameters -> different KL rate); the claim is the ordering, and
     that is what has to be stable. Same logic as the k_max sweep.

  2. PRINCIPLE -- Ljung-Box on the AR residuals: the smallest p that whitens the residuals.

COST. d2 is a (p-1)-dimensional nonlinear optimisation and scales steeply:
p=5 0.55 s, p=10 3.1 s, p=15 12.9 s, p=20 28.3 s, p=30 78 s, p=40 153 s -- 276 s for
one window's full sweep. Two design consequences:

  * the d2 pass is PARALLEL and uses few windows per state;
  * the Ljung-Box pass is essentially free (Yule-Walker is O(p^2)), so it runs on many
    more windows.

    python ar_order_sweep.py               # all 18 sessions
    python ar_order_sweep.py --test        # PPP20 day4 only
    python ar_order_sweep.py --d2-windows 8 --workers 20
Writes data/revision/log/ar_order_sweep.csv and ar_order_ljungbox.csv. ljungbox_df_audit.py imports ar_residuals
from here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))), "methods", "d2"))

from fig1_metrics import (  # noqa: E402
    BIN_MUA_S, GUARD_S, MIN_FIRING_RATE, STATES, binned_counts,
    build_state_intervals, make_windows, slice_sorted,
)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path, session_id  # noqa: E402

OUTDIR = os.path.join(r"D:\Documents\GitHub\Sleep_criticaltiy", "data", "revision", "log")

ORDERS = (5, 10, 15, 20, 30, 40)
ORDER_REF = 20
LB_LAGS = 50            # Ljung-Box horizon: 50 x 40 ms = 2 s
D2_WINDOWS = 4          # per (session, state) -- the expensive pass
LB_WINDOWS = 40         # per (session, state) -- the cheap pass


def ar_residuals(x, phi):
    """e_t = x_t - sum_k phi_k x_{t-k}, for t >= p."""
    p = len(phi)
    if p == 0 or x.size <= p:
        return np.empty(0)
    X = np.lib.stride_tricks.sliding_window_view(x, p)[:-1]
    return x[p:] - X[:, ::-1] @ phi


def thin(df, n):
    return df.iloc[:: int(np.ceil(len(df) / n))] if len(df) > n else df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--d2-windows", type=int, default=D2_WINDOWS)
    ap.add_argument("--lb-windows", type=int, default=LB_WINDOWS)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 4, 2))
    args = ap.parse_args()

    from neuro_py.io import loading
    from d2_fast import fit_ar_yule_walker
    from statsmodels.stats.diagnostic import acorr_ljungbox

    paths = [r"W:\data\PPP\PPP20\day4"] if args.test else FIG1_SESSIONS

    tasks = []           # (key, x, order, dt) for the parallel d2 pass
    lb_rows = []
    t_start = time.time()

    for bp in paths:
        animal, day = session_id(bp)
        try:
            st, cm = loading.load_spikes(bp, brainRegion="CA1")
            state_dict = loading.load_SleepState_states(bp)
        except Exception as e:
            print(f"  [FAIL] {bp}: {type(e).__name__}: {e}", flush=True); continue

        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        dat = np.asarray(st.data, dtype=object)[keep]
        all_spikes = np.sort(np.concatenate([np.asarray(s, float) for s in dat]))
        iv = build_state_intervals(state_dict, GUARD_S)

        for state in STATES:
            if not len(iv[state]):
                continue
            win = make_windows(iv[state], state)
            # a state whose epochs are all shorter than the window (e.g. PPP4 day11
            # THETA, 1.5 min total) yields an empty frame with no columns at all
            if not len(win) or "nonoverlap" not in win.columns:
                continue
            win = win[win.nonoverlap].reset_index(drop=True)
            if not len(win):
                continue

            lb_win = thin(win, args.lb_windows)
            xs = []
            for r in lb_win.itertuples(index=False):
                w = slice_sorted(all_spikes, r.win_start, r.win_stop)
                x = binned_counts(w, r.win_start, r.win_stop, BIN_MUA_S,
                                  presliced=True).astype(float)
                if x.size < max(ORDERS) + 20 or x.std() <= 0:
                    continue
                xs.append(x)

            # --- cheap pass: residual whiteness at every order ---
            for wi, x in enumerate(xs):
                z = (x - x.mean()) / (x.std() + 1e-12)
                rec = {"animal": animal, "day": day, "state": state, "w": wi}
                for p in ORDERS:
                    try:
                        phi, _ = fit_ar_yule_walker(z, p)
                        e = ar_residuals(z, phi)
                        if e.size > LB_LAGS + 10:
                            # without `model_df=p` this uses df = LB_LAGS rather than
                            # LB_LAGS - p, which is anti-conservative, more so at high p;
                            # `ljungbox_df_audit.py` computes both on the same residuals
                            # -> ljungbox_df_audit.csv.
                            pv = float(acorr_ljungbox(
                                e, lags=[LB_LAGS], return_df=True)["lb_pvalue"].iloc[0])
                            rec[f"lb_p{p}"] = pv
                            rec[f"white_p{p}"] = pv > 0.05
                    except Exception:
                        pass
                lb_rows.append(rec)

            # --- expensive pass: queue d2 tasks on a small subset ---
            step = max(int(np.ceil(len(xs) / args.d2_windows)), 1)
            for wi, x in list(enumerate(xs))[::step][:args.d2_windows]:
                for p in ORDERS:
                    tasks.append(((animal, day, state, wi, p), x, p, BIN_MUA_S))

        print(f"  queued {animal}_{day}  (running total: {len(tasks)} d2 tasks)",
              flush=True)

    est = sum({5: 0.55, 10: 3.1, 15: 12.9, 20: 28.3, 30: 78.0, 40: 153.0}[t[2]]
              for t in tasks)
    print(f"\n{len(tasks)} d2 evaluations queued | ~{est/3600:.1f} core-hours "
          f"| {args.workers} workers -> ~{est/args.workers/60:.0f} min", flush=True)

    # ---- Ljung-Box results (available immediately) ----
    lb = pd.DataFrame(lb_rows)
    assert_local_path(OUTDIR); os.makedirs(OUTDIR, exist_ok=True)
    lb.to_csv(assert_local_path(os.path.join(OUTDIR, "ar_order_ljungbox.csv")),
              index=False)
    if len(lb):
        idx = [s for s in STATES if s in set(lb.state)]
        wcols = [f"white_p{p}" for p in ORDERS if f"white_p{p}" in lb.columns]
        print("\n=== residual whiteness: fraction of windows with white residuals "
              "(Ljung-Box p > 0.05, 50 lags = 2 s) ===")
        print(lb.groupby("state")[wcols].mean().reindex(idx).round(3).to_string())
        print("\n  overall: " + "  ".join(
            f"p={p}:{lb[f'white_p{p}'].mean():.3f}" for p in ORDERS
            if f"white_p{p}" in lb.columns))

    # ---- parallel d2 pass ----
    from multiprocessing import Pool
    from d2_worker import d2_one

    res = {}
    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        for i, (key, val) in enumerate(pool.imap_unordered(d2_one, tasks,
                                                           chunksize=1), 1):
            res[key] = val
            if i % 200 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"    {i}/{len(tasks)}  {el/60:.1f} min elapsed, "
                      f"~{el/i*(len(tasks)-i)/60:.0f} min left", flush=True)

    agg = defaultdict(list)
    for (animal, day, state, wi, p), v in res.items():
        if np.isfinite(v):
            agg[(animal, day, state, p)].append(v)
    rows = defaultdict(dict)
    for (animal, day, state, p), vals in agg.items():
        r = rows[(animal, day, state)]
        r.update({"animal": animal, "day": day, "state": state})
        r[f"d2_p{p}"] = float(np.median(vals))
        r[f"n_p{p}"] = len(vals)
    df = pd.DataFrame(list(rows.values()))
    df.to_csv(assert_local_path(os.path.join(OUTDIR, "ar_order_sweep.csv")), index=False)
    print(f"\nwrote -> {OUTDIR}  ({len(df)} rows, "
          f"{(time.time()-t_start)/60:.0f} min total)")

    if not len(df):
        return
    idx = [s for s in STATES if s in set(df.state)]
    dcols = [f"d2_p{p}" for p in ORDERS if f"d2_p{p}" in df.columns]
    piv = df.groupby("state")[dcols].median().reindex(idx)
    print("\n=== median d2 by state and AR order ===")
    print(piv.round(4).to_string())
    print("\n=== state ordering by d2 at each order (low = closer to criticality) ===")
    for c in dcols:
        print(f"   {c:<8s} " + "  <  ".join(piv[c].sort_values().index))
    ref = f"d2_p{ORDER_REF}"
    if ref in piv.columns:
        print(f"\n=== d2 relative to the submitted p={ORDER_REF} ===")
        print(piv.div(piv[ref], axis=0).round(3).to_string())


if __name__ == "__main__":
    main()
