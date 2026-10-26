"""
ljungbox_df_audit.py -- residual whiteness of the AR(p) fit: Ljung-Box with and without the degrees-of-freedom
correction.

Ljung-Box on the residuals of a FITTED AR(p) model is asymptotically chi2 with **df = lags - p**, not df = lags:

    acorr_ljungbox(e, lags=[50], return_df=True)               # df = 50
    acorr_ljungbox(e, lags=[50], model_df=p, return_df=True)   # df = 50 - p

Without model_df the critical value is too large, so too many windows are declared white, and more so the larger
p. Both are computed here on the SAME residuals (40 ms bins, z-scored, Yule-Walker AR(p), p = 5 ... 40, 50 lags =
2 s), so the difference is the df correction alone. Windows: the non-overlapping 30 s windows of every state, 18
Fig-1 sessions (WPS windows per session and state; WPS=0: all of them). white_df (df = 50 - p) is drawn in
Extended Data Fig. 2g.

    -> data/revision/log/ljungbox_df_audit.csv
"""
from __future__ import annotations

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig1_metrics import (  # noqa: E402
    GUARD_S, MIN_FIRING_RATE, STATES, binned_counts, build_state_intervals,
    make_windows, slice_sorted,
)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path, session_id  # noqa: E402
from ar_order_sweep import ar_residuals  # noqa: E402
# fit_ar_yule_walker lives in methods/d2/d2_fast.py and is imported inside the
# worker, after that folder is put on sys.path -- same pattern as ar_order_sweep.py.

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
OUTDIR = os.path.join(REPO, "data", "revision", "log")
BIN = 0.040
ORDERS = (5, 10, 15, 20, 30, 40)
LB_LAGS = 50            # exactly as ar_order_sweep.py uses it
# 0 = use every window. Read from the environment, NOT set from argv: these run
# under multiprocessing.Pool and on Windows the workers re-import this module, so a
# value assigned in main() would silently revert to the default in every child.
WINDOWS_PER_STATE = int(os.environ.get("WPS", "40"))


def one_session(bp):
    _helper = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))), "methods", "d2")
    if _helper not in sys.path:
        sys.path.insert(0, _helper)
    from neuro_py.io import loading
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from d2_fast import fit_ar_yule_walker

    animal, day = session_id(bp)
    try:
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        state_dict = loading.load_SleepState_states(bp)
    except Exception as e:                                        # noqa: BLE001
        return animal, day, None, f"{type(e).__name__}: {e}"

    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    dat = np.asarray(st.data, dtype=object)[keep]
    sp = np.sort(np.concatenate([np.asarray(s, float) for s in dat]))
    iv = build_state_intervals(state_dict, GUARD_S)

    rows = []
    for state in STATES:
        if state not in iv or not len(iv[state]):
            continue
        win = make_windows(iv[state], state)
        if not len(win) or "nonoverlap" not in win.columns:
            continue
        w = win[win.nonoverlap].reset_index(drop=True)
        if WINDOWS_PER_STATE and len(w) > WINDOWS_PER_STATE:
            w = w.iloc[:: int(np.ceil(len(w) / WINDOWS_PER_STATE))]
        for r in w.itertuples(index=False):
            seg = slice_sorted(sp, r.win_start, r.win_stop)
            x = binned_counts(seg, r.win_start, r.win_stop, BIN,
                              presliced=True).astype(float)
            if not x.size or x.std() == 0:
                continue
            z = (x - x.mean()) / (x.std() + 1e-12)
            for p in ORDERS:
                try:
                    phi, _ = fit_ar_yule_walker(z, p)
                    e = ar_residuals(z, phi)
                    if e.size <= LB_LAGS + 10:
                        continue
                    # identical residuals, identical lag -- only df differs
                    pv_no = float(acorr_ljungbox(
                        e, lags=[LB_LAGS], return_df=True)["lb_pvalue"].iloc[0])
                    pv_yes = float(acorr_ljungbox(
                        e, lags=[LB_LAGS], model_df=p,
                        return_df=True)["lb_pvalue"].iloc[0])
                    rows.append(dict(animal=animal, day=day, state=state,
                                     w=int(round(r.win_start)), order=p,
                                     pv_nodf=pv_no, pv_df=pv_yes,
                                     white_nodf=pv_no > 0.05,
                                     white_df=pv_yes > 0.05))
                except Exception:                                 # noqa: BLE001
                    pass
    return animal, day, pd.DataFrame(rows), None


def main():
    t0 = time.time()
    print("Ljung-Box df audit | 18 sessions | 40 ms bin, lags=50\n", flush=True)
    from multiprocessing import Pool
    out = []
    with Pool(processes=min(18, os.cpu_count() or 4)) as pool:
        for animal, day, df, err in pool.imap_unordered(one_session, FIG1_SESSIONS):
            if err:
                print(f"  [FAIL] {animal}_{day}: {err}", flush=True)
                continue
            print(f"  {animal}_{day:9s} {len(df):5d} rows", flush=True)
            out.append(df)
    D = pd.concat(out, ignore_index=True)
    assert_local_path(OUTDIR); os.makedirs(OUTDIR, exist_ok=True)
    D.to_csv(assert_local_path(os.path.join(OUTDIR, "ljungbox_df_audit.csv")),
             index=False)
    print(f"\n{len(D)} window-order fits ({time.time()-t0:.0f} s)\n")

    print("=" * 78)
    print("RESIDUAL WHITENESS, same residuals, lags=50, only the df differs")
    print("  'no model_df' = what ar_order_sweep.py does now  (df = 50)")
    print("  'model_df=p'  = correct for a fitted AR(p)       (df = 50 - p)")
    print("=" * 78)
    for state in ["NREM"] + [s for s in D.state.unique() if s != "NREM"]:
        g = D[D.state == state]
        if not len(g):
            continue
        print(f"\n--- {state}  (n = {g.w.nunique()} windows) ---")
        print(f"{'order':>7}{'% white (no df)':>18}{'% white (df=50-p)':>20}"
              f"{'inflation':>12}")
        for p in ORDERS:
            s = g[g.order == p]
            if not len(s):
                continue
            a, b = s.white_nodf.mean() * 100, s.white_df.mean() * 100
            print(f"{p:>7}{a:>17.1f}%{b:>19.1f}%{a-b:>11.1f} pp")

    n = D[D.state == "NREM"]
    print("\n  The correction grows with p, exactly as expected -- df = 50 - p, so the")
    print("  omission matters more the more coefficients are fitted. That biases the")
    print("  comparison in favour of HIGH orders, which is the direction that flatters")
    print("  the p = 20 choice.")
    for p in (5, 20, 40):
        s = n[n.order == p]
        if len(s):
            print(f"    p={p:<3} NREM: {s.white_nodf.mean()*100:5.1f}% -> "
                  f"{s.white_df.mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
