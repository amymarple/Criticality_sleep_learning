"""
euclid_order_sweep.py -- AR-order sweep for **d2 (2024 Euclidean)**, all states.

This sweeps the d2 definition the published figures actually use (Euclidean, not KL),
across every state rather than NREM only.

The 2024 Euclidean d2 is a **closed-form QR projection** onto the beta=2 hyperplane
(Sooter `calc_dbv1.m`), not an iterative minimisation over a spectral integral, so it is
cheap enough that this parallelises over SESSIONS rather than windows; the wall time is
dominated by loading the shares.

Built-in validation: at p = 20 this should reproduce the MATLAB `d2` already stored in
`data/revision/*.pkl` (from getDbetas_error_bar.m). The script reports that agreement.

Reads the shares; writes only to D:.
    -> data/revision/log/euclid_order_sweep.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig1_metrics import (  # noqa: E402
    BIN_MUA_S, GUARD_S, MIN_FIRING_RATE, STATES, binned_counts,
    build_state_intervals, make_windows, slice_sorted,
)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path, session_id  # noqa: E402

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
OUTDIR = os.path.join(REPO, "data", "revision", "log")
ORDERS = (5, 10, 15, 20, 30, 40)


def _manifold_basis(n, b=2):
    """Columns span the beta manifold. Exact port of calc_dbv1.m's `x`."""
    m = b // 2 - 1
    x = np.zeros((n, n - m))
    for k in range(m + 1, n + 1):
        for t in range(1, k + 1):
            x[t - 1, k - m - 1] = comb(k, t) * (-1) ** (t + 1)
    return x


_BASIS = {}


def euclid_db(kern, b=2):
    """d2 (2024 Euclidean): L2 distance from kern to the beta manifold.

    Port of prox_crit_toolkit/src/calc_dbv1.m (QR least squares -> lstsq).
    """
    kern = np.asarray(kern, float)
    n = kern.size
    key = (n, b)
    if key not in _BASIS:
        x = _manifold_basis(n, b)
        X = x - x[:, [0]]
        _BASIS[key] = (x[:, 0].copy(), X[:, 1:].copy())
    x0, X = _BASIS[key]
    v, *_ = np.linalg.lstsq(X, kern - x0, rcond=None)
    return float(np.sqrt(np.sum((kern - (x0 + X @ v)) ** 2)))


def one_session(bp):
    """Everything for one session. Runs in a worker process."""
    # Windows spawns workers, so methods/d2 is not on the child's sys.path.
    _helper = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))), "methods", "d2")
    if _helper not in sys.path:
        sys.path.insert(0, _helper)
    from d2_fast import fit_ar_yule_walker
    from neuro_py.io import loading

    animal, day = session_id(bp)
    try:
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        state_dict = loading.load_SleepState_states(bp)
    except Exception as e:                                       # noqa: BLE001
        return animal, day, None, f"load failed: {type(e).__name__}: {e}"

    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    dat = np.asarray(st.data, dtype=object)[keep]
    all_spikes = np.sort(np.concatenate([np.asarray(s, float) for s in dat]))
    iv = build_state_intervals(state_dict, GUARD_S)

    try:
        ev = loading.load_ripples_events(basepath=bp)
        rip = (np.sort(np.asarray(ev["start"], float))
               if ev is not None and len(ev) else None)
    except Exception:                                            # noqa: BLE001
        rip = None

    rows = []
    for state in STATES:
        if state not in iv or not len(iv[state]):
            continue
        win = make_windows(iv[state], state)
        if not len(win) or "nonoverlap" not in win.columns:
            continue
        win = win[win.nonoverlap].reset_index(drop=True)
        for r in win.itertuples(index=False):
            x = binned_counts(slice_sorted(all_spikes, r.win_start, r.win_stop),
                              r.win_start, r.win_stop, BIN_MUA_S,
                              presliced=True).astype(float)
            if x.size < max(ORDERS) + 20 or x.std() <= 0:
                continue
            xz = (x - x.mean()) / x.std()
            rec = {"animal": animal, "day": day, "state": state,
                   "w": int(round(r.win_start)),
                   "win_start": float(r.win_start), "win_stop": float(r.win_stop),
                   "MUA_mean": float(x.mean())}
            if rip is not None:
                n_rip = int(np.searchsorted(rip, r.win_stop, "right")
                            - np.searchsorted(rip, r.win_start, "left"))
                rec["ripple_rate"] = n_rip / max(r.win_stop - r.win_start, 1e-9)
            else:
                rec["ripple_rate"] = np.nan
            for p in ORDERS:
                try:
                    phi, _ = fit_ar_yule_walker(xz, p)
                    s = float(np.sum(phi))
                    # EXACT: the beta=2 manifold is the hyperplane sum(phi)=1, whose
                    # unit normal is (1..1)/sqrt(p), so the distance is closed-form.
                    rec[f"d2ex_p{p}"] = abs(1.0 - s) / np.sqrt(p)
                    rec[f"sumphi_p{p}"] = s
                    # QR projection (calc_dbv1.m port); diverges from the exact value at
                    # p>=30, where cond(X) ~ 1e17 exceeds double precision -- valid through p=20.
                    rec[f"d2qr_p{p}"] = euclid_db(phi)
                    rec[f"phinorm_p{p}"] = float(np.linalg.norm(phi))
                    rt = np.abs(np.roots(np.r_[phi[::-1], -1.0]))
                    rec[f"minroot_p{p}"] = float(np.min(rt)) if rt.size else np.nan
                except Exception:                                # noqa: BLE001
                    for k in ("d2ex", "sumphi", "d2qr", "phinorm", "minroot"):
                        rec[f"{k}_p{p}"] = np.nan
            rows.append(rec)
    return animal, day, pd.DataFrame(rows), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(18, os.cpu_count() or 4))
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    paths = FIG1_SESSIONS[:1] if args.test else FIG1_SESSIONS
    t0 = time.time()
    print(f"d2 (2024 Euclidean) order sweep | {len(paths)} sessions | "
          f"{args.workers} workers | orders {ORDERS}\n", flush=True)

    out = []
    if args.workers > 1 and len(paths) > 1:
        from multiprocessing import Pool
        with Pool(processes=args.workers) as pool:
            for animal, day, df, err in pool.imap_unordered(one_session, paths):
                if err:
                    print(f"  [FAIL] {animal}_{day}: {err}", flush=True)
                    continue
                print(f"  {animal}_{day:8s} {len(df):5d} windows", flush=True)
                out.append(df)
    else:
        for bp in paths:
            animal, day, df, err = one_session(bp)
            if err:
                print(f"  [FAIL] {animal}_{day}: {err}", flush=True)
                continue
            print(f"  {animal}_{day:8s} {len(df):5d} windows", flush=True)
            out.append(df)

    M = pd.concat(out, ignore_index=True)
    assert_local_path(OUTDIR)
    os.makedirs(OUTDIR, exist_ok=True)
    dst = assert_local_path(os.path.join(OUTDIR, "euclid_order_sweep.csv"))
    M.to_csv(dst, index=False)
    print(f"\nwrote -> {dst}  ({len(M)} windows, {time.time()-t0:.0f} s)\n")

    print("=== coverage: windows per state ===")
    print(M.groupby("state").size().reindex([s for s in STATES if s in set(M.state)])
           .to_string(), "\n")

    print("=== d2 (2024 Euclidean), median by state and AR order ===")
    piv = M.groupby("state")[[f"d2ex_p{p}" for p in ORDERS]].median()
    print(piv.reindex([s for s in STATES if s in piv.index]).round(4).to_string(), "\n")

    # --- validation against the MATLAB values already stored -----------------
    import glob
    W = []
    for f in sorted(glob.glob(os.path.join(REPO, "data", "revision",
                                           "*_fig1_metrics_nonoverlap.pkl"))):
        w = pd.read_pickle(f)["windows"].copy()
        base = os.path.basename(f).split("_CA1_fig1")[0]
        w["animal"], w["day"] = base.split("_", 1)
        w["w"] = w.win_start.round().astype(int)
        W.append(w[["animal", "day", "state", "w", "d2"]])
    if W:
        J = M.merge(pd.concat(W, ignore_index=True), on=["animal", "day", "state", "w"],
                    how="inner")
        v = J[["d2ex_p20", "d2"]].dropna()
        if len(v):
            print("=== validation: our Python Euclidean p=20 vs stored MATLAB d2 ===")
            print(f"  joined windows : {len(v)}")
            print(f"  Pearson r      : {np.corrcoef(v.d2ex_p20, v.d2)[0,1]:+.4f}")
            print(f"  ratio (ours/MATLAB) median : {(v.d2ex_p20/v.d2).median():.4f}")
            print(f"  ours   median {v.d2ex_p20.median():.4f}")
            print(f"  MATLAB median {v.d2.median():.4f}")


if __name__ == "__main__":
    main()
