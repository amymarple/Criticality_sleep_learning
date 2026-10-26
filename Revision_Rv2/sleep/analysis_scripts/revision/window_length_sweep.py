"""window_length_sweep.py -- d2 (2024 Euclidean, p=20, 40 ms) across window length.

The estimator of `bin_choice_justification.py` (`yw_sumphi`; the 30 s arm reproduces
`bin_choice_windows.csv :: d2ex_fb_b40_p20`), on non-overlapping windows of
10 / 15 / 20 / 30 / 60 s (step = window, which tiles identically to
`make_windows(step=2)[nonoverlap]`), all four states, the 18 Fig-1 sessions.
Per window: d2, N (samples), N/p, spikes per bin, and NREM ripple rate. Summaries:

  (i)   the state ordering at each window length (paired, session = unit)
  (ii)  the NREM ripple correlation (Fisher-z mean +/- SEM)
  (iii) the fraction of windows below the N/p >= 20 adequacy floor
  (iv)  window-to-window SD of d2 vs window length

Windows shorter than 30 s sit below the N/p >= 20 floor at p = 20 (16 s is the boundary;
10 s gives N/p = 12.5); they are computed to measure where the estimator degrades.

    $PY window_length_sweep.py --test            # PPP20 day4 only
    $PY window_length_sweep.py --workers 8

    -> data/revision/log/window_length_sweep_windows.csv    one row per window
    -> data/revision/log/window_length_sweep.csv            summary tables
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fig1_metrics import (                                           # noqa: E402
    GUARD_S, MIN_FIRING_RATE, STATES, binned_counts, build_state_intervals,
    make_windows, slice_sorted)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path, session_id  # noqa: E402
from bin_choice_justification import yw_sumphi                       # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, ".."))
LOG = os.path.join(REPO, "data", "revision", "log")
WINDOWS_S = (10, 15, 20, 30, 60)
BIN, P = 0.040, 20
NP_FLOOR = 20


def one_session(bp):
    _h = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))), "methods", "d2")
    if _h not in sys.path:
        sys.path.insert(0, _h)
    from neuro_py.io import loading

    animal, day = session_id(bp)
    try:
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        state_dict = loading.load_SleepState_states(bp)
    except Exception as e:                                           # noqa: BLE001
        return animal, day, None, f"{type(e).__name__}: {e}"
    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    sp = np.sort(np.concatenate([np.asarray(s, float)
                                 for s in np.asarray(st.data, dtype=object)[keep]]))
    iv = build_state_intervals(state_dict, GUARD_S)
    try:
        ev = loading.load_ripples_events(basepath=bp)
        rip = (np.sort(np.asarray(ev["start"], float))
               if ev is not None and len(ev) else None)
    except Exception:                                                # noqa: BLE001
        rip = None

    rows = []
    for state in STATES:
        if state not in iv or not len(iv[state]):
            continue
        for W in WINDOWS_S:
            win = make_windows(iv[state], state, window=float(W), step=float(W))
            if not len(win):
                continue
            for r in win.itertuples(index=False):
                seg = slice_sorted(sp, r.win_start, r.win_stop)
                x = binned_counts(seg, r.win_start, r.win_stop, BIN,
                                  presliced=True).astype(float)
                rec = dict(animal=animal, day=day, state=state, window_s=W,
                           w=int(round(r.win_start)), n_samples=int(x.size),
                           n_over_p=x.size / P,
                           spikes_per_bin=float(x.mean()) if x.size else np.nan,
                           d2ex=np.nan)
                if state == "NREM":
                    rec["ripple_rate"] = (
                        (np.searchsorted(rip, r.win_stop, "right")
                         - np.searchsorted(rip, r.win_start, "left"))
                        / max(r.win_stop - r.win_start, 1e-9)) if rip is not None else np.nan
                if x.size >= P + 20 and x.std() > 0:
                    z = (x - x.mean()) / x.std()
                    try:
                        rec["d2ex"] = abs(1 - yw_sumphi(z, P)) / np.sqrt(P)
                    except Exception:                                # noqa: BLE001
                        pass
                rows.append(rec)
    return animal, day, pd.DataFrame(rows), None


def fisher(rs):
    r = np.asarray([v for v in rs if np.isfinite(v)], float)
    if r.size < 2:
        return np.nan, np.nan, int(r.size), 0
    z = np.arctanh(np.clip(r, -.999999, .999999))
    return (float(np.tanh(z.mean())), float(z.std(ddof=1) / np.sqrt(z.size)),
            int(r.size), int((r > 0).sum()))


def summarise(W):
    from scipy import stats
    out = []
    for (st, ws), g in W.groupby(["state", "window_s"]):
        per = g.groupby(["animal", "day"])
        out.append(dict(table="coverage", state=st, window_s=ws,
                        n_sessions=per.ngroups, n_windows=len(g),
                        frac_fit_failed=float(g.d2ex.isna().mean()),
                        n_over_p=float(g.n_over_p.median()),
                        below_np_floor=bool(g.n_over_p.median() < NP_FLOOR),
                        median_d2=float(per.d2ex.median().median()),
                        median_within_session_sd=float(per.d2ex.std().median())))
    med = W.groupby(["window_s", "state", "animal", "day"]).d2ex.median().reset_index()
    for ws, g in med.groupby("window_s"):
        piv = g.pivot_table(index=["animal", "day"], columns="state", values="d2ex")
        for other in ("THETA", "nonTHETA", "REM"):
            if other not in piv or "NREM" not in piv:
                continue
            v = piv[["NREM", other]].dropna()
            if len(v) < 3:
                continue
            d = v["NREM"] - v[other]
            out.append(dict(table="ordering", window_s=ws, contrast=f"NREM-{other}",
                            n_sessions=len(v), median_diff=float(d.median()),
                            n_pos=int((d > 0).sum()),
                            p_wilcoxon=float(stats.wilcoxon(v["NREM"], v[other]).pvalue)))
    nr = W[W.state == "NREM"]
    for ws, g in nr.groupby("window_s"):
        rs = []
        for _, s in g.groupby(["animal", "day"]):
            v = s[["d2ex", "ripple_rate"]].dropna()
            if len(v) >= 10 and v.d2ex.std() > 0 and v.ripple_rate.std() > 0:
                rs.append(float(np.corrcoef(v.d2ex, v.ripple_rate)[0, 1]))
        r, se, n, npos = fisher(rs)
        out.append(dict(table="ripple", window_s=ws, r=r, sem_z=se, n_sessions=n,
                        n_pos=npos, r_median=float(np.median(rs)) if rs else np.nan))
    return pd.DataFrame(out)


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sess = [bp for bp in FIG1_SESSIONS if "PPP20\\day4" in bp] if a.test else FIG1_SESSIONS
    t0 = time.time()
    print(f"window-length sweep | {len(sess)} sessions | windows {WINDOWS_S} s | "
          f"{a.workers} workers", flush=True)
    frames = []
    with Pool(processes=min(a.workers, len(sess))) as pool:
        for animal, day, W, err in pool.imap_unordered(one_session, sess):
            if err:
                print(f"  [FAIL] {animal}_{day}: {err}", flush=True)
                continue
            frames.append(W)
            print(f"  {animal}_{day:8s} {len(W):6d} windows  ({time.time()-t0:.0f} s)",
                  flush=True)
    W = pd.concat(frames, ignore_index=True)
    os.makedirs(assert_local_path(LOG), exist_ok=True)
    suffix = "_test" if a.test else ""
    W.to_csv(assert_local_path(os.path.join(
        LOG, f"window_length_sweep_windows{suffix}.csv")), index=False)

    ref = os.path.join(LOG, "bin_choice_windows.csv")
    if os.path.exists(ref):
        B = pd.read_csv(ref)[["animal", "day", "w", "d2ex_fb_b40_p20"]]
        M = W[(W.state == "NREM") & (W.window_s == 30)].merge(B, on=["animal", "day", "w"])
        d = (M.d2ex - M.d2ex_fb_b40_p20).abs()
        print(f"\nVALIDATION vs bin_choice_windows.csv (30 s NREM): {len(M)} matched "
              f"windows, max |diff| = {d.max():.2e}, "
              f"{'REPRODUCES' if d.max() < 1e-9 else 'MISMATCH -- investigate'}")

    S = summarise(W)
    S.to_csv(assert_local_path(os.path.join(
        LOG, f"window_length_sweep{suffix}.csv")), index=False)
    pd.set_option("display.width", 160)
    for tbl in ("coverage", "ordering", "ripple"):
        print(f"\n=== {tbl} ===")
        print(S[S.table == tbl].dropna(axis=1, how="all").to_string(index=False))
    print(f"\ndone in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
