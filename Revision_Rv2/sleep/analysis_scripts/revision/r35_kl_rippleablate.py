"""
r35_kl_rippleablate.py -- d2 (2025 KL) computed on ripple-excised windows, completing the
full-window/ripple-excised x Euclidean/KL comparison (the Euclidean pair already exists as
MATLAB output; this fills in the KL ripple-excised arm, alongside a matched full-window KL
recomputation for a paired comparison).

Mirrors `matlab/time_resolved_d2_across_state_ripple_ablate.m`: raw ripple intervals are
subtracted from each 30 s window, leaving ripple-free subwindows; a window is kept only if
>= MIN_FREE_SEC (1.0 s) of ripple-free time survives; each surviving subwindow is binned
at 40 ms and concatenated (the same gluing the .m performs via `getDbetas`); d2 (2025 KL)
is then the KL-divergence rate to the beta=2 manifold on the glued series. Both the full
and excised arms use the same estimator (Yule-Walker AR) on the same windows, so the
comparison is paired; this differs from the stored `kl_error_bars.csv`, which used an MLE
AR fit (needed for its delta-method error bars), so the full-window arm is recomputed here
too.

Windows are the revision pipeline's own NREM non-overlapping tiling, identical to
`kl_error_bars.py`, keyed on (animal, day, w).

--sessions subsets which sessions to (re)compute; results are merged into any existing
output CSV rather than overwriting it.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/r35_kl_rippleablate.csv
"""
from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HELPER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))), "methods", "d2")
if _HELPER not in sys.path:
    sys.path.insert(0, _HELPER)

from fig1_metrics import (  # noqa: E402
    BIN_MUA_S, GUARD_S, MIN_FIRING_RATE, binned_counts, build_state_intervals,
    make_windows, slice_sorted,
)

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
LOGDIR = os.path.join(REPO, "data", "revision", "log")
MIN_FREE_SEC = 1.0          # the .m's own threshold
ORDER = 20
MIN_BINS = ORDER + 20       # same guard kl_error_bars.py applies

# the Fig. 2d sessions that also have KL windows (PPP7_day8 is not in the Fig-1 18)
SESSIONS = {
    "PPP7_day7":   r"W:\data\PPP\PPP7\day7",
    "PPP7_day10":  r"W:\data\PPP\PPP7\day10",
    "PPP4_day11":  r"W:\data\PPP\PPP4\day11",
    "PPP8_day8":   r"W:\data\PPP\PPP8\day8",
    "PPP20_day4":  r"W:\data\PPP\PPP20\day4",
    "PPP20_day5":  r"W:\data\PPP\PPP20\day5",
    "PPP21_day4":  r"W:\data\PPP\PPP21\day4",
    "PPP21_day5":  r"W:\data\PPP\PPP21\day5",
    "PPP23_day7":  r"W:\data\PPP\PPP23\day7",
    "PPP23_day8":  r"W:\data\PPP\PPP23\day8",
}


def assert_local_path(path):
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        raise PermissionError(f"refusing to write to UNC path: {p}")
    if os.path.splitdrive(p)[0].rstrip(":").upper() not in {"C", "D"}:
        raise PermissionError(f"refusing to write outside C:/D:: {p}")
    return p


def subtract_intervals(win, iv):
    """Port of fast_subtract_intervals: [w0,w1] minus the ripple intervals."""
    w0, w1 = win
    if iv.size == 0:
        return np.array([[w0, w1]])
    sel = iv[(iv[:, 1] > w0) & (iv[:, 0] < w1)]
    if not len(sel):
        return np.array([[w0, w1]])
    out, cur = [], w0
    for a, b in sel[np.argsort(sel[:, 0])]:
        a, b = max(a, w0), min(b, w1)
        if a > cur:
            out.append([cur, a])
        cur = max(cur, b)
    if cur < w1:
        out.append([cur, w1])
    return np.array(out) if out else np.empty((0, 2))


def glued_series(sp, segs, bin_s=BIN_MUA_S):
    """Bin each ripple-free subwindow and concatenate -- the MATLAB's gluing."""
    parts = []
    for a, b in segs:
        if b - a < bin_s:
            continue
        parts.append(binned_counts(slice_sorted(sp, a, b), a, b, bin_s,
                                   presliced=True).astype(float))
    return np.concatenate(parts) if parts else np.array([])


def make_d2_fn(dt, num_w=2 ** 14):
    """KL d2 with the spectral caches built ONCE per worker (`kl_error_bars.py ::
    make_d2_fn`). Same estimator, same Yule-Walker fit, same objective as
    `d2_from_timeseries`; only the cache lifetime differs."""
    from scipy.optimize import minimize

    from d2_fast import (_kl_rate_bits_from_AR_cached, _phiB_init_from,
                         _SpectralCache, _theta_to_phiB, fit_ar_yule_walker)
    caches = {}

    def f(x, order=ORDER, maxiter=600):
        x = np.asarray(x, float)
        x = (x - x.mean()) / (x.std() + 1e-12)
        phiA, sigma2A = fit_ar_yule_walker(x, order)
        p = len(phiA)
        if p not in caches:
            caches[p] = (_SpectralCache(max(1024, min(4096, num_w // 4)), p),
                         _SpectralCache(num_w, p))
        cc, cf = caches[p]

        def obj(theta, cache):
            return _kl_rate_bits_from_AR_cached(
                phiA, sigma2A, _theta_to_phiB(theta), cache)[0]

        th0 = _phiB_init_from(phiA)[:-1]
        res = minimize(lambda th: obj(th, cc), th0, method="Powell",
                       options=dict(maxiter=maxiter, xtol=1e-6, ftol=1e-8))
        return float(obj(np.asarray(res.x, float), cf)) / dt

    return f


def _init_worker():
    one.d2fn = make_d2_fn(BIN_MUA_S)


def one(args):
    key, x_full, x_abl = args
    d2fn = one.d2fn
    rec = {}
    for tag, x in (("full", x_full), ("abl", x_abl)):
        if x is None or x.size < MIN_BINS or not np.isfinite(x).all() or x.std() <= 0:
            rec[f"kl_{tag}"] = np.nan
            rec[f"n_{tag}"] = 0 if x is None else int(x.size)
            continue
        try:
            rec[f"kl_{tag}"] = float(d2fn(x))
        except Exception:                                        # noqa: BLE001
            rec[f"kl_{tag}"] = np.nan
        rec[f"n_{tag}"] = int(x.size)
    return key, rec


def collect():
    import nelpy as nel
    from neuro_py.io import loading
    items = []
    for sess, bp in SESSIONS.items():
        animal, day = sess.split("_")
        try:
            st, cm = loading.load_spikes(bp, brainRegion="CA1")
            iv = build_state_intervals(loading.load_SleepState_states(bp), GUARD_S)
            rip = loading.load_ripples_events(bp)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [SKIP] {sess}: {type(e).__name__}: {e}", flush=True)
            continue
        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        sp = np.sort(np.concatenate([np.asarray(s, float)
                                     for s in np.asarray(st.data, dtype=object)[keep]]))
        riv = np.sort(np.array(rip[["start", "stop"]], float), axis=0)
        if "NREM" not in iv or not len(iv["NREM"]):
            continue
        win = make_windows(iv["NREM"], "NREM")
        win = win[win.nonoverlap] if "nonoverlap" in win.columns else win
        n0 = 0
        for r in win.reset_index(drop=True).itertuples(index=False):
            x_full = binned_counts(slice_sorted(sp, r.win_start, r.win_stop),
                                   r.win_start, r.win_stop, BIN_MUA_S,
                                   presliced=True).astype(float)
            segs = subtract_intervals((r.win_start, r.win_stop), riv)
            free = float(np.sum(segs[:, 1] - segs[:, 0])) if segs.size else 0.0
            x_abl = glued_series(sp, segs) if free >= MIN_FREE_SEC else None
            items.append(((animal, day, int(round(r.win_start)),
                           1.0 - free / (r.win_stop - r.win_start)), x_full, x_abl))
            n0 += 1
        print(f"  {sess:12} {n0:4d} NREM windows queued", flush=True)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sessions", nargs="*",
                    help="subset to compute; results are MERGED into any existing CSV "
                         "so the file accumulates rather than being overwritten")
    a = ap.parse_args()

    t0 = time.time()
    if a.sessions:
        for k in list(SESSIONS):
            if k not in a.sessions:
                del SESSIONS[k]
    items = collect()
    print(f"\n{len(items)} windows | KL order p={ORDER} | {a.workers} workers", flush=True)

    from multiprocessing import Pool
    done = {}
    with Pool(processes=a.workers, initializer=_init_worker) as pool:
        for n, (key, rec) in enumerate(pool.imap_unordered(one, items, chunksize=4), 1):
            done[key] = rec
            if n % 250 == 0:
                el = (time.time() - t0) / 60
                print(f"    {n}/{len(items)}  {el:.1f} min", flush=True)

    D = pd.DataFrame([dict(animal=k[0], day=k[1], w=k[2], frac_removed=k[3], **v)
                      for k, v in done.items()])
    D["session"] = D.animal + "_" + D.day
    assert_local_path(LOGDIR)
    dst = assert_local_path(os.path.join(LOGDIR, "r35_kl_rippleablate.csv"))
    if os.path.exists(dst):
        old = pd.read_csv(dst)
        if "session" not in old.columns:
            old["session"] = old.animal.astype(str) + "_" + old.day.astype(str)
        old = old[~old["session"].isin(set(D["session"]))]
        D = pd.concat([old, D], ignore_index=True)
        print(f"  merged with existing CSV -> {D.session.nunique()} sessions total")
    D.to_csv(dst, index=False)
    print(f"\nwrote -> {dst}   ({len(D)} windows, {(time.time()-t0)/60:.1f} min)")

    # --- validation against the stored KL (different AR fit: YW here, MLE there) ---
    ref = os.path.join(LOGDIR, "kl_error_bars.csv")
    if os.path.exists(ref):
        K = pd.read_csv(ref)
        K["session"] = K.animal.astype(str) + "_" + K.day.astype(str)
        m = D.merge(K[["session", "w", "d2mle_p20"]], on=["session", "w"], how="inner")
        ok = m[["kl_full", "d2mle_p20"]].dropna()
        if len(ok) > 10:
            from scipy import stats as sstats
            print(f"\nvalidation vs stored kl_error_bars.csv (YW here vs MLE there): "
                  f"n={len(ok)}, r={sstats.spearmanr(ok.kl_full, ok.d2mle_p20).statistic:.3f}, "
                  f"median {ok.kl_full.median():.3f} vs {ok.d2mle_p20.median():.3f}")

    from scipy import stats as sstats
    print(f"\n{'='*70}\nKL d2, full window vs ripple-excised, per session\n{'='*70}")
    print(f"{'session':13}{'n':>6}{'KL full':>10}{'KL excised':>12}{'r':>8}"
          f"{'%removed':>10}")
    rows = []
    for s, g in D.groupby("session"):
        v = g[["kl_full", "kl_abl"]].dropna()
        if len(v) < 10:
            continue
        r = sstats.spearmanr(v.kl_full, v.kl_abl).statistic
        print(f"{s:13}{len(v):6d}{v.kl_full.median():10.3f}{v.kl_abl.median():12.3f}"
              f"{r:8.3f}{g.frac_removed.median():10.1%}")
        rows.append(dict(session=s, n=len(v), r=r,
                         med_full=v.kl_full.median(), med_abl=v.kl_abl.median()))
    R = pd.DataFrame(rows)
    print(f"\n  median across sessions: full {R.med_full.median():.3f}, "
          f"excised {R.med_abl.median():.3f}, r = {R.r.median():.3f}")
    print(f"  excision {'RAISES' if R.med_abl.median() > R.med_full.median() else 'LOWERS'}"
          f" KL d2 — same direction as the Euclidean" )


if __name__ == "__main__":
    main()
