"""
barrage_order_kernel_sweep.py -- Figure 3 (barrage) order/kernel sweep, d2 (2024 Euclidean).

Figure 3 uses a different session set from Figures 1-2 (barrage needs CA2 recordings),
and those sessions have far fewer CA1 units (29-64 vs 152-297). The stored Figure 3 d2
`.mat` files use the same parameters as Figure 1 (40 ms, p=20, 30 s, CA1), which gives
1.4-6.7 spikes/bin in most of these sessions; the other figures use a floor of ~10.

The script:
  1. reports spikes/bin per session for CA1 and CA1+CA2 at several bins, to establish
     which (population, bin) combinations are admissible for this session set;
  2. runs the order/kernel sweep at each admissible combination, correlating d2 against
     barrage rate (the Figure 3 event variable) at the parameters where the floor is met.

Populations: CA1 (what Figure 3 used) and CA1+CA2 (pooled; barrages originate in CA2, and
pooling is the only way these sessions clear the spikes/bin floor at a usable bin).

Reads the shares (X: and W:); writes only to D:.
    -> data/revision/log/barrage_sweep_coverage.csv
    -> data/revision/log/barrage_sweep_windows.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig1_metrics import (  # noqa: E402
    GUARD_S, MIN_FIRING_RATE, binned_counts, build_state_intervals,
    make_windows, slice_sorted,
)
from run_fig1_metrics import assert_local_path  # noqa: E402

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
OUTDIR = os.path.join(REPO, "data", "revision", "log")

# Figure 3 barrage set -- from data/ripple_barr/*/barr*, NOT the Figure 1/2 sessions.
BARRAGE_SESSIONS = {
    "NN16_day13": r"X:\data\Barrage\NN16\day13",
    "NN2_day10": r"X:\data\Barrage\NN2\day10",
    "NN2_day10_3": r"X:\data\Barrage\NN2\day10_3",
    "NN2_day24": r"X:\data\Barrage\NN2\day24",
    "NN3_day12": r"X:\data\Barrage\NN3\day12",
    "NN3_day15": r"X:\data\Barrage\NN3\day15",
    "NN3_day16": r"X:\data\Barrage\NN3\day16",
    "NN3_day25": r"X:\data\Barrage\NN3\day25",
    "NN8_day6": r"X:\data\Barrage\NN8\day6",
    "NN8_day7": r"X:\data\Barrage\NN8\day7",
    "PPP7_day7": r"W:\data\PPP\PPP7\day7",
    # PPP7_day12 uses the CA2-curated tree (CA1/CA2 split, with the barrage events file).
    "PPP7_day12": r"W:\data\PPP\CA2\PPP7\day12",
}
POPS = {"CA1": ["CA1"], "CA1+CA2": ["CA1", "CA2"]}
BINS_COVERAGE = (0.040, 0.080, 0.100, 0.200)
SPIKES_FLOOR = 10
FIXED_KERNEL = [(0.040, 20), (0.060, 13), (0.080, 10)]      # ~800 ms
FIXED_ORDER = [(0.040, 20), (0.060, 20), (0.080, 20)]       # 800/1200/1600 ms


def yw_sumphi(xz, p):
    n = xz.size
    r = np.correlate(xz, xz, "full")[n - 1:] / n
    i = np.arange(p)
    return float(np.sum(np.linalg.solve(r[np.abs(i[:, None] - i[None, :])], r[1:p + 1])))


def one_session(item):
    name, bp = item
    _h = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))), "methods", "d2")
    if _h not in sys.path:
        sys.path.insert(0, _h)
    from neuro_py.io import loading

    try:
        state_dict = loading.load_SleepState_states(bp)
        iv = build_state_intervals(state_dict, GUARD_S)
    except Exception as e:                                       # noqa: BLE001
        return name, None, None, f"states: {type(e).__name__}"
    if "NREM" not in iv or not len(iv["NREM"]):
        return name, None, None, "no NREM"

    try:
        bev = loading.load_barrage_events(basepath=bp)
        barr = (np.sort(np.asarray(bev["start"], float))
                if bev is not None and len(bev) else None)
    except Exception:                                            # noqa: BLE001
        barr = None

    win = make_windows(iv["NREM"], "NREM")
    if not len(win) or "nonoverlap" not in win.columns:
        return name, None, None, "no windows"
    win = win[win.nonoverlap].reset_index(drop=True)

    cov, rows = [], []
    for pop, regs in POPS.items():
        try:
            st, cm = loading.load_spikes(bp, brainRegion=regs)
        except Exception:                                        # noqa: BLE001
            continue
        if cm is None or not len(cm):
            continue
        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        if keep.sum() < 5:
            continue
        sp = np.sort(np.concatenate([np.asarray(s, float)
                                     for s in np.asarray(st.data, dtype=object)[keep]]))
        spb = {b: [] for b in BINS_COVERAGE}
        for r in win.itertuples(index=False):
            seg = slice_sorted(sp, r.win_start, r.win_stop)
            for b in BINS_COVERAGE:
                x = binned_counts(seg, r.win_start, r.win_stop, b, presliced=True)
                if x.size:
                    spb[b].append(float(np.mean(x)))
            rec = {"session": name, "pop": pop, "w": int(round(r.win_start))}
            rec["barrage_rate"] = (
                (np.searchsorted(barr, r.win_stop, "right")
                 - np.searchsorted(barr, r.win_start, "left"))
                / max(r.win_stop - r.win_start, 1e-9)) if barr is not None else np.nan
            for tag, combos in (("fk", FIXED_KERNEL), ("fo", FIXED_ORDER)):
                for b, p in combos:
                    x = binned_counts(seg, r.win_start, r.win_stop, b,
                                      presliced=True).astype(float)
                    key = f"d2ex_{tag}_b{int(b*1000)}_p{p}"
                    if x.size < p + 20 or x.std() <= 0:
                        rec[key] = np.nan
                        continue
                    z = (x - x.mean()) / x.std()
                    try:
                        rec[key] = abs(1 - yw_sumphi(z, p)) / np.sqrt(p)
                    except Exception:                            # noqa: BLE001
                        rec[key] = np.nan
            rows.append(rec)
        c = {"session": name, "pop": pop, "n_units": int(keep.sum()),
             "n_windows": len(win),
             "n_barrage": 0 if barr is None else int(barr.size)}
        for b in BINS_COVERAGE:
            c[f"spb_{int(b*1000)}"] = float(np.median(spb[b])) if spb[b] else np.nan
        cov.append(c)
    return name, pd.DataFrame(cov), pd.DataFrame(rows), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    args = ap.parse_args()
    t0 = time.time()
    print(f"barrage sweep | {len(BARRAGE_SESSIONS)} sessions | {args.workers} workers\n",
          flush=True)

    from multiprocessing import Pool
    cov, wins = [], []
    with Pool(processes=args.workers) as pool:
        for name, c, w, err in pool.imap_unordered(one_session,
                                                   list(BARRAGE_SESSIONS.items())):
            if err:
                print(f"  [SKIP] {name}: {err}", flush=True)
                continue
            print(f"  {name:13s} {len(w):5d} rows", flush=True)
            cov.append(c); wins.append(w)

    C = pd.concat(cov, ignore_index=True)
    W = pd.concat(wins, ignore_index=True)
    assert_local_path(OUTDIR); os.makedirs(OUTDIR, exist_ok=True)
    C.to_csv(assert_local_path(os.path.join(OUTDIR, "barrage_sweep_coverage.csv")),
             index=False)
    W.to_csv(assert_local_path(os.path.join(OUTDIR, "barrage_sweep_windows.csv")),
             index=False)
    print(f"\nwrote CSVs ({time.time()-t0:.0f} s)\n")

    print("=== 1. SPIKES PER BIN, NREM (floor ~10). Figure 3 used CA1 @ 40 ms ===")
    for pop in POPS:
        s = C[C["pop"] == pop].set_index("session")
        cols = [f"spb_{int(b*1000)}" for b in BINS_COVERAGE]
        t = s[["n_units"] + cols].rename(
            columns={c: c.replace("spb_", "") + "ms" for c in cols})
        print(f"\n--- {pop} ---")
        print(t.round(1).to_string())
        for b in BINS_COVERAGE:
            n = int((s[f"spb_{int(b*1000)}"] < SPIKES_FLOOR).sum())
            print(f"    {int(b*1000):>3} ms : {n}/{len(s)} sessions below the floor")

    print("\n=== 2. barrage rate vs d2 (2024 Euclidean), per-session Pearson ===")
    for pop in POPS:
        sub = W[W["pop"] == pop]
        if not len(sub):
            continue
        print(f"\n--- population: {pop} ---")
        for tag, combos, lab in (("fk", FIXED_KERNEL, "fixed kernel ~800 ms"),
                                 ("fo", FIXED_ORDER, "fixed order p=20")):
            print(f"  {lab}")
            for b, p in combos:
                c = f"d2ex_{tag}_b{int(b*1000)}_p{p}"
                rs = [np.corrcoef(g.barrage_rate, g[c])[0, 1]
                      for _, g in sub.groupby("session")
                      if len(g[["barrage_rate", c]].dropna()) >= 10
                      and g.barrage_rate.std() > 0 and g[c].std() > 0]
                if not rs:
                    continue
                print(f"    {int(b*1000):>3}ms p={p:<3} kernel {int(b*1000*p):>4}ms   "
                      f"r={np.median(rs):+.3f}   {sum(x>0 for x in rs)}/{len(rs)} positive")


if __name__ == "__main__":
    main()
