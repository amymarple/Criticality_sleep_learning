"""
bin_choice_justification.py -- the 40 ms bin, and the ripple result across bin sizes.

  1. COVERAGE  Empty-bin fraction and mean spikes/bin for the 18 sessions at
     5/15/20/40/60/80 ms, per state: below 40 ms the spike-count series is undersampled,
     and unevenly across states (NREM/nonTHETA are ~10x sparser than REM/THETA), which
     reorders the states rather than only adding noise.

  2. BIN SWEEP in three arms, so bin size is separated from kernel length and order:
       fixed kernel ~800 ms : (5 ms, p=160) ... (40 ms, p=20) (60 ms, p=13) (80 ms, p=10)
       fixed order  p=20    : kernels of 100 to 1600 ms
       fixed bin    40 ms   : p = 5 to 40
     For each, the per-session NREM ripple-rate correlation.

d2 here is d2 (2024 Euclidean) via its closed form |1-sum(phi)|/sqrt(p).

Reads the shares; writes only to D:.
    -> data/revision/log/bin_choice.csv
    -> data/revision/log/bin_choice_windows.csv
    -> reports/figures/bin_choice_justification.png / .svg
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# One BLAS thread per worker, set before numpy is imported: the script runs one session per
# multiprocessing.Pool worker, and a BLAS thread pool in every worker oversubscribes the cores.
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

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUTDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "figures")

# coverage bins, 5 to 80 ms
BINS_COVERAGE = (0.005, 0.015, 0.020, 0.040, 0.060, 0.080)
SPIKES_PER_BIN_FLOOR = 10          # ~10 ISI per bin rule of thumb
# (bin, order) -- kernel = bin*order
# Fixed ~800 ms kernel: p scales as 1/bin, and so does N, so N/p is invariant (~37.5).
FIXED_KERNEL = [(0.005, 160), (0.015, 53), (0.040, 20), (0.060, 13), (0.080, 10)]
# Fixed order p=20: kernel scales with bin, and N/p falls as the bin coarsens
# (80 ms -> N/p 18.8, just under the >= 20 floor).
FIXED_ORDER = [(0.005, 20), (0.015, 20), (0.040, 20), (0.060, 20), (0.080, 20)]
# Fixed bin of 40 ms, order swept.
FIXED_BIN = [(0.040, 5), (0.040, 10), (0.040, 20), (0.040, 30), (0.040, 40)]


def _acf(xz):
    """Autocorrelation via FFT, O(n log n): equal to `np.correlate(xz, xz, "full")[n-1:]/n` to 2e-16
    (machine precision) at n = 750 / 2000 / 6000."""
    n = xz.size
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(xz, m)
    return np.fft.irfft(f * np.conj(f), m)[:n] / n


def yw_sumphi(xz, p):
    r = _acf(xz)
    idx = np.arange(p)
    phi = np.linalg.solve(r[np.abs(idx[:, None] - idx[None, :])], r[1:p + 1])
    return float(np.sum(phi))


def one_session(bp):
    _h = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))), "methods", "d2")
    if _h not in sys.path:
        sys.path.insert(0, _h)
    from neuro_py.io import loading

    animal, day = session_id(bp)
    try:
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        state_dict = loading.load_SleepState_states(bp)
    except Exception as e:                                       # noqa: BLE001
        return animal, day, None, None, f"{type(e).__name__}: {e}"

    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    sp = np.sort(np.concatenate([np.asarray(s, float)
                                 for s in np.asarray(st.data, dtype=object)[keep]]))
    iv = build_state_intervals(state_dict, GUARD_S)
    try:
        ev = loading.load_ripples_events(basepath=bp)
        rip = (np.sort(np.asarray(ev["start"], float))
               if ev is not None and len(ev) else None)
    except Exception:                                            # noqa: BLE001
        rip = None

    cov, wins = [], []
    for state in STATES:
        if state not in iv or not len(iv[state]):
            continue
        win = make_windows(iv[state], state)
        if not len(win) or "nonoverlap" not in win.columns:
            continue
        acc = {b: {"mean": [], "zero": []} for b in BINS_COVERAGE}
        for r in win[win.nonoverlap].reset_index(drop=True).itertuples(index=False):
            seg = slice_sorted(sp, r.win_start, r.win_stop)
            if seg.size < 50:
                continue
            for b in BINS_COVERAGE:
                x = binned_counts(seg, r.win_start, r.win_stop, b, presliced=True)
                if x.size:
                    acc[b]["mean"].append(float(np.mean(x)))
                    acc[b]["zero"].append(float(np.mean(x == 0)))
            if state == "NREM":
                rec = {"animal": animal, "day": day,
                       "w": int(round(r.win_start))}
                rec["ripple_rate"] = (
                    (np.searchsorted(rip, r.win_stop, "right")
                     - np.searchsorted(rip, r.win_start, "left"))
                    / max(r.win_stop - r.win_start, 1e-9)) if rip is not None else np.nan
                for tag, combos in (("fk", FIXED_KERNEL), ("fo", FIXED_ORDER), ("fb", FIXED_BIN)):
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
                        except Exception:                        # noqa: BLE001
                            rec[key] = np.nan
                wins.append(rec)
        for b in BINS_COVERAGE:
            if acc[b]["mean"]:
                cov.append({"animal": animal, "day": day, "state": state,
                            "bin_ms": int(b * 1000),
                            "n_windows": len(acc[b]["mean"]),
                            "spikes_per_bin": float(np.median(acc[b]["mean"])),
                            "empty_frac": float(np.median(acc[b]["zero"]))})
    return animal, day, pd.DataFrame(cov), pd.DataFrame(wins), None


def make_figure(C):
    """Two panels: the criterion (spikes/bin) and the consequence (empty bins)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.8, "xtick.major.width": 0.8,
                         "ytick.major.width": 0.8})

    bins = sorted(C.bin_ms.unique())
    states = [s for s in STATES if s in set(C.state)]
    # bin size is ordered: one sequential hue, light to dark
    ramp = plt.get_cmap("Blues")
    colors = {b: ramp(v) for b, v in zip(bins, np.linspace(0.35, 0.92, len(bins)))}

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), constrained_layout=True)
    specs = [("spikes_per_bin", "mean spikes per bin", SPIKES_PER_BIN_FLOOR,
              "~10 spikes/bin floor"),
             ("empty_frac", "fraction of empty bins", None, None)]

    for ax, (col, ylab, hline, hlab) in zip(axes, specs):
        w = 0.8 / len(bins)
        for bi, b in enumerate(bins):
            xs, ys = [], []
            for si, s in enumerate(states):
                v = C[(C.state == s) & (C.bin_ms == b)][col].to_numpy()
                x0 = si + (bi - (len(bins) - 1) / 2) * w
                ax.bar(x0, np.median(v), width=w * 0.88, color=colors[b],
                       edgecolor="white", linewidth=0.8, zorder=2,
                       label=f"{b} ms" if si == 0 else None)
                jit = (np.random.default_rng(si * 10 + bi).random(v.size) - .5) * w * .5
                xs.append(x0 + jit); ys.append(v)
            ax.scatter(np.concatenate(xs), np.concatenate(ys), s=5, color="0.25",
                       alpha=.55, linewidths=0, zorder=3)
        if hline is not None:
            ax.axhline(hline, color="#B03A2E", lw=1.2, ls="--", zorder=4)
            ax.text(len(states) - .5, hline * 1.06, hlab, ha="right", va="bottom",
                    fontsize=7.5, color="#B03A2E")
        if col == "empty_frac":
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states, fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color="0.9", lw=.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0].legend(title="bin size", frameon=False, fontsize=8, title_fontsize=8,
                   loc="upper left")
    fig.suptitle("Why 40 ms: below it the spike-count series is undersampled, "
                 "and unevenly across states", fontsize=9.5)

    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(
            os.path.join(FIGDIR, f"bin_choice_justification.{ext}")),
            dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(18, os.cpu_count() or 4))
    args = ap.parse_args()
    t0 = time.time()
    print(f"bin-choice justification | 18 sessions | {args.workers} workers\n", flush=True)

    from multiprocessing import Pool
    cov, wins = [], []
    with Pool(processes=args.workers) as pool:
        for animal, day, c, w, err in pool.imap_unordered(one_session, FIG1_SESSIONS):
            if err:
                print(f"  [FAIL] {animal}_{day}: {err}", flush=True)
                continue
            print(f"  {animal}_{day:9s} {len(c):3d} cov rows, {len(w):5d} NREM windows",
                  flush=True)
            cov.append(c); wins.append(w)

    C = pd.concat(cov, ignore_index=True)
    W = pd.concat(wins, ignore_index=True)
    assert_local_path(OUTDIR); os.makedirs(OUTDIR, exist_ok=True)
    C.to_csv(assert_local_path(os.path.join(OUTDIR, "bin_choice.csv")), index=False)
    W.to_csv(assert_local_path(os.path.join(OUTDIR, "bin_choice_windows.csv")),
             index=False)
    print(f"\nwrote CSVs ({time.time()-t0:.0f} s)\n")

    order = [s for s in STATES if s in set(C.state)]
    print("=== 1. MEAN SPIKES PER BIN (median across sessions; floor ~10) ===")
    t = C.pivot_table(index="state", columns="bin_ms", values="spikes_per_bin",
                      aggfunc="median").reindex(order)
    print(t.round(1).to_string())
    print("  sessions BELOW the 10-spike floor, per bin:")
    for b in sorted(C.bin_ms.unique()):
        sub = C[C.bin_ms == b]
        print(f"    {b:>3} ms : " + "  ".join(
            f"{s} {int((sub[sub.state==s].spikes_per_bin < SPIKES_PER_BIN_FLOOR).sum())}"
            f"/{int((sub.state==s).sum())}" for s in order))

    print("\n=== 2. FRACTION OF EMPTY BINS (median across sessions) ===")
    e = C.pivot_table(index="state", columns="bin_ms", values="empty_frac",
                      aggfunc="median").reindex(order)
    print((e * 100).round(2).to_string() + "   [%]")
    print("\n  per-session range at 20 ms (min-max %):")
    for s in order:
        v = C[(C.state == s) & (C.bin_ms == 20)].empty_frac * 100
        print(f"    {s:9s} {v.min():5.1f} - {v.max():5.1f}")

    print("\n=== 3. NREM ripple rate vs d2 (2024 Euclidean), per-session Pearson ===")
    W["sess"] = W.animal + "_" + W.day
    F2 = {"PPP20_day4", "PPP20_day5", "PPP21_day4", "PPP21_day5", "PPP23_day7",
          "PPP23_day8", "PPP4_day11", "PPP7_day10", "PPP8_day8"}
    for tag, combos, lab in (("fk", FIXED_KERNEL, "FIXED KERNEL ~800 ms"),
                             ("fo", FIXED_ORDER, "FIXED ORDER p=20"),
                             ("fb", FIXED_BIN, "FIXED BIN 40 ms, order swept")):
        print(f"\n--- {lab} ---")
        print(f"{'bin':>6}{'order':>7}{'kernel':>9}   {'all n=16':<22}{'Fig2 n=9':<22}")
        for b, p in combos:
            c = f"d2ex_{tag}_b{int(b*1000)}_p{p}"
            outs = []
            for sub in (W, W[W.sess.isin(F2)]):
                rs = [np.corrcoef(g.ripple_rate, g[c])[0, 1]
                      for _, g in sub.groupby("sess")
                      if len(g[["ripple_rate", c]].dropna()) >= 10
                      and g.ripple_rate.std() > 0 and g[c].std() > 0]
                outs.append(f"r={np.median(rs):+.3f}  {sum(x>0 for x in rs)}/{len(rs)}")
            print(f"{int(b*1000):>4}ms{p:>7}{int(b*1000*p):>7}ms   {outs[0]:<22}{outs[1]:<22}")

    make_figure(C)
    print(f"\nfigure -> {os.path.join(FIGDIR, 'bin_choice_justification.png')}")


if __name__ == "__main__":
    main()
