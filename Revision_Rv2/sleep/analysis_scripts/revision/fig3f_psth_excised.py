"""fig3f_psth_excised.py -- Figure 3f, revised: the branching ratio around barrages, on the ripple-
and barrage-excised background.

The published Fig 3f (generate_branching_ratio_mouse.ipynb) is the full-window BR around
barrage starts (10 s windows stepped 2 s), so a window around a barrage contains the
barrage's own spikes; this excises both ripples and barrages from the background instead.

Reuses fig2g_psth_excised.py's functions (br_series, changes, psth, random_times), adapted
to: events = barrage starts in NREM (opto_cck_excised.load_barrage_iv); excised = the
union of ripple and barrage intervals; windows on the raw NREM epochs with no state guard;
sessions = the 12-session barrage cohort (barrage_order_kernel_sweep.BARRAGE_SESSIONS), with
NN2_day10 and NN2_day10_3 pooled into one file mean.

Two window sets for the per-session change: `sliding` (every 2 s window) and `disjoint`
(the same windows thinned to centres >= 10 s apart within each NREM interval). The PSTH
uses the sliding series; every test is on per-session means.

    -> data/revision/log/fig3f_psth_sessions.csv   (per session x arm x window set: mean change, barrage / shuffle)
    -> data/revision/log/fig3f_psth_curves.csv     (per session x arm x lag bin: PSTH, barrage / shuffle, SEM)
    -> data/revision/log/fig3f_psth_validation.csv (full arm vs the stored series and changes)
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pickle                                                        # noqa: E402

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from scipy import stats                                              # noqa: E402

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                                   # noqa: E402
from fig1_metrics import MIN_FIRING_RATE                                        # noqa: E402
from fig2g_psth_excised import br_series, changes, psth, random_times, WIN       # noqa: E402
from barrage_order_kernel_sweep import BARRAGE_SESSIONS                          # noqa: E402
from opto_cck_excised import load_barrage_iv                                     # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
STORED = os.path.join(REPO, "data", "ripple_barr", "branching_ratio", "barr")
OUT_S = os.path.join(LOGDIR, "fig3f_psth_sessions.csv")
OUT_C = os.path.join(LOGDIR, "fig3f_psth_curves.csv")
OUT_V = os.path.join(LOGDIR, "fig3f_psth_validation.csv")
SEED = 0
SESS = os.environ.get("SESS", "")                     # one session (a test run writes *_<SESS>.csv)
NPROC = int(os.environ.get("NPROC", "6"))


def merge(iv):
    """Union of intervals, sorted by start (free_segments needs starts and stops both sorted)."""
    if not len(iv):
        return np.zeros(0), np.zeros(0)
    iv = iv[np.argsort(iv[:, 0])]
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    out = np.asarray(out, float)
    return out[:, 0], out[:, 1]


def disjoint_idx(cen, ivs, win=WIN):
    """Indices of windows whose centres are >= win apart within each NREM interval (greedy, in time order)."""
    keep, last = [], -np.inf
    for a, b in ivs:
        last = -np.inf
        for j in np.flatnonzero((cen >= a) & (cen <= b)):
            if cen[j] >= last + win - 1e-6:
                keep.append(j)
                last = cen[j]
    return np.asarray(keep, int)


def run_session(sess):
    from neuro_py.io import loading
    bp = BARRAGE_SESSIONS[sess]
    rng = np.random.default_rng(SEED + sum(map(ord, sess)))
    st, cm = loading.load_spikes(bp, brainRegion="CA1")
    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    units = [np.sort(np.asarray(s, float)) for s in np.asarray(st.data, dtype=object)[keep]]
    ctype = np.asarray(cm.putativeCellType)[keep]
    e_spikes = [u for u, c in zip(units, ctype) if "Pyramidal" in str(c)]
    sd = loading.load_SleepState_states(bp)
    nrem_raw = np.asarray(sd["NREMstate"], float)
    # windows on the RAW NREM epochs, no state guard (how the stored barrage series was made; its first window
    # starts at the epoch onset). Fig 2f's ripple series used a 5 s guard.
    ivs = [tuple(x) for x in nrem_raw]
    rip = loading.load_ripples_events(bp)
    rip_iv = np.column_stack([np.asarray(rip["start"], float), np.asarray(rip["stop"], float)])
    bar_iv = load_barrage_iv(bp)
    in_nrem = np.zeros(len(bar_iv), bool)
    for a, b in nrem_raw:
        in_nrem |= (bar_iv[:, 0] >= a) & (bar_iv[:, 0] <= b)
    starts = np.sort(bar_iv[in_nrem, 0])
    e0, e1 = merge(np.vstack([rip_iv, bar_iv]))                     # ripples AND barrages out of the background
    cen, full, exc = br_series(e_spikes, ivs, e0, e1)
    shuf = random_times(nrem_raw, bar_iv[:, 0][np.argsort(bar_iv[:, 0])], np.sort(bar_iv[:, 1]), starts, rng)
    dj = disjoint_idx(cen, ivs)
    rows, curves = [], []
    for arm, br in (("full", full), ("excised", exc)):
        for wset, idx in (("sliding", np.arange(len(cen))), ("disjoint", dj)):
            c_, b_ = cen[idx], br[idx]
            rows.append(dict(session=sess, arm=arm, windows=wset, n_windows=len(idx), n_barrages=len(starts),
                             n_units=len(e_spikes), change_barrage=float(np.nanmean(changes(c_, b_, starts))),
                             change_shuffle=float(np.nanmean(changes(c_, b_, shuf))), br_median=float(np.nanmedian(b_))))
        t, p_b, e_b = psth(cen, br, starts)
        _, p_s, e_s = psth(cen, br, shuf)
        for ti, a_, b_, ea, eb in zip(t, p_b, p_s, e_b, e_s):
            curves.append(dict(session=sess, arm=arm, lag_s=ti, br_barrage=a_, br_shuffle=b_, sem_barrage=ea, sem_shuffle=eb))
    # comparison of the full arm against the stored series and the stored per-event changes
    val = dict(session=sess, n_windows=len(cen))
    f = os.path.join(STORED, f"{sess}_CA1.pkl")
    if os.path.exists(f):
        r_ = pickle.load(open(f, "rb"))["results"]
        sc, sb = np.asarray(r_["centers"], float), np.asarray(r_["branching_ratios"], float)
        j = np.clip(np.searchsorted(sc, cen), 0, sc.size - 1)
        ok = (np.abs(sc[j] - cen) < 1e-6) & np.isfinite(full) & np.isfinite(sb[j])
        val.update(stored_windows=int(sc.size), matched=int(ok.sum()),
                   r_series=float(np.corrcoef(full[ok], sb[j][ok])[0, 1]) if ok.sum() > 10 else np.nan,
                   max_abs_diff=float(np.nanmax(np.abs(full[ok] - sb[j][ok]))) if ok.any() else np.nan)
    g = os.path.join(STORED, f"{sess}_CA1_br_changes.pkl")
    if os.path.exists(g):
        q = pickle.load(open(g, "rb"))
        val.update(stored_change=float(np.nanmean(q["barrage_br_changes"])),
                   stored_change_shuffle=float(np.nanmean(q["barrage_br_changes_shuffle"])),
                   stored_n_barrages=int(len(q["barrage_br_changes"])))
    full_sl = [r for r in rows if r["arm"] == "full" and r["windows"] == "sliding"][0]
    val.update(change=full_sl["change_barrage"], change_shuffle=full_sl["change_shuffle"], n_barrages=len(starts))
    print(f"  {sess:12s} {len(cen):5d} windows ({len(dj)} disjoint), {len(starts):4d} NREM barrages, {len(e_spikes)} pyr | "
          + " | ".join(f"{r['arm']}/{r['windows']} {r['change_barrage']:+.4f} vs {r['change_shuffle']:+.4f}" for r in rows)
          + f" | stored series r = {val.get('r_series', np.nan):.3f}, change {val.get('stored_change', np.nan):+.4f}", flush=True)
    return rows, curves, val


def file_key(sess):
    return "_".join(sess.split("_")[:2])                    # NN2_day10_3 -> NN2_day10


def report(R):
    print(f"\n{'=' * 90}\nFig 3f: BR change around barrage starts (centre +/-2.5 s vs baseline), per file mean "
          f"(NN2_day10 + _3 pooled); barrage vs shuffle, paired signed-rank\n{'=' * 90}")
    for (arm, wset), P in R.groupby(["arm", "windows"]):
        P = P.assign(file=P.session.map(file_key)).groupby("file")[["change_barrage", "change_shuffle"]].mean()
        d = P.change_barrage - P.change_shuffle
        print(f"  {arm:8s} {wset:9s} N = {len(P)} | barrage {P.change_barrage.median():+.4f}  shuffle "
              f"{P.change_shuffle.median():+.4f} | barrage > shuffle {int((d > 0).sum())}/{len(d)}, "
              f"signed-rank p = {stats.wilcoxon(d).pvalue:.4f}")


def main():
    sessions = [SESS] if SESS else list(BARRAGE_SESSIONS)
    if len(sessions) == 1:
        out = [run_session(sessions[0])]
    else:
        from multiprocessing import Pool
        with Pool(min(NPROC, len(sessions))) as pool:
            out = pool.map(run_session, sessions)
    R = pd.DataFrame([r for rows, _, _ in out for r in rows])
    C = pd.DataFrame([c for _, curves, _ in out for c in curves])
    V = pd.DataFrame([v for _, _, v in out])
    suf = f"_{SESS}" if SESS else ""
    for D, p in ((R, OUT_S), (C, OUT_C), (V, OUT_V)):
        D.to_csv(assert_local_path(p.replace(".csv", f"{suf}.csv")), index=False)
    print(V.to_string(index=False))
    report(R)


if __name__ == "__main__":
    main()
