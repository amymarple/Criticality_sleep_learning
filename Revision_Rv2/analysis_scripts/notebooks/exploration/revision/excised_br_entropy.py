"""
excised_br_entropy.py -- ripple-excised branching ratio and ISI entropy, in Fig 2's
position, matched to the ripple-excised d2.

On non-overlapping windows, BR and (pooled) ISI entropy track SWR rate and are flat
across replay-probability sextiles, while the ripple-excised d2 rises with replay
probability. The BR / entropy values there were computed over the full window, with
ripples included, while d2 was computed with ripples excised. Here BR and ISI entropy
are recomputed on the ripple-free part of every window, exactly as the MATLAB d2
producer excises (raw ripples.timestamps subtracted from the 30 s window; the
surviving sub-windows are used together, nothing crosses a gap):

    BR_excised   : MR estimator on 4 ms active-unit counts of the PYRAMIDAL population,
                   r_k pooled over the ripple-free sub-windows (state_level_br.rk_curve_multi,
                   no lag regression crosses an excised gap), fit_m at k_max = 200
    isi_excised  : pooled log10(ISI) entropy, 20-bin auto-ranged (fig1_metrics.isi_entropy's
                   histogram), ISIs taken within each sub-window only
The un-excised versions are recomputed the same way (single segment) and checked against
the Fig-1 pickle's BR_pyr_k200 / isi_entropy_pooled, so the excision is the only change.

Windows: Fig 1's non-overlapping NREM tiles, 8 decoded sessions (d2_composition_glm.load_session
+ add_swr for SWR rate and replay probability). Output feeds br_ripple_replay.sextile_curves.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/excised_br_entropy_windows.csv
    -> data/revision/log/excised_br_entropy_sextiles.csv
    -> reports/revision_figs/excised_br_isi_ripple_replay.png/.svg
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

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path                     # noqa: E402
from fig1_metrics import (active_unit_counts, fit_m, slice_sorted, BIN_BR_S,        # noqa: E402
                          MIN_FIRING_RATE, ISI_N_BINS)
from state_level_br import rk_curve_multi                                          # noqa: E402
from r35_glm_ablation import LOGDIR, FIGDIR, WIN_S                                 # noqa: E402
from r35_swr_property_control import SESSION_TEMPLATE                              # noqa: E402
from d2_composition_glm import load_session, add_swr, fig2_basepaths                               # noqa: E402
import br_ripple_replay as brr                                                     # noqa: E402

K_MAX = 200
MIN_FREE_S = 1.0          # as the MATLAB producer (MIN_FREE_SEC)
MIN_SEG_S = 0.1           # a sub-window shorter than this contributes nothing at 4 ms
OUT_WIN = os.path.join(LOGDIR, "excised_br_entropy_windows.csv")
OUT_SEX = os.path.join(LOGDIR, "excised_br_entropy_sextiles.csv")


def free_segments(t0, t1, r0, r1):
    """[t0, t1) minus the ripple intervals (r0, r1 sorted by start)."""
    lo = np.searchsorted(r1, t0, "right")
    hi = np.searchsorted(r0, t1, "left")
    segs, cur = [], t0
    for a, b in zip(r0[lo:hi], r1[lo:hi]):
        if a > cur:
            segs.append((cur, min(a, t1)))
        cur = max(cur, b)
    if cur < t1:
        segs.append((cur, t1))
    return [(a, b) for a, b in segs if b - a >= MIN_SEG_S]


def entropy_from_isis(isis):
    isis = isis[np.isfinite(isis) & (isis > 0)]
    if isis.size < 2:
        return np.nan
    hist, _ = np.histogram(np.log10(isis), bins=ISI_N_BINS)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def window_metrics(e_spikes, all_sorted, segs):
    """BR (pyr, k=200) and pooled ISI entropy over a list of segments."""
    acts = [active_unit_counts(e_spikes, a, b, BIN_BR_S) for a, b in segs]
    K, r_k, se, n = rk_curve_multi(acts, K_MAX + 1)
    br = fit_m(K, r_k, se, K_MAX)
    isis = np.concatenate([np.diff(slice_sorted(all_sorted, a, b)) for a, b in segs]) if segs else np.array([])
    return br, entropy_from_isis(isis), float(sum(b - a for a, b in segs))


def main():
    from neuro_py.io import loading
    parts = []
    for bp in fig2_basepaths():
        sid, W = load_session(bp)
        if W is None or sid not in SESSION_TEMPLATE:
            continue
        W = add_swr(W, sid, bp)
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        units = [np.sort(np.asarray(s, float)) for s in np.asarray(st.data, dtype=object)[keep]]
        ctype = np.asarray(cm.putativeCellType)[keep]
        e_spikes = [u for u, c in zip(units, ctype) if "Pyramidal" in str(c)]
        all_sorted = np.sort(np.concatenate(units))
        rip = loading.load_ripples_events(bp)
        r0 = np.asarray(rip["start"], float); order = np.argsort(r0)
        r0, r1 = r0[order], np.asarray(rip["stop"], float)[order]
        br_ex, isi_ex, free, br_full, isi_full = [], [], [], [], []
        for c in W["center_s"].to_numpy(float):
            t0, t1 = c - WIN_S / 2, c + WIN_S / 2
            b, h, _ = window_metrics(e_spikes, all_sorted, [(t0, t1)])
            br_full.append(b); isi_full.append(h)
            segs = free_segments(t0, t1, r0, r1)
            tot = sum(bb - aa for aa, bb in segs)
            if tot < MIN_FREE_S:
                br_ex.append(np.nan); isi_ex.append(np.nan); free.append(tot); continue
            b, h, f = window_metrics(e_spikes, all_sorted, segs)
            br_ex.append(b); isi_ex.append(h); free.append(f)
        W["BR_excised"], W["isi_excised"] = br_ex, isi_ex
        W["BR_recomp"], W["isi_recomp"], W["free_s"] = br_full, isi_full, free
        ok = W[["BR_recomp", "BR_pyr_k200"]].dropna()
        ok2 = W[["isi_recomp", "isi_entropy_pooled"]].dropna()
        print(f"  {sid:12s} {len(W):4d} windows | recompute vs pickle: r(BR) {np.corrcoef(ok.T.to_numpy())[0,1]:.3f} "
              f"r(ISI) {np.corrcoef(ok2.T.to_numpy())[0,1]:.3f} | excised keeps {W.free_s.median()/WIN_S:.1%} | "
              f"r(BR_ex, BR) {W[['BR_excised','BR_pyr_k200']].corr().iloc[0,1]:.3f} r(isi_ex, isi) "
              f"{W[['isi_excised','isi_entropy_pooled']].corr().iloc[0,1]:.3f}", flush=True)
        W["session"] = sid
        parts.append(W)
    D = pd.concat(parts, ignore_index=True)
    assert_local_path(LOGDIR)
    D.to_csv(assert_local_path(OUT_WIN), index=False)

    brr.METRICS = [("BR_excised", "BR, ripple-excised"), ("isi_excised", "ISI entropy, ripple-excised"),
                   ("d2_excised", "d2, ripple-excised"),
                   ("BR_pyr_k200", "BR, full window"), ("isi_entropy_pooled", "ISI entropy, full window")]
    brr.COLORS.update({"BR_excised": "#2c7fb8", "isi_excised": "#238b45", "BR_pyr_k200": "#9ecae1",
                       "isi_entropy_pooled": "#a1d99b"})
    C = pd.concat([brr.sextile_curves(D, by, m, min_swr=ms) for by, ms in (("ripple_rate", None), ("replay_prob", 1))
                   for m, _ in brr.METRICS], ignore_index=True)
    C.to_csv(assert_local_path(OUT_SEX), index=False)
    print("\nper-session Spearman rho(sextile, metric z), signed-rank across sessions:")
    for by in ("ripple_rate", "replay_prob"):
        for m, lab in brr.METRICS:
            v = C[(C.by == by) & (C.metric == m)].rho.to_numpy()
            tb = (C[(C.by == by) & (C.metric == m)].s6 - C[(C.by == by) & (C.metric == m)].s1).median()
            print(f"  by {by:12s} {lab:30s} rho median {np.median(v):+.3f}  pos {int((v > 0).sum())}/{len(v)}  "
                  f"p = {stats.wilcoxon(v).pvalue:.3f} | top-bottom {tb:+.2f} SD")
    print("\nwithin-session r with replay probability / SWR rate (Fisher mean, sessions positive):")
    for m, lab in brr.METRICS:
        for x in ("replay_prob", "ripple_rate"):
            rs = np.array([np.corrcoef(g[[m, x]].dropna().T.to_numpy())[0, 1] for _, g in D.groupby("session")
                           if g[[m, x]].dropna().shape[0] > 40])
            print(f"  {lab:30s} vs {x:12s} r = {brr.fisher(rs):+.3f}  ({int((rs > 0).sum())}/{len(rs)}, p = {stats.wilcoxon(rs).pvalue:.3f})")
    brr.plot(C, ["BR_excised", "isi_excised", "d2_excised"], "excised_br_isi_ripple_replay")


if __name__ == "__main__":
    main()
