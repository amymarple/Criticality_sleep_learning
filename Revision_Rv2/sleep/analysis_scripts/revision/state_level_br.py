"""
state_level_br.py -- branching ratio per STATE (not per window), W&P-style.

Motivated by Supplementary Note 10 of Wilting & Priesemann 2018, which specifies
for the rat hippocampus recordings:

    "we collapsed the spike times of all recorded neurons into one single train of
     population spike counts a_t, where a_t denotes how many neurons spiked in the
     t-th time bin"                        -> A_t = count of ACTIVE UNITS  (matched)
    "We used dt = 4 ms"                    -> 4 ms                          (matched)
    "k_max = 2500 (corresponding to 10 s) for the rat recordings, ...
     assuring that k_max was always in the order of multiple autocorrelation times"
                                           -> k_max = 10 s, NOT 800 ms      (not matched)

The 30 s windows used here cannot hold a 10 s lag. This script therefore estimates one m per
(session, state) from ALL of that state's data, pooling the lag regressions across
intervals exactly as WiltingPriesemann.get_slopes does (it concatenates x and y over
segments, then runs a single regression per lag). That makes large k_max reachable and
gives a like-for-like comparison with their rat numbers.

Writes data/revision/log/state_level_br.csv. Reads the shares; writes only to D:.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig1_metrics import (  # noqa: E402
    BIN_BR_S, GUARD_S, MIN_FIRING_RATE, STATES, active_unit_counts,
    build_state_intervals, fit_m, fit_m_offset, tau_ms,
)
from run_fig1_metrics import (  # noqa: E402
    FIG1_SESSIONS, assert_local_path, session_id,
)

OUTDIR = os.path.join(r"D:\Documents\GitHub\Sleep_criticaltiy", "data", "revision", "log")

# 800 ms (the window-level choice used here) up to 10 s (W&P's rat choice)
K_MAX_LEVELS = (25, 50, 100, 200, 500, 1000, 2500)
K_MAX_TOP = max(K_MAX_LEVELS)


def rk_curve_multi(segments, k_max=K_MAX_TOP):
    """Pooled lag regressions over many segments -- no regression ever crosses a gap.

    Identical to WiltingPriesemann.get_slopes (which concatenates x and y across
    segments and runs one linregress per lag), but accumulates the six sufficient
    statistics instead, and gets the cross term for every lag at once by FFT.
    """
    K = np.arange(1, k_max)
    n = np.zeros(len(K))
    sx = np.zeros(len(K)); sy = np.zeros(len(K))
    sxx = np.zeros(len(K)); syy = np.zeros(len(K)); sxy = np.zeros(len(K))

    for A in segments:
        N = A.size
        if N < 20:
            continue
        c1 = np.concatenate(([0.0], np.cumsum(A)))
        c2 = np.concatenate(([0.0], np.cumsum(A * A)))
        tot1, tot2 = c1[-1], c2[-1]

        nfft = 1 << int(np.ceil(np.log2(2 * N)))
        F = np.fft.rfft(A, n=nfft)
        acf = np.fft.irfft(F * np.conjugate(F), n=nfft)

        ok = (N - K) >= 2
        if not ok.any():
            continue
        kk = K[ok]
        nn = (N - kk).astype(float)
        n[ok] += nn
        sx[ok] += c1[nn.astype(np.int64)]
        sxx[ok] += c2[nn.astype(np.int64)]
        sy[ok] += tot1 - c1[kk]
        syy[ok] += tot2 - c2[kk]
        sxy[ok] += acf[kk]

    r_k = np.full(len(K), np.nan)
    se = np.full(len(K), np.nan)
    good = n >= 10
    Sxx = np.where(good, sxx - sx * sx / np.where(good, n, 1), np.nan)
    Sxy = np.where(good, sxy - sx * sy / np.where(good, n, 1), np.nan)
    Syy = np.where(good, syy - sy * sy / np.where(good, n, 1), np.nan)
    ok = good & (Sxx > 0)
    r_k[ok] = Sxy[ok] / Sxx[ok]
    resid = np.where(ok, (Syy - r_k * Sxy) / np.maximum(n - 2, 1), np.nan)
    se[ok] = np.sqrt(np.maximum(resid[ok], 0.0) / Sxx[ok])
    return K, r_k, se, n


def main():
    from neuro_py.io import loading

    rows = []
    for bp in FIG1_SESSIONS:
        animal, day = session_id(bp)
        t0 = time.time()
        try:
            st, cm = loading.load_spikes(bp, brainRegion="CA1")
            state_dict = loading.load_SleepState_states(bp)
        except Exception as e:
            print(f"  [FAIL] {bp}: {type(e).__name__}: {e}", flush=True)
            continue

        ct = np.asarray(cm.putativeCellType).astype(str)
        fr = np.asarray(cm.firingRate, float)
        keep = fr > MIN_FIRING_RATE
        is_e = (np.char.find(np.char.lower(ct), "pyramidal") >= 0) & keep
        dat = np.asarray(st.data, dtype=object)
        pops = {
            "pyr": [np.sort(np.asarray(s, float)) for s in dat[is_e]],
            "all": [np.sort(np.asarray(s, float)) for s in dat[keep]],
        }
        iv = build_state_intervals(state_dict, GUARD_S)
        print(f"\n=== {animal}_{day} ===  pyr={len(pops['pyr'])} all={len(pops['all'])}",
              flush=True)

        for state in STATES:
            ivs = iv[state]
            if not len(ivs):
                continue
            dur = float(np.sum(ivs[:, 1] - ivs[:, 0]))
            for pop, units in pops.items():
                if not units:
                    continue
                segs = [active_unit_counts(units, a, b, BIN_BR_S) for a, b in ivs]
                K, r_k, se, n_pairs = rk_curve_multi(segs)
                row = {"animal": animal, "day": day, "state": state, "pop": pop,
                       "n_units": len(units), "n_intervals": len(ivs),
                       "total_dur_s": dur,
                       "longest_interval_s": float(np.max(ivs[:, 1] - ivs[:, 0]))}
                for km in K_MAX_LEVELS:
                    m = fit_m(K, r_k, se, km)
                    row[f"m_k{km}"] = m
                    row[f"tau_ms_k{km}"] = tau_ms(np.array([m]))[0]
                    # W&P reject a recording when r_k carries a constant offset
                    # (H_offset, Suppl. Note 5); a flat tail inflates the plain m
                    m_off, b_off, c_off, sse_off, sse_pl = fit_m_offset(K, r_k, se, km)
                    row[f"moff_k{km}"] = m_off
                    row[f"c_k{km}"] = c_off
                    # offset relative to the fitted amplitude: |c| / (|b| + |c|)
                    row[f"offfrac_k{km}"] = (abs(c_off) / (abs(b_off) + abs(c_off))
                                             if np.isfinite(c_off) and
                                             (abs(b_off) + abs(c_off)) > 0 else np.nan)
                    row[f"sse_ratio_k{km}"] = (sse_off / sse_pl
                                               if sse_pl and np.isfinite(sse_pl) and sse_pl > 0
                                               else np.nan)
                    # how much data actually supports the longest lag in this fit
                    idx = min(km - 2, len(n_pairs) - 1)
                    row[f"npairs_k{km}"] = float(n_pairs[idx])
                rows.append(row)
                print("   %-9s %-4s m: " % (state, pop) +
                      "  ".join("k%-4d=%.3f%s" % (
                          km, row[f"m_k{km}"],
                          # "*" = large descriptive offset fraction, NOT a W&P rejection
                          "~" if (row[f"offfrac_k{km}"] or 0) > 0.2 else " ")
                          for km in K_MAX_LEVELS), flush=True)
        print(f"   [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    assert_local_path(OUTDIR)
    os.makedirs(OUTDIR, exist_ok=True)
    out = assert_local_path(os.path.join(OUTDIR, "state_level_br.csv"))
    df.to_csv(out, index=False)
    print(f"\nwrote -> {out}   ({len(df)} rows)")

    if len(df):
        for pop in ("pyr", "all"):
            sub = df[df["pop"] == pop]
            if not len(sub):
                continue
            idx = [s for s in STATES if s in set(sub.state)]
            print(f"\n=== median m across sessions, {pop} (plain fit, no offset) ===")
            print(sub.groupby("state")[[f"m_k{k}" for k in K_MAX_LEVELS]]
                     .median().reindex(idx).round(4).to_string())
            print(f"--- m with an offset term allowed, {pop} ---")
            print(sub.groupby("state")[[f"moff_k{k}" for k in K_MAX_LEVELS]]
                     .median().reindex(idx).round(4).to_string())
            # This fraction is a descriptive statistic, not a W&P criterion. Their
            # tests (H_offset: 2*SSE_off < SSE_plain; H_tau) are applied in
            # wp_rejection_test.py, which reads the sse_ratio/moff columns below.
            print(f"--- offset fraction |c|/(|b|+|c|)  (descriptive only "
                  f"-- see wp_rejection_test.py for W&P's real H_offset/H_tau) ---")
            print(sub.groupby("state")[[f"offfrac_k{k}" for k in K_MAX_LEVELS]]
                     .median().reindex(idx).round(3).to_string())
            print(f"--- tau (ms), plain fit, {pop} ---")
            print(sub.groupby("state")[[f"tau_ms_k{k}" for k in K_MAX_LEVELS]]
                     .median().reindex(idx).round(1).to_string())


if __name__ == "__main__":
    main()
