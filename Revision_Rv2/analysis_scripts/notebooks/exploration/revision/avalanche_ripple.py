"""
avalanche_ripple.py -- avalanche exponents and DCC as a function of ripple rate.

Avalanches are detected over binned multi-unit spike counts, with the bin set to the
session's own population mean inter-event interval (~2 ms) rather than a fixed value,
and the activity threshold swept over several percentiles (20/30/40th); xmin for the
power-law fit is chosen by the `powerlaw` package's KS procedure and reported, not fixed
at 0. States are WAKE (THETA + nonTHETA merged) and NREM; theta is not split here.

Two views of the fitted exponents against ripple rate:
  A. mixture test -- bins avalanches by size and reports, per size bin, the mean local
     ripple rate and the local CCDF slope;
  B. tercile fit -- splits time into LOW/MID/HIGH local-ripple-rate terciles
     (duration-matched) and refits tau, alpha, beta and DCC in each.
Both the tercile split (B) and the mixture test (A) are run in NREM only: ripples run
far sparser in WAKE, so WAKE ripple-rate terciles are mostly ties at zero; WAKE is
still fit as a whole state for the state contrast. Restricting the rate analysis to
NREM holds state fixed while ripple rate varies.

    python avalanche_ripple.py --basepath "W:\\data\\PPP\\PPP20\\day4" --session PPP20_day4
Writes data/revision/log/avalanche_ripple_<session>.csv  and  _mixture_<session>.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fig1_metrics import GUARD_S, MIN_FIRING_RATE, build_state_intervals  # noqa: E402
from run_fig1_metrics import assert_local_path  # noqa: E402

LOG = os.path.join(r"D:\Documents\GitHub\Sleep_criticaltiy", "data", "revision", "log")
THRESHOLDS = (20, 30, 40)
RIPPLE_SMOOTH_S = 10.0      # window for the local ripple-rate estimate. 10 s not 1 s:
                            # NREM ripples run ~0.95 Hz and WAKE ~0.07 Hz, so a 1 s
                            # window returns 0 or 1 for nearly every avalanche and the
                            # terciles collapse. 10 s gives a graded rate in NREM; WAKE
                            # ripple rate stays near 0/1 at this window.
N_RIPPLE_BINS = 3           # terciles
MIN_AVAL = 200
N_FIT_MAX = 100_000         # powerlaw's KS xmin search scales badly; NREM at the mean-IEI
                            # bin yields ~1.2 M avalanches and the search does not finish.
                            # Exponent SE ~ 1/sqrt(n), so 100 k is already saturated
                            # (SE ~ 0.003). Subsample only for FITTING; all avalanches are
                            # still used for counts, ripple assignment and the mixture test.


def detect(all_spk, intervals, bin_s, pct):
    """Avalanches within intervals only. Returns size, duration, centre time.

    Avalanches cannot cross an interval boundary -- each interval is binned separately.
    """
    S, D, T = [], [], []
    for a, b in intervals:
        n = int(np.floor((b - a) / bin_s))
        if n < 5:
            continue
        lo, hi = np.searchsorted(all_spk, (a, b))
        c = np.histogram(all_spk[lo:hi], bins=n, range=(a, a + n * bin_s))[0]
        thr = np.percentile(c, pct)
        act = (c > thr).view(np.int8)
        if not act.any():
            continue
        d = np.diff(np.r_[0, act, 0])
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        for s_, e_ in zip(st, en):
            S.append(int(c[s_:e_].sum()))
            D.append(int(e_ - s_))
            T.append(a + (s_ + e_) * 0.5 * bin_s)
    return np.array(S), np.array(D), np.array(T)


def ripple_rate_at(times, ripple_starts, half_win=RIPPLE_SMOOTH_S / 2):
    """Local ripple rate (Hz) in a +/- half_win window around each time."""
    lo = np.searchsorted(ripple_starts, times - half_win)
    hi = np.searchsorted(ripple_starts, times + half_win)
    return (hi - lo) / (2 * half_win)


def fit_exponents(sizes, durs):
    """tau, alpha, beta_fit, beta_pred, DCC -- with xmin reported, not assumed."""
    import powerlaw
    out = dict(n=len(sizes))
    if len(sizes) < MIN_AVAL:
        return out
    S_, D_ = sizes[sizes > 0], durs[durs > 0]
    if len(S_) > N_FIT_MAX:
        rs = np.random.default_rng(0)
        S_ = rs.choice(S_, N_FIT_MAX, replace=False)
    if len(D_) > N_FIT_MAX:
        rs = np.random.default_rng(1)
        D_ = rs.choice(D_, N_FIT_MAX, replace=False)
    out["n_fit"] = int(len(S_))
    with np.errstate(all="ignore"):
        fs = powerlaw.Fit(S_, discrete=True, verbose=False)
        fd = powerlaw.Fit(D_, discrete=True, verbose=False)
    out.update(tau=float(fs.alpha), tau_xmin=float(fs.xmin),
               alpha=float(fd.alpha), alpha_xmin=float(fd.xmin))
    # beta from mean size per duration, over durations that are actually populated
    ud = np.unique(durs)
    ms = np.array([sizes[durs == t].mean() for t in ud])
    ok = (ud > 0) & (ms > 0)
    if ok.sum() >= 4:
        b, _ = np.polyfit(np.log10(ud[ok]), np.log10(ms[ok]), 1)
        out["beta_fit"] = float(b)
    if out.get("tau", 1) > 1 and "beta_fit" in out:
        out["beta_pred"] = float((out["alpha"] - 1) / (out["tau"] - 1))
        out["DCC"] = float(abs(out["beta_pred"] - out["beta_fit"]))
    return out


def local_slope_ccdf(sizes, n_win=12):
    """Local log-log slope of the CCDF, in overlapping log-size windows."""
    s = np.sort(sizes[sizes > 0])
    if len(s) < 50:
        return np.zeros(0), np.zeros(0)
    ccdf = 1.0 - np.arange(len(s)) / len(s)
    ls, lc = np.log10(s), np.log10(np.clip(ccdf, 1e-12, None))
    edges = np.linspace(ls.min(), ls.max(), n_win + 1)
    cen, slp = [], []
    for i in range(n_win):
        m = (ls >= edges[i]) & (ls < edges[i + 1])
        if m.sum() >= 20 and np.ptp(ls[m]) > 0:
            b, _ = np.polyfit(ls[m], lc[m], 1)
            cen.append(10 ** (0.5 * (edges[i] + edges[i + 1]))); slp.append(b)
    return np.array(cen), np.array(slp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basepath", default=r"W:\data\PPP\PPP20\day4")
    ap.add_argument("--session", default="PPP20_day4")
    args = ap.parse_args()
    from neuro_py.io import loading

    st, cm = loading.load_spikes(args.basepath, brainRegion="CA1")
    sd = loading.load_SleepState_states(args.basepath)
    rip = loading.load_ripples_events(basepath=args.basepath)
    rip_start = np.sort(np.asarray(rip.start, float))

    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    units = [np.sort(np.asarray(s, float))
             for s in np.asarray(st.data, dtype=object)[keep]]
    allspk = np.sort(np.concatenate(units))
    mean_iei = (allspk[-1] - allspk[0]) / len(allspk)
    print(f"=== {args.session} === {len(units)} units | mean IEI = {mean_iei*1000:.2f} ms "
          f"| {len(rip_start)} ripples")

    iv = build_state_intervals(sd, GUARD_S)
    wake = (np.vstack([iv["THETA"], iv["nonTHETA"]])
            if len(iv["THETA"]) else iv["nonTHETA"])
    wake = wake[np.argsort(wake[:, 0])]
    STATE_IV = {"WAKE": wake, "NREM": iv["NREM"]}

    rows, mixrows = [], []
    for state, ivs in STATE_IV.items():
        if not len(ivs):
            continue
        for pct in THRESHOLDS:
            t0 = time.time()
            S, D, T = detect(allspk, ivs, mean_iei, pct)
            if len(S) < MIN_AVAL:
                print(f"  {state} pct={pct}: only {len(S)} avalanches"); continue
            rr = ripple_rate_at(T, rip_start)

            # ---- whole-state fit ----
            f = fit_exponents(S, D)
            rows.append(dict(session=args.session, state=state, pct=pct,
                             bin_ms=mean_iei * 1000, ripple_group="ALL",
                             ripple_rate_mean=float(rr.mean()), **f))

            # ---- B. terciles of local ripple rate. NREM ONLY: WAKE ripple rate is
            #         0.069 Hz, so its terciles are mostly ties at zero (see docstring).
            #         Equal-count terciles => equal avalanche counts per group, which is
            #         the matching that matters for exponent fits.
            if state != "NREM":
                print(f"  [skip] {state} ripple terciles: rate too sparse to grade")
                continue
            qs = np.quantile(rr, [1 / 3, 2 / 3])
            grp = np.digitize(rr, qs)
            for gi, gname in enumerate(("LOW", "MID", "HIGH")):
                m = grp == gi
                if m.sum() < MIN_AVAL:
                    continue
                fg = fit_exponents(S[m], D[m])
                rows.append(dict(session=args.session, state=state, pct=pct,
                                 bin_ms=mean_iei * 1000, ripple_group=gname,
                                 ripple_rate_mean=float(rr[m].mean()), **fg))

            # ---- A. mixture test: ripple rate and local slope vs avalanche SIZE ----
            if pct == 30 and state == "NREM":
                pos = S > 0
                ls = np.log10(S[pos]); rrp = rr[pos]
                edges = np.linspace(ls.min(), ls.max(), 13)
                cen, slp = local_slope_ccdf(S)
                for i in range(len(edges) - 1):
                    m = (ls >= edges[i]) & (ls < edges[i + 1])
                    if m.sum() < 30:
                        continue
                    size_c = 10 ** (0.5 * (edges[i] + edges[i + 1]))
                    sl = np.interp(size_c, cen, slp) if len(cen) > 1 else np.nan
                    mixrows.append(dict(session=args.session, state=state,
                                        size_bin_centre=size_c, n=int(m.sum()),
                                        mean_ripple_rate=float(rrp[m].mean()),
                                        sem_ripple_rate=float(rrp[m].std() / np.sqrt(m.sum())),
                                        local_ccdf_slope=float(sl)))
            print(f"  {state} pct={pct}: {len(S)} avalanches, "
                  f"tau={f.get('tau', np.nan):.3f} alpha={f.get('alpha', np.nan):.3f} "
                  f"DCC={f.get('DCC', np.nan):.3f}  [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows); mx = pd.DataFrame(mixrows)
    assert_local_path(LOG); os.makedirs(LOG, exist_ok=True)
    df.to_csv(assert_local_path(os.path.join(
        LOG, f"avalanche_ripple_{args.session}.csv")), index=False)
    mx.to_csv(assert_local_path(os.path.join(
        LOG, f"avalanche_ripple_mixture_{args.session}.csv")), index=False)

    if len(df):
        print("\n" + "=" * 88)
        print("B. EXPONENTS AND DCC BY RIPPLE-RATE TERCILE (threshold = 30th pct)")
        print("=" * 88)
        s = df[df.pct == 30]
        cols = ["ripple_rate_mean", "n", "tau", "alpha", "beta_fit", "beta_pred", "DCC"]
        for state in s.state.unique():
            print(f"\n  {state}")
            t = s[s.state == state].set_index("ripple_group")
            print(t[[c for c in cols if c in t.columns]].round(3).to_string())
        print("\n" + "=" * 88)
        print("THRESHOLD SENSITIVITY (whole state)")
        print("=" * 88)
        a = df[df.ripple_group == "ALL"]
        print(a.pivot_table(index="state", columns="pct",
                            values=["tau", "alpha", "DCC"]).round(3).to_string())
    if len(mx):
        print("\n" + "=" * 88)
        print("A. MIXTURE TEST -- ripple rate and local CCDF slope vs avalanche size")
        print("=" * 88)
        for state in mx.state.unique():
            t = mx[mx.state == state]
            print(f"\n  {state}")
            print(f"    {'size':>9}{'n':>8}{'ripple Hz':>12}{'local slope':>13}")
            for r in t.itertuples(index=False):
                print(f"    {r.size_bin_centre:>9.1f}{r.n:>8d}"
                      f"{r.mean_ripple_rate:>12.3f}{r.local_ccdf_slope:>13.3f}")
            from scipy.stats import spearmanr
            if len(t) >= 5:
                rs = spearmanr(t.size_bin_centre, t.mean_ripple_rate)
                print(f"    spearman(size, ripple rate) = {rs.statistic:+.3f} "
                      f"p={rs.pvalue:.3g}   <- flat => exponent is NOT ripple-driven")
    print(f"\nwrote -> {LOG}")


if __name__ == "__main__":
    main()
