"""mixture_proof.py -- the NREM avalanche-size distribution as a two-component mixture, split by
each avalanche's local ripple rate (median split: low vs high), compared against the pooled
single power-law fit.

By total probability the pooled distribution is the weighted sum of the two conditional
distributions. Four further quantities are computed:

  T1  shape comparison between the components (not merely rate/scale): raw sizes and sizes
      normalised by the within-component mean
  T2  family fit per component (power law vs exponential)
  T3  penalised-likelihood comparison of the pooled distribution against the two-component
      mixture and against a single power law
  T4  reconstruction: the mixture of the fitted components against the pooled CCDF, including
      its curvature, versus a single power law

Design:
  * All two-sample comparisons use fixed-n subsamples (5000 each, n ~ 500k otherwise), repeated,
    reporting the KS statistic D as effect size with its resampling spread rather than a p-value
    on the full n.
  * T1 is run on raw sizes and on sizes normalised by the within-component mean, to separate
    shape from scale.
  * Common xmin across components throughout.
  * All likelihoods computed in plain numpy (discrete truncated power law / exponential); no
    scipy.linalg.

    python mixture_proof.py --basepath "W:\\data\\PPP\\PPP20\\day4" --session PPP20_day4
    -> data/revision/log/mixture_proof_<session>.csv (+ _curves)
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
from fig1_metrics import GUARD_S, MIN_FIRING_RATE, build_state_intervals, subtract_intervals  # noqa: E402
from run_fig1_metrics import assert_local_path  # noqa: E402
from avalanche_ripple import detect, ripple_rate_at  # noqa: E402

LOG = os.path.join(r"D:\Documents\GitHub\Sleep_criticaltiy", "data", "revision", "log")
PCT = 30
N_SUB = 5000
N_BOOT = 200
SEED = 0


# ---------------------------------------------------------------------------
# discrete truncated power law and exponential: log-likelihoods in numpy
# ---------------------------------------------------------------------------
def ll_powerlaw(x, alpha, xmin, xmax):
    """Discrete power law on [xmin, xmax], normalised by direct summation."""
    k = np.arange(xmin, xmax + 1, dtype=float)
    Z = np.sum(k ** (-alpha))
    if Z <= 0 or not np.isfinite(Z):
        return -np.inf
    return float(np.sum(-alpha * np.log(x) - np.log(Z)))


def fit_powerlaw(x, xmin, xmax):
    """MLE alpha by 1-D grid + refine. Returns (alpha, ll)."""
    grid = np.linspace(1.05, 8.0, 240)
    lls = np.array([ll_powerlaw(x, a, xmin, xmax) for a in grid])
    i = int(np.argmax(lls))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, 60)
    llf = np.array([ll_powerlaw(x, a, xmin, xmax) for a in fine])
    j = int(np.argmax(llf))
    return float(fine[j]), float(llf[j])


def ll_exponential(x, lam, xmin, xmax):
    """Discrete (geometric-like) exponential on [xmin, xmax]."""
    k = np.arange(xmin, xmax + 1, dtype=float)
    Z = np.sum(np.exp(-lam * k))
    if Z <= 0 or not np.isfinite(Z):
        return -np.inf
    return float(np.sum(-lam * x - np.log(Z)))


def fit_exponential(x, xmin, xmax):
    grid = np.logspace(-4, 0.3, 240)
    lls = np.array([ll_exponential(x, l, xmin, xmax) for l in grid])
    i = int(np.argmax(lls))
    return float(grid[i]), float(lls[i])


def ks_two_sample(a, b):
    """Two-sample KS statistic, plain numpy."""
    allv = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), allv, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), allv, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def ccdf_at(x, grid):
    xs = np.sort(x)
    return 1.0 - np.searchsorted(xs, grid, side="right") / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basepath", default=r"W:\data\PPP\PPP20\day4")
    ap.add_argument("--session", default="PPP20_day4")
    ap.add_argument("--xmax-q", type=float, default=0.999)
    ap.add_argument("--ablate-ripples", action="store_true",
                    help="excise ripple windows (+/- 50 ms) from NREM before detection; "
                         "tests whether the second mixture component IS the ripples")
    args = ap.parse_args()
    from neuro_py.io import loading
    rng = np.random.default_rng(SEED)

    st, cm = loading.load_spikes(args.basepath, brainRegion="CA1")
    sd = loading.load_SleepState_states(args.basepath)
    rip = loading.load_ripples_events(basepath=args.basepath)
    rip_start = np.sort(np.asarray(rip.start, float))
    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    units = [np.sort(np.asarray(s, float))
             for s in np.asarray(st.data, dtype=object)[keep]]
    allspk = np.sort(np.concatenate(units))
    mean_iei = (allspk[-1] - allspk[0]) / len(allspk)
    iv = build_state_intervals(sd, GUARD_S)

    t0 = time.time()
    nrem_iv = iv["NREM"]
    if args.ablate_ripples:
        rip_iv = np.c_[np.asarray(rip.start, float) - 0.05,
                       np.asarray(rip.stop, float) + 0.05]
        nrem_iv = subtract_intervals(nrem_iv, rip_iv)
        print(f"ripples ablated: NREM {float(np.sum(iv['NREM'][:,1]-iv['NREM'][:,0]))/60:.1f} "
              f"-> {float(np.sum(nrem_iv[:,1]-nrem_iv[:,0]))/60:.1f} min")
    S, D, T = detect(allspk, nrem_iv, mean_iei, PCT)
    rr = ripple_rate_at(T, rip_start)
    print(f"=== {args.session} NREM === {len(S)} avalanches, bin {mean_iei*1000:.2f} ms "
          f"[{time.time()-t0:.0f}s]")

    med = np.median(rr)
    lowS, highS = S[rr <= med], S[rr > med]
    print(f"split at median ripple rate {med:.3f} Hz: "
          f"low n={len(lowS)}, high n={len(highS)}")

    # common xmin: KS-chosen once on the pooled distribution;
    # xmax = 99.9th pct of pooled to keep the discrete normalisation tractable
    import powerlaw
    sub = rng.choice(S[S > 0], min(100_000, int((S > 0).sum())), replace=False)
    with np.errstate(all="ignore"):
        fp = powerlaw.Fit(sub, discrete=True, verbose=False)
    xmin = int(fp.xmin)
    xmax = int(np.quantile(S, args.xmax_q)) if args.xmax_q < 1 else int(S.max())
    pool = S[(S >= xmin) & (S <= xmax)].astype(float)
    lo = lowS[(lowS >= xmin) & (lowS <= xmax)].astype(float)
    hi = highS[(highS >= xmin) & (highS <= xmax)].astype(float)
    w_lo = len(lo) / (len(lo) + len(hi))
    print(f"common xmin={xmin} (pooled KS), xmax={xmax} | "
          f"in-range: pooled {len(pool)}, low {len(lo)}, high {len(hi)}, w_low={w_lo:.3f}")

    rows = []

    # ---- T1: shape difference, fixed-n subsampled KS, raw and mean-normalised ----
    for tag, a, b in (("raw", lo, hi),
                      ("mean_normalised", lo / lo.mean(), hi / hi.mean())):
        Ds = [ks_two_sample(rng.choice(a, N_SUB, replace=False),
                            rng.choice(b, N_SUB, replace=False))
              for _ in range(N_BOOT)]
        # null: same-component split of pooled
        Dn = []
        p2 = np.concatenate([a, b]) if tag == "raw" else np.concatenate([a, b])
        for _ in range(N_BOOT):
            pick = rng.choice(p2, 2 * N_SUB, replace=False)
            Dn.append(ks_two_sample(pick[:N_SUB], pick[N_SUB:]))
        Ds, Dn = np.array(Ds), np.array(Dn)
        excess = float(np.mean(Ds > np.quantile(Dn, 0.99)))
        rows.append(dict(test="T1_" + tag, D_med=float(np.median(Ds)),
                         D_null99=float(np.quantile(Dn, 0.99)),
                         frac_above_null99=excess))
        print(f"T1 [{tag:>16s}]  KS D = {np.median(Ds):.4f} "
              f"(null 99th pct {np.quantile(Dn,0.99):.4f})  "
              f"exceeds null in {100*excess:.0f}% of resamples")

    # ---- T2: family, per component, common range ----
    for name, x in (("low", lo), ("high", hi)):
        a_pl, ll_pl = fit_powerlaw(x, xmin, xmax)
        l_ex, ll_ex = fit_exponential(x, xmin, xmax)
        # normalised log-likelihood ratio per observation (Vuong-style, sign only)
        R = (ll_pl - ll_ex) / len(x)
        rows.append(dict(test=f"T2_{name}", alpha=a_pl, lam=l_ex,
                         ll_pl=ll_pl, ll_ex=ll_ex, R_per_obs=float(R)))
        print(f"T2 [{name:>4s}]  alpha={a_pl:.3f}  lambda={l_ex:.4f}  "
              f"(ll_PL - ll_EXP)/n = {R:+.4f}  "
              f"{'power law wins' if R > 0 else 'EXPONENTIAL wins'}")

    # ---- T3: single power law vs 2-component mixture on pooled, AIC ----
    a_pool, ll_pool = fit_powerlaw(pool, xmin, xmax)          # 1 param
    a_lo, _ = fit_powerlaw(lo, xmin, xmax)
    a_hi, _ = fit_powerlaw(hi, xmin, xmax)
    l_hi, _ = fit_exponential(hi, xmin, xmax)

    def mix_ll(pool, comp_hi):
        """Pooled LL under w*PL(a_lo) + (1-w)*comp_hi, only w free on a grid."""
        k = np.arange(xmin, xmax + 1, dtype=float)
        p_lo = k ** (-a_lo) / np.sum(k ** (-a_lo))
        if comp_hi == "PL":
            p_hi = k ** (-a_hi) / np.sum(k ** (-a_hi))
        else:
            p_hi = np.exp(-l_hi * k) / np.sum(np.exp(-l_hi * k))
        idx = (pool - xmin).astype(int)
        best = (-np.inf, np.nan)
        for w in np.linspace(0.05, 0.95, 91):
            ll = float(np.sum(np.log(w * p_lo[idx] + (1 - w) * p_hi[idx])))
            if ll > best[0]:
                best = (ll, w)
        return best

    for comp, npar in (("PL", 3), ("EXP", 3)):
        ll_mix, w = mix_ll(pool, comp)
        d_aic = 2 * (npar - 1) - 2 * (ll_mix - ll_pool)
        rows.append(dict(test=f"T3_mix_{comp}", ll_single=ll_pool, ll_mix=ll_mix,
                         w_low=w, delta_AIC=float(d_aic)))
        print(f"T3 [PL+{comp:>3s}] ll_single={ll_pool:.0f}  ll_mix={ll_mix:.0f}  "
              f"w_low={w:.2f}  deltaAIC(mix-single)={d_aic:+.0f}  "
              f"{'MIXTURE preferred' if d_aic < 0 else 'single preferred'}")

    # ---- T4: reconstruction curves for the figure ----
    grid = np.unique(np.round(np.logspace(np.log10(xmin), np.log10(xmax), 60)))
    k = np.arange(xmin, xmax + 1, dtype=float)
    p_lo = k ** (-a_lo) / np.sum(k ** (-a_lo))
    p_hi = np.exp(-l_hi * k) / np.sum(np.exp(-l_hi * k))
    ll_mix, w = mix_ll(pool, "EXP")
    mix_pmf = w * p_lo + (1 - w) * p_hi
    mix_ccdf = 1 - np.cumsum(mix_pmf)
    single_pmf = k ** (-a_pool) / np.sum(k ** (-a_pool))
    single_ccdf = 1 - np.cumsum(single_pmf)
    curves = pd.DataFrame(dict(
        size=grid,
        ccdf_pooled=ccdf_at(pool, grid),
        ccdf_low=ccdf_at(lo, grid),
        ccdf_high=ccdf_at(hi, grid),
        ccdf_mixture_model=np.interp(grid, k, mix_ccdf),
        ccdf_single_pl=np.interp(grid, k, single_ccdf),
    ))

    df = pd.DataFrame(rows)
    assert_local_path(LOG); os.makedirs(LOG, exist_ok=True)
    df.to_csv(assert_local_path(os.path.join(
        LOG, f"mixture_proof_{args.session}.csv")), index=False)
    curves.to_csv(assert_local_path(os.path.join(
        LOG, f"mixture_proof_curves_{args.session}.csv")), index=False)
    print(f"\nwrote -> {LOG}\\mixture_proof_{args.session}.csv (+ _curves)")


if __name__ == "__main__":
    main()
