"""ks_xmin_convention.py -- the low/high goodness-of-fit comparison under both xmin conventions.

The low/high-ripple result "high ripple rate degrades scale-freeness" is measured by the KS
distance D rather than by the exponent. freexmin_sweep.csv measures it under free (Clauset/auto)
xmin; this script measures it under both xmin conventions, in a way that isolates the convention:

    same session, same bin, same threshold, same detected avalanches --
    the only thing that differs between the two arms is how xmin was chosen.

  common xmin : one xmin from the pooled NREM distribution, applied to LOW and HIGH alike
                (what ripple_robustness.py does -- maximises comparability of exponents)
  free xmin   : Clauset KS search run independently per group
                (what the group CSVs do -- lets each state pick its own scaling onset)

D is computed with the identical call in both arms --
`powerlaw.Fit(..., discrete=True).power_law.KS()` -- so the two numbers are like for like.
freexmin_sweep.py uses that same call, so the free arm here reproduces it.

Reuses the repo's own detection and ripple-rate code (`avalanche_ripple.detect`,
`ripple_rate_at`) so the avalanches are the same avalanches used throughout this analysis.
Bins are per-session multiples of the mean IEI; 1x is the primary bin, 2x the marginal case.

    -> data/revision/log/ks_xmin_convention.csv
"""
from __future__ import annotations

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fig1_metrics import GUARD_S, MIN_FIRING_RATE, build_state_intervals  # noqa: E402
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path, session_id  # noqa: E402
from avalanche_ripple import detect, ripple_rate_at  # noqa: E402

LOG = os.path.join(r"D:\Documents\GitHub\Sleep_criticaltiy", "data", "revision", "log")
BIN_MULT = (1, 2)
PCTS = (20, 30, 40)
N_XMIN_SUB = 30_000
MIN_TAIL = 200


def ks_at(data, xmin, xmax):
    """KS distance of a discrete power law on [xmin, xmax].

    Identical call to freexmin_sweep.py's, so the common- and free-xmin arms are
    measured the same way and only the xmin differs."""
    import powerlaw
    with np.errstate(all="ignore"):
        f = powerlaw.Fit(data, xmin=xmin, xmax=xmax, discrete=True, verbose=False)
        return float(f.power_law.KS()), float(f.power_law.alpha)


def main():
    from neuro_py.io import loading
    import powerlaw
    rng = np.random.default_rng(0)
    rows = []
    for bp in FIG1_SESSIONS:
        a, d = session_id(bp)
        sess = f"{a}_{d}"
        try:
            st, cm = loading.load_spikes(bp, brainRegion="CA1")
            sd = loading.load_SleepState_states(bp)
            rip = loading.load_ripples_events(basepath=bp)
            rip_start = np.sort(np.asarray(rip["start"], float))
        except Exception as e:                                    # noqa: BLE001
            print(f"[FAIL] {sess}: {type(e).__name__}: {e}", flush=True)
            continue
        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        units = [np.sort(np.asarray(u, float))
                 for u in np.asarray(st.data, dtype=object)[keep]]
        allspk = np.sort(np.concatenate(units))
        iei = (allspk[-1] - allspk[0]) / len(allspk)
        nrem = build_state_intervals(sd, GUARD_S)["NREM"]
        if not len(nrem):
            continue

        for mult in BIN_MULT:
            bin_s = iei * mult
            for pct in PCTS:
                t0 = time.time()
                S, D, T = detect(allspk, nrem, bin_s, pct)
                if len(S) < 2000:
                    continue
                rr = ripple_rate_at(T, rip_start)
                med = np.median(rr)

                # --- COMMON xmin: one value from the pooled distribution -------
                sub = S[S > 0]
                if len(sub) > N_XMIN_SUB:
                    sub = rng.choice(sub, N_XMIN_SUB, replace=False)
                with np.errstate(all="ignore"):
                    xmin_common = int(powerlaw.Fit(sub, discrete=True,
                                                   verbose=False).xmin)
                xmax = int(S.max())

                for g, m in (("LOW", rr <= med), ("HIGH", rr > med)):
                    s_ = S[m].astype(float)
                    rec = dict(session=sess, bin_mult=mult, bin_ms=bin_s * 1000,
                               pct=pct, group=g, n=int(len(s_)))

                    # arm 1 -- common xmin
                    sf = s_[(s_ >= xmin_common) & (s_ <= xmax)]
                    if len(sf) >= MIN_TAIL:
                        ks_c, al_c = ks_at(sf, xmin_common, xmax)
                        rec.update(xmin_common=xmin_common, ks_common=ks_c,
                                   alpha_common=al_c, n_common=int(len(sf)))

                    # arm 2 -- free xmin, chosen independently for THIS group
                    try:
                        with np.errstate(all="ignore"):
                            ff = powerlaw.Fit(s_[s_ > 0], discrete=True, verbose=False)
                            xf = int(ff.xmin)
                        nf = int((s_ >= xf).sum())
                        if nf >= MIN_TAIL:
                            rec.update(xmin_free=xf, ks_free=float(ff.power_law.KS()),
                                       alpha_free=float(ff.power_law.alpha), n_free=nf)
                    except Exception:                             # noqa: BLE001
                        pass
                    rows.append(rec)
                print(f"  {sess} x{mult} pct{pct}  [{time.time()-t0:.0f}s]", flush=True)

    out = pd.DataFrame(rows)
    assert_local_path(LOG)
    out.to_csv(assert_local_path(os.path.join(LOG, "ks_xmin_convention.csv")),
               index=False)
    print(f"\nwrote {len(out)} rows\n")
    report(out)


def report(F):
    from scipy.stats import wilcoxon
    print("=" * 96)
    print("KS DISTANCE D, LOW vs HIGH -- SAME avalanches, only the xmin RULE differs")
    print("=" * 96)
    for arm, kcol, xcol in (("COMMON xmin (pooled, shared)", "ks_common", "xmin_common"),
                            ("FREE xmin (Clauset, per group)", "ks_free", "xmin_free")):
        if kcol not in F.columns:
            print(f"\n--- {arm}: no rows ---")
            continue
        print(f"\n--- {arm} ---")
        print(f"{'bin':>5}{'pct':>5}{'n':>4}{'D LOW':>9}{'D HIGH':>9}{'ratio':>7}"
              f"{'p':>10}{'HIGH worse':>12}{'xmin LOW':>10}{'xmin HIGH':>11}")
        for (b, pc), g in F.groupby(["bin_mult", "pct"]):
            w = g.pivot_table(index="session", columns="group",
                              values=[kcol, xcol]).dropna()
            if len(w) < 5:
                continue
            lo, hi = w[(kcol, "LOW")], w[(kcol, "HIGH")]
            p = wilcoxon(lo, hi).pvalue
            star = ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns")
            print(f"{b:>4}x{pc:>5}{len(w):>4}{lo.median():>9.4f}{hi.median():>9.4f}"
                  f"{hi.median()/lo.median():>7.2f}{p:>10.5f}"
                  f"{int((hi>lo).sum()):>7}/{len(w):<4}"
                  f"{w[(xcol,'LOW')].median():>10.0f}{w[(xcol,'HIGH')].median():>11.0f}"
                  f"  {star}")

    print("\n" + "=" * 96)
    print("THE POINT: if the gap holds in BOTH arms, it is not an xmin artefact.")
    print("Contrast with the EXPONENT, which is ~93% determined by xmin (REVISION_STATUS 14.1):")
    print("=" * 96)
    for arm, acol in (("common", "alpha_common"), ("free", "alpha_free")):
        if acol not in F.columns:
            continue
        g = F[F.bin_mult == 1]
        w = g.pivot_table(index=["session", "pct"], columns="group", values=acol).dropna()
        if len(w) < 5:
            continue
        p = wilcoxon(w.LOW, w.HIGH).pvalue
        print(f"  alpha, {arm:6s} xmin, 1x IEI: LOW {w.LOW.median():.3f} "
              f"HIGH {w.HIGH.median():.3f}  p={p:.5f}  "
              f"HIGH<LOW in {int((w.HIGH<w.LOW).sum())}/{len(w)}")


if __name__ == "__main__":
    main()
