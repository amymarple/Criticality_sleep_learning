"""
fig1_nonoverlap_contrasts.py -- the published Figure 1 state contrasts recomputed on
non-overlapping (independent) windows.

Every window-level producer in this folder uses non-overlapping windows, but the published
Figure 1 numbers -- and the paper's own test (Fig1_D_histogram_S1_E.ipynb
cell 4: per-session median d2 per Wake/Sleep group, ranked within session, nested LMM
d2_rank ~ group with animal as the grouping factor) -- still run on all 189 526 sliding
windows (30 s stepped 2 s, ~93 % overlap). This script reruns the same test, with the
same code path as d2_reproduce.py, on every 15th window of each contiguous run of the
published grid (window contents disjoint), and reports session-level paired contrasts
with effect sizes for both d2 definitions:

  d2 (2024 Euclidean, p=20)  from d2_reproduce.csv (the paper's own .mat values)
  d2 (2025 KL, p=20)         from fig1_state_space_kl_join.csv (already non-overlapping
                             by construction; its own tiling, not the MATLAB grid)

Contrasts: NREM - REM, NREM - nonTHETA, NREM - THETA, Sleep - Wake, on per-session
medians; signed-rank p, sessions with the published sign, paired Cohen's d
(mean difference / SD of differences), and the window counts per state.

    -> data/revision/log/fig1_nonoverlap_contrasts.csv
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path      # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
STEP_S = 2.0
STRIDE = 15
STATES = ("THETA", "nonTHETA", "REM", "NREM")
CONTRASTS = [("NREM", "REM"), ("NREM", "nonTHETA"), ("NREM", "THETA"), ("Sleep", "Wake")]


def select_nonoverlap(W):
    W = W.sort_values(["animal_day", "group", "center_s"]).reset_index(drop=True)
    keep = np.zeros(len(W), bool)
    for _, g in W.groupby(["animal_day", "group"], sort=False):
        c = g["center_s"].to_numpy()
        run = np.r_[0, np.cumsum(~np.isclose(np.diff(c), STEP_S, atol=0.05))]
        pos = g.index.to_numpy()
        for r in np.unique(run):
            keep[pos[run == r][::STRIDE]] = True
    return W[keep]


def paper_test(df, label, aggfunc="median"):
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    s = (df.groupby(["animal", "animal_day", "simplified_group"], observed=True)["d2"]
           .agg(aggfunc).reset_index())
    s["d2_rank"] = s.groupby("animal_day")["d2"].rank(method="average")
    res = None
    for method in ("lbfgs", "powell", "nm", "bfgs"):
        try:
            res = smf.mixedlm("d2_rank ~ simplified_group", data=s, groups=s["animal"]).fit(method=method, disp=False)
            break
        except Exception:                                    # noqa: BLE001
            res = None
    if res is None:
        print(f"  {label:<44s} LMM failed under every optimiser")
        return dict(label=label, beta=np.nan, se=np.nan, p=np.nan, n_sessions=s.animal_day.nunique())
    term = "simplified_group[T.Wake]"
    beta, se = res.params[term], res.bse[term]
    p = float(np.exp(np.log(2.0) + norm.logsf(abs(beta / se))))
    print(f"  {label:<44s} n={s.animal_day.nunique():2d} sessions | beta(Wake)={beta:+.4f} SE={se:.4f} p={p:.3g}")
    return dict(label=label, beta=beta, se=se, p=p, n_sessions=s.animal_day.nunique())


def contrasts(W, label, col="d2"):
    per = W.groupby(["animal_day", "group"])[col].median().unstack("group")
    per["Sleep"] = W[W.group.isin(["NREM", "REM"])].groupby("animal_day")[col].median()
    per["Wake"] = W[W.group.isin(["THETA", "nonTHETA"])].groupby("animal_day")[col].median()
    rows = []
    for a, b in CONTRASTS:
        if a not in per or b not in per:
            continue
        d = (per[a] - per[b]).dropna()
        if len(d) < 3:
            continue
        p = stats.wilcoxon(d).pvalue
        rows.append(dict(arm=label, contrast=f"{a} - {b}", n=len(d), median=float(d.median()),
                         pos=int((d > 0).sum()), p=p, cohen_d=float(d.mean() / d.std(ddof=1))))
        print(f"    {a:>5s} - {b:8s} n={len(d):2d}  median {d.median():+.4f}  {int((d > 0).sum())}/{len(d)} pos  "
              f"p={p:.4f}  paired d={d.mean() / d.std(ddof=1):+.2f}")
    return rows


def main():
    W = pd.read_csv(os.path.join(LOGDIR, "d2_reproduce.csv"))
    W = W[W.group.isin(STATES)].copy()
    N = select_nonoverlap(W)
    print(f"Euclidean: {len(W)} sliding windows -> {len(N)} non-overlapping ({len(N) / len(W):.1%})")
    print("  per state: " + "  ".join(f"{s}={int((N.group == s).sum())}" for s in STATES))
    out, cont = [], []
    print("\n=== the paper's LMM test (median per session x group, ranked, mixedlm groups=animal) ===")
    out.append(paper_test(W, "Euclidean, all sliding windows"))
    out.append(paper_test(N, "Euclidean, non-overlapping (every 15th)"))
    print("\n=== paired session-level contrasts, d2 (2024 Euclidean, p=20) ===")
    print("  sliding:")
    cont += contrasts(W, "euclid_sliding")
    print("  non-overlapping:")
    cont += contrasts(N, "euclid_nonoverlap")
    # KL arm
    K = pd.read_csv(os.path.join(LOGDIR, "fig1_state_space_kl_join.csv"))
    K = K[K.nonoverlap & K.d2_kl.notna() & K.state.isin(STATES)].copy()
    K["animal_day"] = K["session_key"]
    K["group"] = K["state"]
    K["simplified_group"] = K.group.map({"THETA": "Wake", "nonTHETA": "Wake", "NREM": "Sleep", "REM": "Sleep"})
    K["d2"] = K["d2_kl"]
    print(f"\nKL: {len(K)} non-overlapping windows, {K.animal_day.nunique()} sessions; per state: "
          + "  ".join(f"{s}={int((K.group == s).sum())}" for s in STATES))
    out.append(paper_test(K, "KL (2025), non-overlapping"))
    print("  non-overlapping, d2 (2025 KL, p=20):")
    cont += contrasts(K, "kl_nonoverlap")
    R = pd.DataFrame(cont)
    T = pd.DataFrame(out)
    assert_local_path(LOGDIR)
    R.to_csv(assert_local_path(os.path.join(LOGDIR, "fig1_nonoverlap_contrasts.csv")), index=False)
    T.to_csv(assert_local_path(os.path.join(LOGDIR, "fig1_nonoverlap_lmm.csv")), index=False)
    print(f"\nwrote {LOGDIR}\\fig1_nonoverlap_contrasts.csv and fig1_nonoverlap_lmm.csv")


if __name__ == "__main__":
    main()
