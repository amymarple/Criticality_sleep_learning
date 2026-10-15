"""
ed4d_replay_prob_lmm.py -- Extended Data Fig. 4d's statistic: the submitted Fig. 2d
linear mixed model, run on non-overlapping windows.

ED 4d (replay probability by within-session d2 sextile, full-window and ripple-excised
d2 (2024 Euclidean, AR order 20, 40 ms)) is tested per session with Spearman and a
signed-rank test across the decoded sessions. The submitted Fig. 2d instead used a
window-level LMM on overlapping sliding windows; the same LMM, conditioned on session,
is also valid on non-overlapping windows, so this script refits it there.

Design -- the submitted recipe unmodified (Previous_Rv1
Fig2_D_E_replayprobality_SWR_replaynon_replay.ipynb cell 6): per session, replay
probability z-scored within session (population SD) and d2 cut into sextiles within
session (pd.qcut on d2, duplicates dropped); one row per window; replay_prob_z ~
sextile_centered with a random intercept per session, and with a random intercept +
slope (the model the submitted result comes from); ML fits (lbfgs). Windows: ED 4d's
own, the non-overlapping decoded NREM tiles of excised_br_entropy_windows.csv with >= 1
SWR; the submitted sliding-window rows are refitted alongside.

Reads only local CSVs; writes only to D:.
    -> data/revision/log/ed4d_replay_prob_lmm.csv   (data, arm, model, beta, se, p, n_windows, n_sessions)
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd                                                       # noqa: E402
import statsmodels.formula.api as smf                                     # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                           # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
WINDOWS = os.path.join(LOGDIR, "excised_br_entropy_windows.csv")
SUBMITTED = os.path.join(os.path.dirname(REPO), "Criticality_sleep_learning", "Previous_Rv1", "csv",
                         "Fig2_D_E_replayprobality_SWR_replaynon_replay", "replay_prob_lmm_rows.csv")
OUT = os.path.join(LOGDIR, "ed4d_replay_prob_lmm.csv")
MODELS = (("random intercept", "1"), ("random intercept + slope", "1 + sextile_centered"))
ARMS = (("full window", "d2_published"), ("ripple-excised", "d2_excised"))


def fit(L, data, arm):
    L = L.copy()
    L["sextile_centered"] = L["d2_sextile"] - L["d2_sextile"].mean()
    rows = []
    for name, re_ in MODELS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("replay_prob_z ~ sextile_centered", data=L, groups=L["file_id"],
                            re_formula=re_).fit(method="lbfgs", reml=False)
        rows.append(dict(data=data, arm=arm, model=name, beta=m.params["sextile_centered"],
                         se=m.bse["sextile_centered"], p=m.pvalues["sextile_centered"],
                         n_windows=len(L), n_sessions=L.file_id.nunique()))
        print(f"  {data:28s} {arm:15s} {name:25s} beta = {rows[-1]['beta']:+.4f} +- {rows[-1]['se']:.4f}  "
              f"p = {rows[-1]['p']:.2g}  ({len(L)} windows, {L.file_id.nunique()} sessions)")
    return rows


def nonoverlap_rows(W, col):
    out = []
    for s, g in W.dropna(subset=[col]).groupby("session"):
        z = (g.replay_prob - g.replay_prob.mean()) / g.replay_prob.std(ddof=0)
        sx = pd.qcut(g[col].astype(float), q=6, labels=False, duplicates="drop") + 1
        out.append(pd.DataFrame({"file_id": s, "d2_sextile": sx.astype(int), "replay_prob_z": z}))
    return pd.concat(out, ignore_index=True)


def main():
    rows = fit(pd.read_csv(SUBMITTED), "submitted (sliding windows)", "full window")
    W = pd.read_csv(WINDOWS).dropna(subset=["replay_prob"])
    W = W[W.ripple_rate * 30 >= 1]
    for arm, col in ARMS:
        rows += fit(nonoverlap_rows(W, col), "non-overlapping (ED 4d)", arm)
    pd.DataFrame(rows).to_csv(assert_local_path(OUT), index=False)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
