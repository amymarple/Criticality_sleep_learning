"""fig2f_backtrace.py -- Figure 2f: reconstructing the per-window input table behind the GLM's
stored per-session summary.

generate_GLM.ipynb (the Fig 2F producer) saves only a summary per session: n_samples, fold
AUCs, the ablation table; the per-window input table itself is not saved. This script
reconstructs the window set from the producer's own recipe and checks it against n_samples,
an exact integer per session:

    d2 file    : newest *_CA1_D2_time_resolved_error_bar.mat / *_CA1_D2_time_error_bar.mat
                 in complex_system (the most recently modified candidate), all states
                 loaded, then filtered to state == NREM
    ripples    : load_ripples_events intersected with NREM
    window     : 30 s centred on center_s, ripple in window if start < c+15 and stop > c-15
    rows kept  : ripple_count > 0 and finite MUA / d2 / replay_prob

For every candidate d2 .mat in each session, counts the NREM windows with >= 1 NREM ripple
and compares with n_samples. A match identifies the d2 file and the state/ripple masks the
producer used; a mismatch on every candidate means the recipe above is not what ran. The same
count is reported for the per-center replay-probability table, to show where it diverges.

    -> data/revision/log/fig2f_backtrace_windows.csv
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.io import loadmat

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                  # noqa: E402
from r35_glm_ablation import BASEPATH, PROB_DIR, LOGDIR, WIN_S   # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
EXPLORE = os.path.join(REPO, "data", "GLM_mice", "explore")
OUT_CSV = os.path.join(LOGDIR, "fig2f_backtrace_windows.csv")
PATS = ["*_CA1_D2_time_resolved_error_bar.mat", "*_CA1_D2_time_error_bar.mat"]


def load_all_states(f):
    """center / d2 / state for every state in the file (cell 2 loads all, cell 14 filters)."""
    m = loadmat(f, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    res = m["results"]
    res = list(res.values()) if isinstance(res, dict) else list(np.atleast_1d(res))
    rows = []
    for r in res:
        rows.append(pd.DataFrame({"center_s": np.atleast_1d(r["center"]).astype(float),
                                  "d2": np.atleast_1d(r["d2"]).astype(float),
                                  "state": str(r["state"])}))
    return pd.concat(rows, ignore_index=True)


def ripple_count(riv, centers):
    w0, w1 = centers - WIN_S / 2, centers + WIN_S / 2
    # (start < w1) & (stop > w0), vectorised over windows via sorted bounds
    s, e = np.sort(riv[:, 0]), np.sort(riv[:, 1])
    n_start_before_w1 = np.searchsorted(s, w1, "left")
    n_stop_before_w0 = np.searchsorted(e, w0, "right")
    return n_start_before_w1 - n_stop_before_w0


def main():
    import nelpy as nel
    from neuro_py.io import loading

    stored = {}
    for f in sorted(glob.glob(os.path.join(EXPLORE, "*_ablation_results_GLM.pkl"))):
        o = pd.read_pickle(f)
        stored[f"{o['animal']}_{o['day']}"] = int(o["n_samples"])

    rows = []
    for sess, bp in BASEPATH.items():
        if sess not in stored:
            continue
        n_target = stored[sess]
        rip = loading.load_ripples_events(bp)
        sd = loading.load_SleepState_states(bp)
        nrem = nel.EpochArray(sd["NREMstate"])
        riv = (nrem & nel.EpochArray(np.array(rip[["start", "stop"]]))).data
        cands = []
        for p in PATS:
            cands += glob.glob(os.path.join(bp, "complex_system", p))
        cands = sorted(set(cands), key=os.path.getmtime)
        print(f"\n{sess}: n_samples stored = {n_target}; {len(cands)} candidate d2 files, "
              f"{len(riv)} NREM ripples")
        for f in cands:
            D = load_all_states(f)
            N = D[D["state"].str.upper().str.contains("NREM")]
            rc = ripple_count(riv, N["center_s"].to_numpy(float))
            n_all = int(len(N))
            n_rip = int(((rc > 0) & np.isfinite(N["d2"].to_numpy(float))).sum())
            newest = f == cands[-1]
            hit = "  <== MATCH" if n_rip == n_target else ""
            print(f"  {os.path.basename(f):58s} mtime {pd.Timestamp(os.path.getmtime(f), unit='s'):%Y-%m-%d %H:%M}"
                  f"  NREM {n_all:5d}  ripple>0 {n_rip:5d}{'  (newest)' if newest else ''}{hit}")
            rows.append(dict(session=sess, source=os.path.basename(f), newest=newest,
                             n_nrem=n_all, n_ripple_gt0=n_rip, n_samples=n_target,
                             match=n_rip == n_target))
        # the per_center table (replay-probability pipeline's window source)
        pf = os.path.join(PROB_DIR, f"{sess}.pkl")
        if os.path.exists(pf):
            d = pd.read_pickle(pf)["per_center"]
            n = d[d["state"].astype(str).str.upper().eq("NREM")]
            n_rip = int((n["ripple_count"].astype(float) > 0).sum())
            print(f"  {'per_center table (replay_probality)':58s} {'':18s}"
                  f"  NREM {len(n):5d}  ripple>0 {n_rip:5d}{'  <== MATCH' if n_rip == n_target else ''}")
            rows.append(dict(session=sess, source="per_center", newest=False, n_nrem=len(n),
                             n_ripple_gt0=n_rip, n_samples=n_target, match=n_rip == n_target))
    R = pd.DataFrame(rows)
    assert_local_path(LOGDIR)
    R.to_csv(assert_local_path(OUT_CSV), index=False)
    print(f"\nwrote {OUT_CSV}")
    m = R[R.match]
    print(f"sessions with an exact n_samples match: {m.session.nunique()}/{R.session.nunique()}")
    print(m[["session", "source", "newest"]].to_string(index=False))


if __name__ == "__main__":
    main()
