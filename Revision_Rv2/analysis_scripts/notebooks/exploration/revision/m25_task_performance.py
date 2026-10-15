"""m25_task_performance.py -- candidate sessions for a behavioural quantification of the figure-8
alternation task that underlies the main results (Figs. 1-3): task schematics and session-level
performance.

For every session of Figs 1-3 -- Fig 1 (run_fig1_metrics.FIG1_SESSIONS), Fig 2 replay set
(r35_swr_property_control.SESSION_TEMPLATE), Fig 3 barrage cohort
(barrage_order_kernel_sweep.BARRAGE_SESSIONS) -- reads the epoch table and
<basename>.animal.behavior.mat and reports, per task block, the number of trials and the fraction
correct. Performance = mean of behavior.trialcorrect over the trials inside the block, the
definition used by the lab's maze scripts (R_Project/PPP/codebook/trialsPeaceMaze.m, which writes
taskPerformance.fig: "% correct trials") and by the pre-sleep analysis notebook
(`Plot all figure_supplementary_final.ipynb`). Whether the lab's taskPerformance figures already
exist on the share is listed too.

    -> data/revision/log/m25_task_performance.csv   (one row per session x task block)
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path, session_id          # noqa: E402
from r35_swr_property_control import SESSION_TEMPLATE                               # noqa: E402
from barrage_order_kernel_sweep import BARRAGE_SESSIONS                             # noqa: E402

LOG = os.path.join(os.path.abspath(os.path.join(_HERE, "..", "..", "..")), "data", "revision", "log")
OUT = os.path.join(LOG, "m25_task_performance.csv")


def sessions():
    S = {}
    for bp in FIG1_SESSIONS:
        S.setdefault("_".join(session_id(bp)), [bp, set()])[1].add("Fig1")
    for s in SESSION_TEMPLATE:
        a, d = s.split("_")
        S.setdefault(s, [rf"W:\data\PPP\{a}\{d}", set()])[1].add("Fig2-replay")
    for s, bp in BARRAGE_SESSIONS.items():
        S.setdefault(s, [bp, set()])[1].add("Fig3-barrage")
    return S


def one(sess, bp, figs):
    from neuro_py.io import loading
    base = dict(session=sess, animal=sess.split("_")[0], figures="+".join(sorted(figs)), basepath=bp)
    try:
        ep = loading.load_epoch(bp).sort_values("startTime").reset_index(drop=True)
    except Exception as e:                                                       # noqa: BLE001
        return [dict(base, note=f"no epoch table ({type(e).__name__})")]
    env = ep.get("environment", pd.Series([""] * len(ep))).astype(str)
    is_sleep = env.str.lower().str.contains("sleep") | ep["name"].astype(str).str.lower().str.contains("sleep")
    seq = " | ".join(f"{n}[{e}]" for n, e in zip(ep["name"], env))
    tasks = ep[~is_sleep]
    base.update(epochs=seq, n_task_blocks=len(tasks), two_mazes=env[~is_sleep].nunique() > 1,
                lab_fig=len(glob.glob(os.path.join(bp, "*taskPerformance*.png")) + glob.glob(os.path.join(bp, "*taskPerformance*.fig"))) > 0)
    f = glob.glob(os.path.join(bp, "*.animal.behavior.mat"))
    if not f:
        return [dict(base, note="no animal.behavior.mat")]
    try:
        beh = sio.loadmat(f[0], simplify_cells=True)["behavior"]
    except NotImplementedError:
        import mat73
        beh = mat73.loadmat(f[0])["behavior"]
    if "trialcorrect" not in beh or "trials" not in beh:
        return [dict(base, note="behavior.mat without trials / trialcorrect")]
    tr = np.asarray(beh["trials"], float).reshape(-1, 2)
    tc = np.asarray(beh["trialcorrect"], float).ravel()
    rows = []
    for k, (_, r) in enumerate(tasks.iterrows(), start=1):
        m = (tr[:, 0] >= r.startTime) & (tr[:, 0] < r.stopTime) & np.isfinite(tc)
        rows.append(dict(base, block=k, block_name=r["name"], block_env=str(r.get("environment", "")),
                         block_min=(r.stopTime - r.startTime) / 60, n_trials=int(m.sum()),
                         frac_correct=float(tc[m].mean()) if m.any() else np.nan))
    return rows or [dict(base, note="no task block")]


def main():
    rows = []
    for sess, (bp, figs) in sessions().items():
        rows += one(sess, bp, figs)
        print(f"  {sess}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(assert_local_path(OUT), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
    cols = ["session", "figures", "n_task_blocks", "two_mazes", "block", "block_name", "block_min", "n_trials",
            "frac_correct", "lab_fig", "note"]
    print(D[[c for c in cols if c in D]].round(3).to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
