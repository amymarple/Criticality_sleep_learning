"""fig2f_swr_property_ladder.py -- Figure 2f: d2's replay coefficient with SWR rate,
amplitude and duration added to the model.

The Fig 2f ladder (fig2f_seat_ladders.py) controls for population rate and SWR rate; this
script adds ripple amplitude, a per-window replay correlate, and duration, the other SWR
property used elsewhere (r35_swr_property_control.py matches sessions on log duration,
amplitude and cell count).

Same windows and model as fig2f_seat_ladders.py: Fig 1's non-overlapping 30 s NREM tiles,
decoded time, 9 sessions (metric_replay_ladder_windows.csv); per-window ripple_amp /
ripple_dur are the means over the window's NREM-clipped ripples of the ripple file's
amplitude / duration (d2_composition_glm.add_swr). Binomial GLM of replay_count / n_swr,
predictors z-scored within session, session as the unit, two-sided signed-rank. Ladder
(cumulative): d2 | + SWR rate | + amplitude | + duration. The published figure is the
ladder alone; JOINT=1 adds a second panel (every coefficient of the final joint model).
CELLS=1 / MUA=1 add further rungs; those tagged runs write suffixed CSVs.

    -> data/revision/log/fig2f_swr_property_ladder.csv          (_cells / _mua when tagged)
    -> reports/revision_figs/fig2f_ladder_d2_swrprops.png
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                       # noqa: E402
from r35_binomial_replay import fit_binomial, pooled_fit, zs         # noqa: E402
from r35_glm_ablation import LOGDIR                                   # noqa: E402
from fig2f_seat_ladders import box_panel, WIN_CSV, FIGDIR, D2        # noqa: E402

OUT_CSV = os.path.join(LOGDIR, "fig2f_swr_property_ladder" + ("_cells" if os.environ.get("CELLS") else "") + ("_mua" if os.environ.get("MUA") else "") + ".csv")
LABEL = {D2: "d2", "rate": "SWR rate", "ripple_amp": "SWR amplitude", "ripple_dur": "SWR duration",
         "n_cells": "cells / SWR", "MUA": "MUA"}
# Candidate SWR-property covariates: duration, amplitude, participating-cell count.
# n_cells = mean number of units firing in the window's SWRs (build_events).
RUNGS = [("d2", [D2]), ("+ SWR rate", [D2, "rate"]), ("+ amplitude", [D2, "rate", "ripple_amp"]),
         ("+ duration", [D2, "rate", "ripple_amp", "ripple_dur"])]
# Main model: d2 + rate + amplitude + duration. Cells / SWR is not a covariate (a mediator
# of the replay call); CELLS=1 adds it anyway as a sensitivity rung. MUA=1 adds MUA.
if os.environ.get("CELLS"):
    RUNGS.append(("+ cells / SWR", RUNGS[-1][1] + ["n_cells"]))
if os.environ.get("MUA"):
    RUNGS.append(("+ MUA", RUNGS[-1][1] + ["MUA"]))
LADDER_DISPLAY = {"d2": "alone", "+ SWR rate": "+ rate", "+ amplitude": "+ rate, amplitude",
                  "+ duration": "+ rate, amplitude, duration", "+ cells / SWR": "+ ... cells / SWR",
                  "+ MUA": "+ ... MUA"}
BOX = ["#c03d3e", "#d4595a", "#e57373", "#ee8f8f", "#f3a6a6", "#f8c0c0"]
JOINT = {D2: "#d62728", "rate": "#7f7f7f", "ripple_amp": "#939393", "ripple_dur": "#a9a9a9", "n_cells": "#c4c4c4", "MUA": "#dfdfdf"}


def fit_all(W, cols, seat_first):
    """Per-session b1 of cols[0] (seat_first=True) -- or of every column, via rotation."""
    z = [f"{c}_z" for c in cols]
    b = np.array([fit_binomial(g, z)["b1"] for _, g in W.groupby("session")])
    return b, pooled_fit(W, z)


def main():
    W = pd.read_csv(WIN_CSV)
    for c in LABEL:
        W[f"{c}_z"] = W.groupby("session")[c].transform(zs)
    rows, ladder = [], []
    for lab, cols in RUNGS:
        b, pf = fit_all(W, cols, True)
        ladder.append((lab, b))
        for s, v in zip(sorted(W.session.unique()), b):
            rows.append(dict(part="ladder", model=lab, variable=D2, session=s, b1=v))
        rows.append(dict(part="ladder", model=lab, variable=D2, session="__pooled__", b1=pf["b1"], p=pf["p"]))
    full = RUNGS[-1][1]
    joint = []
    for v in full:                        # rotate v to the front: b1 is reported for cols[0]
        b, pf = fit_all(W, [v] + [c for c in full if c != v], True)
        joint.append((v, b))
        for s, val in zip(sorted(W.session.unique()), b):
            rows.append(dict(part="joint", model=RUNGS[-1][0], variable=v, session=s, b1=val))
        rows.append(dict(part="joint", model=RUNGS[-1][0], variable=v, session="__pooled__", b1=pf["b1"], p=pf["p"]))
    pd.DataFrame(rows).to_csv(assert_local_path(OUT_CSV), index=False)

    print(f"{W.session.nunique()} sessions, {len(W)} windows; d2 = {'ripple-excised' if D2 == 'd2_excised' else 'full-window'} "
          f"2024 Euclidean (p = 20)\nladder, b1 of d2 per SD (sessions positive, signed-rank p; pooled p):")
    for (lab, b), (_, cols) in zip(ladder, RUNGS):
        pf = pooled_fit(W, [f"{c}_z" for c in cols])
        print(f"  {lab:14s} {np.median(b):+.3f} ({int((b > 0).sum())}/{len(b)}, p={stats.wilcoxon(b).pvalue:.3f})  pooled p={pf['p']:.2g}")
    print("joint model, every coefficient:")
    for v, b in joint:
        print(f"  {LABEL[v]:14s} {np.median(b):+.3f} ({int((b > 0).sum())}/{len(b)}, p={stats.wilcoxon(b).pvalue:.3f})")
    alone = {v: fit_all(W, [v], True)[0] for v in full if v != D2}
    print("alone:", "  ".join(f"{LABEL[v]} {np.median(b):+.3f} ({int((b > 0).sum())}/{len(b)}, p={stats.wilcoxon(b).pvalue:.3f})"
                             for v, b in alone.items()))

    # The published figure is the ladder alone; JOINT=1 redraws the two-panel version
    # (b = every coefficient of the joint model)
    two = bool(os.environ.get("JOINT"))
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(8.4, 3.4) if two else (4.6, 3.4), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.1, 1]) if two else None)
    axes = axes[0]
    # Display labels: "+ SWR rate" etc. name what was added to the model, but every box
    # still plots d2's own coefficient. The CSV keeps the RUNGS names, which other scripts read.
    box_panel(axes[0], [b for _, b in ladder], [LADDER_DISPLAY.get(lab, lab) for lab, _ in ladder], BOX,
              ylabel="d2 coefficient, log-odds per SD\n(ripple-excised d2; binomial, per session)",
              title=("a  " if two else "") + "d2 with SWR properties in the model")
    if two:
        box_panel(axes[1], [b for _, b in joint], [LABEL[v] for v, _ in joint], [JOINT[v] for v, _ in joint],
                  ylabel="log-odds of replay per SD\n(joint model, per session)",
                  title="b  replay ~ " + " + ".join(LABEL[v].replace("SWR ", "").replace("amplitude", "amp.").replace("duration", "dur.").replace("cells / SWR", "cells") for v in full))
    fig.suptitle(f"{W.session.nunique()} sessions, decoded NREM tiles", fontsize=8.5, y=1.0)
    fig.tight_layout()
    tagged = os.environ.get("CELLS") or os.environ.get("MUA") or two
    out = os.path.join(FIGDIR, "backup" if tagged else "", "fig2f_ladder_d2_swrprops" + ("_cells" if os.environ.get("CELLS") else "") + ("_mua" if os.environ.get("MUA") else "") + ("_joint" if two else "") + ".png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(assert_local_path(out), dpi=200, bbox_inches="tight"); fig.savefig(out[:-4] + ".svg", bbox_inches="tight")
    print("  ->", out)


if __name__ == "__main__":
    main()
