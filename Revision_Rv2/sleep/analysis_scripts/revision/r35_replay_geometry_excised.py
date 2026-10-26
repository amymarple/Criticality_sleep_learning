"""r35_replay_geometry_excised.py -- the ripple-excised d2 against the geometry of the decoded
replay events, on Fig 1's tiles (the per-window table of Extended Data Fig. 7).

r35_fig2d_replot.py relates the full-window d2 to the replay events' geometry on the 2 s-step
window grid of the replay-probability tables. This script asks the same question for the
ripple-excised d2 on Fig 1's non-overlapping tiles, and whether each relation survives
partialling out the ripple rate or cells per SWR. Geometry is a per-event content measure, so
it cannot be produced by recruitment the way a replay / non-replay call can.

Windows: metric_replay_ladder_windows.csv (decoded-time Fig 1 NREM tiles, 9 sessions;
d2_excised, d2_published, ripple_rate, n_cells). Per window, the means over the significant
replay events of dur_s, traj_span, traj_net, |slope|, rval and gof (load_replay_events;
r35_fig2d_replot.window_means). Per-session Spearman, and Spearman partial given ripple rate
and given cells per SWR; Fisher-z mean, signed-rank test across sessions.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/r35_replay_geometry_excised.csv           (per-window table)
    -> data/revision/log/r35_replay_geometry_excised_summary.csv
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
from run_fig1_metrics import assert_local_path                                   # noqa: E402
from r35_glm_ablation import LOGDIR                                               # noqa: E402
from r35_swr_property_control import load_replay_events                           # noqa: E402
from r35_fig2d_replot import window_means, partial_spearman, fisher, MEASURES     # noqa: E402

WIN_CSV = os.path.join(LOGDIR, "metric_replay_ladder_windows.csv")
OUT = os.path.join(LOGDIR, "r35_replay_geometry_excised.csv")
OUT_SUM = os.path.join(LOGDIR, "r35_replay_geometry_excised_summary.csv")
GEOM = [m for m in MEASURES if m[0] != "replay_prob"]


def main():
    W = pd.read_csv(WIN_CSV)
    parts = []
    for s, g in W.groupby("session"):
        E = load_replay_events(s)
        S = E[E.is_sig].copy()
        S["dur_s"] = S["stop"] - S["start"]
        S["speed"] = S["slope"].abs()
        ev = S[["start", "stop"]].to_numpy(float)
        g = g.copy()
        c = g["center_s"].to_numpy(float)
        for col, _, _ in GEOM:
            g[col] = window_means(ev, S[col].to_numpy(float), c)
        parts.append(g)
        print(f"  {s:12s} {len(g):4d} tiles, {int(g.dur_s.notna().sum()):4d} with >= 1 replay event", flush=True)
    W = pd.concat(parts, ignore_index=True)
    W.to_csv(assert_local_path(OUT), index=False)

    rows = []
    for arm, d2col in (("ripple-excised", "d2_excised"), ("full window", "d2_published")):
        print(f"\n{'=' * 96}\nd2 (2024 Euclidean, p = 20), {arm} -- per-session Spearman with replay geometry, "
              f"{W.session.nunique()} sessions\n{'=' * 96}")
        print(f"{'measure':20}{'r':>8}{'pos':>6}{'p':>8}{'r|rate':>10}{'pos':>6}{'p':>8}{'r|cells':>10}{'pos':>6}{'p':>8}")
        for col, lbl, unit in GEOM:
            rs, pr, pc = [], [], []
            for s, g in W.groupby("session"):
                v = g[[d2col, col, "ripple_rate", "n_cells"]].dropna()
                if len(v) < 10 or v[col].nunique() <= 2:
                    continue
                rs.append(stats.spearmanr(v[d2col], v[col]).statistic)
                pr.append(partial_spearman(v[d2col].to_numpy(), v[col].to_numpy(), v["ripple_rate"].to_numpy()))
                pc.append(partial_spearman(v[d2col].to_numpy(), v[col].to_numpy(), v["n_cells"].to_numpy()))
            cells = []
            rec = dict(arm=arm, measure=col, label=lbl)
            for tag, vals in (("", rs), ("_rate", pr), ("_cells", pc)):
                vv = np.asarray([x for x in vals if np.isfinite(x)])
                r, se, n = fisher(vv)
                p = stats.wilcoxon(vv).pvalue if vv.size >= 3 else np.nan
                cells.append(f"{r:+8.3f}{int((vv > 0).sum()):>3d}/{vv.size:<2d}{p:8.4f}")
                rec.update({f"r{tag}": r, f"se{tag}": se, f"p{tag}": p, f"n_pos{tag}": int((vv > 0).sum()), f"n{tag}": vv.size})
            print(f"{lbl:20}" + "".join(cells))
            rows.append(rec)
    pd.DataFrame(rows).to_csv(assert_local_path(OUT_SUM), index=False)
    print(f"\nwrote {OUT_SUM}")


if __name__ == "__main__":
    main()
