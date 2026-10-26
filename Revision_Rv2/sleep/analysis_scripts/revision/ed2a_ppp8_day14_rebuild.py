"""
ed2a_ppp8_day14_rebuild.py -- ED2a (novel vs familiar, d2 relative to pre-task sleep) with PPP8_day14's
d2 (2024 Euclidean) computed from its own recording.

data/novel_familiar/novel/PPP8_day14.pkl carries the d2 timeline of PPP8_day12 at day 14's epoch times;
this script computes PPP8_day14's d2 from its own recording. PPP8_day14 enters the novel arm's paired test
(PPP7_day14 has no mid rows and does not).

Method: the producer's own code, as reproduced exactly for all 18 stored pickles by
ed_nonoverlap_ed2a_ed1f.py (MUA_sleepstage cell 8 load_d2_timeline_complete -> newest
`*_CA1_D2_time_resolved_error_bar.mat`; cell 15 annotate_epochs, NREM, first 1200 rows per epoch group,
m_pre = mean presleep d2, d2_rel = (d2 - m_pre) / (d2 + m_pre)), run on W:\data\PPP\PPP8\day14 with its
own epochs. Then:
  * all windows: the table with PPP8_day14's rows replaced;
  * disjoint: the ed_nonoverlap override table plus PPP8_day14 thinned the same way.
FigS2_A_novel_familar's own cells are run on both (LMM S2 - S1 paired by file). PPP7_day14 is kept as in
the table (post rows only; it does not enter the paired test).

    -> data/revision/log/ed2a_ppp8_day14_rebuilt.csv (PPP8_day14 rows, both arms)
    -> data/revision/log/ed2a_corrected_all_windows.csv, ed2a_corrected_stats.csv
    -> data/revision/log/ed_nonoverlap_tables/d2_rel_novel_familiar.csv (disjoint, now with PPP8_day14)
    -> reports/revision_figs/ed_nonoverlap/FigS2_A_novel_familar/_all_windows_corrected/*.png/svg
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ed_nonoverlap_ed2a_ed1f import (CAP_N, ED2A_NAME, ED2A_SHIPPED, L_D2, PPP, disjoint_mask,  # noqa: E402
                                     ed2a_producer, quiet, run_cell94, stack)
from run_fig1_metrics import assert_local_path  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
OVR = os.path.join(LOGDIR, "ed_nonoverlap_tables")
RV1 = r"D:\Documents\GitHub\Criticality_sleep_learning\Previous_Rv1\sleep"
NB_S2A = os.path.join(RV1, "figure_scripts", "FigS2_A_novel_familar.ipynb")
FIG1D = os.path.join(RV1, "csv", "Fig1_D_histogram_S1_E", "d2_windows_18sessions.csv")
FIGOUT = os.path.join(REPO, "reports", "revision_figs", "ed_nonoverlap", "FigS2_A_novel_familar", "_all_windows_corrected")
STEM, ANIMAL, DAY = "PPP8_day14", "PPP8", "day14"


def rebuild():
    from neuro_py.io import loading
    ns, block = ed2a_producer()
    ca1, meta = quiet(ns["load_d2_timeline_complete"], os.path.join(PPP, ANIMAL, DAY), "CA1")
    assert os.path.basename(meta["path"]) == "day14_CA1_D2_time_resolved_error_bar.mat", meta["path"]
    epoch_df = loading.load_epoch(os.path.join(PPP, ANIMAL, DAY))
    first_n = lambda d: ns["first_n_per_epoch"](d, n=CAP_N)  # noqa: E731
    d_all, rel_all, m_all = run_cell94(ns, block, ns["annotate_epochs"](ca1, epoch_df), first_n)
    F = pd.read_csv(FIG1D, float_precision="round_trip")
    F = F[(F.animal == ANIMAL) & (F.day.astype(str) == "14")]
    m = ca1.merge(F[["center_s", "d2"]], on="center_s", suffixes=("", "_fig1d"))
    assert len(m) == len(F) == len(ca1) and np.allclose(m.d2, m.d2_fig1d, atol=1e-12), "not Fig 1d's PPP8 day14 d2"
    old = pd.read_pickle(os.path.join(REPO, "data", "novel_familiar", "novel", f"{STEM}.pkl"))
    print(f"PPP8_day14 own d2: {os.path.basename(meta['path'])}, {len(ca1)} windows (= Fig 1d's {len(F)}); "
          f"ED2a rows {len(rel_all)} (shipped {len(old)}), m_pre {m_all:.5f}", flush=True)
    # disjoint mask, matching ed_nonoverlap_ed2a_ed1f.py
    keep = disjoint_mask(ca1["center_s"].to_numpy(float), L_D2)
    span = set(d_all["center_s"].to_numpy(float))
    in_span = lambda d: d[d["center_s"].isin(span)].sort_values("center_s")  # noqa: E731
    _, rel_dis, m_dis = run_cell94(ns, block, ns["annotate_epochs"](ca1[keep], epoch_df), in_span)
    print(f"  disjoint rows {len(rel_dis)}, m_pre {m_dis:.5f}", flush=True)
    return stack(rel_all, STEM, "novel"), stack(rel_dis, STEM, "novel")


def run_s2a(table, save_dir=None):
    """FigS2_A_novel_familar's own cells on `table`; returns the printed LMM lines."""
    cells = [c for c in json.load(open(NB_S2A, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    clean = lambda s: "\n".join(("pass  # " + l) if re.match(r"\s*[%!]", l) else l for l in s.splitlines())  # noqa: E731
    orig = matplotlib.figure.Figure.savefig

    def save(self, fname, *a, **k):
        if save_dir is None:
            return None
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(str(fname)))[0]
        self.text(0.01, -0.03, "ED2a, all windows, PPP8_day14 recomputed from its own d2 (2024 Euclidean)",
                  fontsize=6, color="0.35", ha="left", va="top")
        k.pop("format", None)
        orig(self, assert_local_path(os.path.join(save_dir, stem + ".svg")), *a, **k)
        orig(self, assert_local_path(os.path.join(save_dir, stem + ".png")), dpi=200, bbox_inches="tight")

    matplotlib.figure.Figure.savefig = save
    out = io.StringIO()
    cwd = os.getcwd()
    os.chdir(os.path.join(RV1, "figure_scripts"))
    try:
        ns = {"__name__": "__main__", "display": print}
        for i, c in enumerate(cells):
            with contextlib.redirect_stdout(out):
                exec(compile(clean("".join(c["source"])), f"s2a_c{i}", "exec"), ns)
            if i == 0:
                ns["read_csv"] = lambda name, **kw: table.copy()
            plt.close("all")
    finally:
        os.chdir(cwd)
        matplotlib.figure.Figure.savefig = orig
    return [l for l in out.getvalue().splitlines() if l.startswith("[novel]") or l.startswith("[control]")]


def per_file_direction(table):
    g = table.groupby(["source", "file", "phase"]).d2_rel.median().unstack("phase").dropna()
    return {s: f"{int((x['post'] > x['mid']).sum())}/{len(x)}" for s, x in g.groupby(level="source")}


def main():
    new_all, new_dis = rebuild()
    pd.concat([new_all.assign(arm="all_windows"), new_dis.assign(arm="disjoint")]).to_csv(
        assert_local_path(os.path.join(LOGDIR, "ed2a_ppp8_day14_rebuilt.csv")), index=False)

    shipped = pd.read_csv(ED2A_SHIPPED, float_precision="round_trip")
    corr_all = pd.concat([shipped[shipped.file != STEM], new_all], ignore_index=True)
    corr_all.to_csv(assert_local_path(os.path.join(LOGDIR, "ed2a_corrected_all_windows.csv")), index=False)

    ovr = os.path.join(OVR, ED2A_NAME)
    os.makedirs(os.path.join(OVR, "backup"), exist_ok=True)
    bak = os.path.join(OVR, "backup", "d2_rel_novel_familiar__PPP8_day14_dropped.csv")
    if not os.path.exists(bak):
        shutil.copy2(ovr, assert_local_path(bak))
    prev = pd.read_csv(bak, float_precision="round_trip")
    assert STEM not in set(prev.file), "override already contains PPP8_day14"
    corr_dis = pd.concat([prev, new_dis], ignore_index=True)
    corr_dis.to_csv(assert_local_path(ovr), index=False)

    rows = []
    for label, T, sd in (("shipped (PPP8_day14 = day-12 data)", shipped, None),
                         ("all windows, PPP8_day14 recomputed", corr_all, FIGOUT),
                         ("disjoint, PPP8_day14 dropped (previous)", prev, None),
                         ("disjoint, PPP8_day14 recomputed", corr_dis, None)):
        lines = run_s2a(T, sd)
        dirs = per_file_direction(T)
        print(f"\n{label}: post > mid novel {dirs.get('novel')}, control {dirs.get('control')}")
        for l in lines:
            print("   " + l)
            mm = re.search(r"\[(\w+)\].*S2−S1 = ([-\d.]+).*p=([\d.eE-]+), N_pairs=(\d+)", l)
            if mm:
                rows.append(dict(table=label, arm=mm.group(1), s2_minus_s1=float(mm.group(2)), p=float(mm.group(3)),
                                 n_pairs=int(mm.group(4)), post_gt_mid=dirs.get(mm.group(1))))
    pd.DataFrame(rows).to_csv(assert_local_path(os.path.join(LOGDIR, "ed2a_corrected_stats.csv")), index=False)


if __name__ == "__main__":
    main()
