"""
ed_nonoverlap.py -- the submitted Extended Data panels (and the main panels drawn by the same
notebooks) redrawn on disjoint windows, by running the Previous_Rv1 figure notebooks themselves.

Every submitted panel was drawn on the 30 s / 2 s sliding grid (~93% overlap); this script
reruns the same figure notebooks with every window-level table thinned to disjoint windows
before anything is plotted.

Previous_Rv1/figure_scripts/*.ipynb read their data only through `read_csv(name)` (cell 0)
and reproduce the submitted figures exactly from Previous_Rv1/csv. Here each notebook's code
cells are executed in one namespace, with two changes:
  * read_csv thins every window-level table to disjoint windows: per session, all states in
    time order, a window is kept only if its centre is >= 30 s after the last kept one
    (`fig1de_nonoverlap.select_disjoint`; no two kept windows share data). Tables without a
    time column are read unchanged, so only figures whose inputs are all thinned are saved
    (ALLOW).
  * Figure.savefig is redirected to reports/revision_figs/ed_nonoverlap/<notebook>/ (svg + png,
    with a footer naming the window rule); figures not in ALLOW are not written.
Tables without a window time are instead read from OVERRIDE_DIR, rebuilt on disjoint windows
by ed_nonoverlap_d2err.py / ed_nonoverlap_ed2a_ed1f.py / ed_nonoverlap_rats_tiles.py /
ed_nonoverlap_ed1b.py.
Env: NO_THIN=1 baseline; ED_SENS=<name> sensitivity folder; ED5C_MIN_N rat scatter cell
threshold. Nothing is written inside Previous_Rv1.

    -> reports/revision_figs/ed_nonoverlap/<notebook>/*.{svg,png}, <notebook>.log
    -> data/revision/log/ed_nonoverlap_thinning.csv  (rows before / after per table)

    ENV=C:/Users/Cornell/.conda/envs/criticality/python.exe
    $ENV ed_nonoverlap.py [notebook ...]
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import traceback
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fig1de_nonoverlap import NONOVL_NOTE, select_disjoint  # noqa: E402
from run_fig1_metrics import assert_local_path  # noqa: E402

REPO = Path(_HERE).parents[2]
RV1 = Path(r"D:\Documents\GitHub\Criticality_sleep_learning\Previous_Rv1")
OUTROOT = REPO / "reports" / "revision_figs" / "ed_nonoverlap"
LOGCSV = REPO / "data" / "revision" / "log" / "ed_nonoverlap_thinning.csv"

OVERRIDE_DIR = REPO / "data" / "revision" / "log" / "ed_nonoverlap_tables"
# Tables with no window time are rebuilt from their producer pickles on disjoint windows by
# ed_nonoverlap_d2err.py, ed_nonoverlap_ed2a_ed1f.py and ed_nonoverlap_rats_tiles.py; a file of
# the shipped name in OVERRIDE_DIR is read instead of the shipped CSV.

# spacing for window lengths other than 30 s (the disjoint rule uses L = window length)
SPACING = {"branching_ratio_windows.csv": 20.0}

# table -> (time column, session-key columns, rows to drop)
THIN = {
    "branching_ratio_windows.csv": ("center", ["epoch"], None),          # ED10: 20 s windows, one session
    "d2_windows_18sessions.csv": ("center_s", ["animal", "day"], None),
    "d2_windows_pre_post.csv": ("center_s", ["animal_day"], None),
    "d2_ripple_ablated_windows.csv": ("center_s", ["file_id"], None),
    "d2_barrage_ablated_windows.csv": ("center_s", ["animal_name", "day"],
                                       lambda d: (d.animal_name.astype(str) + "_" + d.day.astype(str)) == "PPP7_day7"),
    "isi_entropy_windows_barrage.csv": ("center_s", ["animal_name", "day"], None),
    "rat_CA2_CA3_firing_by_window.csv": ("center_s", ["_file"], None),
    "rat_d2_windows.csv": ("center_s", ["animal", "day"], None),
    "rat_d2_mua_pairs.csv": ("time", ["animal_day"], None),
    "rat_d2_windows_by_state.csv": ("center_s", ["animal_day"], None),
}
ROW_ALIGNED = {"rat_isi_entropy_by_swr_rate.csv": ("rat_d2_windows_by_state.csv", "ripple_rate", "ripple_rate")}


def retile_rat_isi_swr(d):
    """ED6d's precomputed columns, from the producer: isi_entropy_norm = per-file z
    (ddof 0) of the window's ISI entropy, tile = per-file
    qcut(ripple_rate, 6, duplicates='drop') + 1. Recomputed on the thinned rows."""
    d = d.copy()
    g = d.groupby("file_id")
    d["isi_entropy_norm"] = g["_isi"].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    d["tile"] = g["ripple_rate"].transform(lambda s: pd.qcut(s, 6, labels=False, duplicates="drop") + 1)
    return d.drop(columns=["_isi"])


RETILE = {"rat_isi_entropy_by_swr_rate.csv": ("isi_entropy", retile_rat_isi_swr)}

# notebook -> saved figure basenames (prefix match) drawn only from thinned tables
ALLOW = {
    "Fig1_D_histogram_S1_E": ["fig1_D_d2_density_sleep_wake", "figS1_B_wake_sleep"],
    "Fig2_A_FigS2_B_FigS3_A": ["fig2A_S2B_pre_vs_postsleep_perfile_paired", "fig3S_A_d2_pre_post_matched_ripple"],
    "FigS3_B_S3C_S3D_ripple_ablate": ["S3_C_", "S2_B_", "S3_D_"],
    "Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar": ["S3_E_", "fig3_H_"],
    "FigS3_G_F_CA3_CA2_pyr_model_surpport_fig": ["S3_F_", "S3_G_"],
    "FigS5_S6_rats": ["figS5_B_rat_d2_density", "figS5_C_rat_NREM_REM_MUA", "figS5_rat_D_", "fig2_A_pre_vs_postsleep",
                      "figS6_G_Rat_isi_entropy_by_d2_tile_pooled",
                      "figS6_C_", "figS6_F_", "figS6_G_barrage_rate_by_isi"],          # from OVERRIDE_DIR tables
    "FigS1_C_SNR_S2C_d2 uncertainty": ["S2_C_CA1_snr_box_30s_vs_60s", "S2_C_ripple_rate_by_d2_tile_pooled_std"],
    "FigS2_A_novel_familar": ["figS2_A_"],
    "FigS1_F_branchingratio_histogram": ["S1_F_"],
    "FigS10_eplipsy": ["branching_ratio_comparison_epilepsy"],
    "FigS1_B_Explain variance_D_statetransition": ["figS1B_explain_variance"],   # ED1b, from ed_nonoverlap_ed1b.py
}
# a panel in ALLOW whose table is an OVERRIDE is saved only if that override file exists
NEEDS_OVERRIDE = {"figS6_C_": "rat_replay_nonreplay_tiles.csv", "figS6_F_": "rat_d2_by_barrage_rate.csv",
                  "figS6_G_barrage_rate_by_isi": "rat_isi_entropy_by_barrage_rate.csv",
                  "S2_C_CA1_snr_box_30s_vs_60s": "d2_err_30s_60s_18sessions.csv",
                  "S2_C_ripple_rate_by_d2_tile_pooled_std": "d2_err_by_swr_rate_sextile.csv",
                  "figS2_A_": "d2_rel_novel_familiar.csv", "S1_F_": "branching_ratio_by_state.csv",
                  "figS1B_explain_variance": "explained_variance_by_window.csv"}
# panels that are not window statistics (example traces, transition snippets, avalanche fits, event-level
# BR changes) or have no window times anywhere (ED1b): saved without thinning, in <notebook>/_as_submitted/
AS_SUBMITTED = {
    "FigS1_B_Explain variance_D_statetransition": ["figS1B_explain_variance", "S1_D_nrem_d2_transition"],
    "FigS2_D_E_F_low_avalanche_example_avalanche_sizeduradcc_": ["figS2_E_", "S2_E_", "S2_F_"],
    "FigS5_A_Example_plot_rat": ["HP18_"],
    "Fig2_B_Example_replay": ["fig2_B_Example_"],                      # replay examples (Fig 2B)
    "FigS5_S6_rats": ["figS5_B_ripple_br_changes", "figS6E_"],
}

# window-count thresholds written for the 2 s grid, rescaled to the same duration on disjoint windows
# (x 2/30): ED5c keeps (session, state) with >= MIN_N = 150 sliding windows (300 s) -> 10 disjoint ones
ED5C_MIN = int(os.environ.get("ED5C_MIN_N", "10"))          # 10 = round(150 * 2 / 30); ED5C_MIN_N for sensitivity
PATCH = {
    "FigS5_S6_rats": [("keep_keys = counts[counts >= MIN_N].index",
                       f"keep_keys = counts[counts >= {ED5C_MIN}].index  # disjoint: MIN_N = 150 on the 2 s grid "
                       f"(300 s) -> 10 disjoint windows by default")],
}

THIN_LOG: list = []
STATE = {"nb": None, "noted": set()}
BASELINE = os.environ.get("NO_THIN") == "1"     # same run, nothing thinned, nothing saved: the printout to compare against
if BASELINE:
    OUTROOT = OUTROOT / "_baseline_all_windows"
SENS = os.environ.get("ED_SENS")                 # sensitivity runs write to their own folder, never the thinning log
if SENS:
    OUTROOT = OUTROOT / "_sensitivity" / SENS


def disjoint(d, tcol, keys, spacing=30.0):
    keys = keys + [c for c in ("window", "region") if c in d.columns and c not in keys]
    k = d[keys].astype(str).agg("|".join, axis=1)
    d = d.assign(_key=k.values, _t=d[tcol].astype(float).values)
    if spacing == 30.0:
        out = select_disjoint(d, key="_key", time_col="_t")
    else:                                            # same greedy rule, L = this table's window length
        d = d.sort_values(["_key", "_t"]).reset_index(drop=True)
        keep = np.zeros(len(d), bool)
        t = d["_t"].to_numpy(float)
        for idx in d.groupby("_key", sort=False).indices.values():
            last = -np.inf
            for j in idx:
                if t[j] >= last + spacing - 1e-6:
                    keep[j], last = True, t[j]
        out = d[keep]
    return out.drop(columns=["_key", "_t"]).reset_index(drop=True)


def make_reader(orig_read, csv_dir):
    def read_csv(name, **kw):
        d = orig_read(name, **kw)
        if BASELINE:
            return d
        if (OVERRIDE_DIR / name).exists():
            n0 = len(d)
            d = pd.read_csv(OVERRIDE_DIR / name, float_precision="round_trip", **kw)
            THIN_LOG.append(dict(notebook=STATE["nb"], table=name, rows_all=n0, rows_disjoint=len(d)))
            print(f"[override] {name}: {n0} shipped rows -> {len(d)} rows from {OVERRIDE_DIR / name}", flush=True)
            return d
        if name in THIN:
            tcol, keys, drop = THIN[name]
            n0 = len(d)
            if drop is not None:
                d = d[~drop(d)].reset_index(drop=True)
            d = disjoint(d, tcol, keys, SPACING.get(name, 30.0))
            THIN_LOG.append(dict(notebook=STATE["nb"], table=name, rows_all=n0, rows_disjoint=len(d)))
            print(f"[thin] {name}: {n0} -> {len(d)} rows (disjoint)", flush=True)
        elif name in ROW_ALIGNED:
            src, ca, cb = ROW_ALIGNED[name]
            B = pd.read_csv(csv_dir / src, float_precision="round_trip")
            assert len(B) == len(d) and np.allclose(d[ca].to_numpy(float), B[cb].to_numpy(float), equal_nan=True), name
            tcol, keys, _ = THIN[src]
            d = d.assign(_t_src=B[tcol].values, **{f"_k{i}": B[k].values for i, k in enumerate(keys)})
            if name in RETILE:
                d["_isi"] = B[RETILE[name][0]].values
            n0 = len(d)
            d = disjoint(d, "_t_src", [f"_k{i}" for i in range(len(keys))])
            d = d.drop(columns=["_t_src"] + [f"_k{i}" for i in range(len(keys))])
            if name in RETILE:
                d = RETILE[name][1](d)                     # re-z and re-tile on the thinned windows
            THIN_LOG.append(dict(notebook=STATE["nb"], table=name, rows_all=n0, rows_disjoint=len(d)))
            print(f"[thin] {name}: {n0} -> {len(d)} rows (disjoint, times from {src} by row)", flush=True)
        return d
    return read_csv


_orig_savefig = matplotlib.figure.Figure.savefig


def guarded_savefig(self, fname, *args, **kw):
    name = os.path.basename(str(fname))
    stem = os.path.splitext(name)[0]
    if BASELINE:
        return None
    allowed = [a for a in ALLOW.get(STATE["nb"], []) if stem.startswith(a)]
    allowed = [a for a in allowed if a not in NEEDS_OVERRIDE or (OVERRIDE_DIR / NEEDS_OVERRIDE[a]).exists()]
    as_sub = any(stem.startswith(a) for a in AS_SUBMITTED.get(STATE["nb"], []))
    if not allowed and not as_sub:
        print(f"[skip] {name}: inputs not thinned, not saved", flush=True)
        return None
    out = OUTROOT / STATE["nb"] / ("_as_submitted" if not allowed else "")
    out.mkdir(parents=True, exist_ok=True)
    if id(self) not in STATE["noted"]:
        note = (f"Previous_Rv1 {STATE['nb']} rerun on {NONOVL_NOTE}" if allowed else
                f"Previous_Rv1 {STATE['nb']}: as submitted (not a window statistic, or no window times exist)")
        self.text(0.01, -0.03, note, fontsize=6, color="0.35", ha="left", va="top")
        STATE["noted"].add(id(self))
    p = assert_local_path(str(out / (stem + ".svg")))
    kw.pop("format", None)
    import time
    for target, kws in ((p, kw), (assert_local_path(str(out / (stem + ".png"))), dict(dpi=200, bbox_inches="tight"))):
        for i in range(8):                       # retry on a transient OSError
            try:
                _orig_savefig(self, target, *args, **kws)
                break
            except OSError:
                if i == 7:
                    raise
                time.sleep(4)
    print(f"[save] {p} (+ .png)", flush=True)
    return None


def clean(src):
    return "\n".join(("pass  # " + l) if re.match(r"\s*[%!]", l) else l for l in src.splitlines())


def run_notebook(stem):
    STATE["nb"] = stem
    nb = json.load(open(RV1 / "figure_scripts" / f"{stem}.ipynb", encoding="utf-8"))
    run = OUTROOT / "_run"
    run.mkdir(parents=True, exist_ok=True)
    logf = open(assert_local_path(str(OUTROOT / f"{stem}.log")), "w", encoding="utf-8")
    cwd = os.getcwd()
    os.chdir(run)                                   # '../figure_output' resolves inside OUTROOT
    ns = {"__name__": "__main__", "display": print}
    errors = 0
    try:
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        patches = [] if BASELINE else PATCH.get(stem, [])
        hits = {old: 0 for old, _ in patches}
        for i, c in enumerate(code):
            src = "".join(c["source"])
            for old, new in patches:
                if old in src:
                    hits[old] += src.count(old)
                    src = src.replace(old, new)
                    print(f"[patch] cell {i}: {old!r} -> {new!r}", file=logf, flush=True)
            print(f"\n######## code cell {i} ########", file=logf, flush=True)
            try:
                with contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
                    exec(compile(clean(src), f"{stem}:cell{i}", "exec"), ns)
            except Exception:                                                  # noqa: BLE001
                errors += 1
                logf.write(traceback.format_exc())
            if i == 0:                                   # cell 0 defines CSV / read_csv: point them at Previous_Rv1, thin
                ns["CSV"] = RV1 / "csv" / ns["CSV"].name
                sys.path.insert(0, str(RV1 / "analysis_scripts"))
                ns["read_csv"] = make_reader(ns["read_csv"], ns["CSV"])
            plt.close("all")
        assert all(v == 1 for v in hits.values()), f"patch did not apply exactly once: {hits}"
    finally:
        os.chdir(cwd)
        logf.close()
    print(f"{stem}: done, {errors} cell error(s); log {OUTROOT / (stem + '.log')}", flush=True)
    return errors


def main():
    matplotlib.figure.Figure.savefig = guarded_savefig
    stems = sys.argv[1:] or list(dict.fromkeys(list(ALLOW) + list(AS_SUBMITTED)))
    for s in stems:
        run_notebook(s)
    matplotlib.figure.Figure.savefig = _orig_savefig
    if BASELINE or SENS:
        return
    T = pd.DataFrame(THIN_LOG).drop_duplicates()
    if LOGCSV.exists():                                  # keep the rows of notebooks not rerun this time
        old = pd.read_csv(LOGCSV)
        T = pd.concat([old[~old.notebook.isin(stems)], T], ignore_index=True)
    T.to_csv(assert_local_path(str(LOGCSV)), index=False)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
