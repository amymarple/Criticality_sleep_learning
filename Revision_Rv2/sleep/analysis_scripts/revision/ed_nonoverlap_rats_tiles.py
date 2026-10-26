"""
ed_nonoverlap_rats_tiles.py -- the three rat Extended Data input tables that carry no
window time (ED6c, ED6f, ED6g of FigS5_S6_rats), rebuilt from their window-level sources
on disjoint windows, schema unchanged, so that ed_nonoverlap.py can hand them to the
unchanged figure notebook in place of the shipped CSVs.

ed_nonoverlap.py thins every shipped table that has a window-time column (the 30 s / 2 s
grid overlaps ~93%). These three are computed from all windows: ED6c
(rat_replay_nonreplay_tiles.csv) is already a session x SWR-rate-tile aggregate, and
ED6f / ED6g (rat_d2_by_barrage_rate.csv, rat_isi_entropy_by_barrage_rate.csv) carry a
per-file z and a per-file rate tile computed on all windows. They are rebuilt from the
windows by the producers' own functions (taken directly, via ast, from the figure
notebooks), run once on all windows (must equal the shipped CSV, gated before writing)
and once after the disjoint rule is applied to the windows, so the producers' own
per-file z, tiling and aggregation are then recomputed on what is left.

Disjoint rule: within one session, all windows of the session's d2 .mat (every state;
the source tables hold NREM only, so the all-state grid comes from the .mat whose d2
matches the table's) in time order; a window is kept if its centre is >= L after the
last kept one, L = the .mat's window_size (30 s, or 60 s for the sessions that use it).
Thinning happens before any per-file z, tiling or aggregation.

Gate: each table is written only if, on all windows, the rebuild equals the shipped CSV
(same rows, values within 1e-9; also checked byte for byte).

    -> data/revision/log/ed_nonoverlap_tables/rat_replay_nonreplay_tiles.csv         (ED6c)
    -> data/revision/log/ed_nonoverlap_tables/rat_d2_by_barrage_rate.csv             (ED6f)
    -> data/revision/log/ed_nonoverlap_tables/rat_isi_entropy_by_barrage_rate.csv    (ED6g)
    stdout: per-session rows all -> disjoint (and, for reference, if only the table's own NREM rows were thinned)

    ENV=C:/Users/Cornell/.conda/envs/criticality/python.exe
    $ENV ed_nonoverlap_rats_tiles.py            (reads V:, S:, Z: -- read-only; ED_TMP=<dir> sets the temp folder)
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import pickle
import sys
import tempfile
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nelpy as nel  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from neuro_py.io import loading  # noqa: E402
from neuro_py.io.loading import _add_manual_events, load_mat  # noqa: E402
from scipy.io import loadmat  # noqa: E402

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fig1de_nonoverlap import select_disjoint  # noqa: E402
from run_fig1_metrics import assert_local_path  # noqa: E402

REPO = Path(_HERE).parents[0]
RV1 = Path(r"D:\Documents\GitHub\Criticality_sleep_learning\Previous_Rv1\sleep")
FIG_NB = RV1 / "figure_scripts" / "FigS5_S6_rats.ipynb"
REPLAY_NB = RV1 / "analysis_scripts" / "session_tables" / "MUA_sleepstage_rat_replay.ipynb"
SHIPPED = RV1 / "csv" / "FigS5_S6_rats"
OUTDIR = REPO / "data" / "revision" / "log" / "ed_nonoverlap_tables"
ISI_DIR = REPO / "data" / "rats" / "ripple_barr" / "isi"
WIN_DIR = REPO / "data" / "rats" / "replay_nonreplay"
SIG_RNR = REPO / "data" / "replay" / "replay_nonreplay"     # the producer's SIG_DIR
SIG_TOP = REPO / "data" / "replay"                          # also holds hp15 day31/35, hp18 day52
TMP_ROOT = os.environ.get("ED_TMP") or tempfile.gettempdir()
TOL = 1e-9
ED6C, ED6F, ED6G = "rat_replay_nonreplay_tiles.csv", "rat_d2_by_barrage_rate.csv", "rat_isi_entropy_by_barrage_rate.csv"

M80, M60, M40 = "_CA1_D2_time_resolved_error_bar_80ms.mat", "_CA1_D2_time_resolved_error_bar_60ms.mat", \
    "_CA1_D2_time_resolved_error_bar.mat"
# ED6c sessions: d2 .mat file, replay-interval folder, and the SWR inputs of the tiles. `added` / `flagged` =
# indices into ripples.added / ripples.flagged that are included (None = all, as neuro_py's loader reads the file).
ED6C_SESSIONS = {
    "hp15_day31_20250130": dict(mat=M80, sig=SIG_TOP),
    "hp15_day35_20250205": dict(mat=M80, sig=SIG_TOP),
    "hp17_day23_20250507": dict(mat=M80),
    "hp17_day40_20250526": dict(mat=M80),
    "hp18_day13_20250417": dict(mat=M60, added=[]),                      # none of the 13 manual SWRs
    "hp18_day17_20250421": dict(mat=M60, added=[0]),                     # 1 of 26 manual SWRs
    "hp18_day36_20250510": dict(mat=M60, states=".SleepState.states_old.mat"),
    "hp18_day48_20250522": dict(mat=M40),
    "hp18_day49_20250523": dict(mat=M60, added=list(range(2, 47)),       # manual SWRs 2-46;
                                flagged=[2] + list(range(5, 18))),       # flags 2, 5-17
    "hp18_day52_20250526": dict(mat=M80, sig=SIG_TOP, ripples=".oldripples.events.mat", added=[], flagged=[]),
    "rm10_day10_20250814": dict(mat=M40),
    "rm10_day11_20250815": dict(mat=M40),
    "rm10_day12_20250817": dict(mat=M40),
    "rm10_day5_20250809": dict(mat=M40, added=[5, 9, 15]),              # 3 of 16 manual SWRs
    "rm10_day6_20250810": dict(mat=M40),
}
# ED6f / g sessions (file_id = pickle stem): d2 .mat on Z: whose d2 equals per_center's at every window
ED6FG_MAT = {k: ("_CA1_D2_time_resolved_error_bar_60.mat" if k in ("AB4_day08", "AB4_day09") else M40)
             for k in ["AB4_day07", "AB4_day08", "AB4_day09", "AB4_day11", "AYA10_day12", "AYA10_day25", "AYA10_day31",
                       "AYA10_day34", "AYA6_day17", "AYA6_day19", "AYA6_day20", "AYA7_day25"]}


# ---------------------------------------------------------------- producer code, taken directly from the notebooks
def nb_defs(nb_path, cells, names=None):
    """Source of the function definitions in `cells` (in order, later ones override as in the notebook)."""
    nb = json.load(open(nb_path, encoding="utf-8"))
    body = []
    for c in cells:
        tree = ast.parse("".join(nb["cells"][c]["source"]))
        body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)]
    if names is not None:
        assert {n.name for n in body} == set(names), f"missing in {nb_path.name}: {set(names) - {n.name for n in body}}"
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


NS_FIG = {"os": os, "pickle": pickle, "np": np, "pd": pd, "Path": Path}
exec(nb_defs(FIG_NB, [11], ["_force_sextiles", "load_pool_rate_binned"]), NS_FIG)
exec(nb_defs(FIG_NB, [13], ["load_pool_rate_binned_isi"]), NS_FIG)
exec(nb_defs(FIG_NB, [28], ["load_all_tile_pkls"]), NS_FIG)
NS_REP = {"os": os, "np": np, "pd": pd, "plt": plt}
exec(nb_defs(REPLAY_NB, [7, 8]), NS_REP)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        out = fn(*a, **k)
    plt.close("all")
    return out


# ---------------------------------------------------------------- windows and the disjoint rule
def mat_grid(path):
    """Every window of a MATLAB d2 (2024 Euclidean) file, all states -- the producers' load_d2_timeline_complete
    (per state: centre / d2 / d2_err truncated to their common length)."""
    try:
        m = loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    except NotImplementedError:
        import mat73
        m = mat73.loadmat(path)
    res, par = m["results"], m.get("params", {})
    items = list(res.values()) if isinstance(res, dict) else list(np.atleast_1d(res))
    parts = []
    for r in items:
        c = np.atleast_1d(np.asarray(r.get("center", []), float)).ravel()
        d = np.atleast_1d(np.asarray(r.get("d2", []), float)).ravel()
        e = np.atleast_1d(np.asarray(r.get("d2_err", []), float)).ravel()
        sizes = [c.size, d.size] + ([e.size] if e.size and np.any(np.isfinite(e)) else [])
        n = min(sizes)
        if n:
            parts.append(pd.DataFrame({"center_s": c[:n], "d2": d[:n], "state": str(r.get("state"))}))
    g = pd.concat(parts, ignore_index=True).sort_values("center_s").reset_index(drop=True)
    assert g.center_s.is_unique, path
    return g, float(par["window_size"]), float(par["step_size"])


def disjoint_centres(t, L):
    """Greedy in time: keep a centre if it is >= L after the last kept one (fig1de_nonoverlap.select_disjoint)."""
    t = np.sort(np.asarray(t, float))
    keep, last = np.zeros(t.size, bool), -np.inf
    for j, x in enumerate(t):
        if x >= last + L - 1e-6:
            keep[j], last = True, x
    if L == 30.0:
        ref = select_disjoint(pd.DataFrame({"k": 0, "t": t}), key="k", time_col="t")["t"].to_numpy(float)
        assert np.array_equal(ref, t[keep]), "disagrees with fig1de_nonoverlap.select_disjoint"
    return t[keep]


def grid_for(table_centres, table_d2, mat_path, expect_L=None):
    """The session's all-state grid, after checking that every table window is a grid window with the same d2."""
    g, L, step = mat_grid(mat_path)
    m = pd.DataFrame({"center_s": np.asarray(table_centres, float), "d2": np.asarray(table_d2, float)}) \
        .merge(g, on="center_s", how="left", suffixes=("", "_mat"), validate="one_to_one")
    assert m.state.notna().all(), f"{mat_path}: table window not on the grid"
    assert m.state.eq("NREM").all(), f"{mat_path}: table window not NREM in the .mat"
    same = (m.d2 == m.d2_mat) | (m.d2.isna() & m.d2_mat.isna())
    assert same.all(), f"{mat_path}: d2 differs at {int((~same).sum())} windows"
    if expect_L is not None:
        assert L == expect_L, (mat_path, L)
    return g, L, step


# ---------------------------------------------------------------- comparison helpers
def fl64(s):
    """float64 array of a column (nullable Int64 NA -> NaN)."""
    return pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()


def same_frame(a, b, keys, vals):
    """Row count, keys identical (after sort), values within TOL (NaN where NaN)."""
    if len(a) != len(b):
        return False, f"rows {len(a)} vs {len(b)}"
    a = a.sort_values(keys, kind="mergesort").reset_index(drop=True)
    b = b.sort_values(keys, kind="mergesort").reset_index(drop=True)
    for k in keys:
        if not (a[k].astype(str).values == b[k].astype(str).values).all():
            return False, f"key {k} differs"
    worst = 0.0
    for v in vals:
        x, y = fl64(a[v]), fl64(b[v])
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            return False, f"{v}: NaN pattern differs"
        ok = ~np.isnan(x)
        worst = max(worst, float(np.max(np.abs(x[ok] - y[ok]), initial=0.0)))
    return worst <= TOL, f"max |diff| {worst:.3g}"


def csv_text(df):
    return df.to_csv(index=False, lineterminator="\n")


def shipped_text(name):
    return (SHIPPED / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def write_table(df, name):
    p = assert_local_path(str(OUTDIR / name))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    back = pd.read_csv(p, float_precision="round_trip")         # the notebook's read_csv
    assert list(back.columns) == list(df.columns) and len(back) == len(df), name
    for c in df.columns:
        if df[c].dtype == object:
            assert (back[c].astype(str).values == df[c].astype(str).values).all(), (name, c)
        else:
            assert np.array_equal(fl64(back[c]), fl64(df[c]), equal_nan=True), (name, c)
    print(f"[write] {p} ({len(df)} rows; reads back exactly)", flush=True)


# ---------------------------------------------------------------- ED6f / ED6g
def run_barrage_loaders(pkl_dir):
    a = quiet(NS_FIG["load_pool_rate_binned"], str(pkl_dir), rate_col="barrage_rate", state="NREM", n_bins=6)
    b = quiet(NS_FIG["load_pool_rate_binned_isi"], str(pkl_dir), rate_col="barrage_rate", state="NREM", n_bins=6)
    return a.reset_index(drop=True), b.reset_index(drop=True)


def ed6fg():
    print("\n=== ED6f / ED6g (barrage-rate tiles) ===", flush=True)
    a_all, b_all = run_barrage_loaders(ISI_DIR)
    gate = {}
    for name, df, v in [(ED6F, a_all, "d2_norm"), (ED6G, b_all, "isi_entropy_norm")]:
        ship = pd.read_csv(SHIPPED / name, float_precision="round_trip")
        ok, msg = same_frame(df.assign(_i=np.arange(len(df))), ship.assign(_i=np.arange(len(ship))),
                             ["file_id", "_i"], [v, "barrage_rate", "tile"])
        byte = csv_text(df) == shipped_text(name)
        print(f"[gate] {name}: all-window rebuild {len(df)} rows vs shipped {len(ship)}; {msg}; "
              f"byte-identical CSV: {byte}", flush=True)
        gate[name] = ok and list(df.columns) == list(ship.columns)
    # which branch of _force_sextiles each file took on all windows, and the sibling table's plain-qcut formula
    for fid, g in a_all.groupby("file_id", sort=False):
        q = pd.qcut(g.barrage_rate, 6, labels=False, duplicates="drop")
        print(f"   {fid}: plain qcut gives {q.nunique()} bins -> {'qcut' if q.nunique() == 6 else 'rank fallback'}; "
              f"tile == qcut+1 for {np.mean(q.values + 1 == g.tile.values) * 100:.1f} % of rows", flush=True)

    counts = []
    with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
        for fid, sfx in ED6FG_MAT.items():
            with open(ISI_DIR / f"{fid}.pkl", "rb") as f:
                d = pickle.load(f)
            pc = d["per_center"]
            an, dy = fid.split("_")
            g, L, step = grid_for(pc.center_s, pc.d2, Path(r"Z:\Data\AYAold") / an / dy / "complex_system" / f"{dy}{sfx}",
                                  expect_L=60.0 if fid in ("AB4_day08", "AB4_day09") else 30.0)
            kept = disjoint_centres(g.center_s, L)
            pct = pc[pc.center_s.isin(kept)]
            # reference only: rows kept if (a) only the table's own NREM rows were thinned, (b) windows with NaN d2
            # were removed from the grid before thinning (counted on the rows ED6f keeps: finite d2 and rate)
            alt_nrem = len(disjoint_centres(pc.center_s, L))
            fin = pc[np.isfinite(pc.d2) & np.isfinite(pc.barrage_rate)]
            alt_nan = int(fin.center_s.isin(disjoint_centres(g.center_s[g.d2.notna()], L)).sum()) \
                if g.d2.isna().any() else None
            counts.append(dict(file_id=fid, L=L, step=step, grid_all=len(g), grid_disjoint=len(kept),
                               per_center_all=len(pc), per_center_disjoint=len(pct),
                               per_center_disjoint_if_nrem_rows_only=alt_nrem,
                               ed6f_rows_disjoint_if_nan_d2_windows_dropped=alt_nan))
            with open(assert_local_path(os.path.join(tmp, f"{fid}.pkl")), "wb") as f:
                pickle.dump({**d, "per_center": pct}, f)
        a_dis, b_dis = run_barrage_loaders(tmp)
    C = pd.DataFrame(counts)
    for name, df, src in [(ED6F, a_dis, a_all), (ED6G, b_dis, b_all)]:
        n0, n1 = src.groupby("file_id").size(), df.groupby("file_id").size()
        C[name + ":rows_all"] = C.file_id.map(n0).fillna(0).astype(int)
        C[name + ":rows_disjoint"] = C.file_id.map(n1).fillna(0).astype(int)
        if gate[name]:
            assert list(df.columns) == list(src.columns)
            write_table(df, name)
        else:
            print(f"[STOP] {name}: all-window rebuild does not equal the shipped table -- not written", flush=True)
    print(C.to_string(index=False), flush=True)
    return C


# ---------------------------------------------------------------- ED6c
def ripple_intervals(bp, fname=".ripples.events.mat", added=None, flagged=None):
    """neuro_py.load_ripples_events (start / stop) with the curation lists restricted to the given indices."""
    r = load_mat(os.path.join(bp, os.path.basename(bp) + fname))["ripples"]
    ts = np.asarray(r.get("timestamps", r.get("times")), float)
    ts = ts.reshape(1, -1) if ts.ndim == 1 else ts
    df = pd.DataFrame(ts, columns=["start", "stop"])
    df["peaks"] = np.asarray(r.get("peaks", np.nan), float).reshape(-1)
    fl = np.asarray(r.get("flagged", []), dtype=int).reshape(-1)
    fl = fl if flagged is None else fl[np.asarray(flagged, int)]
    if fl.size:
        df = df.drop(index=fl - 1, errors="ignore").reset_index(drop=True)
    if "added" in r:
        ad = np.asarray(r["added"], dtype=float).reshape(-1)
        ad = ad if added is None else ad[np.asarray(added, int)]
        df = _add_manual_events(df, ad.tolist())
    return df[["start", "stop"]].to_numpy(float)


def session_inputs(sess, cfg):
    an = sess.split("_")[0]
    bp = os.path.join(r"S:\data\latentseq\RM10", sess) if an == "rm10" else \
        os.path.join(r"V:\data\hpc_ctx_project", an.upper(), sess)
    if "states" in cfg:
        nrem = load_mat(os.path.join(bp, os.path.basename(bp) + cfg["states"]))["SleepState"]["ints"]["NREMstate"]
    else:
        nrem = loading.load_SleepState_states(bp)["NREMstate"]
    nrem_epochs = nel.EpochArray(nrem)
    epoch_df = loading.load_epoch(bp)
    rip = ripple_intervals(bp, cfg.get("ripples", ".ripples.events.mat"), cfg.get("added"), cfg.get("flagged"))
    if not any(k in cfg for k in ("ripples", "added", "flagged")):
        ref = loading.load_ripples_events(basepath=bp)[["start", "stop"]].to_numpy(float)
        assert np.array_equal(ref, rip), f"{sess}: ripple_intervals != neuro_py loader"
    ripple_epochs = nel.EpochArray(rip)
    ripple_epochs = nrem_epochs & ripple_epochs
    mask = (epoch_df["environment"] == "sleep") & (epoch_df["manipulation"].isna())
    sleep_no_manip_epochs = nel.EpochArray(np.array(epoch_df.loc[mask, ["startTime", "stopTime"]], dtype=float))
    ripple_epochs = sleep_no_manip_epochs & ripple_epochs
    return bp, ripple_epochs


def replay_tiles(bp, ripple_epochs, w, sig_dir):
    NS_REP["SIG_DIR"] = str(sig_dir)
    return quiet(NS_REP["run_replay_d2_for_session"], basepath=bp, ripple_epochs=ripple_epochs,
                 centers=w.center.values, d2_values=w.d2.values, d2_errors=w.d2_err.values, d2_window_size=30,
                 n_bins=6, n_boot=500, tile_mode="quantile", replace=True, seed=0)


def long_table(tiles):
    """load_all_tile_pkls (FigS5_S6_rats), run on a folder of the given tile frames."""
    with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
        for fname, t in tiles.items():
            t.to_pickle(assert_local_path(os.path.join(tmp, fname)))
        return NS_FIG["load_all_tile_pkls"](tmp, pattern="*_tile.pkl")


def ed6c():
    print("\n=== ED6c (replay / non-replay SWR-rate tiles) ===", flush=True)
    tiles_all, tiles_dis, counts, fail = {}, {}, [], []
    for sess, cfg in ED6C_SESSIONS.items():
        an = sess.split("_")[0].upper()
        stem = f"{an}_{sess}_CA1"
        w = pd.read_pickle(WIN_DIR / f"{stem}.pkl")
        t_stored = pd.read_pickle(WIN_DIR / f"{stem}_tile.pkl")
        bp, rep = session_inputs(sess, cfg)
        sig = cfg.get("sig", SIG_RNR)
        # 1) rebuild on all windows
        dfw, dft = replay_tiles(bp, rep, w, sig)
        if len(dfw) == len(w) and len(dft) == len(t_stored):
            chk = {c: float(np.max(np.abs(fl64(dfw[c]) - fl64(w[c]))))
                   for c in ("center", "d2", "rate_in", "rate_out_mean", "rate_out_sd")}
            for c in ("in_d2_mean", "in_d2_sem", "out_d2_mean", "out_d2_lo", "out_d2_hi"):
                x, y = fl64(dft[c]), fl64(t_stored[c])
                chk[c] = float(np.nanmax(np.abs(x - y))) if np.array_equal(np.isnan(x), np.isnan(y)) else np.inf
            worst = max(chk.values())
        else:
            worst = np.inf
        if not worst <= TOL:
            fail.append(sess)
        tiles_all[stem + "_tile.pkl"] = dft
        # 2) disjoint windows (grid = every window of the session's d2 .mat, all states)
        g, L, step = grid_for(w.center, w.d2, os.path.join(bp, "complex_system", sess + cfg["mat"]), expect_L=30.0)
        kept = disjoint_centres(g.center_s, L)
        wk = w[w.center.isin(kept)].reset_index(drop=True)
        _, dft_k = replay_tiles(bp, rep, wk, sig)
        tiles_dis[stem + "_tile.pkl"] = dft_k
        counts.append(dict(session=sess, d2_mat=sess + cfg["mat"], grid_all=len(g), grid_disjoint=len(kept),
                           windows_all=len(w), windows_disjoint=len(wk),
                           windows_disjoint_if_nrem_rows_only=len(disjoint_centres(w.center, L)),
                           rebuild_max_abs_diff=worst))
        print(f"   {sess}: rebuild vs stored windows / tiles max|diff| {worst:.2g}; windows {len(w)} -> {len(wk)}",
              flush=True)
    C = pd.DataFrame(counts)
    L_all = long_table(tiles_all)
    ship = pd.read_csv(SHIPPED / ED6C, float_precision="round_trip")
    ok, msg = same_frame(L_all, ship, ["file_id", "group", "tile_num"], ["d2_mean", "d2_se"])
    byte = csv_text(L_all) == shipped_text(ED6C)
    print(f"[gate] {ED6C}: all-window rebuild {len(L_all)} rows vs shipped {len(ship)}; {msg}; "
          f"byte-identical CSV: {byte}; sessions not reproduced: {fail or 'none'}", flush=True)
    if ok and not fail and list(L_all.columns) == list(ship.columns):
        L_dis = long_table(tiles_dis)
        assert list(L_dis.columns) == list(ship.columns) and len(L_dis) == len(ship)
        write_table(L_dis, ED6C)
    else:
        print(f"[STOP] {ED6C}: all-window rebuild does not equal the shipped table -- not written", flush=True)
    print(C.to_string(index=False), flush=True)
    return C


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    ed6fg()
    ed6c()


if __name__ == "__main__":
    main()
