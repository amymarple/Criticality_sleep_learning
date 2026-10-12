"""
make_figure_csvs.py -- how the CSVs in ../csv/ were made, and how ../figure_scripts/ were derived.

The figure notebooks of the submitted manuscript read pickles, MATLAB .mat files and raw recordings on
the lab servers (W:, V:, X:, Z:). For this repository each notebook is copied with ONE kind of change:
the block that loads data from those sources is replaced by reading a CSV, and the original loading code
is kept below it as comments. Every other line (processing, statistics, plotting) is the author's code,
unchanged. Save paths are pointed at ../figure_output/.

For each notebook (see SPECS below) this script
  1. runs the ORIGINAL notebook from the lab data, cell by cell, and at the end of each loading block
     ("cut") writes the variables the rest of the notebook uses to ../csv/<notebook>/<name>.csv;
  2. writes the copied notebook to ../figure_scripts/ with each cut replaced by the CSV read;
  3. runs the copy and compares everything it prints after the first cut with the original run.
While the original runs, every file write outside this repository (savefig, to_csv, to_pickle, open(...,
'w'), makedirs) is diverted to a scratch folder, so the source data and the lab's figure folders are
never modified.

Needs the lab servers mounted and the original repository (Sleep_criticaltiy) at ORIG_REPO. A reader of
the paper does not need this script: the CSVs are already here.

    python make_figure_csvs.py <notebook stem> [<notebook stem> ...]     (or: all)
"""
from __future__ import annotations

import builtins
import contextlib
import difflib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.figure import Figure                # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

ORIG_REPO = Path(r"D:\Documents\GitHub\Sleep_criticaltiy")
ORIG_FIG = ORIG_REPO / "notebooks" / "figures"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent                                   # Previous_Rv1
FIG_DIR = REPO / "figure_scripts"
CSV_ROOT = REPO / "csv"              # what the figure notebooks read (slim)
CSV_FULL = REPO / "csv_full"         # everything each loading block returned (not in git)
SCRATCH = Path(tempfile.gettempdir()) / "previous_rv1_build"

# path prefixes of the lab's figure folders -> ../figure_output
SAVE_PREFIX = re.compile(r"D:\\Documents\\GitHub\\Sleep_criticaltiy\\reports"
                         r"(?:\\(?:new_planc|supplementary|individual_fig_main|individual_fig_supp|nature|"
                         r"figure_plan|new_plan|figures|rats)(?=[\\\"']))*", re.I)


# ----------------------------------------------------------------------------- export / import helpers
def export_var(name, value, path: Path):
    """Write one variable to CSV. DataFrames as they are; arrays as columns; (N, 2) interval arrays as
    start / stop; dicts of arrays as long tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, pd.DataFrame):
        df = value
    elif isinstance(value, pd.Series):
        df = value.to_frame(name)
    elif hasattr(value, "data") and value.__class__.__name__ == "EpochArray":
        df = pd.DataFrame(np.asarray(value.data, float).reshape(-1, 2), columns=["start", "stop"])
    else:
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[1] == 2:
            df = pd.DataFrame(arr, columns=["start", "stop"])
        elif arr.ndim == 2:                                  # a matrix, e.g. files x sextiles
            df = pd.DataFrame(arr, columns=[f"c{k + 1}" for k in range(arr.shape[1])])
        else:
            df = pd.DataFrame({name: arr.ravel()})
    _ORIG_TO_CSV(df, path, index=False)
    return len(df)


# ----------------------------------------------------------------------------- write guard
_ORIG_TO_CSV = pd.DataFrame.to_csv
_ALLOWED = [REPO.resolve()]


def _allowed(p) -> bool:
    try:
        rp = Path(p).resolve()
    except Exception:
        return False
    return any(str(rp).lower().startswith(str(a).lower()) for a in _ALLOWED + [SCRATCH.resolve()])


def _divert(p) -> str:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return str(SCRATCH / os.path.basename(str(p)))


@contextlib.contextmanager
def guarded_writes():
    saved = dict(savefig=Figure.savefig, show=plt.show, open=builtins.open, makedirs=os.makedirs,
                 mkdir=Path.mkdir,
                 **{m: getattr(pd.DataFrame, m) for m in ("to_csv", "to_pickle", "to_excel", "to_parquet")})

    def savefig(self, fname, *a, **k):
        if not hasattr(fname, "write") and not _allowed(fname):
            fname = _divert(fname)
        return saved["savefig"](self, fname, *a, **k)

    def wrap(meth):
        def w(self, path=None, *a, **k):
            if path is not None and not hasattr(path, "write") and not _allowed(path):
                path = _divert(path)
            return saved[meth](self, path, *a, **k)
        return w

    def opener(file, mode="r", *a, **k):
        if any(c in mode for c in "wax+") and isinstance(file, (str, Path)) and not _allowed(file):
            file = _divert(file)
        return saved["open"](file, mode, *a, **k)

    def makedirs(name, *a, **k):
        if _allowed(name) or os.path.isdir(name):
            return saved["makedirs"](name, *a, **k) if _allowed(name) else None
        print(f"[guard] makedirs refused: {name}")

    def mkdir(self, *a, **k):
        if _allowed(self) or self.is_dir():
            return saved["mkdir"](self, *a, **k) if _allowed(self) else None
        print(f"[guard] mkdir refused: {self}")

    Figure.savefig, plt.show, builtins.open, os.makedirs, Path.mkdir = savefig, (lambda *a, **k: None), opener, makedirs, mkdir
    for m in ("to_csv", "to_pickle", "to_excel", "to_parquet"):
        setattr(pd.DataFrame, m, wrap(m))
    try:
        yield
    finally:
        Figure.savefig, plt.show, builtins.open, os.makedirs, Path.mkdir = (saved["savefig"], saved["show"], saved["open"],
                                                                             saved["makedirs"], saved["mkdir"])
        for m in ("to_csv", "to_pickle", "to_excel", "to_parquet"):
            setattr(pd.DataFrame, m, saved[m])


# ----------------------------------------------------------------------------- notebook helpers
def load_nb(path):
    return json.load(open(path, encoding="utf-8"))


def cell_lines(cell):
    return "".join(cell.get("source", [])).splitlines()


def clean(src: str) -> str:
    """IPython magics / shell lines are not Python; keep them visible but inert for exec."""
    return "\n".join(("pass  # " + l) if re.match(r"\s*[%!]", l) else l for l in src.splitlines())


def run_cells(nb, ns, cuts=None, csv_dir=None, log=None):
    """Execute code cells. Each cut is marked by two calls inserted at the block's own indentation
    (so cuts inside an `if` block work too): one before it, and one after it that writes the CSVs."""
    cuts = cuts or []
    by_cell = {}
    for c in cuts:
        by_cell.setdefault(c["cell"], []).append(c)

    def cut_begin():
        print("######## BEGIN-CUT ########", flush=True)

    def cut_end(k):
        c = cuts[k]
        # the variables visible where the cut ends: module level, or a function's locals
        scope = dict(ns)
        scope.update(sys._getframe(1).f_locals)
        if csv_dir is not None:
            for var, fname in c["export"].items():
                if "," in var:                         # several 1-D arrays of equal length -> one table
                    val = pd.DataFrame({v: np.asarray(scope[v]).ravel() for v in var.split(",")})
                else:
                    val = scope[var]
                n = export_var(var, val, csv_dir / fname)
                print(f"[export] {var} -> {fname} ({n} rows)", flush=True)
            if c.get("export_code"):                  # custom export (e.g. only the plotted events)
                csv_dir.mkdir(parents=True, exist_ok=True)
                scope["CSV_OUT"] = csv_dir
                exec(c["export_code"], scope)
                print(f"[export] custom export written to {csv_dir.name}", flush=True)
        print(f"######## CUT {c['cell']}:{c['end']} ########", flush=True)

    ns["__cut_begin__"], ns["__cut_end__"] = cut_begin, cut_end
    ns["_EPOCH_ACC"] = {}            # for cuts inside a loop that collect one table per iteration
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = cell_lines(cell)
        for c in sorted(by_cell.get(i, []), key=lambda c: -c["start"]):
            ind = re.match(r"\s*", lines[c["start"]]).group(0)
            k = cuts.index(c)
            lines = (lines[:c["start"]] + [f"{ind}__cut_begin__()"] + lines[c["start"]:c["end"] + 1]
                     + [f"{ind}__cut_end__({k})"] + lines[c["end"] + 1:])
        print(f"\n######## cell {i} ########", file=log, flush=True)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            exec(compile(clean("\n".join(lines)), f"cell{i}", "exec"), ns)
    plt.close("all")


def build_copy(nb, spec, out_path):
    nb = json.loads(json.dumps(nb))
    stem = spec["stem"]
    head = [f"# Data for this notebook: ../csv/{stem}/  (exported from the lab data by",
            "# ../analysis_scripts/make_figure_csvs.py; the original loading code is kept below as comments)",
            "import os, glob, pickle, re, sys   # standard imports (some notebooks relied on them being loaded already)",
            "from pathlib import Path",
            "sys.path.insert(0, str(Path('..') / 'analysis_scripts'))   # the lab's `criticality` package, shipped there",
            "import pandas as pd",
            "import numpy as np",
            f"CSV = Path('..') / 'csv' / '{spec.get('csv_stem', stem)}'",
            "OUT = Path('..') / 'figure_output'",
            "OUT.mkdir(exist_ok=True)",
            "",
            "",
            "def read_csv(name, **kw):",
            "    \"\"\"Read one of this notebook's CSVs; round_trip parsing gives back exactly the numbers written.\"\"\"",
            "    return pd.read_csv(CSV / name, float_precision='round_trip', **kw)"]
    if spec.get("kernel_state"):
        head += ["", "", "# The original notebook relied on these names already existing in its Jupyter kernel",
                 "# (left over from an earlier run); defined here so the notebook runs from a fresh kernel."]
        head += spec["kernel_state"]
    by_cell = {}
    for c in spec["cuts"]:
        by_cell.setdefault(c["cell"], []).append(c)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = cell_lines(cell)
        for c in sorted(by_cell.get(i, []), key=lambda c: -c["start"]):
            block = lines[c["start"]:c["end"] + 1]
            ind = re.match(r"\s*", block[0]).group(0) if block else ""
            new = [f"{ind}# ---- data: read from CSV (replaces the loading code below) ----"]
            new += [ind + l.replace("pd.read_csv(CSV / ", "read_csv(") for l in c["load"]]
            new += [f"{ind}# ---- original loading code (needs the lab servers), kept for reference ----"]
            new += [f"{ind}# {l[len(ind):] if l.startswith(ind) else l}" for l in block]
            lines = lines[:c["start"]] + new + lines[c["end"] + 1:]
        for a, b in spec.get("replace", {}).get(i, []):
            txt = "\n".join(lines)
            assert a in txt, f"replace target not found in cell {i}: {a[:60]}"
            lines = txt.replace(a, b).splitlines()
        txt = SAVE_PREFIX.sub(lambda m: "../figure_output", "\n".join(lines))
        cell["source"] = [l + "\n" for l in txt.splitlines()]
        cell["source"][-1:] = [cell["source"][-1].rstrip("\n")] if cell["source"] else []
        cell["outputs"], cell["execution_count"] = [], None
    first = next(k for k, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    nb["cells"].insert(first, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                               "source": [l + "\n" for l in head[:-1]] + [head[-1]]})
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return nb


def norm_log(text, start_marker):
    """Lines printed after the first cut, without the guard / timing / path noise."""
    if start_marker:
        text = text.split(start_marker, 1)[-1]
    # drop what the loading blocks themselves print (the copy does not run them)
    text = re.sub(r"######## BEGIN-CUT ########.*?(?=######## CUT )", "", text, flags=re.S)
    out = []
    for l in text.splitlines():
        if re.search(r"\[guard\]|\[export\]|Saved|saved|savefig|Folder|Warning:|warnings\.warn|\d+\.\d+ ?s\b|elapsed|it/s|%\|", l):
            continue
        if l.startswith("######## cell"):
            continue
        out.append(re.sub(r"(?:[A-Za-z]:\\|\.\./|\.\.\\)\S+", "<path>", l))   # absolute or ../ save paths
    return out


def svg_signature(path: Path) -> str:
    """SVG content without the run-specific parts matplotlib writes (date, random object ids)."""
    t = path.read_text(encoding="utf-8", errors="ignore")
    t = re.sub(r"<metadata>.*?</metadata>", "", t, flags=re.S)

    def png_pixels(m):                      # embedded rasters: compare decoded pixels, not PNG bytes
        import base64, hashlib
        from matplotlib import image as mpimg
        arr = mpimg.imread(io.BytesIO(base64.b64decode(re.sub(r"\s", "", m.group(1)))), format="png")
        return "data:image/png;pixels=" + hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest() + '"'
    t = re.sub(r'data:image/png;base64,([^"]+)"', png_pixels, t)
    ids = {}
    t = re.sub(r'(id="|#)([a-z]+[0-9a-f]{8,12})\b',
               lambda m: m.group(1) + ids.setdefault(m.group(2), f"ID{len(ids)}"), t)
    return t


def compare_figures(before: set):
    """Every file the copy saved in ../figure_output this run vs the same file saved (diverted to
    SCRATCH) by the original run."""
    out_dir = REPO / "figure_output"
    rows = []
    for f in sorted(out_dir.iterdir()):
        if f.name in before or not (SCRATCH / f.name).exists():
            continue
        if f.suffix.lower() == ".svg":
            same = svg_signature(f) == svg_signature(SCRATCH / f.name)
        else:
            same = f.read_bytes() == (SCRATCH / f.name).read_bytes()
        rows.append((f.name, same))
    return rows


# Some exploration cells draw random numbers without a seed (the plotted windows themselves are
# hard-coded). For the comparison only, both runs start from the same seed.
SEED = "import random as _rnd, numpy as _np\n_rnd.seed(0)\n_np.random.seed(0)"


def fresh_kernel_state():
    """Each run starts like a new Jupyter kernel: default matplotlib settings, and modules that set
    global style on import (the lab's `criticality` package) imported afresh."""
    matplotlib.rcdefaults()
    plt.close("all")
    for m in [m for m in sys.modules if m == "criticality" or m.startswith("criticality.")]:
        del sys.modules[m]


def slim(cstem, full_dir: Path, since: float):
    """Write the CSVs the notebooks read (../csv/) from the full exports (../csv_full/): identical,
    except where SLIM has a rule (bulk data no panel needs)."""
    out_dir = CSV_ROOT / cstem
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(full_dir.glob("*.csv")):
        if f.stat().st_mtime < since:
            continue                                   # not exported by this run (e.g. a stale file)
        df = pd.read_csv(f, float_precision="round_trip")
        rule = SLIM.get((cstem, f.name))
        if rule is not None:
            df = rule(df, full_dir)
        _ORIG_TO_CSV(df, out_dir / f.name, index=False)


def drop_cells(text, cells, offset=0):
    """Remove what the given (original-numbered) cells printed from a run log."""
    for c in cells:
        text = re.sub(rf"######## cell {c + offset} ########.*?(?=######## cell |\Z)", "", text, flags=re.S)
    return text


def process(spec):
    import time
    stem = spec["stem"]
    cstem = spec.get("csv_stem", stem)                 # a notebook may share another notebook's CSVs
    src = ORIG_REPO / "notebooks" / spec.get("src", f"figures/{stem}.ipynb")
    csv_dir = CSV_FULL / cstem
    nb = load_nb(src)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    print(f"=== {stem}: running the original from the lab data", flush=True)
    log_a = io.StringIO()
    t_run = time.time() - 1
    with guarded_writes():
        cwd = os.getcwd(); os.chdir(src.parent)
        try:
            fresh_kernel_state()
            ns = {"__name__": "__main__"}
            # the kernel state the notebooks were run in: standard modules already imported
            exec("import os, glob, pickle, re\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd", ns)
            exec(SEED, ns)
            exec("\n".join(spec.get("kernel_state", [])), ns)
            run_cells(nb, ns, spec["cuts"], csv_dir, log_a)
        finally:
            os.chdir(cwd)
    (SCRATCH / f"{stem}.original.log").write_text(log_a.getvalue(), encoding="utf-8")
    slim(cstem, csv_dir, t_run)
    out_nb = FIG_DIR / f"{stem}.ipynb"
    copy = build_copy(nb, spec, out_nb)
    print(f"=== {stem}: running the copy from ../csv", flush=True)
    log_b = io.StringIO()
    t0 = time.time() - 1
    with guarded_writes():
        cwd = os.getcwd(); os.chdir(FIG_DIR)
        try:
            fresh_kernel_state()
            ns_b = {"__name__": "__main__"}
            exec(SEED, ns_b)
            run_cells(copy, ns_b, None, None, log_b)
        finally:
            os.chdir(cwd)
    (SCRATCH / f"{stem}.copy.log").write_text(log_b.getvalue(), encoding="utf-8")
    first = min(spec["cuts"], key=lambda c: (c["cell"], c["end"]))
    skip = spec.get("unverified_cells", [])            # cells whose printout depends on slimmed bulk data
    a = norm_log(drop_cells(log_a.getvalue(), skip), f"######## CUT {first['cell']}:{first['end']} ########")
    b = [l for l in norm_log(drop_cells(log_b.getvalue(), skip, offset=1), None) if not l.startswith("######## CUT")]
    # every line the original prints after the first cut must appear, in the same order, in the copy
    diff, j = [], 0
    for line in a:
        if line.startswith("######## CUT"):
            continue
        try:
            j = b.index(line, j) + 1
        except ValueError:
            diff.append(line)
    status = f"IDENTICAL ({len(a)} lines)" if not diff else f"{len(diff)} of {len(a)} lines not reproduced"
    print(f"=== {stem}: output after the cut vs original: {status}  (logs in {SCRATCH})", flush=True)
    for d in diff[:20]:
        print("    missing:", d[:200])
    # figures: each file the copy saved this run vs the original run's file (same name, diverted)
    out_dir = REPO / "figure_output"
    figs = []
    for f in sorted(out_dir.iterdir()):
        if f.is_dir() or f.stat().st_mtime < t0:
            continue
        if not (SCRATCH / f.name).exists():
            figs.append((f.name, None))
        elif f.suffix.lower() == ".svg":
            figs.append((f.name, svg_signature(f) == svg_signature(SCRATCH / f.name)))
        else:
            figs.append((f.name, f.read_bytes() == (SCRATCH / f.name).read_bytes()))
    for name, same in figs:
        print(f"    figure {name}: " + ("identical to the original run" if same else
                                         "no original to compare" if same is None else "DIFFERS"))
    with open(HERE / "make_figure_csvs_check.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M')}  {stem}: printed output after the cut {status}; "
                 + "; ".join(f"{n} {'same' if s else 'n/a' if s is None else 'DIFFERS'}" for n, s in figs) + "\n")
    return not diff and all(s is not False for _, s in figs)


# ----------------------------------------------------------------------------- SPECS
# cell = index in the ORIGINAL notebook; start / end = 0-based line numbers (inclusive) of the loading
# block inside that cell; export = {variable: csv file}; load = the lines that replace the block.
SPECS = {}


def spec(stem, cuts, **kw):
    SPECS[stem] = dict(stem=stem, cuts=cuts, **kw)


# SLIM: bulk data no panel needs, removed from ../csv/ (the full exports stay in ../csv_full/).
# {(csv folder, file): rule(df, full_dir) -> df}
SLIM = {}


def _crop_time(col, t0, t1):
    return lambda df, full: df[(df[col] >= t0) & (df[col] <= t1)]


def _crop_like(ref_file, col, t0, t1):
    """Same row positions as the time crop of ref_file (for a series binned in step with it)."""
    def rule(df, full):
        ref = pd.read_csv(full / ref_file, float_precision="round_trip")
        idx = np.flatnonzero((ref[col] >= t0) & (ref[col] <= t1))
        return df.iloc[idx[0]: idx[-1] + 1]
    return rule


# Fig. 2c draws 14 350.6-14 530.6 s (hard-coded in the notebook); the 40 ms rate series are smoothed over
# 30 s before they are sampled there, so 14 200-14 700 s keeps every value the panel uses.
_FIG2C = ("Fig2_C_Example_replayprobalit", (14200.0, 14700.0))
SLIM[(_FIG2C[0], "mua_40ms.csv")] = _crop_time("time_bins", *_FIG2C[1])
SLIM[(_FIG2C[0], "rate_E_40ms.csv")] = _crop_time("time_bins", *_FIG2C[1])
SLIM[(_FIG2C[0], "rate_I_40ms.csv")] = _crop_like("rate_E_40ms.csv", "time_bins", *_FIG2C[1])
# Fig. 2g inset uses only the start, stop and peak of each SWR
SLIM[("Fig2_G_Example_branchingratio", "ripple_events.csv")] = lambda df, full: df[["start", "stop", "peaks"]]


spec("Fig1_D_histogram_S1_E", [
    dict(cell=0, start=523, end=554, export={"df_all": "d2_windows_18sessions.csv"},
         load=["pooled, missing = [], []",
               "df_all = pd.read_csv(CSV / 'd2_windows_18sessions.csv')"]),
    # cell 3 (Extended Data Fig. 1e) repeats the same 18-session loop; its export is byte-identical,
    # so both cuts read the same file
    dict(cell=3, start=458, end=489, export={"df_all": "d2_windows_18sessions.csv"},
         load=["pooled, missing = [], []",
               "df_all = pd.read_csv(CSV / 'd2_windows_18sessions.csv')"]),
])


spec("Fig1_C_Example_plot_mouse", [
    # session loading: spikes, sleep states and epochs of the example session W:\data\PPP\PPP20\day5
    dict(cell=0, start=25, end=76,
         export={"nrem_epochs": "states_NREM.csv", "awake_epochs": "states_WAKE.csv",
                 "rem_epochs": "states_REM.csv", "epoch_df": "epochs.csv"},
         load=["save_dir = str(OUT)",
               "nrem_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_NREM.csv')[['start', 'stop']].to_numpy())",
               "awake_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_WAKE.csv')[['start', 'stop']].to_numpy())",
               "rem_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_REM.csv')[['start', 'stop']].to_numpy())",
               "epoch_df = pd.read_csv(CSV / 'epochs.csv')",
               "spike_times = None   # raw spikes are not shipped; the 40 ms MUA is read in the binning cell"]),
    # time-resolved d2 (2024 Euclidean, AR 20, 40 ms bins, 30 s windows) from the session's MATLAB .mat
    dict(cell=2, start=208, end=217, export={"ca1_complete_df": "d2_timeline_CA1.csv"},
         load=["ca1_complete_df = pd.read_csv(CSV / 'd2_timeline_CA1.csv')",
               "ca1_complete_meta = {}"]),
    # MUA: all CA1 units, 40 ms bins (the notebook's own bin_spikes)
    dict(cell=8, start=10, end=10, export={"time_bins,firing_rate": "mua_40ms.csv"},
         load=["_mua = pd.read_csv(CSV / 'mua_40ms.csv')",
               "time_bins, firing_rate = _mua['time_bins'].to_numpy(), _mua['firing_rate'].to_numpy()"]),
    # ripple / barrage events: loaded by the original notebook but not drawn in its panels
    dict(cell=11, start=0, end=27, export={},
         load=["pass  # ripple and barrage events are not used by the panels of this notebook"]),
])

spec("Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange", [
    # d2-MUA window pairs of the 18 Fig. 1 sessions (data/cross_state/pairs/*.pkl, made by
    # generate_pairs_df.ipynb) + task-epoch labels from each session's epoch table on W:
    dict(cell=0, start=6, end=91, export={"all_pairs": "d2_mua_pairs_18sessions.csv"},
         load=["all_pairs = pd.read_csv(CSV / 'd2_mua_pairs_18sessions.csv')"]),
])


# ---------------------------------------------------------------------------------------- Figure 2
spec("Fig2_A_FigS2_B_FigS3_A", [
    # d2 windows of the pre/post-sleep sessions (data/ripple_barr/manipulation/pre_post/*.pkl,
    # made by generate_pre_post.ipynb), pooled
    dict(cell=0, start=68, end=82, export={"full_df": "d2_windows_pre_post.csv"},
         load=["full_df = pd.read_csv(CSV / 'd2_windows_pre_post.csv')"]),
])

spec("Fig2_D_E_replayprobality_SWR_replaynon_replay", [
    # panel E: d2 (z within file) by tile of replay / non-replay SWR rate, all files pooled
    # (data/ripple_barr/replay_non_replay/*.pkl, made by generate_replay_non_replay.ipynb)
    dict(cell=1, start=130, end=193, export={"pooled": "d2_by_replay_nonreplay_swr_rate_tiles.csv"},
         load=["rows = []",
               "pooled = pd.read_csv(CSV / 'd2_by_replay_nonreplay_swr_rate_tiles.csv')"]),
    dict(cell=4, start=55, end=57, export={},
         load=["save_dir = str(OUT)",
               "pkl_files = []   # the per-file data are read from ../csv below"]),
    # panel D: per-file curve of replay probability (z within file) by d2 sextile
    # (data/replay/replay_probality/*.pkl, made by generate_replay_probablity_tile.ipynb)
    dict(cell=4, start=103, end=146,
         export={"per_file_curves": "replay_prob_by_d2_sextile_per_file.csv", "meta_df": "replay_prob_files.csv"},
         load=["per_file_curves = pd.read_csv(CSV / 'replay_prob_by_d2_sextile_per_file.csv').to_numpy(float)",
               "meta_df = pd.read_csv(CSV / 'replay_prob_files.csv')"]),
    # panel D statistics: one row per window (d2 sextile, replay probability z) for the mixed model
    dict(cell=5, start=6, end=56, export={"lmm_df": "replay_prob_lmm_rows.csv"},
         load=["rows = []",
               "lmm_df = pd.read_csv(CSV / 'replay_prob_lmm_rows.csv')"]),
])

spec("Fig2_E_replay_non_replay", [
    dict(cell=0, start=130, end=193, export={"pooled": "d2_by_replay_nonreplay_swr_rate_tiles.csv"},
         load=["rows = []",
               "pooled = pd.read_csv(CSV / 'd2_by_replay_nonreplay_swr_rate_tiles.csv')"]),
], csv_stem="Fig2_D_E_replayprobality_SWR_replaynon_replay")   # the same table as Fig2_D_E's panel e

spec("Fig2_F_GLM", [
    # per-session GLM ablation results (data/GLM_mice/explore/*_ablation_results_GLM.pkl, made by
    # generate_GLM.ipynb), pooled by the notebook's load_and_pool_glm_pickles
    dict(cell=0, start=227, end=227,
         export={"pooled": "glm_all_models.csv", "ab_df": "glm_ablations.csv", "gain_df": "glm_gain.csv",
                 "glm_files": "glm_files.csv"},
         load=["pooled = pd.read_csv(CSV / 'glm_all_models.csv')",
               "ab_df = pd.read_csv(CSV / 'glm_ablations.csv')",
               "gain_df = pd.read_csv(CSV / 'glm_gain.csv')",
               "glm_files = pd.read_csv(CSV / 'glm_files.csv')['glm_files'].tolist()"]),
])

spec("Fig2_G_branchingratio", [
    # branching-ratio change around each SWR / barrage and shuffle, every event of every session
    # (data/ripple_barr/branching_ratio/ripple/changes/*_br_changes_.pkl, made by generate_br_change_ripple.ipynb)
    dict(cell=0, start=49, end=105, export={"df": "br_change_per_event.csv"},
         load=["df = pd.read_csv(CSV / 'br_change_per_event.csv')"]),
])

spec("Fig2_H_entropy", [
    # ISI entropy per window with its SWR-rate sextile, pooled (data/ripple_barr/ISI_entropy/ripple/*.pkl,
    # made by generate_ISI_entropy_ripple.ipynb), as returned by the notebook's load_pool_rate_binned_isi
    dict(cell=0, start=237, end=240, export={"df_ripple_pooled_isi": "isi_entropy_by_swr_rate_sextile.csv"},
         load=["df_ripple_pooled_isi = pd.read_csv(CSV / 'isi_entropy_by_swr_rate_sextile.csv')"]),
])


# the setup cell shared by the example-session notebooks (Fig1_C, Fig2_B, Fig2_C): states and epochs
def _example_session_setup(cell=0):
    return dict(cell=cell, start=25, end=76,
                export={"nrem_epochs": "states_NREM.csv", "awake_epochs": "states_WAKE.csv",
                        "rem_epochs": "states_REM.csv", "epoch_df": "epochs.csv"},
                load=["save_dir = str(OUT)",
                      "nrem_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_NREM.csv')[['start', 'stop']].to_numpy())",
                      "awake_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_WAKE.csv')[['start', 'stop']].to_numpy())",
                      "rem_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_REM.csv')[['start', 'stop']].to_numpy())",
                      "epoch_df = pd.read_csv(CSV / 'epochs.csv')",
                      "sleep_epochs = nel.EpochArray(np.array([epoch_df.startTime, epoch_df.stopTime]).T)",
                      "spike_times = None   # raw spikes are not shipped"])


_FIG2B_SIG, _FIG2B_NONSIG, _FIG2B_CH = 1731, 934, 255

spec("Fig2_B_Example_replay", [
    _example_session_setup(0),
    dict(cell=2, start=0, end=27, export={},
         load=["pass  # ripple and barrage events are not used by the panels of this notebook"]),
    # decoded replay events of PPP21 day 4 (R_Project replay_results/*_replay_sequences.mat): every
    # significant event's interval, fit and slope; the decoded posterior only for the plotted event
    dict(cell=4, start=0, end=88, export={}, export_code=f"""
_rows = []
for _side, _mats, _ivs, _gf, _sl in (("left", left_replay_matrix, left_intervals, left_replay_goodfit, left_replay_slope),
                                     ("right", right_replay_matrix, right_intervals, right_replay_goodfit, right_replay_slope)):
    for _k in range(len(_mats)):
        _iv = np.asarray(_ivs[_k], float).ravel()
        _rows.append(dict(side=_side, start=_iv[0], stop=_iv[1], goodfit=float(_gf[_k]), slope=float(_sl[_k])))
pd.DataFrame(_rows).to_csv(CSV_OUT / "replay_events_significant.csv", index=False)
_nL = len(left_replay_matrix)
_m = left_replay_matrix[{_FIG2B_SIG}] if {_FIG2B_SIG} < _nL else right_replay_matrix[{_FIG2B_SIG} - _nL]
pd.DataFrame(np.asarray(_m, float)).to_csv(CSV_OUT / "replay_matrix_event{_FIG2B_SIG}.csv", index=False)
""", load=[
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "# every significant replay event (left then right, as in the original lists)",
        "_ev = pd.read_csv(CSV / 'replay_events_significant.csv')",
        "left_replay_matrix, right_replay_matrix, left_intervals, right_intervals = [], [], [], []",
        "left_replay_goodfit, right_replay_goodfit, left_replay_slope, right_replay_slope = [], [], [], []",
        "for _r in _ev.itertuples():",
        "    _L = _r.side == 'left'",
        "    (left_replay_matrix if _L else right_replay_matrix).append(None)",
        "    (left_intervals if _L else right_intervals).append(np.array([_r.start, _r.stop]))",
        "    (left_replay_goodfit if _L else right_replay_goodfit).append(_r.goodfit)",
        "    (left_replay_slope if _L else right_replay_slope).append(_r.slope)",
        f"# the decoded posterior is shipped only for the plotted event ({_FIG2B_SIG})",
        f"_m = pd.read_csv(CSV / 'replay_matrix_event{_FIG2B_SIG}.csv').to_numpy(float)",
        f"if {_FIG2B_SIG} < len(left_replay_matrix): left_replay_matrix[{_FIG2B_SIG}] = _m",
        f"else: right_replay_matrix[{_FIG2B_SIG} - len(left_replay_matrix)] = _m"]),
    # non-significant events: interval and p-values of each; the posterior only for the plotted one
    dict(cell=5, start=0, end=40, export={}, export_code=f"""
_rows = [dict(start=float(np.ravel(d['interval'])[0]), stop=float(np.ravel(d['interval'])[1]),
              pval_left=float(np.ravel(d['pval'])[0]), pval_right=float(np.ravel(d['pval'])[1])) for d in non_sig_matrices]
pd.DataFrame(_rows).to_csv(CSV_OUT / "replay_events_nonsignificant.csv", index=False)
pd.DataFrame(np.asarray(non_sig_matrices[{_FIG2B_NONSIG}]['left'], float)).to_csv(
    CSV_OUT / "nonreplay_matrix_event{_FIG2B_NONSIG}_left.csv", index=False)
""", load=[
        "_ns = pd.read_csv(CSV / 'replay_events_nonsignificant.csv')",
        "non_sig_matrices = [{'left': None, 'right': None, 'interval': np.array([r.start, r.stop]),",
        "                     'pval': np.array([r.pval_left, r.pval_right])} for r in _ns.itertuples()]",
        f"non_sig_matrices[{_FIG2B_NONSIG}]['left'] = pd.read_csv(CSV / 'nonreplay_matrix_event{_FIG2B_NONSIG}_left.csv').to_numpy(float)",
        "non_sig_intervals = [d['interval'] for d in non_sig_matrices]"]),
    # LFP: only the ripple channel, only around the two plotted events
    dict(cell=6, start=1, end=4, export={}, export_code=f"""
_nL = len(left_replay_matrix)
_iv = left_intervals[{_FIG2B_SIG}] if {_FIG2B_SIG} < _nL else right_intervals[{_FIG2B_SIG} - _nL]
_parts = []
for _c in (float(np.mean(_iv)), float(np.mean(non_sig_matrices[{_FIG2B_NONSIG}]['interval']))):
    _i0, _i1 = np.searchsorted(ts, _c - 0.3), np.searchsorted(ts, _c + 0.3)
    _parts.append(pd.DataFrame({{"ts": np.asarray(ts[_i0:_i1], float), "lfp": np.asarray(lfp[_i0:_i1, chan], float)}}))
pd.concat(_parts).sort_values("ts").to_csv(CSV_OUT / "lfp_ch{_FIG2B_CH}_around_plotted_events.csv", index=False)
pd.DataFrame(dict(nChannels=[nChannels], fs=[fs], fs_dat=[fs_dat])).to_csv(CSV_OUT / "lfp_meta.csv", index=False)
""", load=[
        "epoch_df = pd.read_csv(CSV / 'epochs.csv')",
        "_meta = pd.read_csv(CSV / 'lfp_meta.csv')",
        "nChannels, fs, fs_dat = int(_meta.nChannels[0]), _meta.fs[0], _meta.fs_dat[0]",
        "# LFP: only channel `chan`, only +-0.3 s around the two plotted events (the full .dat is GBs)",
        f"_lfp = pd.read_csv(CSV / 'lfp_ch{_FIG2B_CH}_around_plotted_events.csv')",
        "ts = _lfp['ts'].to_numpy(float)",
        "lfp = np.zeros((len(ts), nChannels)); lfp[:, chan] = _lfp['lfp'].to_numpy(float)"]),
    dict(cell=7, start=0, end=5, export={},
         load=["pass  # (LFP size printout: the shipped LFP holds only the plotted segments)"]),
])


spec("Fig2_C_Example_replayprobalit", [
    _example_session_setup(0),                                   # PPP8 day 8
    # time-resolved d2 (CA1, *_D2_time_resolved.mat)
    dict(cell=2, start=208, end=217, export={"ca1_complete_df": "d2_timeline_CA1.csv"},
         load=["ca1_complete_df = pd.read_csv(CSV / 'd2_timeline_CA1.csv')",
               "ca1_complete_meta = {}"]),
    # MUA: all CA1 units, 40 ms bins, whole session (smoothed over 30 s by the notebook)
    dict(cell=11, start=10, end=10, export={"time_bins,firing_rate": "mua_40ms.csv"},
         load=["_mua = pd.read_csv(CSV / 'mua_40ms.csv')",
               "time_bins, firing_rate = _mua['time_bins'].to_numpy(), _mua['firing_rate'].to_numpy()"]),
    # pyramidal (E) and interneuron (I) rates, 40 ms bins, for the E/I ratio
    dict(cell=14, start=24, end=25, export={"time_bins,fr_E": "rate_E_40ms.csv", "fr_I": "rate_I_40ms.csv"},
         load=["_E = pd.read_csv(CSV / 'rate_E_40ms.csv')",
               "time_bins, fr_E = _E['time_bins'].to_numpy(), _E['fr_E'].to_numpy()",
               "fr_I = pd.read_csv(CSV / 'rate_I_40ms.csv')['fr_I'].to_numpy()"]),
    # SWR peak times of the session (barrage events are not used by the panels)
    dict(cell=16, start=0, end=27, export={"ripple_peaks": "ripple_peaks.csv"},
         load=["ripple_peaks = pd.read_csv(CSV / 'ripple_peaks.csv')['ripple_peaks']",
               "barrage_epochs = []"]),
    # decoded replay events (R_Project replay_results/*_replay_sequences_new.mat): times and intervals
    dict(cell=18, start=0, end=94,
         export={"left_times": "replay_times_left.csv", "right_times": "replay_times_right.csv",
                 "all_intervals": "replay_intervals.csv"},
         load=["import numpy as np",
               "import matplotlib.pyplot as plt",
               "left_times = pd.read_csv(CSV / 'replay_times_left.csv')['left_times'].tolist()",
               "right_times = pd.read_csv(CSV / 'replay_times_right.csv')['right_times'].tolist()",
               "all_intervals = [np.array(iv) for iv in pd.read_csv(CSV / 'replay_intervals.csv')[['start', 'stop']].to_numpy()]"]),
# cell 14 prints a session-wide mean E/I ratio; the E/I series is not in any panel and is cropped (SLIM),
# so that printout is not compared (it is reproduced from csv_full/)
], unverified_cells=[14])

spec("Fig2_G_Example_branchingratio", [
    # sleep states of the example session PPP21 day 5
    dict(cell=1, start=24, end=45,
         export={"nrem_epochs": "states_NREM.csv", "awake_epochs": "states_WAKE.csv", "rem_epochs": "states_REM.csv"},
         load=["nrem_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_NREM.csv')[['start', 'stop']].to_numpy())",
               "awake_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_WAKE.csv')[['start', 'stop']].to_numpy())",
               "rem_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_REM.csv')[['start', 'stop']].to_numpy())"]),
    # SWR events of the session
    dict(cell=4, start=17, end=17, export={"ripple_events": "ripple_events.csv"},
         load=["ripple_events = pd.read_csv(CSV / 'ripple_events.csv')"]),
    # branching ratio per 10 s window in NREM (data/ripple_barr/branching_ratio/ripple/*.pkl, made by
    # generate_br_change_ripple.ipynb)
    dict(cell=5, start=6, end=17, export={"centers_time,branching_ratios": "branching_ratio_windows_NREM.csv"},
         load=["_br = pd.read_csv(CSV / 'branching_ratio_windows_NREM.csv')",
               "centers_time, branching_ratios = _br['centers_time'].to_numpy(), _br['branching_ratios'].to_numpy()"]),
])


# ---------------------------------------------------------------------------------------- Figure 3
spec("Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar", [
    # panel e: d2 windows of the barrage sessions (z within file) by barrage-rate sextile, pooled
    # (data/ripple_barr/mice_d2_barr_ripple/*.pkl, made by MUA_sleepstage.ipynb)
    dict(cell=1, start=151, end=156, export={"df_barrage_pooled": "d2_by_barrage_rate_sextile.csv"},
         load=["df_barrage_pooled = pd.read_csv(CSV / 'd2_by_barrage_rate_sextile.csv')"]),
    # panel f: branching-ratio change around every barrage and at shuffled times
    # (data/ripple_barr/branching_ratio/barr/*_br_changes.pkl, made by branching_ratio.ipynb)
    dict(cell=3, start=43, end=96, export={"df": "br_change_per_barrage.csv"},
         load=["files = []",
               "df = pd.read_csv(CSV / 'br_change_per_barrage.csv')"]),
    # panel g: d2 windows of the CCK-silencing (Sncg) sessions, pre- and post-task sleep
    # (data/ripple_barr/manipulation/sncg/*.pkl, made by generate_barr_manipulation.ipynb)
    dict(cell=6, start=36, end=37, export={"pooled_df": "d2_windows_sncg.csv"},
         load=["all_pkls = []",
               "pooled_df = pd.read_csv(CSV / 'd2_windows_sncg.csv')"]),
    # panel h: ISI entropy per window with barrage rate (data/ripple_barr/ISI_entropy/barrage/*.pkl,
    # made by generate_ISI_entropy_ripple.ipynb)
    dict(cell=9, start=247, end=248, export={"pooled_df": "isi_entropy_windows_barrage.csv"},
         load=["pooled_df = pd.read_csv(CSV / 'isi_entropy_windows_barrage.csv')"]),
    # Extended Data (barrage-ablated d2 by barrage rate): d2 per window with the barrages removed
    # (data/ripple_barr/barr_ablate/*.pkl, made by MUA_sleepstage_ablate_barr.ipynb)
    dict(cell=10, start=409, end=409, export={"pooled_df": "d2_barrage_ablated_windows.csv"},
         load=["pooled_df = pd.read_csv(CSV / 'd2_barrage_ablated_windows.csv')"]),
# panel h's plot_tiles_line returns a ripple figure `fig_rip` that the notebook no longer makes
], kernel_state=["fig_rip = None"])


# ------------------------------------------------------------------------------ Extended Data 1, 2, 3
spec("FigS1_B_Explain variance_D_statetransition", [
    # ED 1b: AR-model explained variance per window length, per animal (R_Project
    # coarse_grain_bin_explore/20250324timebin_grainedmouse.mat), reshaped by the notebook into groupData
    dict(cell=0, start=52, end=112, export={}, export_code="""
_rows = []
for _key, _animals in groupData.items():
    for _a, _ad in enumerate(_animals):
        if not hasattr(_ad, '__iter__'):
            _ad = [_ad]
        elif isinstance(_ad, np.ndarray):
            _ad = _ad.tolist()
        for _e in _ad:
            try:
                _v = _e.explain_variance
            except AttributeError:
                _v = _e['explain_variance']
            _rows += [(_key, _a, float(_x)) for _x in np.array(_v).flatten()]
pd.DataFrame(_rows, columns=["window", "animal", "explain_variance"]).to_csv(
    CSV_OUT / "explained_variance_by_window.csv", index=False)
""", load=["# explained variance of the AR fit, one row per value: window length key, animal, value",
           "_ev = pd.read_csv(CSV / 'explained_variance_by_window.csv')",
           "groupData, group_values_dict, allData, groupLabels = {}, {}, [], []",
           "for key in _ev['window'].unique():            # original order of the .mat fields",
           "    g = _ev[_ev['window'] == key]",
           "    groupData[key] = [a['explain_variance'].to_numpy() for _, a in g.groupby('animal', sort=False)]",
           "    group_values_dict[key] = g['explain_variance'].to_numpy()",
           "    allData.extend(group_values_dict[key].tolist())",
           "    groupLabels.extend([key] * len(g))"]),
    # ED 1d: state-labelled d2 windows around state transitions, 14 sessions (*_state_transition.mat)
    dict(cell=2, start=215, end=221, export={"df_all": "d2_windows_state_transition.csv"},
         load=["dfs = []",
               "df_all = pd.read_csv(CSV / 'd2_windows_state_transition.csv')"]),
])

spec("FigS1_C_SNR_S2C_d2 uncertainty", [
    # ED 1c: d2 and its error (delta method) for 30 s and 60 s windows, 18 sessions
    # (*_error_bar.mat and *_60sec.mat on the server)
    dict(cell=0, start=263, end=273, export={},
         export_code='pd.concat(pooled_all["CA1"], ignore_index=True).to_csv(CSV_OUT / "d2_err_30s_60s_18sessions.csv", index=False)',
         load=["missing = []",
               "pooled_all = {'CA1': [pd.read_csv(CSV / 'd2_err_30s_60s_18sessions.csv')]}"]),
    # ED 2c: d2 and its error by SWR-rate sextile (data/ripple_barr/d2_uncertainty/*.pkl, made by
    # MUA_sleepstage.ipynb); the notebook loads the same table twice
    dict(cell=2, start=217, end=224, export={"df_ripple_pooled": "d2_err_by_swr_rate_sextile.csv"},
         load=["df_ripple_pooled = pd.read_csv(CSV / 'd2_err_by_swr_rate_sextile.csv')"]),
    dict(cell=2, start=404, end=411, export={"df_ripple_pooled": "d2_err_by_swr_rate_sextile.csv"},
         load=["df_ripple_pooled = pd.read_csv(CSV / 'd2_err_by_swr_rate_sextile.csv')"]),
])

spec("FigS1_F_branchingratio_histogram", [
    # ED 1f: branching ratio per window and state, 18 sessions (data/cross_state/branching_ratio_1/
    # branching_ratio_allstates_*.pkl, made by generate_branching_ratio_mouse.ipynb), with wake windows
    # inside sleep epochs removed (epoch tables on W:); weights = window durations
    dict(cell=0, start=111, end=174, export={}, export_code="""
pd.concat([pd.DataFrame({"state": _st, "br": pooled_vals[_st], "weight": pooled_wts[_st]}) for _st in STATES],
          ignore_index=True).to_csv(CSV_OUT / "branching_ratio_by_state.csv", index=False)
""", load=["_p = pd.read_csv(CSV / 'branching_ratio_by_state.csv')",
           "pooled_vals = {st: _p.loc[_p['state'] == st, 'br'].to_numpy() for st in STATES}",
           "pooled_wts = {st: _p.loc[_p['state'] == st, 'weight'].to_numpy() for st in STATES}"]),
], src="manuscript_code/FigS1_F_branchingratio_histogram.ipynb")

spec("FigS2_A_novel_familar", [
    # ED 2a: d2 relative to pre-task sleep, mid- and post-task sleep, familiar (control) and novel sessions
    # (data/novel_familiar/{control,novel}/*.pkl; no producer in the source repository)
    dict(cell=0, start=48, end=72, export={"all_rel": "d2_rel_novel_familiar.csv"},
         load=["rows = []",
               "all_rel = pd.read_csv(CSV / 'd2_rel_novel_familiar.csv')"]),
])

spec("FigS3_B_S3C_S3D_ripple_ablate", [
    # ED 3b-d: d2 with the ripples removed, per window, with SWR rate and ISI entropy
    # (data/ripple_barr/ripple_ablate/*.pkl, made by generate_ablate_ripple.ipynb)
    dict(cell=1, start=74, end=87, export={"ca1_all_df": "d2_ripple_ablated_windows.csv"},
         load=["all_dfs = []",
               "ca1_all_df = pd.read_csv(CSV / 'd2_ripple_ablated_windows.csv')"]),
])

_S3FG_EXPORT = ('pd.concat([d.assign(_file=k, _name=os.path.basename(pkl_files[k])) for k, d in enumerate(all_dfs)],'
                ' ignore_index=True).to_csv(CSV_OUT / "rat_CA2_CA3_firing_by_window.csv", index=False)')
_S3FG_LOAD = ["# one table per session, in the original file order (column _file; _name = the session's file)",
              "_t = pd.read_csv(CSV / 'rat_CA2_CA3_firing_by_window.csv')",
              "all_dfs = [g.drop(columns=['_file', '_name']).reset_index(drop=True) for _, g in _t.groupby('_file', sort=True)]",
              "pkl_files = [g['_name'].iloc[0] for _, g in _t.groupby('_file', sort=True)]"]
spec("FigS3_G_F_CA3_CA2_pyr_model_surpport_fig", [
    # ED 3f, 3g (rats): CA3 and CA2 firing per window with SWR / barrage rate
    # (data/rats/barr_CA2_CA3/*.pkl, made by MUA_sleepstage_barr_model_supple_ver2.ipynb)
    dict(cell=1, start=22, end=32, export={}, export_code=_S3FG_EXPORT, load=_S3FG_LOAD),
    dict(cell=4, start=22, end=32, export={}, export_code=_S3FG_EXPORT, load=_S3FG_LOAD),
])


spec("FigS2_D_E_F_low_avalanche_example_avalanche_sizeduradcc_", [
    # example session PPP20 day 4: sleep states and epochs (raw spikes are not shipped, see below)
    dict(cell=3, start=24, end=59,
         export={"nrem_epochs": "states_NREM.csv", "awake_epochs": "states_WAKE.csv",
                 "rem_epochs": "states_REM.csv", "epoch_df": "epochs.csv"},
         load=["from types import SimpleNamespace",
               "save_dir = str(OUT)",
               "nrem_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_NREM.csv')[['start', 'stop']].to_numpy())",
               "awake_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_WAKE.csv')[['start', 'stop']].to_numpy())",
               "rem_epochs = nel.EpochArray(pd.read_csv(CSV / 'states_REM.csv')[['start', 'stop']].to_numpy())",
               "epoch_df = pd.read_csv(CSV / 'epochs.csv')",
               "sleep_epochs = nel.EpochArray(np.array([epoch_df.startTime, epoch_df.stopTime]).T)",
               "# raw spikes are not shipped: the avalanches they give are read inside run_dcc_pipeline_forplot",
               "st = SimpleNamespace(time=None)",
               "spike_times = None"]),
    dict(cell=6, start=3, end=3, export={"ripple_events": "ripple_events.csv"},
         load=["ripple_events = pd.read_csv(CSV / 'ripple_events.csv')"]),
    dict(cell=6, start=7, end=47, export={},
         load=["# barrage, delta-wave and all_d2 files: loaded by the original notebook, not used by its panels",
               "barrage_epochs, delta_events, all_d2 = [], [], []"]),
    # SWR rate in 1 s bins within the pre-, mid- and post-task NREM (the only output of this cell the panels use)
    dict(cell=8, start=15, end=58,
         export={"pre_timestamps,pre_ripple_rate": "swr_rate_1s_pre.csv",
                 "mid_timestamps,mid_ripple_rate": "swr_rate_1s_mid.csv",
                 "post_timestamps,post_ripple_rate": "swr_rate_1s_post.csv"},
         load=["_r = pd.read_csv(CSV / 'swr_rate_1s_pre.csv'); pre_timestamps, pre_ripple_rate = _r['pre_timestamps'].to_numpy(), _r['pre_ripple_rate'].to_numpy()",
               "_r = pd.read_csv(CSV / 'swr_rate_1s_mid.csv'); mid_timestamps, mid_ripple_rate = _r['mid_timestamps'].to_numpy(), _r['mid_ripple_rate'].to_numpy()",
               "_r = pd.read_csv(CSV / 'swr_rate_1s_post.csv'); post_timestamps, post_ripple_rate = _r['post_timestamps'].to_numpy(), _r['post_ripple_rate'].to_numpy()"]),
    # avalanches (all CA1 units, 50 ms bins, 30th percentile) in the low- and high-SWR-rate intervals of
    # pre-task NREM: what coarse_grain_avalanche_detection_all_units_intervals returns for the call below
    dict(cell=13, start=36, end=39,
         export={"aval_sizes_low,aval_durs_low": "avalanches_low_swr_rate.csv",
                 "aval_sizes_high,aval_durs_high": "avalanches_high_swr_rate.csv"},
         load=["_a = read_csv('avalanches_low_swr_rate.csv')",
               "aval_sizes_low, aval_durs_low = _a['aval_sizes_low'].to_numpy(), _a['aval_durs_low'].to_numpy()",
               "_a = read_csv('avalanches_high_swr_rate.csv')",
               "aval_sizes_high, aval_durs_high = _a['aval_sizes_high'].to_numpy(), _a['aval_durs_high'].to_numpy()"]),
    # ED 2e, 2f: power-law exponents and DCC of every session (data/l_h_ripples/log/*.csv and R_Project
    # avalanche/dcc_all_all_sleep.csv, written by low_high_ripple_avalanche.ipynb)
    dict(cell=15, start=60, end=61, export={"df_size": "av_size_all_sleep.csv", "df_duration": "av_duration_all_sleep.csv"},
         load=["df_size = pd.read_csv(CSV / 'av_size_all_sleep.csv')",
               "df_duration = pd.read_csv(CSV / 'av_duration_all_sleep.csv')"]),
    dict(cell=15, start=233, end=233, export={"df": "dcc_all_all_sleep.csv"},
         load=["df = pd.read_csv(CSV / 'dcc_all_all_sleep.csv')"]),
])
SLIM[("FigS2_D_E_F_low_avalanche_example_avalanche_sizeduradcc_", "ripple_events.csv")] = \
    lambda df, full: df[["start", "stop", "peaks"]]


# ------------------------------------------------------------------------------ Extended Data 5, 6, 10
# ED 5a is the Fig. 1c notebook run on a rat session (HP18 day 36): same cells, same cuts
spec("FigS5_A_Example_plot_rat", [_cut for _cut in SPECS["Fig1_C_Example_plot_mouse"]["cuts"]])


def _epoch_loop_cut(cell, line, name):
    """Replace one `epoch_df = loading.load_epoch(str(basepath))` inside a per-session loop: the epoch
    tables of all sessions go into one CSV (column `session`), and each iteration reads its own rows."""
    return dict(cell=cell, start=line, end=line, export={}, export_code=f"""
_EPOCH_ACC.setdefault("{name}", []).append(epoch_df.assign(session=str(basepath)))
pd.concat(_EPOCH_ACC["{name}"], ignore_index=True).to_csv(CSV_OUT / "{name}", index=False)
""", load=[f"_ep = pd.read_csv(CSV / '{name}')   # epoch tables of all sessions, one CSV",
           "epoch_df = _ep[_ep['session'] == str(basepath)].drop(columns='session').dropna(axis=1, how='all').reset_index(drop=True)"])


spec("FigS5_S6_rats", [
    # ED 6b: branching-ratio change around every SWR and at shuffled times, rats
    # (data/rats/branching_ratio/*_br_changes*.pkl, made by generate_branching_ratio_rats.ipynb)
    dict(cell=1, start=45, end=100, export={"df": "rat_br_change_per_swr.csv"},
         load=["files = []", "df = pd.read_csv(CSV / 'rat_br_change_per_swr.csv')"]),
    # ED 5b: d2 windows of the rat sessions with state and task epoch (*_error_bar.mat + epochs on the servers)
    dict(cell=4, start=533, end=564, export={"df_all": "rat_d2_windows.csv"},
         load=["pooled, missing = [], []", "df_all = pd.read_csv(CSV / 'rat_d2_windows.csv')"]),
    # ED 6d: ISI entropy by SWR-rate sextile (data/rats/*.pkl, made by MUA_sleepstage_rat.ipynb)
    dict(cell=8, start=183, end=186, export={"df_ripple_pooled_isi": "rat_isi_entropy_by_swr_rate.csv"},
         load=["df_ripple_pooled_isi = pd.read_csv(CSV / 'rat_isi_entropy_by_swr_rate.csv')"]),
    # ED 6f: d2 by barrage-rate sextile (data/rats/ripple_barr/isi/*.pkl)
    dict(cell=10, start=225, end=228, export={"df_barrage_pooled": "rat_d2_by_barrage_rate.csv"},
         load=["df_barrage_pooled = pd.read_csv(CSV / 'rat_d2_by_barrage_rate.csv')"]),
    # ED 6g: ISI entropy by barrage-rate sextile (data/rats/ripple_barr/isi/*.pkl)
    dict(cell=12, start=184, end=187, export={"df_barrage_pooled_isi": "rat_isi_entropy_by_barrage_rate.csv"},
         load=["df_barrage_pooled_isi = pd.read_csv(CSV / 'rat_isi_entropy_by_barrage_rate.csv')"]),
    # ED 5c: d2-MUA window pairs of the rat sessions (data/rats/cross_state/pairs/*.pkl)
    dict(cell=14, start=22, end=48, export={"all_pairs": "rat_d2_mua_pairs.csv"},
         load=["files = []", "all_pairs = pd.read_csv(CSV / 'rat_d2_mua_pairs.csv')"]),
    _epoch_loop_cut(14, 127, "rat_epoch_tables_ED5c.csv"),
    # ED 5d / 6a: d2 windows by state of the rat sessions (data/rats/*.pkl)
    dict(cell=21, start=25, end=95, export={"all_pairs": "rat_d2_windows_by_state.csv"},
         load=["files = []", "all_pairs = pd.read_csv(CSV / 'rat_d2_windows_by_state.csv')"]),
    _epoch_loop_cut(21, 177, "rat_epoch_tables_ED6a.csv"),
    # sessions whose first epoch is the pre-task sleep (from each session's epoch table)
    dict(cell=22, start=37, end=41, export={"paths_with_presleep_first": "rat_sessions_presleep_first.csv"},
         load=["paths_with_presleep_first = pd.read_csv(CSV / 'rat_sessions_presleep_first.csv')['paths_with_presleep_first'].tolist()"]),
    # ED 6c: d2 by tile of replay / non-replay SWR rate per session (data/rats/replay_nonreplay/*_tile.pkl)
    dict(cell=27, start=168, end=169, export={"long_all": "rat_replay_nonreplay_tiles.csv"},
         load=["long_all = pd.read_csv(CSV / 'rat_replay_nonreplay_tiles.csv')"]),
    # ED 6e: branching-ratio change around SWRs and barrages (non-overlapping events), rats
    dict(cell=30, start=57, end=112, export={"df": "rat_br_change_per_event_nonoverlap.csv"},
         load=["files = []", "df = pd.read_csv(CSV / 'rat_br_change_per_event_nonoverlap.csv')"]),
])

# ED 10: epilepsy session AYA9 day 28 (Z:\Data\AYAold). Spike times are not shipped. The copy reads the
# unit groups of general\matlab.mat, keeps the two hand-set epochs (epileptic 108-118 min, control
# 14-24 min) as written in the notebook, and reads what the notebook computes from the spikes:
# avalanche sizes/durations and the branching ratio of every 20 s window
_ED10_GROUPS = ("ca1int", "ca1pyr", "ca2int", "ca2pyr", "ca3int", "ca3pyr")
_NO_SPIKES = "  # spike times are not shipped"
spec("FigS10_eplipsy", [
    dict(cell=1, start=24, end=29, export={},
         load=["pass  # (the original creates a working folder next to the raw data)"]),
    dict(cell=2, start=7, end=66, export={}, export_code=f"""
pd.concat([pd.DataFrame({{"group": _gname, "matlab_index": np.asarray(_garr, int).ravel()}})
           for _gname, _garr in zip({_ED10_GROUPS!r}, (ca1int, ca1pyr, ca2int, ca2pyr, ca3int, ca3pyr))],
          ignore_index=True).to_csv(CSV_OUT / "unit_groups.csv", index=False)
""", load=["# unit groups of general\\matlab.mat (MATLAB indices, 1-based)",
           "_grp = pd.read_csv(CSV / 'unit_groups.csv')",
           f"ca1int, ca1pyr, ca2int, ca2pyr, ca3int, ca3pyr = [_grp.loc[_grp['group'] == g, 'matlab_index'].to_numpy(int) for g in {_ED10_GROUPS!r}]",
           "ca1int_py, ca1pyr_py, ca2int_py, ca2pyr_py, ca3int_py, ca3pyr_py = (ca1int - 1, ca1pyr - 1, ca2int - 1, ca2pyr - 1, ca3int - 1, ca3pyr - 1)",
           "spike_times = ca1_spike_trains = ca1pyr_spike_trains = None" + _NO_SPIKES]),
    dict(cell=4, start=0, end=30, export={},
         load=["pass  # (lists the spike count of every unit)" + _NO_SPIKES]),
    dict(cell=5, start=24, end=25, export={}, load=["ca1_units = ca1_all_spikes = None" + _NO_SPIKES]),
    dict(cell=5, start=50, end=50, export={}, load=["edges = pop_counts = pop_rates = None" + _NO_SPIKES]),
    dict(cell=5, start=163, end=163, export={}, load=["sliding_epi = None  # (not used by any panel)"]),
    dict(cell=5, start=195, end=195, export={}, load=["total_dur = None  # (not used by any panel)"]),
    # avalanches of the CA1 units inside each epoch (40 ms bins, threshold at the 50th percentile)
    dict(cell=8, start=17, end=29, export={}, export_code="""
pd.concat([pd.DataFrame({"epoch": _n, "size": np.asarray(_s).ravel(), "duration": np.asarray(_d).ravel()})
           for _n, _s, _d in (("epileptic", sizes_epi, dur_epi), ("control", sizes_ctl, dur_ctl))],
          ignore_index=True).to_csv(CSV_OUT / "avalanches_epileptic_control.csv", index=False)
""", load=["# avalanches of the CA1 units in each epoch (40 ms bins, threshold at the 50th percentile)",
           "_av = pd.read_csv(CSV / 'avalanches_epileptic_control.csv')",
           "sizes_epi, dur_epi = (_av.loc[_av['epoch'] == 'epileptic', c].to_numpy() for c in ('size', 'duration'))",
           "sizes_ctl, dur_ctl = (_av.loc[_av['epoch'] == 'control', c].to_numpy() for c in ('size', 'duration'))"]),
    # branching ratio (MR estimator, CA1 population counts in 10 ms bins, k_max = 100 i.e. lags up to
    # 1 s) of every 20 s window stepped 2 s inside each epoch
    dict(cell=9, start=121, end=153, export={}, export_code="""
pd.concat([pd.DataFrame({"epoch": _n, "start": [a for a, b in _iv], "end": [b for a, b in _iv], "center": _r["centers"],
                         "branching_ratio": [np.nan if v is None else v for v in _r["branching_ratios"]]})
           for _n, _iv, _r in (("epileptic", epi_intervals, epi_results), ("control", ctl_intervals, ctl_results))],
          ignore_index=True).to_csv(CSV_OUT / "branching_ratio_windows.csv", index=False)
""", load=["# branching ratio (MR estimator; CA1 population counts in 10 ms bins, k_max = 100, i.e. lags up to 1 s)",
           "# of every 20 s window, stepped 2 s, inside the epileptic and the control epoch",
           "_br = pd.read_csv(CSV / 'branching_ratio_windows.csv')",
           "epi_intervals, ctl_intervals = (list(zip(_br.loc[_br['epoch'] == e, 'start'], _br.loc[_br['epoch'] == e, 'end'])) for e in ('epileptic', 'control'))",
           "epi_results, ctl_results = ({'centers': _br.loc[_br['epoch'] == e, 'center'].tolist(),",
           "                             'branching_ratios': _br.loc[_br['epoch'] == e, 'branching_ratio'].tolist(),",
           "                             'Goodfit': []} for e in ('epileptic', 'control'))"]),
])


if __name__ == "__main__":
    todo = list(SPECS) if sys.argv[1:] in (["all"], []) else sys.argv[1:]
    ok = {s: process(SPECS[s]) for s in todo}
    print("\nsummary:", {k: ("ok" if v else "DIFFERS") for k, v in ok.items()})
