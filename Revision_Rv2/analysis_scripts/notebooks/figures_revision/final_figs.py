"""
final_figs.py -- shared helpers for the revision's final-figure notebooks (notebooks/figures_revision/).

Convention: every panel of the revised figures has its own output file, and is drawn by a
  notebook here; panels on the same theme or data share one notebook, named after its panels:
  <Fig>_<letters>_<theme>.ipynb  (e.g. Fig1_CD_manifold_structure_index.ipynb);
  each panel is saved as reports/revision_final/<Fig>/<Fig>_<letter>_<short_name>.svg / .png
where <Fig> is Fig1, Fig2, ... or FigS_<...> for supplementary panels (folder `supplementary`).
A notebook redraws its panel from the producer's saved tables (or reruns the producer), never from a
copied picture, except schematics, which are copied from their vector source.

Index of panels and their sources: reports/revision_final/README.md.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REV = os.path.join(REPO, "notebooks", "exploration", "revision")
FIGS = os.path.join(REPO, "reports", "revision_figs")
FINAL = os.path.join(REPO, "reports", "revision_final")
INKSCAPE = r"C:\Program Files\Inkscape\bin\inkscape.exe"
if REV not in sys.path:
    sys.path.insert(0, REV)

from run_fig1_metrics import assert_local_path  # noqa: E402


def folder(fig_id):
    """Fig1_B_... -> reports/revision_final/Fig1 ; FigS_... -> reports/revision_final/supplementary"""
    head = fig_id.split("_")[0]
    if fig_id.startswith("Letter"):                    # Letter01_... -> letter (figures outside the Fig/FigS numbering)
        sub = "letter"
    elif fig_id.startswith("FigS"):                      # FigS1_A_... -> supplementary/FigS1; FigS_... -> supplementary
        sub = os.path.join("supplementary", head) if head[4:].isdigit() else "supplementary"
    else:
        sub = head
    d = os.path.join(FINAL, sub)
    os.makedirs(assert_local_path(d), exist_ok=True)
    return d


def stem(fig_id):
    return os.path.join(folder(fig_id), fig_id)


def _retry(fn, *a, **k):
    for i in range(8):                       # Windows refuses writes while an editor preview holds the file
        try:
            return fn(*a, **k)
        except OSError:
            if i == 7:
                raise
            time.sleep(4)


def save(fig, fig_id, dpi=300, tight=True):
    """Save a matplotlib figure as <stem>.svg and <stem>.png."""
    s = stem(fig_id)
    bb = dict(bbox_inches="tight", pad_inches=0.01) if tight else {}   # tight=False: exactly the figure size (fixed axes)
    _retry(fig.savefig, assert_local_path(s + ".svg"), dpi=600, **bb)   # images inside: 600 dpi (Nature >= 300); vectors and text unaffected
    _retry(fig.savefig, assert_local_path(s + ".png"), dpi=dpi, **bb)   # print margin, whatever the rcParams
    print(f"saved {os.path.relpath(s, REPO)}.svg / .png")
    return s + ".png"


def settle(fig, tight=True, dpi=300):
    """Draw the figure as save() does, without writing it. Called before drop_counts, so that the layout is solved
    with the session counts present, as for the published panels."""
    import io
    bb = dict(bbox_inches="tight", pad_inches=0.01) if tight else {}
    fig.savefig(io.BytesIO(), format="svg", dpi=600, **bb)
    fig.savefig(io.BytesIO(), format="png", dpi=dpi, **bb)


def copy(src, fig_id):
    """Copy a producer's output (src without extension, or a file) to <stem>.svg / .png."""
    base = os.path.splitext(src)[0]
    s, done = stem(fig_id), []
    for ext in (".svg", ".png"):
        if os.path.exists(base + ext):
            _retry(shutil.copy2, base + ext, assert_local_path(s + ext))
            done.append(ext)
    if ".svg" in done and ".png" not in done and os.path.exists(INKSCAPE):
        subprocess.run([INKSCAPE, s + ".svg", "--export-type=png", "--export-dpi=300",
                        f"--export-filename={assert_local_path(s + '.png')}"], check=True, capture_output=True)
        done.append(".png (inkscape)")
    assert done, f"nothing to copy at {base}"
    print(f"copied {os.path.basename(base)} -> {os.path.relpath(s, REPO)} {done}")
    return s + ".png"


class retry_savefig:
    """Context manager: while a producer saves its own figures, retry a write that Windows refuses
    (OSError 22, e.g. an editor preview holding the file)."""

    def __enter__(self):
        import matplotlib.figure
        self._orig = matplotlib.figure.Figure.savefig
        orig = self._orig
        matplotlib.figure.Figure.savefig = lambda fig, *a, **k: _retry(orig, fig, *a, **k)
        return self

    def __exit__(self, *exc):
        import matplotlib.figure
        matplotlib.figure.Figure.savefig = self._orig
        return False


def run_rv1(notebook):
    """Run a Previous_Rv1 figure notebook through ed_nonoverlap (window tables thinned to disjoint
    windows; tables without window times read from ed_nonoverlap_tables/). Returns its output folder."""
    import matplotlib
    import matplotlib.figure
    import ed_nonoverlap as E
    orig = matplotlib.figure.Figure.savefig
    matplotlib.figure.Figure.savefig = E.guarded_savefig
    try:
        E.run_notebook(notebook)
    finally:
        matplotlib.figure.Figure.savefig = orig
    return os.path.join(E.OUTROOT, notebook)


def crop_svg(src, fig_id, x0, y0, w, h, drop_ids=(), unit_mm=25.4 / 96):
    """A region of an assembled SVG (its user units, e.g. an Inkscape figure) as a panel file of its own, at its own
    size: the root's viewBox becomes the region and width / height its size in mm; the elements entirely outside the
    region (the rest of the source figure) are removed (drop_offpage).
    drop_ids: elements to leave out (e.g. the old panel letters, redrawn by the assembler). Returns the svg path."""
    import re
    s = open(src, encoding="utf-8").read()
    for i in drop_ids:                                       # a <text> element (with its tspans), or one <tspan>
        m = (re.search(r'<text\b[^>]*\bid="%s"(?:(?!</text>).)*</text>' % re.escape(i), s, flags=re.S)
             or re.search(r'<tspan\b[^>]*\bid="%s"[^>]*>[^<]*</tspan>' % re.escape(i), s, flags=re.S))
        assert m, f"{i} not found in {src}"
        s = s[:m.start()] + s[m.end():]
    tag = re.search(r"<svg\b[^>]*>", s).group(0)
    new = re.sub(r'\s(width|height|viewBox)="[^"]*"', "", tag)
    new = new.replace("<svg", f'<svg width="{w * unit_mm:.4f}mm" height="{h * unit_mm:.4f}mm" viewBox="{x0} {y0} {w} {h}"', 1)
    s = s.replace(tag, new, 1)
    out = assert_local_path(stem(fig_id) + ".svg")
    _retry(lambda: open(out, "w", encoding="utf-8").write(s))
    if os.path.exists(INKSCAPE):
        drop_offpage(out, w * unit_mm, h * unit_mm)
        png = assert_local_path(stem(fig_id) + ".png")
        subprocess.run([INKSCAPE, out, "--export-type=png", "--export-dpi=600", "--export-background=#ffffff",
                        f"--export-filename={png}"], capture_output=True)
    print(f"cropped {os.path.basename(src)} [{x0}, {y0}, {w} x {h}] -> {os.path.relpath(out, REPO)} ({w * unit_mm:.1f} x {h * unit_mm:.1f} mm)")
    return out


def drop_offpage(path, w_mm, h_mm):
    """Remove from an SVG every element drawn entirely outside its page of w_mm x h_mm, so that the file holds only
    what it shows. Bounding boxes from Inkscape (svg_bboxes, px from the page corner); the root, elements without an id
    and anything inside defs / clipPath / mask / pattern / marker / symbol / gradients / filters are kept. A text-level
    edit: the rest of the file is unchanged. Returns the number of elements removed."""
    import re
    keep = {"defs", "clipPath", "mask", "pattern", "marker", "symbol", "linearGradient", "radialGradient", "filter",
            "style", "metadata", "title", "desc"}
    s = open(path, encoding="utf-8").read()
    q = svg_bboxes(path, unit_px=1.0)
    W, H = w_mm * 96 / 25.4, h_mm * 96 / 25.4

    def off(i):
        if i not in q:
            return False
        x, y, w, h = q[i]
        return x > W or y > H or x + w < 0 or y + h < 0

    tok = re.compile(r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|<!DOCTYPE[^>]*>"
                     r"|<(/?)([\w:.-]+)((?:[^>\"']|\"[^\"]*\"|'[^']*')*?)(/?)>", re.S)
    stack, cut = [], []                                       # stack: (start, tag, id, kept: a kept element or in one)
    for m in tok.finditer(s):
        if m.group(2) is None:                                # comment, CDATA, prolog
            continue
        close, name, attrs, single = m.groups()
        if close:
            start, local, i, kept = stack.pop()
            if not kept and local != "svg" and off(i):
                cut.append((start, m.end()))
            continue
        local = name.split(":")[-1]
        i = re.search(r'\bid="([^"]+)"', attrs)
        entry = (m.start(), local, i.group(1) if i else None, local in keep or bool(stack and stack[-1][3]))
        if not single:
            stack.append(entry)
        elif not entry[3] and local != "svg" and off(entry[2]):
            cut.append((m.start(), m.end()))
    assert not stack, f"unbalanced tags in {path}"
    outer = []                                                # outermost only: children go with their parent
    for start, end in sorted(cut):
        if not outer or start >= outer[-1][1]:
            outer.append((start, end))
    for start, end in reversed(outer):
        s = s[:start] + s[end:]
    _retry(lambda: open(assert_local_path(path), "w", encoding="utf-8").write(s))
    return len(outer)


def svg_bboxes(path, unit_px=96 / 25.4):
    """{element id: (x, y, w, h)} from Inkscape's --query-all, in the document's units when unit_px is its px per
    unit (96 / 25.4 for a document in mm)."""
    import csv
    import io
    out = subprocess.run([INKSCAPE, "--query-all", path], capture_output=True, text=True).stdout
    q = {}
    for r in csv.reader(io.StringIO(out)):
        if len(r) == 5:
            try:
                q[r[0]] = tuple(float(v) / unit_px for v in r[1:])
            except ValueError:
                pass
    return q


def recolor_svg(path, rules, png_dpi=600):
    """Recolour elements of an SVG in place: rules = [(ids, {"#old": "#new"}), ...]; each mapping applies only to the
    fill / stroke / stop-color of the elements whose id is in ids (e.g. those inside one panel of a copied figure).
    Re-exports the PNG beside it. Returns {(old, new): number of replacements}."""
    import re
    s = open(path, encoding="utf-8").read()
    done = {}
    for ids, mapping in rules:
        mapping = {k.lower(): v for k, v in mapping.items()}

        def fix(m):
            tag = m.group(0)
            i = re.search(r'\bid="([^"]+)"', tag)
            if not i or i.group(1) not in ids:
                return tag

            def col(mm):
                old = mm.group(2).lower()
                if old in mapping:
                    done[(old, mapping[old])] = done.get((old, mapping[old]), 0) + 1
                    return mm.group(1) + mapping[old]
                return mm.group(0)
            return re.sub(r'((?:fill|stroke|stop-color)\s*[:=]\s*"?)(#[0-9a-fA-F]{6})', col, tag)
        s = re.sub(r"<[a-zA-Z][^<>]*\bid=\"[^\"]+\"[^<>]*>", fix, s)
    _retry(lambda: open(assert_local_path(path), "w", encoding="utf-8").write(s))
    png = os.path.splitext(path)[0] + ".png"
    subprocess.run([INKSCAPE, path, "--export-type=png", f"--export-dpi={png_dpi}", "--export-background=#ffffff",
                    f"--export-filename={assert_local_path(png)}"], capture_output=True)
    return done


def rv1_table(notebook, after, var, out_csv):
    """Run a Previous_Rv1 figure notebook through ed_nonoverlap (disjoint windows) and write the DataFrame
    `var`, as it stands right after the source line `after` (which must occur once in the notebook), to
    out_csv: the exact table a panel plots, for a redraw at final size. The run's own figures are not saved."""
    import tempfile
    from pathlib import Path
    import matplotlib.figure
    import ed_nonoverlap as E
    out_csv = assert_local_path(out_csv)
    patches, root = E.PATCH.get(notebook), E.OUTROOT
    E.PATCH[notebook] = (patches or []) + [(after, after + f"\n{var}.to_csv(r'{out_csv}', index=False)")]
    E.OUTROOT = Path(tempfile.mkdtemp(prefix="rv1_table_"))           # its log and _run folder, not the ED outputs
    orig = matplotlib.figure.Figure.savefig
    matplotlib.figure.Figure.savefig = lambda *a, **k: None
    try:
        errors = E.run_notebook(notebook)
    finally:
        matplotlib.figure.Figure.savefig = orig
        E.OUTROOT = root
        if patches is None:
            E.PATCH.pop(notebook)
        else:
            E.PATCH[notebook] = patches
    assert os.path.exists(out_csv), f"{var} not written; see the log in the temp folder ({errors} cell errors)"
    print(f"{notebook}: {var} -> {out_csv}")
    return out_csv


def run_nb(path, after_cell=None, overrides="", replace=(), timeout=3600):
    """Execute a notebook in memory -- the file itself is not changed. `overrides` is inserted as a code cell after
    cell `after_cell` (index over all cells), and each (old, new) of `replace` must occur exactly once in its code.
    For final-size renders of an exploration notebook whose knobs set size and output path."""
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(path, as_version=4)
    for old, new in replace:
        hits = [c for c in nb.cells if c.cell_type == "code" and old in c.source]
        assert len(hits) == 1 and hits[0].source.count(old) == 1, f"{old!r} found in {len(hits)} cells"
        hits[0].source = hits[0].source.replace(old, new)
    if overrides:
        nb.cells.insert(after_cell + 1, nbformat.v4.new_code_cell(overrides))
    NotebookClient(nb, timeout=timeout, kernel_name="criticality",          # the env, not PATH python
                   resources={"metadata": {"path": os.path.dirname(path)}}).execute()
    for c in nb.cells:                                       # echo the notebook's printed output
        for o in c.get("outputs", []):
            if o.get("name") == "stdout":
                print(o["text"], end="")
    return nb


# ---- final size (Nature Neuroscience: 180 mm or 88 mm wide, Arial 5-7 pt)
MM = 1 / 25.4                     # inches per mm, for figsize
FIG_W = 180.0                     # mm, two-column figure width


def final_style(size=6.0):
    """rcParams for a panel drawn at its printed size: Arial `size` pt text (ticks 0.5 pt smaller), thin lines."""
    import matplotlib as mpl
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "custom",
                         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                         "font.size": size, "axes.labelsize": size, "axes.titlesize": size,
                         "xtick.labelsize": size - 0.5, "ytick.labelsize": size - 0.5, "legend.fontsize": size - 0.5,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2,
                         "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2,
                         "axes.titlepad": 3, "lines.linewidth": 0.8, "patch.linewidth": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "legend.handlelength": 1.2, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
                         "axes.unicode_minus": False, "savefig.pad_inches": 0.01})


# the exploration figures' settings (fig2_new / fig2_composite main): panels drawn by those modules' functions are
# drawn under these, at their final figsize, and then restyled by shrink()
NATIVE_RC = {"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "sans-serif",
             "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 11, "axes.labelsize": 11,
             "axes.titlesize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
             "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.major.size": 3.5,
             "ytick.major.size": 3.5, "axes.unicode_minus": False, "savefig.pad_inches": 0.01}


COUNT = r":?\s*\b\d+\s*/\s*\d+\b(?:\s*(?:sessions?|mice))?\s*[↑↓]?"   # "9/9 ↑", ": 7/9 ↑", "18/18 sessions"


def drop_counts(fig):
    """Formal panels show no sessions-in-direction counts (they go in the legend instead): remove every "k/n"
    (with its arrow / "sessions") from the figure's texts, keeping the stars or p."""
    import re
    from matplotlib.text import Text
    n = 0
    for t in set(fig.findobj(Text)):
        s = t.get_text()
        new = re.sub(COUNT, "", s).strip()
        if new != s.strip():
            new = re.sub(r",\s*,", ",", new)                         # "r = −0.40, , p = …" -> "r = −0.40, p = …"
            t.set_text(re.sub(r"\s{2,}", " ", new) if "\n" not in new else new)
            n += 1
    return n


def capture_figures(run, keys):
    """Run a producer (run(): a callable) with Figure.savefig intercepted: a figure it saves under a basename starting
    with one of `keys` is kept and returned instead of written; nothing at all is written by its savefig calls, and
    plt.close is suspended so the figures stay alive. Returns {key: Figure}."""
    import matplotlib.figure
    import matplotlib.pyplot as plt
    got = {}
    orig, close = matplotlib.figure.Figure.savefig, plt.close

    def grab(fig, fname, *a, **k):
        st = os.path.splitext(os.path.basename(str(fname)))[0]
        for key in keys:
            if st.startswith(key) and key not in got:
                got[key] = fig
    matplotlib.figure.Figure.savefig = grab
    plt.close = lambda *a, **k: None
    try:
        run()
    finally:
        matplotlib.figure.Figure.savefig, plt.close = orig, close
    missing = [k for k in keys if k not in got]
    assert not missing, f"not saved by the producer: {missing}"
    return got


def run_rv1_capture(notebook, keys, override_dir=None):
    """Run a Previous_Rv1 figure notebook as ed_nonoverlap does (window tables thinned to disjoint windows, override
    tables from ed_nonoverlap_tables/, or from `override_dir` for this run only) and return its figures saved under
    `keys` (see capture_figures)."""
    import tempfile
    from pathlib import Path
    import ed_nonoverlap as E
    global LAST_RV1_LOG
    root, odir = E.OUTROOT, E.OVERRIDE_DIR
    E.OUTROOT = Path(tempfile.mkdtemp(prefix="rv1_capture_"))         # its log and _run folder, not the ED outputs
    if override_dir is not None:
        E.OVERRIDE_DIR = Path(override_dir)
    LAST_RV1_LOG = str(E.OUTROOT / f"{notebook}.log")                 # its printed statistics, for the panel's stars
    try:
        return capture_figures(lambda: E.run_notebook(notebook), keys)
    finally:
        E.OUTROOT, E.OVERRIDE_DIR = root, odir


LAST_RV1_LOG = None


def stars(p):
    """Significance stars: *p < 0.05, **p < 0.01, ***p < 0.001, ****p < 0.0001."""
    return "****" if p < 1e-4 else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def bracket(ax, x0, x1, y, text, h=None, fs=6, lw=0.5):
    """A significance bracket from x0 to x1 at height y (data units), with `text` (stars) above it."""
    lo, hi = ax.get_ylim()
    h = 0.02 * (hi - lo) if h is None else h
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color="k", lw=lw, clip_on=False)
    ax.text((x0 + x1) / 2, y + 0.3 * h, text, ha="center", va="bottom", fontsize=fs)


def restyle(fig, w_mm, h_mm, font=6 / 14, line=0.45, marker=0.45, drop_texts=()):
    """A producer's exploration-size figure at its printed size: resized to w x h mm, every text Arial and x `font`
    (clipped to 5-7 pt), lines and markers scaled (shrink), figure-level texts containing any of drop_texts removed
    (footnotes go to the legend), constrained layout. The drawing itself is the producer's."""
    from matplotlib.text import Text
    for t in list(fig.texts):
        if any(s in t.get_text() for s in drop_texts):
            t.remove()
    fig.set_size_inches(w_mm * MM, h_mm * MM)
    for t in fig.findobj(Text):
        t.set_fontfamily("Arial")
    shrink(fig, font=font, line=line, marker=marker)
    fig.set_layout_engine("constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
    return fig


def sparse_yticks(ax, every=2):
    """Label every `every`-th y tick of the laid-out figure (major step x every); the others stay as unlabelled
    minor ticks."""
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    ax.figure.canvas.draw()                                   # the ticks of the final layout, not the first guess
    step = float(np.median(np.diff(ax.get_yticks())))
    ax.yaxis.set_major_locator(MultipleLocator(every * step))
    ax.yaxis.set_minor_locator(MultipleLocator(step))


def native_style():
    import matplotlib as mpl
    mpl.rcParams.update(NATIVE_RC)


def shrink(fig, font=6 / 11, line=0.5, marker=0.6, min_font=5.0, max_font=7.0):
    """Restyle, for its printed size, a figure drawn under NATIVE_RC by a producer's own panel function (whose
    sizes are hard-coded): every text x `font`, clipped to [min_font, max_font] pt (11 -> 6, 10 -> 5.5, 9 -> 5);
    every line, edge and spine x `line`; every marker x `marker` (scatter areas x marker**2); ticks likewise;
    legends rebuilt at the new size in the same place.
    The producer's drawing code -- and the statistics it computes on the way -- is used unchanged."""
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    fs = lambda s: min(max(s * font, min_font), max_font)                       # noqa: E731
    for ax in set(fig.findobj(Axes)):          # a legend sizes its handle boxes when made: rebuild it at the new size
        leg = ax.get_legend()
        if leg is not None:
            h, lab = ax.get_legend_handles_labels()
            texts = [t.get_text() for t in leg.get_texts()]
            pool = {}                                   # a legend over twin axes (handles of both): its entries,
            for a in ax.get_shared_x_axes().get_siblings(ax):   # in its order, from every axes sharing x
                for hh, ll in zip(*a.get_legend_handles_labels()):
                    pool.setdefault(ll, hh)
            if texts and all(t in pool for t in texts):
                h, lab = [pool[t] for t in texts], texts
            if not h:                                   # a legend of proxy handles (legend(handles=[...])): its own
                h = list(getattr(leg, "legend_handles", None) or leg.legendHandles)
                lab = [t.get_text() for t in leg.get_texts()]
            bb = leg._bbox_to_anchor                                            # its anchor, back in axes fractions
            if bb is not None:
                (x0, y0), (x1, y1) = ax.transAxes.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
                bb = (x0, y0) if np.isclose(x0, x1) and np.isclose(y0, y1) else (x0, y0, x1 - x0, y1 - y0)
            ax.legend(h, lab, loc=leg._loc, bbox_to_anchor=bb, ncols=leg._ncols, frameon=leg.get_frame_on(),
                      fontsize=fs(leg.get_texts()[0].get_fontsize() if leg.get_texts() else NATIVE_RC["legend.fontsize"]))
    seen = set()
    for o in fig.findobj():
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, Text):
            o.set_fontsize(fs(o.get_fontsize()))
        elif isinstance(o, Line2D):
            o.set_linewidth(o.get_linewidth() * line)
            o.set_markersize(o.get_markersize() * marker)
            o.set_markeredgewidth(o.get_markeredgewidth() * line)
        elif isinstance(o, Patch):
            o.set_linewidth(o.get_linewidth() * line)
        elif isinstance(o, Collection):
            o.set_linewidths([w * line for w in o.get_linewidths()])
            if hasattr(o, "get_sizes") and len(o.get_sizes()):
                o.set_sizes(o.get_sizes() * marker ** 2)
    for ax in set(fig.findobj(Axes)):          # after the texts, so tick labels are not shrunk twice; ticks made
        ax.tick_params(which="major", labelsize=fs(NATIVE_RC["xtick.labelsize"]),   # later use these settings
                       width=NATIVE_RC["xtick.major.width"] * line, length=NATIVE_RC["xtick.major.size"] * marker, pad=1.5)
        ax.tick_params(which="minor", width=0.8 * line, length=2 * marker)
        ax.xaxis.labelpad = ax.yaxis.labelpad = 2


def rv1_namespace(notebook, upto=None, defs_from=()):
    """The data a Previous_Rv1 figure notebook draws its panels from, for a final-size redraw: its code cells
    [0, upto) executed as ed_nonoverlap runs them (shipped CSVs; window tables thinned to disjoint windows), plus
    the function definitions (only) of the code cells in `defs_from`. No figure is saved. Returns the namespace."""
    import ast
    import contextlib
    import io
    import json
    import tempfile
    import matplotlib.figure
    import matplotlib.pyplot as plt
    import ed_nonoverlap as E
    nb = json.load(open(E.RV1 / "figure_scripts" / f"{notebook}.ipynb", encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__main__", "display": lambda *a, **k: None}
    run = os.path.join(tempfile.mkdtemp(prefix="rv1_ns_"), "_run")      # '../figure_output' lands in the temp folder
    os.makedirs(run)
    cwd, orig = os.getcwd(), matplotlib.figure.Figure.savefig
    E.STATE["nb"] = notebook
    matplotlib.figure.Figure.savefig = lambda *a, **k: None
    log = io.StringIO()
    try:
        os.chdir(run)
        for i, src in enumerate(code[:len(code) if upto is None else upto]):
            with contextlib.redirect_stdout(log):
                exec(compile(E.clean(src), f"{notebook}:cell{i}", "exec"), ns)
            if i == 0:
                ns["CSV"] = E.RV1 / "csv" / ns["CSV"].name
                sys.path.insert(0, str(E.RV1 / "analysis_scripts"))
                ns["read_csv"] = E.make_reader(ns["read_csv"], ns["CSV"])
        for i in defs_from:
            tree = ast.parse(E.clean(code[i]))
            keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
            exec(compile(ast.Module(body=keep, type_ignores=[]), f"{notebook}:cell{i}:defs", "exec"), ns)
    finally:
        os.chdir(cwd)
        matplotlib.figure.Figure.savefig = orig
        plt.close("all")
    return ns


def style(size=11):
    import matplotlib as mpl
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": size, "axes.labelsize": size, "xtick.labelsize": size - 1,
                         "ytick.labelsize": size - 1, "axes.linewidth": 1.2, "legend.fontsize": size - 2,
                         "xtick.major.width": 1.2, "ytick.major.width": 1.2})


def show(png):
    try:
        from IPython.display import Image, display
        display(Image(filename=png, width=700))
    except Exception:                                                     # noqa: BLE001
        pass
