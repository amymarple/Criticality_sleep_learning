"""
si_panels.py -- a multi-panel figure drawn by a producer's drawing code, turned into single panels: the drawing code
runs, its figure is captured (nothing written), then each axes becomes a panel file of its own at print size.
"""
from __future__ import annotations

import importlib
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

import final_figs as F

COUNT = re.compile(r"\b\d+\s*/\s*\d+\b(?:\s*(?:sessions?|mice))?\s*[↑↓]?")
WAKE = {"Wake-θ": "Wake θ", "Wake-nonθ": "Wake non-θ", "theta wake": "Wake θ", "non-theta wake": "Wake non-θ"}


class no_csv:
    """While drawing code runs only to draw its figure: its DataFrame.to_csv calls are skipped."""

    def __enter__(self):
        self.orig = pd.DataFrame.to_csv
        pd.DataFrame.to_csv = lambda df, path=None, *a, **k: None
        return self

    def __exit__(self, *exc):
        pd.DataFrame.to_csv = self.orig
        return False


def producer(name, env=None):
    """a drawing module with its environment knobs set (they are read at import)"""
    for k, v in (env or {}).items():
        os.environ[k] = v
    mod = sys.modules.get(name)
    return importlib.reload(mod) if mod else importlib.import_module(name)


def patch(mod, fname, reps):
    """Redefine one function of a module with exact text replacements in its source (each must occur once)."""
    import inspect
    src = inspect.getsource(getattr(mod, fname))
    for old, new in reps:
        assert src.count(old) == 1, f"{fname}: {old!r} occurs {src.count(old)} times"
        src = src.replace(old, new)
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)


def grab(call, key):
    """the figure the drawing code saves under basename `key`, nothing written; its rcParams undone after"""
    with plt.rc_context(), no_csv():
        return F.capture_figures(call, [key])[key]


def formal(ax):
    """the panel's in-panel text without the sessions-in-direction counts ("9/9**" -> "**", "16/16 ↑ ****" ->
    "****", "14/16\\np < 0.001" -> "p < 0.001"); a count without a test result becomes n.s. ("6/9"); the counts on
    their own ("18/18 < 0") and a per-session summary ("per-session r = ...") go to the legend"""
    for t in list(ax.texts):
        s = t.get_text()
        if s.startswith("per-session r") or re.fullmatch(r"\s*\d+/\d+\s*<\s*0\s*", s):
            t.remove()
        elif COUNT.search(s):
            new = COUNT.sub("", s).strip()
            t.set_text(new if new else "n.s.")


def panel(fig, i, name, w, h, title="", xlabel=None, ylabel=None, ticks=None):
    """axes i of a captured figure as a panel file of its own, at w x h mm: a copy of the figure with the other axes
    removed, the axes over the whole figure, title / labels set (short; the details are in the legend), tick labels
    mapped (ticks: str -> str, default the paper's wake-state names), Extended Data text sizes (ticks 5 pt, labels
    and titles 6 pt, in-panel text and legend 5 pt). The layout is solved with the counts present, then the counts
    are removed (formal) and the panel is saved."""
    f = pickle.loads(pickle.dumps(fig))
    keep = f.axes[i]
    for a in [a for a in f.axes if a is not keep]:
        f.delaxes(a)
    for t in list(f.texts):
        t.remove()
    f._suptitle = None
    keep.set_subplotspec(GridSpec(1, 1, figure=f)[0])
    keep.tick_params(labelleft=True, labelbottom=True)   # a panel that shared an axis keeps its numbers
    for loc in ("left", "center", "right"):
        keep.set_title("", loc=loc)
    if title:
        keep.set_title(title, loc="left")
    if xlabel is not None:
        keep.set_xlabel(xlabel)
    if ylabel is not None:
        keep.set_ylabel(ylabel)
    labels = [t.get_text() for t in keep.get_xticklabels()]
    new = [(ticks or (lambda s: WAKE.get(s, s)))(s) for s in labels]
    if new != labels:
        keep.set_xticks(keep.get_xticks(), new)
    F.restyle(f, w, h, font=6 / 9)
    keep.tick_params(labelsize=5)
    for lab in (keep.xaxis.label, keep.yaxis.label, keep.title, keep._left_title):
        lab.set_size(6)
    for t in keep.texts + (keep.get_legend().get_texts() if keep.get_legend() else []):
        t.set_fontsize(5)
    F.settle(f)
    formal(keep)
    F.save(f, name)
    plt.close(f)
