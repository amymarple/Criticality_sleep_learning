"""
r24_dissociation_figure.py -- the d2-vs-branching-ratio dissociation by state.

The sum of the AR coefficients is a multi-lag branching ratio, so d2 could be read as one minus a
branching ratio. The two metrics rank the brain states differently: d2 (2024 Euclidean, p=20, 40 ms)
is largest in NREM; the branching ratio m (pyramidal, 4 ms bins, k_max = 200 => the same 800 ms
kernel) is largest in THETA.

Reads only stored session medians -- no refitting, no share access, no heavy compute:
    data/revision/log/fig1_session_medians_centre.csv   (LABEL=centre; LABEL=wholewindow:
        fig1_session_medians.csv), 18 sessions x 4 states

Writes:
    reports/revision_figs/backup/r24_d2_vs_br_dissociation.png / .svg   (3 panels incl. tau; r24_figure.py
        uses STATES, LABELS, XPOS and, with TOP=states, state_box_panel())
    data/revision/log/letter_table_state_d2_br.csv

Single-threaded; BLAS threads pinned in case a Pool is added.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from scipy import stats                                              # noqa: E402

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
# LABEL=centre (default): wake split by the window centre in THETA on non-overlapping .mat windows,
# the paper's Fig 1D rule (r24_state_centre.py); LABEL=wholewindow: the revision tiles
LABEL = os.environ.get("LABEL", "centre")
SRC = os.path.join(REPO, "data", "revision", "log",
                   "fig1_session_medians_centre.csv" if LABEL == "centre" else "fig1_session_medians.csv")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
LOGDIR = os.path.join(REPO, "data", "revision", "log")

STATES = ["NREM", "REM", "THETA", "nonTHETA"]      # sleep pair, then wake pair
XPOS = [0, 1, 2.5, 3.5]                             # gap between the pairs; contrasts stay within a pair
CONTRAST_LABEL = {"NREM-REM": "sleep: NREM vs REM", "THETA-nonTHETA": "wake: θ vs non-θ"}
LABELS = {"THETA": "Wake-θ", "REM": "REM",
          "nonTHETA": "Wake-nonθ", "NREM": "NREM"}
# POP=all: m and tau on all CA1 units, the population d2 uses
POP = os.environ.get("POP", "pyr")
_SFX = "" if POP == "pyr" else f"_{POP}"
LAB_SFX = "" if LABEL == "centre" else f"_{LABEL}"          # figure names: centre is unsuffixed
TAB_SFX = _SFX + ("_centre" if LABEL == "centre" else "")   # table: the unsuffixed name is whole-window
PANELS = [
    ("d2", "$d_2$ (2024 Euclidean, $p$=20, 40 ms)",
     "larger = FURTHER from criticality", "#c0392b"),
    (f"BR_{POP}_k200", f"branching ratio $m$ ({'pyr' if POP == 'pyr' else 'all units'}, 4 ms, $k_{{max}}$=200)",
     "larger = CLOSER to criticality", "#2c3e50"),
    (f"tau_ms_{POP}_k200", r"correlation time $\tau$ = $-\Delta t/\ln m$ (ms)",
     "larger = CLOSER to criticality", "#1f7a8c"),
]


def assert_local_path(path):
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        raise PermissionError(f"refusing to write to UNC path: {p}")
    if os.path.splitdrive(p)[0].rstrip(":").upper() not in {"C", "D"}:
        raise PermissionError(f"refusing to write outside C:/D:: {p}")
    return p


def paired(T, col, a, b):
    """Paired Wilcoxon across sessions having BOTH states. Session is the unit."""
    w = T.pivot(index="session", columns="state", values=col)
    if a not in w or b not in w:
        return dict(n=0)
    w = w[[a, b]].dropna()
    if len(w) < 3:
        return dict(n=len(w))
    d = w[a].to_numpy(float) - w[b].to_numpy(float)
    return dict(n=len(w), median_diff=float(np.median(d)),
                p=float(stats.wilcoxon(w[a], w[b]).pvalue),
                n_pos=int((d > 0).sum()))


# the two contrasts that carry the dissociation: each metric resolves the one the other
# cannot -- within sleep and within wake only, no sleep-vs-wake
KEY = [("NREM", "REM"), ("THETA", "nonTHETA")]


def contrast_table(T):
    con = []
    for col, name, _, _ in PANELS:
        for a, b in [("NREM", "REM"), ("THETA", "nonTHETA"),          # the two the figure shows
                     ("NREM", "THETA"), ("THETA", "REM")]:            # sleep-vs-wake, kept in the table only
            r = paired(T, col, a, b)
            con.append(dict(metric=col, contrast=f"{a}-{b}", **r))
    return pd.DataFrame(con)


def state_box_panel(ax, T, C, col, name, arrow, c, stats_offset=-30, fontsize=8.2):
    """Box per state (sessions having that state), one dot per session, sleep | wake divider,
    and the within-group paired contrasts in a box below the axis. Returns the stats box
    (excluded from layout; pass it to savefig's bbox_extra_artists)."""
    import matplotlib
    w = T.pivot(index="session", columns="state", values=col).reindex(columns=STATES)
    x = np.array(XPOS, float)
    data = [w[s].dropna().to_numpy(float) for s in STATES]
    bp = ax.boxplot(data, positions=x, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="k", lw=1.8),
                    whiskerprops=dict(color="k"), capprops=dict(color="k"))
    for patch in bp["boxes"]:
        patch.set_facecolor(matplotlib.colors.to_rgba(c, 0.35))
        patch.set_edgecolor("k")
    rng = np.random.default_rng(0)
    for i, v in enumerate(data):
        ax.scatter(x[i] + rng.uniform(-0.13, 0.13, v.size), v, s=12,
                   color=c, edgecolor="none", alpha=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in STATES], fontsize=8.5)
    ax.set_title(name, fontsize=9.5, pad=18)
    ax.text(0.5, 1.02, arrow, transform=ax.transAxes, fontsize=7.8,
            ha="center", va="bottom", color="0.35", style="italic")
    lines = []
    for a, b in KEY:
        r = C[(C.metric == col) & (C.contrast == f"{a}-{b}")].iloc[0]
        k = int(r.n_pos) if r.median_diff > 0 else int(r.n) - int(r.n_pos)
        sig = r.p < 0.05
        lines.append(("✓ " if sig else "✗ ") +
                     f"{CONTRAST_LABEL[f'{a}-{b}']}: p = {r.p:.4f}" +
                     (f", {k}/{int(r.n)}" if sig else "  (n.s.)"))
    n = int(w.notna().any(axis=1).sum())
    lines.append(f"n = {n} sessions, paired Wilcoxon")
    box = ax.annotate("\n".join(lines), xy=(0.5, 0), xycoords="axes fraction",
                      xytext=(0, stats_offset), textcoords="offset points",
                      ha="center", va="top", fontsize=fontsize, family="monospace",
                      bbox=dict(boxstyle="round,pad=0.45", fc="#f5f5f5",
                                ec="0.75", lw=0.8))
    box.set_in_layout(False)
    ax.axvline(np.mean(XPOS[1:3]), color="0.6", lw=0.8, ls=":")   # sleep | wake
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return box


def main():
    T = pd.read_csv(SRC)
    T = T[T.state.isin(STATES)].copy()
    print(f"{T.session.nunique()} sessions, {len(T)} session-state rows")

    # ---- table -----------------------------------------------------------
    rows = []
    for col, name, _, _ in PANELS:
        for s in STATES:
            v = T.loc[T.state == s, col].dropna()
            rows.append(dict(metric=col, label=name.split(" (")[0], state=s,
                             n_sessions=len(v), median=v.median(),
                             q25=v.quantile(.25), q75=v.quantile(.75),
                             rank_high_to_low=np.nan))
    S = pd.DataFrame(rows)
    for col in S.metric.unique():                 # rank states within each metric
        m = S.metric == col
        S.loc[m, "rank_high_to_low"] = (
            S.loc[m, "median"].rank(ascending=False).astype(int))

    C = contrast_table(T)

    os.makedirs(assert_local_path(LOGDIR), exist_ok=True)
    # a POP=all run writes its own table
    out = os.path.join(LOGDIR, f"letter_table_state_d2_br{TAB_SFX}.csv")
    with open(assert_local_path(out), "w", encoding="utf8", newline="") as f:
        f.write("# state-resolved medians, session = unit of analysis\n")
        S.to_csv(f, index=False)
        f.write("\n# paired Wilcoxon across sessions with both states\n")
        C.to_csv(f, index=False)
    print(f"table -> {out}")

    print("\nstate ranking, highest first")
    for col, name, _, _ in PANELS:
        order = S[S.metric == col].sort_values("median", ascending=False)
        print(f"  {col:18} " + "  >  ".join(
            f"{r.state} {r.median:.4g}" for r in order.itertuples()))
    print()
    print(C.to_string(index=False))

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    boxes = []
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.8))
    for ax, (col, name, arrow, c) in zip(axes, PANELS):
        boxes.append(state_box_panel(ax, T, C, col, name, arrow, c))
    axes[2].set_yscale("log")
    fig.suptitle(
        "State-resolved $d_2$, $m$ and $\\tau$ (R2.4 / R3.4); contrasts within sleep and within wake only\n"
        + ("wake-θ = window centre in a θ epoch (paper's Fig 1D), non-overlapping .mat windows"
           if LABEL == "centre" else "wake-θ = whole window inside θ (revision tiles)"), fontsize=10.5)
    fig.tight_layout(rect=(0, 0.22, 1, 0.95))
    # the 3-panel figure (with tau) goes to backup/
    out_dir = os.path.join(FIGDIR, "backup")
    os.makedirs(assert_local_path(out_dir), exist_ok=True)
    for ext in ("png", "svg"):
        p = assert_local_path(os.path.join(
            out_dir, f"r24_d2_vs_br_dissociation{_SFX}{LAB_SFX}.{ext}"))
        fig.savefig(p, dpi=200, bbox_inches="tight",
                    bbox_extra_artists=boxes + [fig._suptitle])
    print(f"\nfigure -> {out_dir}\\r24_d2_vs_br_dissociation{_SFX}{LAB_SFX}.png/.svg")


if __name__ == "__main__":
    main()
