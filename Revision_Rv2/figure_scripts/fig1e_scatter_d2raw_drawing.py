"""Drawing functions from analysis_scripts/notebooks/exploration/revision/ (fig1de_nonoverlap.py, fig1e_scatter_d2raw.py, run_fig1_metrics.py) used by the
figure notebooks. LOGDIR, FIGDIR: the folder of the notebook's tables, set by the notebook.
"""
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

LOGDIR = "."
FIGDIR = "."


# --- fig1e_scatter_d2raw.py
WAKE_PAL = {"Wake θ": "#d95f0e", "Wake non-θ": "#fdb863"}


# --- fig1e_scatter_d2raw.py
SLEEP_PAL = {"NREM": "#7b3294", "REM": "#d7b5d8"}


# --- fig1e_scatter_d2raw.py
WAKE_LINE = {"Wake θ": "#97550a", "Wake non-θ": "#b87f44"}


# --- fig1e_scatter_d2raw.py
SLEEP_LINE = {"NREM": "#56236a", "REM": "#976695"}


# --- fig1e_scatter_d2raw.py
MIN_WIN = 5


# --- fig1e_scatter_d2raw.py
WAKE = os.environ.get("WAKE", "task")


# --- fig1e_scatter_d2raw.py
META_MIN_N = 4


# --- fig1e_scatter_d2raw.py
def per_session(d):
    rows = []
    for s, g in d.groupby("animal_day"):
        if len(g) < MIN_WIN or g.mua_z.std() == 0:
            continue
        x, y = g.mua_z.to_numpy(float), g.d2.to_numpy(float)
        b = np.polyfit(x, y, 1)[0]
        lo, hi = np.percentile(x, [5, 95])
        rows.append(dict(session=s, n=len(g), r=pearsonr(x, y)[0], slope_per_sd=b, d2_change_5_95=b * (hi - lo),
                         d2_median=float(np.median(y))))
    return pd.DataFrame(rows)


# --- fig1e_scatter_d2raw.py
def meta_r(d, min_n=META_MIN_N):
    """Per-session correlations pooled across sessions, weighted by each session's window count (a session
    with few windows gives an unreliable per-session fit): Fisher z_i = arctanh(r_i) with variance 1 / (n_i - 3).
    Fixed effect: inverse-variance weighted mean of z_i. Random effects (DerSimonian-Laird, primary): weights
    1 / (v_i + tau^2), so a few large sessions cannot carry the result if sessions genuinely differ. Two-sided z
    tests; the pooled r = tanh(pooled z)."""
    from scipy.stats import norm
    rows = []
    for s, g in d.groupby("animal_day"):
        if len(g) < min_n or g.mua_z.std() == 0 or g.d2.std() == 0:
            continue
        rows.append((len(g), pearsonr(g.mua_z.to_numpy(float), g.d2.to_numpy(float))[0]))
    if len(rows) < 2:
        return {}
    n = np.array([a for a, _ in rows], float)
    y = np.arctanh(np.clip(np.array([b for _, b in rows], float), -0.999999, 0.999999))
    v = 1.0 / (n - 3)
    w = 1.0 / v
    y_fe = float(np.sum(w * y) / np.sum(w)); se_fe = float(np.sqrt(1.0 / np.sum(w)))
    k = len(y)
    Q = float(np.sum(w * (y - y_fe) ** 2))
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    ws = 1.0 / (v + tau2)
    y_re = float(np.sum(ws * y) / np.sum(ws)); se_re = float(np.sqrt(1.0 / np.sum(ws)))
    return dict(meta_k=k, meta_windows=int(n.sum()), meta_min_n=min_n,
                r_fe=float(np.tanh(y_fe)), p_fe=float(2 * norm.sf(abs(y_fe / se_fe))),
                r_re=float(np.tanh(y_re)), r_re_lo=float(np.tanh(y_re - 1.96 * se_re)),
                r_re_hi=float(np.tanh(y_re + 1.96 * se_re)), z_re=float(y_re / se_re),
                p_re=float(2 * norm.sf(abs(y_re / se_re))), tau2=tau2, Q=Q,
                I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0)


# --- fig1e_scatter_d2raw.py
def summarise(state, d, P):
    r = P.r.to_numpy()
    b, a = np.polyfit(d.mua_z, d.d2, 1)
    s = dict(state=state, n_windows=len(d), n_sessions_all=int(d.animal_day.nunique()), n_sessions=len(P),
             r_fisher_mean=float(np.tanh(np.arctanh(np.clip(r, -0.999999, 0.999999)).mean())),
             n_pos=int((r > 0).sum()), p_signed_rank=float(wilcoxon(r).pvalue) if len(P) >= 3 else np.nan,
             slope_per_sd_median=float(P.slope_per_sd.median()), d2_change_5_95_median=float(P.d2_change_5_95.median()),
             d2_median_of_sessions=float(P.d2_median.median()),
             slope_pooled=float(b), intercept_pooled=float(a), r_pooled=float(pearsonr(d.mua_z, d.d2)[0]))
    s.update(meta_r(d))
    if "r_re" in s:
        print(f"  {state}: per-session Pearson r pooled by random-effects meta-analysis, r = {s['r_re']:+.2f}, "
              f"p = {s['p_re']:.2g}")
    return s


# --- fig1e_scatter_d2raw.py
def local_limits(x, pad_frac=0.1):
    lo, hi = np.percentile(np.asarray(x, float), [1, 99])
    pad = pad_frac * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


# --- run_fig1_metrics.py
WRITABLE_DRIVES = {"C", "D"}


# --- run_fig1_metrics.py
def assert_local_path(path):
    """Raise unless `path` is on a local writable drive. Called before every write."""
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        raise PermissionError(f"refusing to write to UNC path: {p}")
    drive = os.path.splitdrive(p)[0].rstrip(":").upper()
    if drive not in WRITABLE_DRIVES:
        raise PermissionError(
            f"refusing to write outside {sorted(WRITABLE_DRIVES)}: {p}"
        )
    return p


# --- fig1de_nonoverlap.py
NONOVL_NOTE = ("disjoint windows: per session, a window is kept only if its centre is >= 30 s from the last kept "
               "window of any state (no two share data)")


# --- fig1e_scatter_d2raw.py (a drawing section, as a function)
def draw_main(T):
    """Draw the Fig 1e d2-vs-MUA scatter (wake theta/non-theta, NREM, REM) with per-session slopes, from table T."""
    task = WAKE != "all"
    wake = (lambda st: T[(T.group == st) & T.epoch_name.notna()]) if task else (lambda st: T[T.group == st])
    panels = [(("Task wake (T-maze)" if task else "Wake (all)") + "\nWake θ + Wake non-θ", [
                  ("Wake non-θ", wake("Wake non-θ"), WAKE_PAL, WAKE_LINE, 0.1),
                  ("Wake θ", wake("Wake θ"), WAKE_PAL, WAKE_LINE, 0.1)]),
              ("Sleep\nNREM + REM", [
                  ("NREM", T[T.group == "NREM"], SLEEP_PAL, SLEEP_LINE, 0.3),
                  ("REM", T[T.group.isin(["REM", "REM4"])], SLEEP_PAL, SLEEP_LINE, 0.05)])]
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "axes.linewidth": 1.5, "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2))
    stats, persess = [], []
    print("f (left): d2 vs MUA (z within session), non-overlapping windows")
    for ax, (title, groups) in zip(axes, panels):
        handles = []
        for st, d, pal, lpal, pad in groups:
            d = d.dropna(subset=["d2", "mua_z"])
            ax.scatter(d.mua_z, d.d2, s=4, alpha=0.8, color=pal[st], edgecolors="none", rasterized=True)
            P = per_session(d)
            s = summarise(st, d, P)
            stats.append(s)
            persess.append(P.assign(state=st))
            # Line = the median within-session slope through the state's centroid (descriptive).
            # REM: points only (125 disjoint windows, 1-14 per session) -- no trend line.
            if st != "REM":
                lo, hi = local_limits(d.mua_z, pad)
                xs = np.linspace(lo, hi, 200)
                cx, cy = float(d.mua_z.mean()), float(d.d2.mean())
                ax.plot(xs, cy + s["slope_per_sd_median"] * (xs - cx), color=lpal[st], linestyle="--", lw=1.8)
                handles += [Line2D([0], [0], marker="o", linestyle="None", color=pal[st], label=st),
                            Line2D([0], [0], color=pal[st], linestyle="--", lw=1.8,
                                   label=f"r = {s['r_fisher_mean']:+.2f}, {s['n_pos']}/{s['n_sessions']} sessions, "
                                         f"p = {s['p_signed_rank']:.2g}")]
            else:
                handles += [Line2D([0], [0], marker="o", linestyle="None", color=pal[st],
                                   label=f"REM ({len(d)} windows, {d.animal_day.nunique()} sessions): no line, "
                                         f"{s['n_pos']}/{s['n_sessions']} > 0")]
        ax.set_title(title)
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
        ax.set_xlabel("Mean population firing rate (z, within session)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(False)
    fig.supylabel("d2 distance (2024 Euclidean)")
    fig.tight_layout()
    import textwrap
    note = ("Fig 1e scatter: d2 raw (p = 20, 40 ms, full window), MUA z-scored within session x Wake/Sleep; "
            f"dashed = median within-session slope through the state centroid (descriptive); r = per-session Pearson, "
            f"Fisher mean, sessions with >= {MIN_WIN} windows, signed-rank; {NONOVL_NOTE}")
    fig.text(0.01, -0.02, textwrap.fill(note, 170), fontsize=7, color="0.35", ha="left", va="top")
    sfx = "" if task else "_allwake"
    out = os.path.join(FIGDIR, "fig1e_scatter_d2raw_muaz_nonoverlap" + sfx)
    fig.savefig(assert_local_path(out + ".svg"), dpi=300, bbox_inches="tight")
    fig.savefig(assert_local_path(out + ".png"), dpi=200, bbox_inches="tight")
    pd.DataFrame(stats).to_csv(assert_local_path(os.path.join(LOGDIR, f"fig1e_scatter_d2raw_stats{sfx}.csv")), index=False)
    pd.concat(persess, ignore_index=True).to_csv(
        assert_local_path(os.path.join(LOGDIR, f"fig1e_scatter_d2raw_per_session{sfx}.csv")), index=False)
