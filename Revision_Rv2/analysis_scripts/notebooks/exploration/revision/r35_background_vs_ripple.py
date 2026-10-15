"""
r35_background_vs_ripple.py -- d2 under ripple-time exclusion from the fitting window,
compared with d2 on the full window.

d2 characterises the background network regime rather than within-ripple dynamics; SWRs
and replay are events the regime permits rather than what d2 measures directly. This
compares d2 computed with ripple time excised from the fitting window against d2 on the
full window, and the SWR-rate / replay relationship under both.

Every session carries a ripple-ablated d2 (*_CA1_D2_time_resolved_rippleablate.mat,
computed on the same 30 s / 2 s window grid as the published *_CA1_D2_time_resolved.mat):
ripple time intervals are subtracted from each 30 s window and the surviving ripple-free
subwindows are glued and fitted together (MATLAB producer:
notebooks/helper/time_resolved_d2_across_state_ripple_ablate.m); only NREM differs from
the standard file. Both d2 arms are read from the .mat files rather than from the
pickles, so every session is on the same footing. The script also reports the percentage
of each fitting window removed by the ablation.

Two quantities: (1) r(d2, d2_ablate) per session; (2) the SWR-rate / replay-probability
relationship on ripple-free d2. Also re-runs the GLM ablation (r35_glm_ablation.py) with
d2_ablate as the target, on identical windows and folds, alongside the standard d2.

NREM, non-overlapping windows (every 15th of the 2 s sliding grid), Fig. 2d sessions,
session as the unit of analysis. Metric is d2 (2024 Euclidean, p=20) in both arms.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/r35_background_vs_ripple.csv
    -> reports/revision_figs/fig2_background_vs_ripple.png / .svg
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r35_glm_ablation import (BASEPATH, LOGDIR, PREDICTORS, assert_local_path,  # noqa: E402
                              build, cv_r2)

FIGDIR = r"D:\Documents\GitHub\Sleep_criticaltiy\reports\revision_figs"
N_FOLD = 5
N_SHUF = 200
SEED = 20260819


def load_d2_mat(basepath, tag):
    """-> DataFrame(center, d2, d2_err) for NREM. `tag` selects the variant."""
    day = os.path.basename(basepath.rstrip("\\/"))
    f = os.path.join(basepath, "complex_system", f"{day}_CA1_D2_time_resolved{tag}.mat")
    if not os.path.exists(f):
        return None
    m = loadmat(f, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    rows = []
    for r in m["results"]:
        st = str(r["state"])
        if "NREM" not in st.upper():
            continue
        rows.append(pd.DataFrame({
            "center": np.atleast_1d(r["center"]).astype(float),
            "d2_x": np.atleast_1d(r["d2"]).astype(float),
            "d2_err_x": (np.atleast_1d(r["d2_err"]).astype(float)
                         if "d2_err" in r else np.nan)}))
    if not rows:
        return None
    return pd.concat(rows).sort_values("center").reset_index(drop=True)


def fisher(v):
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan, 0
    z = np.arctanh(np.clip(v, -0.999999, 0.999999))
    return (float(np.tanh(z.mean())),
            float(z.std(ddof=1) / np.sqrt(z.size)) if z.size > 1 else np.nan, z.size)


def frac_window_removed(basepath, centers, win=30.0):
    """Fraction of each 30 s fitting window excised by ripple subtraction.

    Mirrors `fast_subtract_intervals(curr_win, ripple_iv)` in the MATLAB producer, which
    uses the RAW `ripples.timestamps` (not NREM-clipped)."""
    from neuro_py.io import loading
    rip = loading.load_ripples_events(basepath)
    iv = np.sort(np.array(rip[["start", "stop"]], float), axis=0)
    a, b = iv[:, 0], iv[:, 1]
    out = np.zeros(len(centers))
    for i, c in enumerate(centers):
        w0, w1 = c - win / 2, c + win / 2
        lo = np.maximum(a, w0)
        hi = np.minimum(b, w1)
        ov = np.clip(hi - lo, 0, None)
        out[i] = ov.sum() / win
    return out


# The KL pair comes from r35_kl_rippleablate.py, which computes the FULL-window and
# RIPPLE-EXCISED KL with the same Yule-Walker estimator on the same windows, so the KL
# arms are paired exactly as the Euclidean arms are.
KL_CSV = os.path.join(LOGDIR, "r35_kl_rippleablate.csv")
KL_ORDER = 20
ARMS = [("d2_std", "Euclidean, full"), ("d2_ablate", "Euclidean, ripple-excised"),
        ("d2_kl", "KL, full"), ("d2_kl_abl", "KL, ripple-excised")]


def kl_centers(session):
    """KL window centres for a session (w + 15) plus both KL arms, or (None, None) if
    the session has no KL windows — only the Fig-1 18-session set does, so PPP7_day8
    returns None."""
    if not os.path.exists(KL_CSV):
        return None, None
    K = pd.read_csv(KL_CSV)
    if "session" not in K.columns:
        K["session"] = K["animal"].astype(str) + "_" + K["day"].astype(str)
    k = K[K["session"] == session]
    if not len(k):
        return None, None
    k = k[["w", "kl_full", "kl_abl"]].copy()
    k["center_s"] = k["w"].astype(float) + 15.0
    k = k.rename(columns={"kl_full": "d2_kl", "kl_abl": "d2_kl_abl"})
    k = k[["center_s", "d2_kl", "d2_kl_abl"]].sort_values("center_s")
    return k["center_s"].to_numpy(float), k


def main():
    rng = np.random.default_rng(SEED)
    rows, frames = [], []
    print("session       n_win   r(std, ablate)  med std  med abl  %win removed  pkl=abl?")
    for s, bp in BASEPATH.items():
        kc, ktab = kl_centers(s)
        try:
            W = build(s, extra_centers=kc)     # predictors on non-overlap + KL centres
        except Exception as e:                                 # noqa: BLE001
            print(f"  [SKIP] {s}: {type(e).__name__}: {e}")
            continue
        A = load_d2_mat(bp, "_rippleablate")
        S = load_d2_mat(bp, "")
        if A is None or S is None:
            print(f"  [SKIP] {s}: missing a d2 mat")
            continue
        # BOTH arms come from the mats; the pickle's d2 is only used to flag provenance
        m = pd.merge_asof(W.sort_values("center_s"),
                          S.rename(columns={"center": "center_s", "d2_x": "d2_std",
                                            "d2_err_x": "d2_std_err"}),
                          on="center_s", tolerance=0.6, direction="nearest")
        m = pd.merge_asof(m.sort_values("center_s"),
                          A.rename(columns={"center": "center_s", "d2_x": "d2_ablate",
                                            "d2_err_x": "d2_ablate_err"}),
                          on="center_s", tolerance=0.6, direction="nearest")
        if ktab is not None:
            m = pd.merge_asof(m.sort_values("center_s"), ktab,
                              on="center_s", tolerance=1.0, direction="nearest")
        else:
            m["d2_kl"] = np.nan
            m["d2_kl_abl"] = np.nan
        m["frac_removed"] = frac_window_removed(bp, m["center_s"].to_numpy(float))
        m["session"] = s
        # which mat does the published pickle actually hold?
        pkl_is_abl = np.mean(np.isclose(m["d2"], m["d2_ablate"], equal_nan=True))
        pkl_is_std = np.mean(np.isclose(m["d2"], m["d2_std"], equal_nan=True))
        frames.append(m)
        v = m[["d2_std", "d2_ablate"]].dropna()
        r = (stats.spearmanr(v["d2_std"], v["d2_ablate"]).statistic
             if len(v) >= 10 else np.nan)
        tag = "ABLATED!" if pkl_is_abl > pkl_is_std else "std"
        print(f"{s:13}{len(v):6d}{r:16.3f}{v['d2_std'].median():9.4f}"
              f"{v['d2_ablate'].median():9.4f}{m.frac_removed.median():13.1%}   {tag}")

        rec = dict(session=s, n=len(v), r_d2_vs_ablate=r,
                   med_d2=v["d2_std"].median(), med_d2_ablate=v["d2_ablate"].median(),
                   frac_removed=m.frac_removed.median(),
                   frac_removed_max=m.frac_removed.max(),
                   pkl_matches_ablate=pkl_is_abl, pkl_matches_std=pkl_is_std)
        m = m.rename(columns={"d2": "d2_pkl"})
        rec["n_kl"] = int(m["d2_kl"].notna().sum()) if "d2_kl" in m else 0
        # every arm: bivariate correlations AND the GLM ablation, same predictors,
        # same folds, so the three targets are directly comparable
        cols = [c for c, _ in PREDICTORS]
        for tgt, _lbl in ARMS:
            if tgt not in m.columns or m[tgt].notna().sum() < 10:
                continue
            for c, _ in PREDICTORS:
                sub = m[[tgt, c]].dropna()
                rec[f"r_{tgt}_{c}"] = (stats.spearmanr(sub[tgt], sub[c]).statistic
                                       if len(sub) >= 10 else np.nan)
            sub = m[[tgt] + cols].dropna()
            if len(sub) >= 40:
                y = sub[tgt].to_numpy(float)
                X = sub[cols].to_numpy(float)
                full = cv_r2(X, y, scheme="interleaved")
                rec[f"r2_full_{tgt}"] = full
                for j2, (c, _) in enumerate(PREDICTORS):
                    rec[f"gain_{tgt}_{c}"] = full - cv_r2(X, y, ablate=j2,
                                                          scheme="interleaved")
        rows.append(rec)

    D = pd.DataFrame(rows)
    assert_local_path(LOGDIR)
    D.to_csv(assert_local_path(os.path.join(
        LOGDIR, "r35_background_vs_ripple.csv")), index=False)
    pd.concat(frames, ignore_index=True).to_csv(assert_local_path(
        os.path.join(LOGDIR, "r35_background_vs_ripple_windows.csv")), index=False)

    print(f"\n{'='*78}\nIS d2 THE BACKGROUND, OR THE RIPPLES?   n={len(D)} sessions, "
          f"non-overlapping NREM windows\n{'='*78}")
    r, se, n = fisher(D.r_d2_vs_ablate)
    print(f"\n  1. r(d2, ripple-ablated d2) = {r:+.3f} +/- {se:.3f}   "
          f"(min {D.r_d2_vs_ablate.min():+.3f}, max {D.r_d2_vs_ablate.max():+.3f})")
    print(f"     median d2 {D.med_d2.median():.4f} vs ablated "
          f"{D.med_d2_ablate.median():.4f}")

    print("\n  2. correlation of each predictor with each d2 definition")
    hdr = "".join(f"{l:>20}" for _, l in ARMS)
    print(f"     {'predictor':22}{hdr}   p(std vs excised)")
    for c, lbl in PREDICTORS:
        cells = ""
        a, b = D.get(f"r_d2_std_{c}"), D.get(f"r_d2_ablate_{c}")
        for tgt, _ in ARMS:
            col = D.get(f"r_{tgt}_{c}")
            cells += (f"{fisher(col)[0]:+20.3f}" if col is not None else f"{'n/a':>20}")
        ok = a.notna() & b.notna()
        pw = (stats.wilcoxon(np.arctanh(a[ok].clip(-0.999, 0.999)),
                             np.arctanh(b[ok].clip(-0.999, 0.999))).pvalue
              if ok.sum() >= 3 else float("nan"))
        print(f"     {lbl:22}{cells}{pw:19.4f}")
    print(f"     {'(n sessions)':22}" + "".join(
        f"{int(D[f'r_{t}_MUA'].notna().sum()):>20}" for t, _ in ARMS
        if f'r_{t}_MUA' in D.columns))

    print("\n  3. GLM ablation, same predictors and folds, per d2 definition")
    for tgt, lbl in ARMS:
        if f"r2_full_{tgt}" not in D.columns:
            continue
        print(f"     --- {lbl}: full-model R2 = {D[f'r2_full_{tgt}'].median():+.4f}")
        for c, cl in PREDICTORS:
            v = D[f"gain_{tgt}_{c}"].dropna()
            if not len(v):
                continue
            pw = stats.wilcoxon(v).pvalue if len(v) >= 3 else float("nan")
            print(f"         {cl:22}{np.median(v):+9.4f}  p={pw:.4f}  "
                  f"{int((v>0).sum())}/{len(v)}")
    plot(D)


def plot(D):
    """Main panel = predictor correlations, one bar per d2 definition.

    The title states the comparison directly: MUA vs SWR-rate once ripple time is
    excised from the fitting window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "svg.fonttype": "none", "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "axes.unicode_minus": False, "axes.linewidth": 1.2,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    })
    COL = {"d2_std": "#b0b7bc", "d2_ablate": "#691c1b",
           "d2_kl": "#7fb3c8", "d2_kl_abl": "#1f6f8b"}
    arms = [(t, l) for t, l in ARMS if f"r_{t}_MUA" in D.columns]
    y = np.arange(len(PREDICTORS))

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.3),
                             gridspec_kw={"width_ratios": [1.6, 1.1, 0.85]})

    ax = axes[0]
    h = 0.8 / max(len(arms), 1)
    for k, (tgt, lbl) in enumerate(arms):
        r = [fisher(D[f"r_{tgt}_{c}"])[0] for c, _ in PREDICTORS]
        e = [fisher(D[f"r_{tgt}_{c}"])[1] for c, _ in PREDICTORS]
        n = int(D[f"r_{tgt}_MUA"].notna().sum())
        ax.barh(y + (k - (len(arms) - 1) / 2) * h, r, h, xerr=e, color=COL[tgt],
                error_kw=dict(lw=1.0), label=f"{lbl}  (n={n})")
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l in PREDICTORS], fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_xlabel("Spearman r with d2   (Fisher-z mean $\\pm$ SEM across sessions)")
    ax.set_title("RIPPLE TIME EXCISED from the fitting window\n"
                 "the MUA confound collapses; the SWR-rate relationship survives",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = axes[1]
    have = [(t, l) for t, l in arms if f"r2_full_{t}" in D.columns]
    h2 = 0.8 / max(len(have), 1)
    for k, (tgt, lbl) in enumerate(have):
        g = [np.median(D[f"gain_{tgt}_{c}"].dropna())
             if f"gain_{tgt}_{c}" in D.columns else np.nan for c, _ in PREDICTORS]
        ax.barh(y + (k - (len(have) - 1) / 2) * h2, g, h2, color=COL[tgt], label=lbl)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_xlabel("$\\Delta R^2$ when ablated")
    ax.set_title("GLM ablation\n(unique contribution)", fontsize=10)

    ax = axes[2]
    x = np.arange(len(D))
    ax.bar(x, D.frac_removed * 100, color="#2c3e50")
    ax.set_xticks(x)
    ax.set_xticklabels(D.session, rotation=90, fontsize=7)
    ax.set_ylabel("% of fitting window excised")
    ax.set_title(f"median {D.frac_removed.median():.1%} of each\n30 s window is ripple",
                 fontsize=10)

    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    assert_local_path(FIGDIR)
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(
            FIGDIR, f"fig2_background_vs_ripple.{ext}")), dpi=300, bbox_inches="tight")
    print(f"\nfigure -> {FIGDIR}\\fig2_background_vs_ripple.png/.svg")
    print(f"tables -> {LOGDIR}\\r35_background_vs_ripple*.csv")


if __name__ == "__main__":
    main()
