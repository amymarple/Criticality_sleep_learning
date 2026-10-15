"""fig3g_reproduce.py -- Figure 3g statistic (manuscript) and seven model/window-source
variants, computed from stored artifacts.

Published pipeline, from notebooks/figures/Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar.ipynb:
  * load data/ripple_barr/manipulation/sncg/*.pkl (7 files; columns epoch, d2) with the
    notebook's load_plot_df, in sorted-glob order;
  * rng = default_rng(0); per file x epoch draw N_target = 1000 rows, with replacement when the
    epoch has fewer;
  * MixedLM d2 ~ C(epoch) + C(animal), groups = animal_day, re_formula "1", lbfgs.

Model variants (each row changes one thing relative to the row above):
  A  published: pkl rows, resampled to 1000 per file x epoch, LMM on windows
  B  pkl rows, no resampling (the real windows), same LMM
  C  pkl rows, session-level: median per session x epoch, paired Wilcoxon / sign (n = 7)
  D  pkl's own d2 source (.mat) but all postsleep windows instead of the first N: the 80 ms /
     60 s CA1+CA2 .mat for six sessions, the CA1 40 ms / 30 s .mat for SNCG06_day27, NREM
     windows whose centre lies in the presleep / postsleep epoch of load_epoch, LMM without
     resampling
  E  same as D at the session level
  F  same as D but resampled to 1000 per file x epoch
  G  presleep-all vs the last N postsleep windows (mirror of the pkl's first-N truncation)

    -> data/revision/log/fig3g_reproduce.csv
"""
from __future__ import annotations

import glob
import os
import pickle
import re
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest, wilcoxon  # noqa: E402

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import opto_cck_metrics as ocm  # noqa: E402
from opto_sncg_metrics import SNCG_SESSIONS, load_d2_80ms, load_epochs  # noqa: E402
from r35_background_vs_ripple import load_d2_mat  # noqa: E402
from run_fig1_metrics import assert_local_path  # noqa: E402

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
SN_DIR = os.path.join(REPO, "data", "ripple_barr", "manipulation", "sncg")
ORDER = ["presleep", "postsleep"]
N_TARGET = 1000
STORED = dict(coef=0.0055, ci_lo=0.0047, ci_hi=0.0063, p=1.91e-39)


# ---- the notebook's own loader function ------------------------------------------------
def load_plot_df(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, dict):
        df = obj.get("plot_df")
        if not isinstance(df, pd.DataFrame):
            df = next((v for v in obj.values() if isinstance(v, pd.DataFrame)), None)
    else:
        raise ValueError("unsupported PKL content")
    stem = os.path.splitext(os.path.basename(pkl_path))[0]
    df["animal_day"] = stem
    m = re.match(r"(?P<animal>.+)_day(?P<day>\d+)$", stem, flags=re.IGNORECASE)
    df["animal"] = m.group("animal") if m else stem
    df["day"] = int(m.group("day")) if m else pd.NA
    return df


def resample(plot_df, seed=0):
    """Per file x epoch, N_TARGET rows, with replacement if short; file order = glob order."""
    rng = np.random.default_rng(seed)
    rows = []
    for f, df_f in plot_df.groupby("animal_day", sort=False):
        for ep in ORDER:
            idx = df_f.index[df_f["epoch"] == ep].to_numpy()
            if idx.size == 0:
                continue
            choice = rng.choice(idx, size=N_TARGET, replace=idx.size < N_TARGET)
            rows.append(df_f.loc[choice])
    return pd.concat(rows, ignore_index=True)


def lmm(df):
    """d2 ~ C(epoch) + C(animal), random intercept per animal_day."""
    import statsmodels.formula.api as smf
    d = df.copy()
    d["animal"] = d["animal"].astype("category")
    d["epoch"] = pd.Categorical(d["epoch"], categories=ORDER, ordered=True)
    d = d.dropna(subset=["d2", "epoch", "animal", "animal_day"]).reset_index(drop=True)
    res = smf.mixedlm("d2 ~ C(epoch) + C(animal)", data=d, groups=d["animal_day"], re_formula="1").fit(method="lbfgs")
    t = "C(epoch)[T.postsleep]"
    lo, hi = res.conf_int().loc[t].to_list()
    return dict(coef=float(res.params[t]), ci_lo=lo, ci_hi=hi, p=float(res.pvalues[t]),
                n_rows=int(res.nobs), n_groups=int(d.animal_day.nunique()),
                re_var=float(np.diag(np.atleast_2d(res.cov_re))[0]))


def session_level(df):
    med = df.groupby(["animal_day", "epoch"], observed=True).d2.median().unstack("epoch")
    d = (med["postsleep"] - med["presleep"]).dropna()
    return dict(n_sessions=len(d), median_delta=float(d.median()), n_up=int((d > 0).sum()),
                p_sign=float(binomtest(int((d > 0).sum()), len(d), 0.5).pvalue),
                p_wilcoxon=float(wilcoxon(med.loc[d.index, "postsleep"], med.loc[d.index, "presleep"]).pvalue),
                per_session="; ".join(f"{k}:{v:+.4f}" for k, v in d.items()))


def mat_epoch_series(bp, tag):
    """The pkl's own d2 source, all NREM windows per epoch."""
    if tag == "SNCG06_day27":
        v = load_d2_mat(bp, "").rename(columns={"center": "center_s", "d2_x": "d2"})
        src = "CA1 40 ms / 30 s"
    else:
        v = load_d2_80ms(bp)
        src = "CA1+CA2 80 ms / 60 s"
    ep = load_epochs(bp)
    out = {}
    for key in ORDER:
        iv = ep[key]
        out[key] = v[(v.center_s >= iv[0, 0]) & (v.center_s < iv[-1, 1])].d2.to_numpy()
    return out, src


def frame(series_by_session, mode="all"):
    rows = []
    for tag, (ser, _) in series_by_session.items():
        pre, post = ser["presleep"], ser["postsleep"]
        n = min(len(pre), len(post))
        if mode == "firstN":
            pre, post = pre[:n], post[:n]
        elif mode == "lastN":
            pre, post = pre[:n], post[-n:]
        for ep, arr in (("presleep", pre), ("postsleep", post)):
            rows.append(pd.DataFrame({"epoch": ep, "d2": arr, "animal_day": tag, "animal": tag.split("_")[0]}))
    return pd.concat(rows, ignore_index=True)


def main():
    out = []
    # ---------------- published pipeline ----------------
    pkls = sorted(glob.glob(os.path.join(SN_DIR, "*.pkl")))
    pooled = pd.concat([load_plot_df(p) for p in pkls], ignore_index=True)
    plot_df = pooled[pooled["epoch"].isin(ORDER)].copy()
    plot_df["epoch"] = pd.Categorical(plot_df["epoch"], categories=ORDER, ordered=True)
    balanced = resample(plot_df, seed=0)
    A = lmm(balanced)
    print("=== Stage 1: published pipeline from the pickles ===")
    print(f"  files: {[os.path.basename(p) for p in pkls]}")
    print(f"  this run : coef {A['coef']:.4f}  CI [{A['ci_lo']:.4f}, {A['ci_hi']:.4f}]  p = {A['p']:.3g}  "
          f"(n_rows {A['n_rows']}, groups {A['n_groups']}, RE var {A['re_var']:.2e})")
    print(f"  stored   : coef {STORED['coef']:.4f}  CI [{STORED['ci_lo']:.4f}, {STORED['ci_hi']:.4f}]  p = {STORED['p']:.3g}")
    ok = abs(A["coef"] - STORED["coef"]) < 5e-4 and abs(np.log10(A["p"]) - np.log10(STORED["p"])) < 1.0
    print(f"  REPRODUCED: {ok}")
    out.append(dict(stage="A", what="published: pkl rows resampled to 1000/file x epoch, window LMM", **A, **session_level(balanced)))

    # ---------------- model-variant ladder (B-G) ----------------
    B = lmm(plot_df)
    out.append(dict(stage="B", what="pkl rows, no resampling, window LMM", **B, **session_level(plot_df)))
    out.append(dict(stage="C", what="pkl rows, session level (medians, paired)", **session_level(plot_df)))

    series, srcs = {}, {}
    for bp in SNCG_SESSIONS:
        tag = ocm.session_tag(bp)
        ser, src = mat_epoch_series(bp, tag)
        series[tag] = (ser, src)
        pk = plot_df[plot_df.animal_day == tag]
        for key in ORDER:
            pv = pk[pk.epoch == key].d2.to_numpy()
            assert len(ser[key]) >= len(pv) and np.allclose(ser[key][:len(pv)], pv), f"{tag} {key}: pkl != first-N of its .mat"
        print(f"  {tag:12s} pkl == first-N of {src:22s} | windows pre {len(ser['presleep']):5d} post {len(ser['postsleep']):5d} "
              f"(pkl {len(pk[pk.epoch=='presleep']):4d} / {len(pk[pk.epoch=='postsleep']):4d})")

    for stage, mode, what in (("D", "all", ".mat source, ALL postsleep windows, window LMM (no resampling)"),
                              ("G", "lastN", ".mat source, presleep vs LAST-N postsleep windows, window LMM")):
        df = frame(series, mode)
        out.append(dict(stage=stage, what=what, **lmm(df), **session_level(df)))
        if stage == "D":
            out.append(dict(stage="E", what=".mat source, all windows, session level", **session_level(df)))
            out.append(dict(stage="F", what=".mat source, all windows, resampled to 1000/file x epoch, window LMM",
                            **lmm(resample(df, seed=0)), **session_level(resample(df, seed=0))))
    chk = frame(series, "firstN")
    out.append(dict(stage="A'", what=".mat source cut to first-N (should equal B)", **lmm(chk), **session_level(chk)))

    O = pd.DataFrame(out)
    O = O.set_index("stage").loc[["A", "A'", "B", "C", "D", "E", "F", "G"]].reset_index()
    O.to_csv(assert_local_path(os.path.join(ocm.LOGDIR, "fig3g_reproduce.csv")), index=False)
    pd.set_option("display.width", 250)
    cols = ["stage", "what", "n_rows", "coef", "p", "n_sessions", "median_delta", "n_up", "p_sign", "p_wilcoxon"]
    print("\n=== ladder ===")
    print(O[cols].round(5).to_string(index=False))
    print("\nper-session deltas (postsleep - presleep):")
    for _, r in O.iterrows():
        print(f"  {r.stage:2s} {r.per_session}")
    print("\nwrote fig3g_reproduce.csv")


if __name__ == "__main__":
    main()
