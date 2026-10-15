"""
r35_replay_full_covariates.py -- d2's relation to replay conditioned on all the
event-classifier covariates, not only the replay-ablation pair.

The replay ablation (r35_replay_ablation_lrt.py) conditions on Fig 2f's two covariates,
background MUA and across-neuron FR_std. The event-type classifier uses a different four:
MUA, MUA Fano, population synchrony and E/I. This script puts d2 into the replay model
with that four-covariate set, and with all six combined, so both covariate sets are
conditioned on the same things.

Fano, synchrony and E/I are computed on the ripple-excised part of each replay window by
`r16_event_excised_windows.metrics`, the same routine used for the event-type classifier,
with the ripple intervals as the events. Its MUA_bg is cross-checked against the replay
builder's MUA_bg. Windows, sessions and d2 (py_excised, 2024 Euclidean, p=20) are those of
`r35_extended_windows.csv`.

Model: replay_count of n_swr, binomial, pooled with session fixed effects, predictors
z-scored within session. Reported for each predictor: drop-one LR chi2, cluster-robust z,
and the per-session sign count with signed-rank.

    -> data/revision/log/r35_replay_full_covariates_windows.csv
    -> data/revision/log/r35_replay_full_covariates.csv
"""
from __future__ import annotations

import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
import statsmodels.api as sm                                          # noqa: E402
from scipy import stats                                               # noqa: E402

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                        # noqa: E402
from fig1_metrics import MIN_FIRING_RATE                              # noqa: E402
from r35_extend_replay_sessions import SESSIONS, EXCLUDE, LOGDIR, WIN_S  # noqa: E402
from r16_event_excised_windows import metrics, intervals              # noqa: E402
from r35_replay_ablation_lrt import glm, dev_expl, zs                 # noqa: E402

SRC = os.path.join(LOGDIR, "r35_extended_windows.csv")
OUT_WIN = os.path.join(LOGDIR, "r35_replay_full_covariates_windows.csv")
OUT = os.path.join(LOGDIR, "r35_replay_full_covariates.csv")
SETS = {
    "Fig 2F set (6.2)":        ["py_excised", "MUA_bg", "FR_std_bg"],
    "classifier set (17.4)":   ["py_excised", "MUA_bg", "fano_bg", "sync_bg", "EI_bg"],
    "all six":                 ["py_excised", "MUA_bg", "FR_std_bg", "fano_bg", "sync_bg", "EI_bg"],
}
LABEL = {"py_excised": "d2", "MUA_bg": "MUA", "FR_std_bg": "FR std", "fano_bg": "MUA Fano",
         "sync_bg": "synchrony", "EI_bg": "E/I"}


def build():
    from neuro_py.io import loading
    W = pd.read_csv(SRC)
    W = W[~W.session.isin(EXCLUDE)]
    out, t0 = [], time.time()
    for sess, g in W.groupby("session", sort=False):
        bp, _ = SESSIONS[sess]
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        units = [np.sort(np.asarray(u, float)) for u in np.asarray(st.data, dtype=object)[keep]]
        ct = np.asarray(cm.putativeCellType.astype(str))[keep]
        e_sp = np.sort(np.concatenate([u for u, t in zip(units, ct)
                                       if t == "Pyramidal Cell"] or [np.zeros(0)]))
        i_sp = np.sort(np.concatenate([u for u, t in zip(units, ct)
                                       if "Interneuron" in t] or [np.zeros(0)]))
        n_e = int((ct == "Pyramidal Cell").sum())
        n_i = int(sum("Interneuron" in t for t in ct))
        rip = intervals(loading, bp, "load_ripples_events")
        n_ok = 0
        for r in g.itertuples(index=False):
            m = metrics(units, e_sp, i_sp, n_e, n_i, r.center_s - WIN_S / 2, rip)
            if m is None:
                continue
            n_ok += 1
            out.append(dict(session=sess, center_s=r.center_s, fano_bg=m["fano"],
                            sync_bg=m["sync"], EI_bg=m["EI"], MUA_bg_check=m["MUA"]))
        print(f"  {sess:12s} {n_ok:4d}/{len(g)} windows | {len(units):3d} units "
              f"({n_e} E, {n_i} I)", flush=True)
    X = pd.DataFrame(out)
    D = W.merge(X, on=["session", "center_s"], how="inner")
    r = np.median([np.corrcoef(h.MUA_bg, h.MUA_bg_check)[0, 1] for _, h in D.groupby("session")])
    print(f"\n{len(D)} windows, {D.session.nunique()} sessions, {time.time() - t0:.0f} s; "
          f"MUA_bg cross-check vs the replay builder: median r = {r:.4f}")
    assert_local_path(LOGDIR)
    D.to_csv(assert_local_path(OUT_WIN), index=False)
    return D


def analyse(D):
    rows = []
    for setname, cols in SETS.items():
        W = D.dropna(subset=cols).copy()
        for c in cols:
            W[f"{c}_z"] = W.groupby("session")[c].transform(zs)
        zc = [f"{c}_z" for c in cols]
        full, fullc, base = glm(W, zc), glm(W, zc, cluster=True), glm(W, [])
        print(f"\n{'=' * 96}\n{setname}: {len(W)} windows, {W.session.nunique()} sessions; "
              f"block explains {dev_expl(full, base):.2f}% of deviance\n{'=' * 96}")
        print(f"  {'predictor':11s}{'coef/SD':>9s}{'LR chi2':>9s}{'p':>10s}"
              f"{'robust z':>10s}{'p':>10s}{'sessions same sign':>20s}{'signed-rank p':>15s}")
        for j, c in enumerate(cols):
            red = glm(W, [x for i, x in enumerate(zc) if i != j])
            lr = 2 * (full.llf - red.llf)
            p = float(stats.chi2.sf(lr, 1))
            z = float(fullc.params[j] / fullc.bse[j])
            pz = float(2 * stats.norm.sf(abs(z)))
            per = []
            for s, g in W.groupby("session"):
                e = np.column_stack([g.n_replay, g.n_swr - g.n_replay])
                f = sm.GLM(e, sm.add_constant(g[zc].to_numpy(float), has_constant="add"),
                           family=sm.families.Binomial()).fit()
                per.append(float(f.params[1 + j]))
            per = np.array(per)
            same = max(int((per > 0).sum()), int((per < 0).sum()))
            pw = float(stats.wilcoxon(per).pvalue)
            print(f"  {LABEL[c]:11s}{float(fullc.params[j]):+9.3f}{lr:9.1f}{p:10.1e}"
                  f"{z:+10.2f}{pz:10.1e}{same:15d}/{per.size:<5d}{pw:15.4f}")
            rows.append(dict(covset=setname, predictor=c, label=LABEL[c],
                             coef=float(fullc.params[j]), se=float(fullc.bse[j]),
                             lr_chi2=lr, p_lr=p, z_cluster=z, p_cluster=pz,
                             same_sign=same, n_sessions=int(per.size), p_signrank=pw,
                             dev_full=dev_expl(full, base)))
    pd.DataFrame(rows).to_csv(assert_local_path(OUT), index=False)
    print(f"\n-> {OUT}")


def main():
    D = pd.read_csv(OUT_WIN) if os.path.exists(OUT_WIN) and os.environ.get("REBUILD", "0") == "0" \
        else build()
    analyse(D)


if __name__ == "__main__":
    main()
