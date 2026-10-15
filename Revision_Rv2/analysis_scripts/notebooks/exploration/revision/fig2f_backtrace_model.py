"""fig2f_backtrace_model.py -- Figure 2f: rebuilding the GLM producer's input table and
rerunning its ablation model.

fig2f_backtrace.py established that n_samples equals the number of NREM windows in the
newest *_CA1_D2_time_resolved_error_bar.mat for all 9 sessions: the producer
(generate_GLM.ipynb) keeps every NREM window, not only the ones with a ripple. This script
rebuilds that table from the notebook's own steps and reruns its model, then compares with
the stored per-session pickles (data/GLM_mice/explore/*_ablation_results_GLM.pkl): fold AUCs
of the full and reduced models and the three ablation deltas. Folds are
KFold(5, shuffle=True, random_state=0) on a table sorted by center_s, so an exact rebuild
gives the stored numbers exactly.

Features: d2; MUA (40 ms population rate over all CA1 units, 30 s centred rolling mean,
interpolated to window centres); FR_std (across-neuron SD, ddof=0, of each unit's rate in
the same 30 s window). Ripples: load_ripples_events intersected with NREM. Replay events:
<sess>_replay_sequences_new.mat, an event counted as replay if either decoding direction has
pvalue < .05 (the p_and_r variant also requires rvalue < .05 in that direction). Rows: NREM
windows, tiled into sextiles of ripple_count. The predictive-gain model
(cv_predictive_gain_logistic) is shared with fig2f_producer_glm.py.

Two choices the stored pickle does not pin down, tried as variants: which decoded
epoch/template combo(s) feed the replay labels (`epochs`: 'all' uses every combo in the
session's replay file, 'committed' uses only the epoch3/epoch5 combo named for that
session), and whether significance requires both a p-value and an r-value threshold or a
p-value alone (`sig`).

    -> data/revision/log/fig2f_backtrace_model.csv
"""
from __future__ import annotations

import glob
import os
import re
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import scipy.io

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                          # noqa: E402
from r35_glm_ablation import BASEPATH, LOGDIR, WIN_S                     # noqa: E402
from r35_swr_property_control import SESSION_TEMPLATE, RES               # noqa: E402
from fig2f_backtrace import EXPLORE, PATS, load_all_states               # noqa: E402
from fig2f_producer_glm import cv_predictive_gain_logistic               # noqa: E402

OUT_CSV = os.path.join(LOGDIR, "fig2f_backtrace_model.csv")
BIN = 0.04
FEATURES = ("d2", "MUA", "FR_std")
SESS = os.environ.get("SESS", "")            # optional: single session for a quick check


# ----------------------------------------------------------------- cells 4-7
def population_mua(spike_times, centers):
    """40 ms bins over all units, 30 s centred rolling mean/std, interpolated to centres."""
    st = np.sort(np.concatenate([np.asarray(s, float).ravel() for s in spike_times if len(s)]))
    t0, t1 = float(np.floor(st.min())), float(np.ceil(st.max()))
    edges = np.arange(t0, t1 + BIN, BIN)
    counts, _ = np.histogram(st, bins=edges)
    fr = counts / BIN
    t = edges[:-1] + BIN / 2
    dt = float(np.median(np.diff(t)))
    win_n = max(int(round(WIN_S / dt)), 1)
    ser = pd.Series(fr, index=pd.Index(t, name="t"))
    roller = ser.rolling(window=win_n, center=True, min_periods=max(1, win_n // 4))
    mean30, std30 = roller.mean().to_numpy(), roller.std().to_numpy()
    return (np.interp(centers, t, mean30, left=np.nan, right=np.nan),
            np.interp(centers, t, std30, left=np.nan, right=np.nan))


def fr_std_across_neurons(spike_times, centers):
    """Per-neuron rate in [c-15, c+15) (searchsorted 'left' both ends), SD across neurons, ddof=0."""
    ts, te = centers - WIN_S / 2, centers + WIN_S / 2
    rates = np.zeros((len(centers), len(spike_times)))
    for i, spk in enumerate(spike_times):
        s = np.sort(np.asarray(spk, float).ravel())
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        rates[:, i] = (np.searchsorted(s, te, "left") - np.searchsorted(s, ts, "left")) / WIN_S
    return rates.std(axis=1, ddof=0)


# ----------------------------------------------------------------- cell 11
def replay_intervals(sess, epochs="all", sig="p_and_r", which="new"):
    """`epochs`: 'all' | 'committed' | an explicit list of '<epoch>_<template>' combos.
    `which`: 'new' -> <sess>_replay_sequences_new.mat, 'old' -> _replay_sequences.mat."""
    suffix = "_new" if which == "new" else ""
    d = scipy.io.loadmat(os.path.join(RES, f"{sess}_replay_sequences{suffix}.mat"))
    combos = sorted({re.match(r"(epoch\d+_template\d+)", k).group(1)
                     for k in d if re.match(r"epoch\d+_template\d+_eventtimes$", k)})
    if isinstance(epochs, (list, tuple)):
        combos = [c for c in combos if c in epochs]
    elif epochs == "committed":
        t = SESSION_TEMPLATE.get(sess, "template04")
        combos = [c for c in combos if c.endswith(t) and c.split("_")[0] in ("epoch3", "epoch5")]
    out = []
    for k in combos:
        ev = np.squeeze(d[k + "_eventtimes"]).reshape(-1, 2)
        n = len(ev)
        pv = np.squeeze(d[k + "_pvalue"]).reshape(n, -1)[:, :2]
        rv = np.squeeze(d[k + "_rvalues"]).reshape(n, -1)[:, :2]
        if sig == "p_and_r":
            ok = ((pv[:, 0] < .05) & (rv[:, 0] < .05)) | ((pv[:, 1] < .05) & (rv[:, 1] < .05))
        else:
            ok = (pv[:, 0] < .05) | (pv[:, 1] < .05)
        out.append(ev[ok])
    return np.concatenate(out) if out else np.zeros((0, 2)), combos


# ----------------------------------------------------------------- cell 13-14
def overlap_counts(iv, centers, mask=None):
    w0, w1 = centers - WIN_S / 2, centers + WIN_S / 2
    a, b = iv[:, 0], iv[:, 1]
    n = np.zeros(len(centers), int)
    for i in range(len(centers)):
        inw = (a < w1[i]) & (b > w0[i])
        n[i] = int((inw & mask).sum() if mask is not None else inw.sum())
    return n


def build_table(sess, bp, epochs, sig, cache, which="new"):
    import nelpy as nel
    from neuro_py.io import loading

    if "base" not in cache:
        cands = []
        for p in PATS:
            cands += glob.glob(os.path.join(bp, "complex_system", p))
        f = max(set(cands), key=os.path.getmtime)
        D = load_all_states(f).sort_values("center_s").reset_index(drop=True)
        c = D["center_s"].to_numpy(float)
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        units = [np.asarray(u, float) for u in np.asarray(st.data, dtype=object)]
        D["FR_std"] = fr_std_across_neurons(units, c)
        D["MUA"], D["MUA_std"] = population_mua(units, c)
        rip = loading.load_ripples_events(bp)
        sd = loading.load_SleepState_states(bp)
        nrem = nel.EpochArray(sd["NREMstate"])
        riv = (nrem & nel.EpochArray(np.array(rip[["start", "stop"]]))).data
        D["ripple_count"] = overlap_counts(riv, c)
        cache["base"], cache["riv"], cache["file"] = D, riv, os.path.basename(f)
    D, riv = cache["base"].copy(), cache["riv"]
    S, combos = replay_intervals(sess, epochs, sig, which)
    has = np.fromiter((np.any((r[0] < S[:, 1]) & (r[1] > S[:, 0])) if len(S) else False
                       for r in riv), bool, len(riv))
    D["replay_count"] = overlap_counts(riv, D["center_s"].to_numpy(float), has)
    # cell 14: prepare_sextile_df
    d = D[D["state"].str.upper().eq("NREM")].copy()
    d["replay_present"] = (d["replay_count"].astype(float) > 0).astype(int)
    d = d.dropna(subset=["d2", "MUA", "ripple_count", "replay_present"]).copy()
    d["tile"] = pd.qcut(d["ripple_count"].astype(float), q=6, labels=False, duplicates="drop") + 1
    return d, dict(n_sig_events=len(S), combos=",".join(combos), n_replay_swr=int(has.sum()))


def search(sessions):
    """For sessions that match neither variant: every replay file x every non-empty
    subset of its combos, p-only significance (the r-value threshold is redundant given
    the p-value threshold)."""
    import itertools
    stored = {}
    for f in sorted(glob.glob(os.path.join(EXPLORE, "*_ablation_results_GLM.pkl"))):
        o = pd.read_pickle(f)
        stored[f"{o['animal']}_{o['day']}"] = o
    for sess in sessions:
        o, bp, cache = stored[sess], BASEPATH[sess], {}
        s_full = o["auc_folds"]["full"]
        print(f"\n===== SEARCH {sess}  stored AUC full {s_full.mean():.4f}", flush=True)
        for which in ("new", "old"):
            _, combos_all = replay_intervals(sess, "all", "p_only", which)
            for r in range(1, len(combos_all) + 1):
                for sub in itertools.combinations(combos_all, r):
                    d, info = build_table(sess, bp, list(sub), "p_only", cache, which)
                    res, det = cv_predictive_gain_logistic(
                        d, features=FEATURES, tile_col="tile", replay_col="replay_present",
                        n_bins=6, group_col="session", n_splits=5, C=1.0, random_state=0)
                    r_full = det["full"]["auc"]
                    dfull = float(np.max(np.abs(r_full - s_full)))
                    print(f"  [{which:3s}] {'+'.join(c.replace('template', 't') for c in sub):45s} "
                          f"neg={int((d.replay_present == 0).sum()):4d} AUC full {r_full.mean():.4f} "
                          f"max|dfold| {dfull:.4f}{'  <== EXACT' if dfull < 1e-6 else ''}", flush=True)


# The epoch/template combos that reproduce each stored pickle's fold AUCs and ablation
# deltas to within 1e-6 (found via the default run plus the SEARCH mode above).
MATCHED = {
    "PPP4_day11": ["epoch5_template02"],
    "PPP7_day8":  ["epoch5_template02"],
    "PPP8_day8":  ["epoch3_template02", "epoch5_template02"],
    "PPP20_day4": ["epoch1_template04", "epoch3_template04", "epoch5_template04"],
    "PPP20_day5": ["epoch1_template04", "epoch3_template04", "epoch5_template04"],
    "PPP21_day4": ["epoch1_template04", "epoch3_template04"],
    "PPP21_day5": ["epoch1_template04", "epoch3_template04", "epoch5_template04"],
    "PPP23_day7": ["epoch5_template04"],
    "PPP23_day8": ["epoch3_template04", "epoch5_template04"],
}


def candidate_coverage(sess, combos):
    """Time spans [first candidate start, last candidate stop] of the decoded combos: the
    intervals over which replay labels are defined. replay_present is 0 by construction
    outside these spans."""
    d = scipy.io.loadmat(os.path.join(RES, f"{sess}_replay_sequences_new.mat"))
    spans = []
    for k in combos:
        ev = np.squeeze(d[k + "_eventtimes"]).reshape(-1, 2)
        spans.append((float(ev[:, 0].min()), float(ev[:, 1].max())))
    return spans


def coverage():
    """Rerun the producer's model (a) as stored, (b) on windows inside decoded time only,
    and redo the panel's group statistics on both."""
    from scipy import stats as sps
    from statsmodels.stats.multitest import multipletests
    rows = []
    for sess, bp in BASEPATH.items():
        if sess not in MATCHED:
            continue
        cache = {}
        d, info = build_table(sess, bp, MATCHED[sess], "p_only", cache)
        spans = candidate_coverage(sess, MATCHED[sess])
        c = d["center_s"].to_numpy(float)
        inside = np.zeros(len(d), bool)
        for a, b in spans:
            inside |= (c >= a - WIN_S / 2) & (c <= b + WIN_S / 2)
        neg = d["replay_present"].to_numpy() == 0
        out = {"session": sess, "combos": "+".join(k.replace("template", "t") for k in MATCHED[sess]),
               "n": len(d), "neg": int(neg.sum()), "neg_outside_decoded": int((neg & ~inside).sum()),
               "n_inside": int(inside.sum()), "neg_inside": int((neg & inside).sum()),
               "frac_session_decoded": float(inside.mean())}
        for tag, sub in (("stored", d), ("inside", d[inside].copy())):
            if sub["replay_present"].nunique() < 2 or len(sub) < 200:
                for f in FEATURES:
                    out[f"imp_{f}_{tag}"] = np.nan
                continue
            sub = sub.copy()
            sub["tile"] = pd.qcut(sub["ripple_count"].astype(float), q=6, labels=False,
                                  duplicates="drop") + 1
            try:
                res, det = cv_predictive_gain_logistic(
                    sub, features=FEATURES, tile_col="tile", replay_col="replay_present",
                    n_bins=6, group_col="session", n_splits=5, C=1.0, random_state=0)
            except ValueError as e:          # a fold with one class: too few negatives
                print(f"    [{sess} {tag}: unfittable -- {str(e)[:60]}]", flush=True)
                for f in FEATURES:
                    out[f"imp_{f}_{tag}"] = np.nan
                continue
            for m, v in zip(res["Model"], res["ΔAUC_vs_Full"]):
                if m.startswith("Ablate"):
                    out[f"imp_{m.replace('Ablate ', '').split('_z')[0]}_{tag}"] = -float(v)
            out[f"auc_full_{tag}"] = float(det["full"]["auc"].mean())
        rows.append(out)
        print(f"  {sess:11s} {out['combos']:28s} n={out['n']:5d} neg={out['neg']:5d} "
              f"(outside decoded time: {out['neg_outside_decoded']:5d}) | decoded {out['frac_session_decoded']:.0%} "
              f"of windows | d2 imp stored {out.get('imp_d2_stored', np.nan):+.4f} -> inside {out.get('imp_d2_inside', np.nan):+.4f}",
              flush=True)
    R = pd.DataFrame(rows)
    outp = os.path.join(LOGDIR, "fig2f_backtrace_coverage.csv")
    R.to_csv(assert_local_path(outp), index=False)
    print(f"\nwrote {outp}")
    for tag in ("stored", "inside"):
        print(f"\n--- panel statistics, {tag} ---")
        P = R.dropna(subset=[f"imp_d2_{tag}"])
        for f in FEATURES:
            v = P[f"imp_{f}_{tag}"].to_numpy()
            print(f"  {f:7s} median {np.median(v):+.5f}  pos {(v > 0).sum()}/{len(v)}  "
                  f"p(greater) {sps.wilcoxon(v, alternative='greater').pvalue:.4f}")
        ps, lab = [], []
        for a, b in (("d2", "MUA"), ("d2", "FR_std"), ("MUA", "FR_std")):
            ps.append(sps.wilcoxon(P[f"imp_{a}_{tag}"] - P[f"imp_{b}_{tag}"]).pvalue)
            lab.append(f"{a} vs {b}")
        for l, p, pa in zip(lab, ps, multipletests(ps, method="holm")[1]):
            print(f"  {l:14s} p {p:.4f}  Holm {pa:.4f}")


def main():
    if os.environ.get("SEARCH"):
        search(os.environ["SEARCH"].split(","))
        return
    if os.environ.get("COVERAGE"):
        coverage()
        return
    stored = {}
    for f in sorted(glob.glob(os.path.join(EXPLORE, "*_ablation_results_GLM.pkl"))):
        o = pd.read_pickle(f)
        stored[f"{o['animal']}_{o['day']}"] = o
    rows = []
    for sess, bp in BASEPATH.items():
        if sess not in stored or (SESS and sess != SESS):
            continue
        o = stored[sess]
        s_full, s_red = o["auc_folds"]["full"], o["auc_folds"]["reduced"]
        s_abl = {m.replace("Ablate ", "").split("_z")[0]: v for m, v in
                 zip(o["summary"]["Model"], o["summary"]["ΔAUC_vs_Full"]) if m.startswith("Ablate")}
        print(f"\n===== {sess}  stored: n={o['n_samples']}  AUC full {s_full.mean():.4f}  "
              f"reduced {s_red.mean():.4f}  dAUC d2 {s_abl['d2']:+.4f} MUA {s_abl['MUA']:+.4f} "
              f"FR_std {s_abl['FR_std']:+.4f}", flush=True)
        cache = {}
        for epochs in ("all", "committed"):
            for sig in ("p_and_r", "p_only"):
                d, info = build_table(sess, bp, epochs, sig, cache)
                res, det = cv_predictive_gain_logistic(
                    d, features=FEATURES, tile_col="tile", replay_col="replay_present",
                    n_bins=6, group_col="session", n_splits=5, C=1.0, random_state=0)
                r_full, r_red = det["full"]["auc"], det["reduced"]["auc"]
                r_abl = {m.replace("Ablate ", "").split("_z")[0]: v for m, v in
                         zip(res["Model"], res["ΔAUC_vs_Full"]) if m.startswith("Ablate")}
                dfull = float(np.max(np.abs(r_full - s_full))) if len(r_full) == len(s_full) else np.nan
                row = dict(session=sess, d2_file=cache["file"], epochs=epochs, sig=sig,
                           n=len(d), n_samples=o["n_samples"], neg=int((d.replay_present == 0).sum()),
                           n_sig_events=info["n_sig_events"], n_replay_swr=info["n_replay_swr"],
                           combos=info["combos"],
                           auc_full=r_full.mean(), auc_full_stored=s_full.mean(),
                           auc_red=r_red.mean(), auc_red_stored=s_red.mean(),
                           max_fold_abs_diff_full=dfull,
                           dauc_d2=r_abl.get("d2"), dauc_d2_stored=s_abl["d2"],
                           dauc_MUA=r_abl.get("MUA"), dauc_MUA_stored=s_abl["MUA"],
                           dauc_FR_std=r_abl.get("FR_std"), dauc_FR_std_stored=s_abl["FR_std"])
                rows.append(row)
                print(f"  [{epochs:9s}|{sig:7s}] n={len(d)} neg={row['neg']:4d} sig events {info['n_sig_events']:5d}"
                      f" | AUC full {r_full.mean():.4f} red {r_red.mean():.4f} | max|dfold| {dfull:.4f}"
                      f" | dAUC d2 {r_abl.get('d2', np.nan):+.4f} MUA {r_abl.get('MUA', np.nan):+.4f}"
                      f" FR_std {r_abl.get('FR_std', np.nan):+.4f}", flush=True)
    R = pd.DataFrame(rows)
    assert_local_path(LOGDIR)
    R.to_csv(assert_local_path(OUT_CSV), index=False)
    print(f"\nwrote {OUT_CSV}")
    R["exact"] = R.max_fold_abs_diff_full < 1e-6
    R["close"] = R.max_fold_abs_diff_full < 5e-3
    print("\nexact fold-AUC reproduction by variant:")
    print(R.groupby(["epochs", "sig"])[["exact", "close"]].sum().to_string())


if __name__ == "__main__":
    main()
