"""
opto_cck_metrics.py -- branching ratio, ISI entropy and MUA in the uLED CCK-silencing sessions, per
stimulation block (baseline vs stimulation).

Conditions come from <basename>.uLEDtimes_HC.mat block boundaries ({pre,post}_{base,stim}; the
short *_stepStim_* calibration blocks are excluded), verified against
<basename>.uLEDPulses.events.mat (pulses present inside every stim block, none inside baseline
blocks). 5/6 sessions carry four blocks with counterbalanced order (stim-first or base-first);
NN13_day23 has pre only. Windows: NREM (SleepState, 5 s guard) intersected with each block, 30 s
non-overlapping (make_windows + nonoverlap flag), so no window straddles a block boundary. Metrics
per window: MUA (40 ms), branching ratio m via fig1_metrics.fit_m (4 ms bins, k_max 200 = 800 ms
kernel, pyramidal primary + all-unit), tau, ISI entropy (pooled, 20 bins of log10 ISI,
rate-invariant), n_ISI; stored d2 (2024 Euclidean, p=20) joined from
data/ripple_barr/manipulation/background_uled/*.pkl on center_s (nearest within 1 s), never
refitted. Statistics: session x phase is the paired unit (base vs stim within the same sleep
phase), and session-level (phases averaged) reported alongside -- paired Wilcoxon, sign
consistency, median delta, delta / within-window SD.

Inputs: X:\\data\\Barrage\\<animal>\\<day> raw sessions (OPTO_SESSIONS, 6 sessions).

    ENV=C:/Users/Cornell/.conda/envs/criticality/python.exe
    $ENV opto_cck_metrics.py --test            # NN3_day12 only
    $ENV opto_cck_metrics.py --all
    $ENV opto_cck_metrics.py --all --population CA1CA2

Reads X: read-only; writes only to D:.
    -> data/revision/log/opto_cck_windows.csv
    -> data/revision/log/opto_cck_session.csv
    -> data/revision/log/opto_cck_contrasts.csv
    -> reports/revision_figs/opto_cck_metrics.png / .svg
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import wilcoxon, binomtest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fig1_metrics import (  # noqa: E402
    BIN_BR_S, BIN_MUA_S, GUARD_S, MIN_FIRING_RATE, active_unit_counts,
    binned_counts, build_state_intervals, fit_m, intersect_intervals,
    isi_entropy, make_windows, rk_curve, slice_sorted, tau_ms,
)
from run_fig1_metrics import assert_local_path  # noqa: E402

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
PKLDIR = os.path.join(REPO, "data", "ripple_barr", "manipulation", "background_uled")

OPTO_SESSIONS = [
    r"X:\data\Barrage\NN3\day12",
]
K_MAX = 200
CONDITION_BLOCKS = ("pre_base", "pre_stim", "post_base", "post_stim")
METRICS = ["d2", "BR_pyr_k200", "tau_ms_pyr_k200", "BR_all_k200",
           "isi_entropy_pooled", "MUA_mean"]
METRIC_LABEL = {"d2": "d2 (2024 Euclidean, p=20)",
                "BR_pyr_k200": "m (pyr, 4 ms, k=200)",
                "tau_ms_pyr_k200": "tau (ms, pyr)",
                "BR_all_k200": "m (all units, 4 ms, k=200)",
                "isi_entropy_pooled": "ISI entropy (bits)",
                "MUA_mean": "MUA (counts / 40 ms)"}


def session_tag(bp):
    return os.path.basename(os.path.dirname(bp)) + "_" + os.path.basename(bp)


def _loadmat(f):
    try:
        return scipy.io.loadmat(f, simplify_cells=True)
    except NotImplementedError:
        import mat73
        return mat73.loadmat(f)


def load_blocks(bp):
    u = _loadmat(glob.glob(os.path.join(bp, "*uLEDtimes_HC.mat"))[0])["uLEDtimes"]
    out = {}
    for k, v in u.items():
        if k in CONDITION_BLOCKS:
            v = np.asarray(v, float).ravel()
            out[k] = (float(v[0]), float(v[1]))
    return out


def load_units(bp, population="CA1"):
    from neuro_py.io import loading
    st, cm = loading.load_spikes(basepath=bp)
    state_dict = loading.load_SleepState_states(bp)
    # normalise single-bout state arrays to shape (n, 2)
    for k, v in list(state_dict.items()):
        a = np.asarray(v, float)
        if a.ndim == 1 and a.size == 2:
            state_dict[k] = a.reshape(1, 2)
    reg = np.asarray(cm.brainRegion).astype(str)
    in_reg = np.char.find(reg, "CA1") >= 0
    if population == "CA1CA2":
        in_reg |= np.char.find(reg, "CA2") >= 0
    keep = in_reg & (np.asarray(cm.firingRate, float) > MIN_FIRING_RATE)
    units = [np.sort(np.asarray(s, float))
             for s in np.asarray(st.data, dtype=object)[keep]]
    ctype = np.asarray(cm.putativeCellType).astype(str)[keep]
    is_pyr = np.char.find(ctype, "Pyr") >= 0
    return units, is_pyr, state_dict


def load_stored_d2(tag):
    f = os.path.join(PKLDIR, f"{tag}.pkl")
    if not os.path.exists(f):
        return None
    d = pickle.load(open(f, "rb"))
    d = d[d.state == "NREM"][["center_s", "d2", "d2_err"]].sort_values("center_s")
    return d.reset_index(drop=True)


def join_d2(win, d2df, tol=1.0):
    win = win.copy()
    win["d2"] = np.nan
    win["d2_err"] = np.nan
    if d2df is None or not len(win):
        return win, 0.0
    c = d2df.center_s.to_numpy()
    idx = np.searchsorted(c, win.center_s.to_numpy())
    idx = np.clip(idx, 1, len(c) - 1)
    lo, hi = idx - 1, idx
    pick = np.where(np.abs(c[lo] - win.center_s) <= np.abs(c[hi] - win.center_s), lo, hi)
    ok = np.abs(c[pick] - win.center_s.to_numpy()) <= tol
    win.loc[ok, "d2"] = d2df.d2.to_numpy()[pick[ok]]
    win.loc[ok, "d2_err"] = d2df.d2_err.to_numpy()[pick[ok]]
    return win, float(ok.mean())


def process_session(bp, population="CA1", verbose=True, sliding=False):
    t0 = time.time()
    tag = session_tag(bp)
    print(f"\n=== {tag} ===", flush=True)
    blocks = load_blocks(bp)
    units, is_pyr, state_dict = load_units(bp, population)
    n_units, n_pyr = len(units), int(is_pyr.sum())
    print(f"  units ({population}, >{MIN_FIRING_RATE} Hz): {n_units} | pyr {n_pyr}", flush=True)
    nrem = build_state_intervals(state_dict, GUARD_S)["NREM"]

    parts = []
    for name, (b0, b1) in sorted(blocks.items(), key=lambda kv: kv[1][0]):
        iv = intersect_intervals(nrem, np.array([[b0, b1]]))
        w = make_windows(iv, "NREM")
        if not len(w):
            print(f"  {name:10s} NREM {0:6.1f} min -> 0 windows")
            continue
        if not sliding:   # statistics use the tiling; --sliding is for LMM/trace only
            w = w[w.nonoverlap].reset_index(drop=True)
        w["block"] = name
        w["phase"] = name.split("_")[0]
        w["condition"] = name.split("_")[1]
        parts.append(w)
        print(f"  {name:10s} NREM {float(np.sum(iv[:,1]-iv[:,0]))/60:6.1f} min "
              f"-> {len(w):4d} non-overlapping windows", flush=True)
    if not parts:
        return None
    win = pd.concat(parts, ignore_index=True)

    all_sp = np.sort(np.concatenate(units))
    pyr = [u for u, p in zip(units, is_pyr) if p]
    rows = []
    for r in win.itertuples(index=False):
        a, b = r.win_start, r.win_stop
        d = {}
        w_all = slice_sorted(all_sp, a, b)
        mua = binned_counts(w_all, a, b, BIN_MUA_S, presliced=True)
        d["MUA_mean"] = float(mua.mean()) if mua.size else np.nan
        d["isi_entropy_pooled"] = isi_entropy(w_all)
        d["n_ISI"] = max(len(w_all) - 1, 0)
        for pop, sp in (("pyr", pyr), ("all", units)):
            if not sp:
                continue
            act = active_unit_counts(sp, a, b, BIN_BR_S)
            k, r_k, se = rk_curve(act, K_MAX)
            m = fit_m(k, r_k, se, K_MAX)
            d[f"BR_{pop}_k{K_MAX}"] = m
            d[f"tau_ms_{pop}_k{K_MAX}"] = tau_ms(np.array([m]))[0]
        rows.append(d)
    win = pd.concat([win, pd.DataFrame(rows)], axis=1)

    win, rate = join_d2(win, load_stored_d2(tag))
    print(f"  stored d2 (2024 Euclidean) join: {rate:.0%}", flush=True)
    win.insert(0, "session", tag)
    win["n_units"] = n_units
    win["n_pyr"] = n_pyr
    win["population"] = population
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return win


def contrasts(win):
    """Paired base-vs-stim: session x phase pairs, and session-level (phase mean)."""
    keys = ["session", "phase", "condition"]
    med = win.groupby(keys)[METRICS].median().reset_index()
    n = win.groupby(keys).size().rename("n_windows").reset_index()
    sess = med.merge(n, on=keys)
    within_sd = win.groupby(["session"])[METRICS].std()

    rows = []
    for level in ("session_phase", "session"):
        if level == "session_phase":
            piv = sess.pivot_table(index=["session", "phase"], columns="condition",
                                   values=METRICS)
        else:
            piv = (sess.groupby(["session", "condition"])[METRICS].mean().reset_index()
                       .pivot_table(index="session", columns="condition", values=METRICS))
        for met in METRICS:
            if (met, "base") not in piv.columns or (met, "stim") not in piv.columns:
                continue
            p = piv[met][["base", "stim"]].dropna()
            if len(p) < 2:
                continue
            delta = p["stim"] - p["base"]
            sess_ids = p.index.get_level_values(0) if level == "session_phase" else p.index
            sd = within_sd.loc[sess_ids, met].to_numpy()
            n_up = int((delta > 0).sum())
            row = {"level": level, "metric": met, "n_pairs": len(p),
                   "median_base": float(p["base"].median()),
                   "median_stim": float(p["stim"].median()),
                   "median_delta": float(delta.median()),
                   "median_delta_over_sd": float(np.median(delta.to_numpy() / sd)),
                   "n_stim_greater": n_up,
                   "p_sign": float(binomtest(n_up, len(p), 0.5).pvalue)}
            try:
                row["p_wilcoxon"] = float(wilcoxon(p["stim"], p["base"]).pvalue)
            except ValueError:
                row["p_wilcoxon"] = np.nan
            rows.append(row)
    return sess, pd.DataFrame(rows)


def figure(sess, con, path, note="NREM, 30 s non-overlapping windows, session-phase medians"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.3 * len(METRICS), 3.4))
    piv = sess.pivot_table(index=["session", "phase"], columns="condition",
                           values=METRICS)
    for ax, met in zip(axes, METRICS):
        if (met, "base") not in piv.columns:
            continue
        p = piv[met][["base", "stim"]].dropna()
        for (s, ph), r in p.iterrows():
            ax.plot([0, 1], [r["base"], r["stim"]], "-o", ms=3.5, lw=0.8,
                    color="#7b3294" if ph == "pre" else "#d95f0e", alpha=0.8)
        c = con[(con.level == "session_phase") & (con.metric == met)]
        if len(c):
            c = c.iloc[0]
            ax.set_title(f"{c.n_stim_greater}/{c.n_pairs} up, "
                         f"\u0394={c.median_delta:+.3g}\np_w={c.p_wilcoxon:.3f}",
                         fontsize=7.5)
        ax.set_xticks([0, 1], ["baseline", "CCK\nsilenced"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel(METRIC_LABEL[met], fontsize=7.5)
    fig.text(0.01, 0.005, "one line per session x sleep phase (purple pre, orange post); "
             + note, fontsize=6.5, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(path + "." + ext), dpi=250)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="NN3_day12 only")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--population", default="CA1", choices=["CA1", "CA1CA2"])
    ap.add_argument("--sliding", action="store_true",
                    help="keep the 2 s sliding grid (~15x windows; for LMM / traces, "
                         "NOT for paired tests) -> *_sliding.csv")
    args = ap.parse_args()
    if args.test:
        paths = [p for p in OPTO_SESSIONS if p.endswith(r"NN3\day12")]
    elif args.sessions:
        paths = [p for p in OPTO_SESSIONS if session_tag(p) in set(args.sessions)]
    else:
        paths = OPTO_SESSIONS

    out = []
    for p in paths:
        try:
            w = process_session(p, args.population, sliding=args.sliding)
            if w is not None:
                out.append(w)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  [FAIL] {p}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    if not out:
        return
    win = pd.concat(out, ignore_index=True)
    sess, con = contrasts(win)

    assert_local_path(LOGDIR)
    os.makedirs(LOGDIR, exist_ok=True)
    sfx = "" if args.population == "CA1" else "_" + args.population
    if args.sliding:
        sfx += "_sliding"
    win.to_csv(assert_local_path(os.path.join(LOGDIR, f"opto_cck_windows{sfx}.csv")), index=False)
    sess.to_csv(assert_local_path(os.path.join(LOGDIR, f"opto_cck_session{sfx}.csv")), index=False)
    con.to_csv(assert_local_path(os.path.join(LOGDIR, f"opto_cck_contrasts{sfx}.csv")), index=False)
    os.makedirs(FIGDIR, exist_ok=True)
    figure(sess, con, os.path.join(FIGDIR, f"opto_cck_metrics{sfx}"))

    pd.set_option("display.width", 200)
    print("\n=== session x phase x condition medians ===")
    print(sess.round(4).to_string(index=False))
    print("\n=== paired contrasts, stim - base ===")
    print(con.round(4).to_string(index=False))
    print(f"\nwrote -> {LOGDIR}\\opto_cck_*{sfx}.csv, {FIGDIR}\\opto_cck_metrics{sfx}.*")


if __name__ == "__main__":
    main()
