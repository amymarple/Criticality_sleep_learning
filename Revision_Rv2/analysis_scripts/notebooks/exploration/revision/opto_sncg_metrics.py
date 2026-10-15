"""
opto_sncg_metrics.py -- Fig. 3g cohort: branching ratio, ISI entropy, MUA and d2 under Sncg-Flp x
Arch3.3 CCK silencing, presleep (baseline) vs postsleep (silenced).

Fig. 3g is built from data/ripple_barr/manipulation/sncg/*.pkl (generate_barr_manipulation.ipynb):
seven sessions of three Sncg mice, d2 (2024 Euclidean, p=20) in NREM, presleep vs postsleep. This
script computes the same window metrics on exactly those sessions, reusing opto_cck_metrics's
windowing and metric code:

  * windows: NREM (5 s guard) within the presleep / postsleep epochs of load_epoch (name contains
    "presleep" / "postsleep", case-insensitive), 30 s, non-overlapping unless --sliding (2 s step,
    for the window-level figure);
  * per window: MUA (40 ms), ISI entropy (pooled), branching ratio m (pyr and all units, 4 ms,
    k_max=200) and tau;
  * d2 (2024 Euclidean, p=20, 40 ms): joined from the session's own
    complex_system/<day>_CA1_D2_time_resolved.mat (the Fig. 3g source; deltaT 0.04 in all seven),
    nearest centre within 1 s, never refitted;
  * units: CA1, firingRate > 0.2 Hz. SNCG06_day27 has 1 such unit and SNCG08_day17 has 4; contrasts
    are reported with and without these two (QC set = the other five).

Conditions are stored as base (presleep) / stim (postsleep) with phase = "sleep" so that
opto_cck_metrics.contrasts / figure run unchanged; the paired unit is the session.

    ENV=C:/Users/Cornell/.conda/envs/criticality/python.exe
    $ENV opto_sncg_metrics.py [--sliding] [--sessions SNCG08_day14 ...]

Writes only new files:
    -> data/revision/log/opto_sncg_{windows,session,contrasts}[_sliding].csv
    -> data/revision/log/opto_sncg_contrasts_qc[_sliding].csv
    -> reports/revision_figs/opto_sncg_metrics[_sliding].png/.svg
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import opto_cck_metrics as ocm  # noqa: E402
from fig1_metrics import (  # noqa: E402
    BIN_BR_S, BIN_MUA_S, GUARD_S, active_unit_counts, binned_counts,
    build_state_intervals, fit_m, intersect_intervals, isi_entropy, make_windows,
    rk_curve, slice_sorted, tau_ms,
)
from r35_background_vs_ripple import load_d2_mat  # noqa: E402
from run_fig1_metrics import assert_local_path  # noqa: E402

SNCG_SESSIONS = [
    r"X:\data\Barrage\SNCG06\day27",
    r"X:\data\Barrage\SNCG08\day14",
    r"X:\data\Barrage\SNCG08\day16",
    r"X:\data\Barrage\SNCG08\day17",
    r"X:\data\Barrage\SNCG09\day10",
    r"X:\data\Barrage\SNCG09\day12",
    r"X:\data\Barrage\SNCG09\day13",
]
QC_EXCLUDE = ("SNCG06_day27", "SNCG08_day17")     # 1 and 4 CA1 units > 0.2 Hz
K_MAX = ocm.K_MAX
COND = {"presleep": "base", "postsleep": "stim"}


def load_epochs(bp):
    from neuro_py.io import loading
    ep = loading.load_epoch(bp)
    out = {}
    for n, a, b in zip(ep.name.astype(str), ep.startTime, ep.stopTime):
        nl = n.lower()
        for key in COND:
            if key in nl:
                out.setdefault(key, []).append((float(a), float(b)))
    return {k: np.array(v) for k, v in out.items()}


def load_d2_80ms(bp):
    """The variant Fig. 3g's pickles were cut from (6 of 7 sessions):
    <day>_CA1_CA2_D2_time_resolved_error_bar_80ms_60second.mat -- CA1+CA2, 80 ms bins,
    60 s windows stepped 5 s, order 20. -> DataFrame(center_s, d2, d2_err) NREM, or None."""
    import scipy.io
    day = os.path.basename(bp.rstrip("\\/"))
    f = os.path.join(bp, "complex_system", f"{day}_CA1_CA2_D2_time_resolved_error_bar_80ms_60second.mat")
    if not os.path.exists(f):
        return None
    m = scipy.io.loadmat(f, simplify_cells=True)
    res = m["results"] if isinstance(m["results"], (list, np.ndarray)) else [m["results"]]
    rows = [pd.DataFrame({"center_s": np.atleast_1d(r["center"]).astype(float),
                          "d2": np.atleast_1d(r["d2"]).astype(float),
                          "d2_err": np.atleast_1d(r["d2_err"]).astype(float) if "d2_err" in r else np.nan})
            for r in res if "NREM" in str(r["state"]).upper()]
    return pd.concat(rows).sort_values("center_s").reset_index(drop=True) if rows else None


def process_session(bp, sliding=False, population="CA1"):
    t0 = time.time()
    tag = ocm.session_tag(bp)
    print(f"\n=== {tag} ===", flush=True)
    units, is_pyr, state_dict = ocm.load_units(bp, population)
    n_units, n_pyr = len(units), int(is_pyr.sum())
    print(f"  units ({population}, >0.2 Hz): {n_units} | pyr {n_pyr}", flush=True)
    nrem = build_state_intervals(state_dict, GUARD_S)["NREM"]
    epochs = load_epochs(bp)

    parts = []
    for key, iv_ep in sorted(epochs.items(), key=lambda kv: kv[1][0, 0]):
        iv = intersect_intervals(nrem, iv_ep)
        w = make_windows(iv, "NREM")
        if not len(w):
            print(f"  {key:9s} NREM 0 min -> 0 windows")
            continue
        if not sliding:
            w = w[w.nonoverlap].reset_index(drop=True)
        w["block"] = key
        w["phase"] = "sleep"
        w["condition"] = COND[key]
        parts.append(w)
        print(f"  {key:9s} {iv_ep[0,0]/3600:5.2f}-{iv_ep[-1,1]/3600:5.2f} h, NREM "
              f"{float(np.sum(iv[:,1]-iv[:,0]))/60:6.1f} min -> {len(w):4d} windows", flush=True)
    if not parts:
        return None
    win = pd.concat(parts, ignore_index=True)

    all_sp = np.sort(np.concatenate(units)) if units else np.zeros(0)
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
            if len(sp) < 3:
                continue
            act = active_unit_counts(sp, a, b, BIN_BR_S)
            k, r_k, se = rk_curve(act, K_MAX)
            m = fit_m(k, r_k, se, K_MAX)
            d[f"BR_{pop}_k{K_MAX}"] = m
            d[f"tau_ms_{pop}_k{K_MAX}"] = tau_ms(np.array([m]))[0]
        rows.append(d)
    win = pd.concat([win, pd.DataFrame(rows)], axis=1)
    for c in ("BR_pyr_k200", "tau_ms_pyr_k200", "BR_all_k200", "tau_ms_all_k200"):
        if c not in win.columns:
            win[c] = np.nan

    d2 = load_d2_mat(bp, "")
    if d2 is not None:
        d2 = d2.rename(columns={"center": "center_s", "d2_x": "d2", "d2_err_x": "d2_err"})
    win, rate = ocm.join_d2(win, d2)
    print(f"  d2 (2024 Euclidean, p=20, CA1 40 ms/30 s) join from .mat: {rate:.0%}", flush=True)
    # the Fig. 3g variant: CA1+CA2, 80 ms, 60 s windows stepped 5 s -> nearest centre within 2.5 s
    v = load_d2_80ms(bp)
    win["d2_80ms_ca1ca2"] = np.nan
    if v is not None and len(win):
        c = v.center_s.to_numpy()
        i = np.searchsorted(c, win.center_s.to_numpy())
        i0, i1 = np.clip(i - 1, 0, len(c) - 1), np.clip(i, 0, len(c) - 1)
        pick = np.where(np.abs(c[i0] - win.center_s) <= np.abs(c[i1] - win.center_s), i0, i1)
        ok = np.abs(c[pick] - win.center_s.to_numpy()) <= 2.5
        win.loc[ok, "d2_80ms_ca1ca2"] = v.d2.to_numpy()[pick[ok]]
        print(f"  d2 (2024 Euclidean, p=20, CA1+CA2 80 ms/60 s — Fig. 3g variant) join: {ok.mean():.0%}", flush=True)
    win.insert(0, "session", tag)
    win["n_units"] = n_units
    win["n_pyr"] = n_pyr
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sliding", action="store_true")
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--population", default="CA1", choices=["CA1", "CA1CA2"],
                    help="unit set for MUA / entropy / BR (d2 columns are joined regardless); "
                         "CA1CA2 -> *_CA1CA2 suffix")
    args = ap.parse_args()
    paths = SNCG_SESSIONS
    if args.sessions:
        paths = [p for p in paths if ocm.session_tag(p) in set(args.sessions)]
    out = []
    for p in paths:
        try:
            w = process_session(p, sliding=args.sliding, population=args.population)
            if w is not None:
                out.append(w)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  [FAIL] {p}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    if not out:
        return
    win = pd.concat(out, ignore_index=True)
    sfx = ("" if args.population == "CA1" else "_CA1CA2") + ("_sliding" if args.sliding else "")
    ocm.METRICS = ["d2", "d2_80ms_ca1ca2", "BR_pyr_k200", "tau_ms_pyr_k200", "BR_all_k200",
                   "isi_entropy_pooled", "MUA_mean"]
    ocm.METRIC_LABEL["d2_80ms_ca1ca2"] = "d2 (2024 Euclidean, p=20)\nCA1+CA2, 80 ms, 60 s (Fig. 3g)"
    sess, con = ocm.contrasts(win)
    sess_qc, con_qc = ocm.contrasts(win[~win.session.isin(QC_EXCLUDE)])
    win.to_csv(assert_local_path(os.path.join(ocm.LOGDIR, f"opto_sncg_windows{sfx}.csv")), index=False)
    sess.to_csv(assert_local_path(os.path.join(ocm.LOGDIR, f"opto_sncg_session{sfx}.csv")), index=False)
    con.to_csv(assert_local_path(os.path.join(ocm.LOGDIR, f"opto_sncg_contrasts{sfx}.csv")), index=False)
    con_qc.to_csv(assert_local_path(os.path.join(ocm.LOGDIR, f"opto_sncg_contrasts_qc{sfx}.csv")), index=False)
    grid = "30 s windows stepped 2 s" if args.sliding else "30 s non-overlapping windows"
    ocm.figure(sess, con, os.path.join(ocm.FIGDIR, f"opto_sncg_metrics{sfx}"),
               note=f"Sncg x Arch CCK silencing (Fig. 3g cohort), NREM, {grid}; one line per session, "
                    f"presleep -> postsleep; session medians")
    pd.set_option("display.width", 220)
    print("\n=== session x condition medians ===")
    print(sess.round(4).to_string(index=False))
    print("\n=== paired contrasts, postsleep - presleep, all 7 sessions (session level) ===")
    print(con[con.level == "session"].round(4).to_string(index=False))
    print(f"\n=== same, yield gate (excl {', '.join(QC_EXCLUDE)}) ===")
    print(con_qc[con_qc.level == "session"].round(4).to_string(index=False))
    print(f"\nwrote -> opto_sncg_*{sfx}.csv, opto_sncg_metrics{sfx}.*")


if __name__ == "__main__":
    main()
