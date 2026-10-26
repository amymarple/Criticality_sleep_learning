"""opto_cck_excised.py -- the CCK-silencing contrast with ripple time removed from every window.

opto_cck_metrics.py reports full-window metrics. This script recomputes the same contrast with
ripple (or barrage) time excised from every window. The ripple-excised d2 (2024 Euclidean)
control used elsewhere (r35_background_vs_ripple.py) already exists for every opto session on
the share:
`X:\\data\\Barrage\\<a>\\<d>\\complex_system\\<d>_CA1_D2_time_resolved_rippleablate.mat`
(MATLAB, ripple intervals subtracted from each 30 s window, ripple-free subwindows glued,
NREM only, >= 1 s free time). Nothing is recomputed for d2 from this file -- both .mat arms
are joined. The other metrics are recomputed on the same excision:

  * ripple intervals (`ripples.events.mat` via neuro_py) are subtracted from each window
    with `r35_kl_rippleablate.subtract_intervals` (port of fast_subtract_intervals);
  * MUA_exc: 40 ms counts on the glued ripple-free series;
  * isi_entropy_exc: ISIs pooled within ripple-free segments only (no ISI spans a gap);
  * BR_{pyr,all}_exc_k200 / tau: active-unit counts at 4 ms per segment, glued (the gluing is
    the MATLAB's; several junctions per window at typical ripple rates), then the same
    rk_curve -> fit_m as the full-window arm;
  * frac_excised, n_ripples per window -- so a change in ripple rate under silencing
    (which would remove different amounts of time in base vs stim) is visible.

d2 arms joined per window (nearest centre within 1 s, as opto_cck_metrics.join_d2):
  d2_pkl          the stored background_uled pickle (what opto_cck_metrics reports)
  d2_plain_mat    the current plain .mat on the share
  d2_excised_mat  the _rippleablate .mat
d2_pkl is contemporaneous with the excised .mat, so it is the paired full-window arm for the
excised contrast; d2_plain_mat is also reported. `params` of both .mat arms are printed so a
unit-set change is visible.

Windows: the same NREM ∩ block, 30 s non-overlapping grid as opto_cck_metrics (CA1,
firingRate > 0.2 Hz). Same paired contrasts (session x phase; session).

    opto_cck_excised.py --test | --all
    -> data/revision/log/opto_cck_excised_{windows,session,contrasts}.csv
    -> reports/revision_figs/opto_cck_excised.png/.svg
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
import scipy.io  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import opto_cck_metrics as ocm  # noqa: E402
from fig1_metrics import (  # noqa: E402
    BIN_BR_S, BIN_MUA_S, GUARD_S, active_unit_counts, binned_counts,
    build_state_intervals, fit_m, intersect_intervals, isi_entropy, make_windows,
    rk_curve, slice_sorted, tau_ms,
)
from r35_background_vs_ripple import load_d2_mat  # noqa: E402
from r35_kl_rippleablate import MIN_FREE_SEC, subtract_intervals  # noqa: E402
from r35_sham_excision import BIN as D2_BIN, bin_mask, d2_of, glue  # noqa: E402
from run_fig1_metrics import assert_local_path  # noqa: E402

# --excise: which events are subtracted from every window. "barrages" uses the repo's
# standard loader (neuro_py load_barrage_events -> <day>.HSEn2.events.mat, NREM-restricted),
# the same definition as the Figure 3 barrage cohort (r16_event_excised_windows.py).
# d2_excised_py / d2_full_py: d2 (2024 Euclidean, p = 20) recomputed in Python on the
# 40 ms population count with event-overlapping bins dropped and the runs glued
# (r35_sham_excision.bin_mask / glue / d2_of -- the producer behind r16),
# so the barrage arms have a d2 on the same excision; d2_excised_mat stays ripple-only.
EXCISE_SUFFIX = {"ripples": "", "barrages": "_barr", "both": "_ripbarr"}
K_MAX = ocm.K_MAX
METRICS = ["d2_excised_mat", "d2_excised_py", "d2_full_py", "d2_pkl", "d2_plain_mat",
           "BR_pyr_exc_k200", "tau_ms_pyr_exc_k200", "BR_all_exc_k200",
           "isi_entropy_exc", "MUA_exc", "frac_excised"]
METRIC_LABEL = {"d2_excised_mat": "d2 (2024 Euclidean, p=20)\nripple-EXCISED .mat",
                "d2_excised_py": "d2 (2024 Euclidean, p=20)\nevent-EXCISED, Python",
                "d2_full_py": "d2 (2024 Euclidean, p=20)\nfull window, Python",
                "d2_pkl": "d2 (2024 Euclidean, p=20)\nfull window, stored pkl",
                "d2_plain_mat": "d2 (2024 Euclidean, p=20)\nfull window, current .mat",
                "BR_pyr_exc_k200": "m (pyr, 4 ms, k=200)\nripple-excised",
                "tau_ms_pyr_exc_k200": "tau (ms, pyr)\nripple-excised",
                "BR_all_exc_k200": "m (all units)\nripple-excised",
                "isi_entropy_exc": "ISI entropy (bits)\nripple-excised",
                "MUA_exc": "MUA (counts / 40 ms)\nripple-excised",
                "frac_excised": "fraction of window\nexcised (ripple time)"}


def load_ripple_iv(bp):
    from neuro_py.io import loading
    rip = loading.load_ripples_events(basepath=bp)
    iv = np.column_stack([np.asarray(rip["start"], float), np.asarray(rip["stop"], float)])
    return iv[np.argsort(iv[:, 0])]


def load_barrage_iv(bp):
    """Barrages exactly as the Figure 3 cohort loads them (HSEn2.events.mat, NREM)."""
    from neuro_py.io import loading
    ev = loading.load_barrage_events(basepath=bp)
    if ev is None or len(ev) == 0:
        return np.empty((0, 2))
    iv = np.column_stack([np.asarray(ev["start"], float), np.asarray(ev["stop"], float)])
    return iv[np.argsort(iv[:, 0])]


def load_event_iv(bp, excise):
    parts = []
    if excise in ("ripples", "both"):
        parts.append(load_ripple_iv(bp))
    if excise in ("barrages", "both"):
        parts.append(load_barrage_iv(bp))
    iv = np.vstack([p for p in parts if len(p)]) if any(len(p) for p in parts) else np.empty((0, 2))
    return iv[np.argsort(iv[:, 0])] if len(iv) else iv


def d2_py_pair(all_sp, a, b, ev_iv):
    """d2 (2024 Euclidean, p=20) on the 40 ms population count of [a, b): full window and
    with event-overlapping bins dropped + runs glued (r16 recipe, r35_sham_excision code)."""
    nb = int(round((b - a) / D2_BIN))
    edges = a + D2_BIN * np.arange(nb + 1)
    cnt = (np.searchsorted(all_sp, edges[1:], "left")
           - np.searchsorted(all_sp, edges[:-1], "left")).astype(float)
    drop = bin_mask(edges, ev_iv) if len(ev_iv) else np.zeros(nb, bool)
    return d2_of(cnt), d2_of(glue(cnt, drop))


def mat_params(bp, tag):
    day = os.path.basename(bp.rstrip("\\/"))
    f = os.path.join(bp, "complex_system", f"{day}_CA1_D2_time_resolved{tag}.mat")
    if not os.path.exists(f):
        return None
    p = scipy.io.loadmat(f, simplify_cells=True).get("params", {})
    return {k: p.get(k) for k in ("n_cells", "total_spikes", "order", "deltaT", "quant")}


def nearest_join(win, df, col, tol=1.0):
    """df: center, d2_x -> win[col] by nearest centre within tol s."""
    win[col] = np.nan
    if df is None or not len(win):
        return
    c = df.center.to_numpy()
    w = win.center_s.to_numpy()
    i = np.searchsorted(c, w)
    i0 = np.clip(i - 1, 0, len(c) - 1)
    i1 = np.clip(i, 0, len(c) - 1)
    pick = np.where(np.abs(c[i0] - w) <= np.abs(c[i1] - w), i0, i1)
    ok = np.abs(c[pick] - w) <= tol
    win.loc[ok, col] = df.d2_x.to_numpy()[pick[ok]]


def excised_metrics(a, b, segs, all_sp, pyr, units):
    d = {"free_s": float(np.sum(segs[:, 1] - segs[:, 0])) if len(segs) else 0.0}
    d["frac_excised"] = 1.0 - d["free_s"] / (b - a)
    if d["free_s"] < MIN_FREE_SEC:
        return d
    # MUA on the glued 40 ms series; ISIs pooled within segments
    mua_parts, isis = [], []
    for s0, s1 in segs:
        if s1 - s0 < BIN_MUA_S:
            continue
        sp = slice_sorted(all_sp, s0, s1)
        mua_parts.append(binned_counts(sp, s0, s1, BIN_MUA_S, presliced=True).astype(float))
        if sp.size > 1:
            isis.append(np.diff(np.unique(sp)))
    mua = np.concatenate(mua_parts) if mua_parts else np.array([])
    d["MUA_exc"] = float(mua.mean()) if mua.size else np.nan
    isis = np.concatenate(isis) if isis else np.array([])
    d["n_ISI_exc"] = int(isis.size)
    d["isi_entropy_exc"] = isi_entropy(np.cumsum(isis)) if isis.size > 2 else np.nan
    # BR on glued 4 ms active-unit counts
    for pop, sp in (("pyr", pyr), ("all", units)):
        if not sp:
            continue
        parts = [active_unit_counts(sp, s0, s1, BIN_BR_S) for s0, s1 in segs
                 if s1 - s0 >= BIN_BR_S]
        act = np.concatenate(parts) if parts else np.array([])
        if act.size <= K_MAX + 1:
            continue
        k, r_k, se = rk_curve(act, K_MAX)
        m = fit_m(k, r_k, se, K_MAX)
        d[f"BR_{pop}_exc_k{K_MAX}"] = m
        d[f"tau_ms_{pop}_exc_k{K_MAX}"] = tau_ms(np.array([m]))[0]
    return d


def process_session(bp, population="CA1", excise="ripples"):
    t0 = time.time()
    tag = ocm.session_tag(bp)
    print(f"\n=== {tag} ===", flush=True)
    blocks = ocm.load_blocks(bp)
    units, is_pyr, state_dict = ocm.load_units(bp, population)
    rip_iv = load_event_iv(bp, excise)
    print(f"  units {len(units)} | pyr {int(is_pyr.sum())} | excised events ({excise}) {len(rip_iv)}", flush=True)
    for t in ("", "_rippleablate"):
        print(f"  .mat params {t or 'plain':13s}: {mat_params(bp, t)}", flush=True)
    nrem = build_state_intervals(state_dict, GUARD_S)["NREM"]

    parts = []
    for name, (b0, b1) in sorted(blocks.items(), key=lambda kv: kv[1][0]):
        iv = intersect_intervals(nrem, np.array([[b0, b1]]))
        w = make_windows(iv, "NREM")
        if not len(w):
            continue
        w = w[w.nonoverlap].reset_index(drop=True)
        w["block"] = name
        w["phase"] = name.split("_")[0]
        w["condition"] = name.split("_")[1]
        parts.append(w)
    if not parts:
        return None
    win = pd.concat(parts, ignore_index=True)

    all_sp = np.sort(np.concatenate(units))
    pyr = [u for u, p in zip(units, is_pyr) if p]
    rows = []
    for r in win.itertuples(index=False):
        a, b = r.win_start, r.win_stop
        segs = subtract_intervals((a, b), rip_iv)
        d = excised_metrics(a, b, segs, all_sp, pyr, units)
        d["n_events"] = int(np.sum((rip_iv[:, 0] >= a) & (rip_iv[:, 0] < b))) if len(rip_iv) else 0
        d["d2_full_py"], d["d2_excised_py"] = d2_py_pair(all_sp, a, b, rip_iv)
        rows.append(d)
    win = pd.concat([win, pd.DataFrame(rows)], axis=1)

    # d2 arms -- joined, never refitted
    pk = ocm.load_stored_d2(tag)
    if pk is not None:
        pk = pk.rename(columns={"center_s": "center", "d2": "d2_x"})
    nearest_join(win, pk, "d2_pkl")
    nearest_join(win, load_d2_mat(bp, ""), "d2_plain_mat")
    nearest_join(win, load_d2_mat(bp, "_rippleablate"), "d2_excised_mat")
    for c in ("d2_pkl", "d2_plain_mat", "d2_excised_mat"):
        print(f"  {c:15s} join {win[c].notna().mean():.0%}", flush=True)
    win.insert(0, "session", tag)
    win["n_units"] = len(units)
    win["n_pyr"] = int(is_pyr.sum())
    print(f"  windows {len(win)} | median frac excised {win.frac_excised.median():.3f} "
          f"| done in {time.time()-t0:.0f}s", flush=True)
    return win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="NN3_day12 only")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--excise", default="ripples", choices=list(EXCISE_SUFFIX),
                    help="events subtracted from every window; output suffix "
                         "'' / _barr / _ripbarr (existing ripple-only files untouched)")
    args = ap.parse_args()
    if args.test:
        paths = [p for p in ocm.OPTO_SESSIONS if p.endswith("NN3\\day12")]
    elif args.sessions:
        paths = [p for p in ocm.OPTO_SESSIONS if ocm.session_tag(p) in set(args.sessions)]
    else:
        paths = ocm.OPTO_SESSIONS
    sfx = EXCISE_SUFFIX[args.excise]

    out = []
    for p in paths:
        try:
            w = process_session(p, excise=args.excise)
            if w is not None:
                out.append(w)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  [FAIL] {p}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    if not out:
        return
    win = pd.concat(out, ignore_index=True)
    win["excise"] = args.excise
    ocm.METRICS = METRICS            # contrasts()/figure() read their module globals
    ocm.METRIC_LABEL = METRIC_LABEL
    sess, con = ocm.contrasts(win)

    assert_local_path(ocm.LOGDIR)
    win.to_csv(assert_local_path(os.path.join(ocm.LOGDIR, f"opto_cck_excised_windows{sfx}.csv")), index=False)
    sess.to_csv(assert_local_path(os.path.join(ocm.LOGDIR, f"opto_cck_excised_session{sfx}.csv")), index=False)
    con.to_csv(assert_local_path(os.path.join(ocm.LOGDIR, f"opto_cck_excised_contrasts{sfx}.csv")), index=False)
    ocm.figure(sess, con, os.path.join(ocm.FIGDIR, f"opto_cck_excised{sfx}"),
               note=f"NREM, 30 s non-overlapping windows, session-phase medians; {args.excise} excised")

    pd.set_option("display.width", 220)
    print("\n=== session x phase x condition medians ===")
    print(sess.round(4).to_string(index=False))
    print(f"\n=== paired contrasts, stim - base ({args.excise} excised) ===")
    print(con.round(4).to_string(index=False))
    print(f"\nwrote -> {ocm.LOGDIR}\\opto_cck_excised_*{sfx}.csv, {ocm.FIGDIR}\\opto_cck_excised{sfx}.*")


if __name__ == "__main__":
    main()
