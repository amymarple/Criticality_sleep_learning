"""
run_fig1_metrics.py -- batch driver for the Figure 1 revision metrics.

    ENV=C:/Users/Cornell/.conda/envs/criticality/python.exe

    $ENV run_fig1_metrics.py --test          # PPP20 day4 only, then inspect
    $ENV run_fig1_metrics.py --all           # all 18 Figure 1 sessions
    $ENV run_fig1_metrics.py --sessions PPP20_day4 PPP21_day5

Writes data/revision/<animal>_<day>_<region>_fig1_metrics.pkl (local D: only).
Raw sessions are read from the network shares; nothing is written there.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig1_metrics import (  # noqa: E402
    BIN_BR_S, BIN_MUA_S, BIN_SYNC_S, GUARD_S, ISI_N_BINS, K_MAX_REF,
    K_MAX_SWEEP, MIN_FIRING_RATE, STATES, STEP_S, SYNC_AGG, WINDOW_S,
    active_matrix, active_unit_counts, binned_counts, build_state_intervals,
    fit_m, isi_entropy, join_d2, load_d2_timeline, make_windows, rk_curve,
    slice_sorted, synchrony_from_matrix, synchrony_subsampled_from_matrix, tau_ms,
)

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
OUTDIR = os.path.join(REPO, "data", "revision")

# --- read-only guarantee for the network shares ----------
WRITABLE_DRIVES = {"C", "D"}


def assert_local_path(path):
    """Raise unless `path` is on a local writable drive.

    The PreToolUse hook cannot see writes made *inside* a running process, so the
    guarantee that this script never touches the ayadata shares is enforced here.
    Called before every write.
    """
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        raise PermissionError(f"refusing to write to UNC path: {p}")
    drive = os.path.splitdrive(p)[0].rstrip(":").upper()
    if drive not in WRITABLE_DRIVES:
        raise PermissionError(
            f"refusing to write outside {sorted(WRITABLE_DRIVES)}: {p}"
        )
    return p

# The 18 Figure 1 sessions -- 6 mice, CA1
FIG1_SESSIONS = [
    r"W:\data\PPP\PPP4\day11",
    r"W:\data\PPP\PPP7\day7",  r"W:\data\PPP\PPP7\day10",
    r"W:\data\PPP\PPP7\day12", r"W:\data\PPP\PPP7\day14",
    r"W:\data\PPP\PPP8\day8",  r"W:\data\PPP\PPP8\day14",
    r"W:\data\PPP\PPP20\day4", r"W:\data\PPP\PPP20\day5",
    r"W:\data\PPP\PPP20\day7", r"W:\data\PPP\PPP20\day8",
    r"W:\data\PPP\PPP20\day9",
    r"W:\data\PPP\PPP21\day4", r"W:\data\PPP\PPP21\day5",
    r"W:\data\PPP\PPP21\day7", r"W:\data\PPP\PPP21\day8",
    r"W:\data\PPP\PPP23\day7", r"W:\data\PPP\PPP23\day8",
]


def session_id(basepath):
    p = basepath.replace("/", "\\").rstrip("\\").split("\\")
    return p[-2], p[-1]


# Per-state window caps, applied to the non-overlapping set.
#   NREM   — uncapped (Figs 2 and 3 are NREM analyses).
#   THETA / REM — uncapped: already scarce (median 7 and 15 windows per session), and
#            only 7/18 and 12/18 sessions clear the 10-window threshold a within-session
#            correlation needs. Any thinning here deletes the state.
#   nonTHETA — the only state thinned, to 150 windows (far more than any within-session
#            statistic requires).
STATE_WINDOW_CAP = {"THETA": None, "nonTHETA": 150, "REM": None, "NREM": None}


def apply_state_caps(win, caps=None):
    """Thin each state independently, evenly spaced across the session.

    Even spacing (not head-truncation) so a capped state still spans the whole
    recording and cannot pick up a time-of-session bias.
    """
    caps = STATE_WINDOW_CAP if caps is None else caps
    out = []
    for s, sub in win.groupby("state", sort=False):
        c = caps.get(s)
        if c and len(sub) > c:
            idx = np.unique(np.linspace(0, len(sub) - 1, c).round().astype(int))
            sub = sub.iloc[idx]
        out.append(sub)
    return (pd.concat(out)
              .sort_values(["state", "center_s"])
              .reset_index(drop=True))


def load_session(basepath, region="CA1"):
    """All units in `region` (not Pyr-only — matches the d2 population)."""
    from neuro_py.io import loading
    st, cm = loading.load_spikes(basepath, brainRegion=region)
    state_dict = loading.load_SleepState_states(basepath)

    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    # sorted once here — every window slice downstream uses searchsorted
    unit_spikes = [np.sort(np.asarray(s, float))
                   for s in np.asarray(st.data, dtype=object)[keep]]
    ctype = np.asarray(cm.putativeCellType)[keep]

    is_e = np.array([("Pyramidal" in str(c)) for c in ctype])
    is_i = np.array([("Interneuron" in str(c)) for c in ctype])
    return unit_spikes, is_e, is_i, state_dict


def process_session(basepath, region="CA1", seed=0, verbose=True,
                    nonoverlap_only=False, max_windows=None):
    t_start = time.time()
    animal, day = session_id(basepath)
    tag = f"{animal}_{day}_{region}"
    if verbose:
        print(f"\n=== {tag} ===", flush=True)

    unit_spikes, is_e, is_i, state_dict = load_session(basepath, region)
    n_units, n_e, n_i = len(unit_spikes), int(is_e.sum()), int(is_i.sum())
    if verbose:
        print(f"  units: {n_units} total | {n_e} E | {n_i} I", flush=True)
    if n_units == 0:
        return None

    all_spikes = np.sort(np.concatenate(unit_spikes)) if n_units else np.zeros(0)
    e_spikes = [unit_spikes[i] for i in np.where(is_e)[0]]
    i_spikes = [unit_spikes[i] for i in np.where(is_i)[0]]
    e_flat = np.sort(np.concatenate(e_spikes)) if e_spikes else np.zeros(0)
    i_flat = np.sort(np.concatenate(i_spikes)) if i_spikes else np.zeros(0)

    state_iv = build_state_intervals(state_dict, GUARD_S)
    if verbose:
        for s in STATES:
            iv = state_iv[s]
            dur = float(np.sum(iv[:, 1] - iv[:, 0])) if len(iv) else 0.0
            print(f"  {s:<9} {len(iv):3d} intervals  {dur/60:7.1f} min", flush=True)

    win = pd.concat([make_windows(state_iv[s], s) for s in STATES if len(state_iv[s])],
                    ignore_index=True)
    if not len(win):
        print("  [skip] no windows"); return None

    rng = np.random.default_rng(seed)
    sync_target = max(5, int(np.floor(0.8 * len(e_spikes)))) if e_spikes else None

    n_slide = len(win)
    if nonoverlap_only:
        # the non-overlapping subset -- ~15x fewer windows than sliding
        win = win[win.nonoverlap].reset_index(drop=True)
    win = apply_state_caps(win)
    if max_windows and len(win) > max_windows:
        # global thinning, not stratified: a uniform stride over a state-ordered
        # table thins every state proportionally, reducing THETA (median 7
        # windows/session) and REM (15) most. STATE_WINDOW_CAP applies caps per
        # state instead.
        win = win.iloc[:: int(np.ceil(len(win) / max_windows))].reset_index(drop=True)
    if verbose:
        print(f"  windows: {n_slide} sliding -> processing {len(win)} "
              f"({'non-overlapping only' if nonoverlap_only else 'all'})", flush=True)

    rows = []
    t_loop = time.time()
    for wi, r in enumerate(win.itertuples(index=False)):
        if verbose and wi and wi % 2000 == 0:
            el = time.time() - t_loop
            print(f"    {wi}/{len(win)}  {el:.0f}s elapsed, "
                  f"~{el/wi*(len(win)-wi):.0f}s left", flush=True)
        t0, t1 = r.win_start, r.win_stop
        dur = t1 - t0
        d = {"state": r.state, "iv_idx": r.iv_idx, "w_idx": r.w_idx,
             "win_start": t0, "win_stop": t1, "center_s": r.center_s,
             "nonoverlap": r.nonoverlap}

        # slice once per window (searchsorted, not a boolean scan of the session)
        w_all = slice_sorted(all_spikes, t0, t1)

        # --- MUA at 40 ms (identical binning to the d2 input) ---
        mua = binned_counts(w_all, t0, t1, BIN_MUA_S, presliced=True)
        d["MUA_mean"] = float(mua.mean()) if mua.size else np.nan
        d["MUA_std"] = float(mua.std()) if mua.size else np.nan
        d["MUA_fano"] = float(mua.var() / mua.mean()) if mua.size and mua.mean() > 0 else np.nan

        # --- E/I over the whole window (per-1s ratios blow up when I -> 0) ---
        ne = int(slice_sorted(e_flat, t0, t1).size) if e_flat.size else 0
        ni = int(slice_sorted(i_flat, t0, t1).size) if i_flat.size else 0
        d["n_E_spikes"], d["n_I_spikes"] = ne, ni
        d["E_rate"], d["I_rate"] = ne / dur, ni / dur
        d["EI_logratio"] = float(np.log((ne / dur + 1e-12) / (ni / dur + 1e-12))) if ni > 0 else np.nan

        # --- synchrony at 50 ms: one activity matrix, raw + subsampled from it ---
        act_mat = active_matrix(e_spikes, t0, t1)
        d["synchrony_raw_mean"], d["synchrony_raw_std"] = synchrony_from_matrix(act_mat)
        d["synchrony_sub_mean"], d["synchrony_sub_std"] = \
            synchrony_subsampled_from_matrix(act_mat, sync_target, rng)

        # --- ISI entropy, raw (pooled and per-unit; decision deferred) ---
        d["isi_entropy_pooled"] = isi_entropy(w_all)
        d["n_ISI"] = max(len(w_all) - 1, 0)
        per_unit = [isi_entropy(slice_sorted(u, t0, t1)) for u in unit_spikes]
        per_unit = [v for v in per_unit if np.isfinite(v)]
        d["isi_entropy_perunit_mean"] = float(np.mean(per_unit)) if per_unit else np.nan
        d["n_units_with_isi"] = len(per_unit)

        # --- branching ratio at 4 ms, r_k once, exponential re-fit per k_max ---
        # Two populations: including interneurons moves THETA m by ~0.06 and
        # reorders the states.
        #   pyr = the population behind the stored per-session results (Fig. 2g)
        #   all = matched to d2's population, for the d2-vs-BR comparison
        for pop, spikes in (("pyr", e_spikes), ("all", unit_spikes)):
            if not spikes:
                continue
            act = active_unit_counts(spikes, t0, t1, BIN_BR_S)
            k, r_k, se = rk_curve(act, K_MAX_REF)
            for km in K_MAX_SWEEP:
                m = fit_m(k, r_k, se, km)
                d[f"BR_{pop}_k{km}"] = m
                d[f"tau_ms_{pop}_k{km}"] = tau_ms(np.array([m]))[0]
            d[f"BR_{pop}_r1"] = float(r_k[0]) if np.isfinite(r_k[0]) else np.nan
        rows.append(d)

    wdf = pd.DataFrame(rows)

    # --- optional d2 join ---
    d2_df, d2_src = load_d2_timeline(basepath, region)
    wdf, join_rate = join_d2(wdf, d2_df)
    if verbose:
        print(f"  d2 source: {d2_src} | join rate {join_rate:.1%}", flush=True)

    # --- state-level metrics ---
    state_rows = []
    for s in STATES:
        iv = state_iv[s]
        if not len(iv):
            continue
        mask = np.zeros(all_spikes.shape, bool)
        for a, b in iv:
            mask |= (all_spikes >= a) & (all_spikes < b)
        dur = float(np.sum(iv[:, 1] - iv[:, 0]))
        rates = []
        for u in unit_spikes:
            c = binned_counts(u, 0, max(1.0, float(all_spikes.max())), 1.0)
            if c.size:
                rates.append(c)
        fr = np.concatenate(rates) if rates else np.zeros(0)
        p5, p95 = (np.percentile(fr, [5, 95]) if fr.size else (np.nan, np.nan))
        state_rows.append({
            "state": s, "n_intervals": len(iv), "total_dur_s": dur,
            "n_spikes": int(mask.sum()),
            "fr_range_log10": float(np.log10((p95 + 1e-9) / (p5 + 1e-9))) if fr.size else np.nan,
        })
    sdf = pd.DataFrame(state_rows)

    out = {
        "meta": {
            "animal_name": animal, "day": day, "region": region, "basepath": basepath,
            "n_units": n_units, "n_E_units": n_e, "n_I_units": n_i,
            "window_size_s": WINDOW_S, "step_s": STEP_S, "guard_s": GUARD_S,
            "bin_mua_s": BIN_MUA_S, "bin_br_s": BIN_BR_S, "bin_sync_s": BIN_SYNC_S,
            "sync_agg": SYNC_AGG, "sync_subsample_n": sync_target,
            "isi_n_bins": ISI_N_BINS, "min_firing_rate": MIN_FIRING_RATE,
            "k_max_sweep": list(K_MAX_SWEEP), "k_max_ref": K_MAX_REF,
            "n_windows_sliding_total": n_slide, "nonoverlap_only": nonoverlap_only,
            "d2_source": d2_src, "d2_join_rate": join_rate,
            "created": datetime.now().isoformat(timespec="seconds"),
        },
        "windows": wdf,
        "state_level": sdf,
    }

    assert_local_path(OUTDIR)
    os.makedirs(OUTDIR, exist_ok=True)
    suffix = "_nonoverlap" if nonoverlap_only else ""
    path = assert_local_path(os.path.join(OUTDIR, f"{tag}_fig1_metrics{suffix}.pkl"))
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"  windows: {len(wdf)} ({int(wdf.nonoverlap.sum())} non-overlapping)")
        print(f"  saved -> {path}   [{time.time()-t_start:.1f}s]", flush=True)
    return out


def summarize(out):
    w = out["windows"]
    print("\n--- per-state summary (median over windows) ---")
    cols = ["d2", "BR_pyr_k200", "BR_all_k200", "tau_ms_pyr_k200", "MUA_mean",
            "EI_logratio", "synchrony_raw_mean", "isi_entropy_pooled"]
    cols = [c for c in cols if c in w.columns]
    print(w.groupby("state")[cols].median().round(3).to_string())
    for pop in ("pyr", "all"):
        kc = [f"BR_{pop}_k{k}" for k in K_MAX_SWEEP if f"BR_{pop}_k{k}" in w.columns]
        if kc:
            print(f"\n--- k_max sweep, {pop} units (median m by state) ---")
            print(w.groupby("state")[kc].median().round(3).to_string())
    print("\n--- n windows ---")
    print(w.groupby("state")["nonoverlap"].agg(["size", "sum"]).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="PPP20 day4 only")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--region", default="CA1")
    ap.add_argument("--nonoverlap-only", action="store_true",
                    help="only non-overlapping windows (~15x fewer; the R2.2 subset)")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap window count by stratified thinning")
    args = ap.parse_args()

    if args.test:
        paths = [r"W:\data\PPP\PPP20\day4"]
    elif args.sessions:
        want = set(args.sessions)
        paths = [p for p in FIG1_SESSIONS
                 if "_".join(session_id(p)) in want or os.path.basename(p) in want]
    else:
        paths = FIG1_SESSIONS

    print(f"{len(paths)} session(s) -> {OUTDIR}")
    ok = 0
    for p in paths:
        try:
            out = process_session(p, args.region,
                                  nonoverlap_only=args.nonoverlap_only,
                                  max_windows=args.max_windows)
            if out is not None:
                ok += 1
                if args.test:
                    summarize(out)
        except Exception as e:
            import traceback
            print(f"  [FAIL] {p}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    print(f"\ndone: {ok}/{len(paths)}")


if __name__ == "__main__":
    main()
