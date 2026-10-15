"""
fig1_dropin_metrics.py -- drop-in window columns for the Figure 1 analyses.

Adds three metric families on the EXACT window grid of the existing
data/revision/*_fig1_metrics_nonoverlap.pkl files (join key: session, state,
center_s), without touching those files:

  sw_broadband        buzcode SleepScoreMaster's broadbandSlowWave, window mean.
                      For these sessions SWweights='PSS' with IRASA=1, i.e. it is
                      itself a power-spectrum-slope score, min-max NORMALIZED 0-1
                      PER SESSION; renamed here from the notebook's "SWSlope"
                      column name, used within-session only.
  psd_slope_<range>   raw aperiodic (1/f) exponent: Welch PSD of the SleepScore
                      slow-wave channel per 30 s window, log-log linear fit over
                      a fixed frequency range (55-65 Hz line noise excluded).
                      Cross-session comparable, unlike sw_broadband.
  synchrony_sub81_*   fraction-active synchrony subsampled to a FIXED 81 E units
                      (the minimum n_E across the 18 sessions), so the column is
                      comparable across sessions. The existing synchrony_sub_*
                      used 0.8*n_E, a per-session target.

LFP comes from <basename>.SleepScoreLFP.LFP.mat (full-length SW channel at
1250 Hz, present in all 18 sessions) -- the 8-19 GB .lfp files are only ever
opened as a read-only memmap for a 1000-sample channel-identity check
(SWchanID is 1-based MATLAB; .lfp column = SWchanID - 1).

scipy.optimize.curve_fit and sklearn fail in this environment (missing LAPACK
DLLs); fits use scipy.stats.linregress.

    ENV=C:/Users/Cornell/.conda/envs/criticality/python.exe
    $ENV fig1_dropin_metrics.py --test          # PPP20 day4, prints summary
    $ENV fig1_dropin_metrics.py --all           # 18 sessions
    $ENV fig1_dropin_metrics.py --all --skip-lfp

Writes data/revision/<animal>_<day>_<region>_fig1_dropin_nonoverlap.pkl and
data/revision/log/fig1_dropin_windows.csv. Local D: only; shares are read-only.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.io
import scipy.stats
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig1_metrics import (  # noqa: E402
    GUARD_S, STATES, SYNC_SUBSAMPLE_DRAWS, active_matrix, build_state_intervals,
    make_windows, synchrony_subsampled_from_matrix,
)
from run_fig1_metrics import (  # noqa: E402
    FIG1_SESSIONS, OUTDIR, apply_state_caps, assert_local_path, load_session,
    session_id,
)

LOGDIR = os.path.join(OUTDIR, "log")

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
SYNC_TARGET_FIXED = 81      # = min n_E over the 18 sessions
NPERSEG_S = 2.0             # Welch segment length
NOTCH_HZ = (55.0, 65.0)     # line noise, always excluded from fits
# Fit ranges. PRIMARY = 20-45 Hz: chosen for split-half reliability and
# agreement with the expected (Gao 2017) direction of the E/I-ratio relation.
# The literature default 30-45 Hz (Gao 2017 / Lendner 2020) performs poorly in
# CA1 -- slow gamma sits inside the band -- and is kept only as the
# convention-matched variant. Never fit <20 Hz (theta harmonics) or >90 Hz
# (CA1 ripple band ~100-250 Hz).
FIT_RANGES = {
    "20_45": (20.0, 45.0),   # primary (audit-selected)
    "30_45": (30.0, 45.0),   # literature convention, gamma-contaminated in CA1
    "30_55": (30.0, 55.0),
    "65_90": (65.0, 90.0),
}


# ----------------------------------------------------------------------------
# MATLAB-side loaders (read-only)
# ----------------------------------------------------------------------------
def _find_file(basepath, suffix):
    hits = sorted(glob.glob(os.path.join(basepath, "*" + suffix)))
    return hits[0] if hits else None


def _loadmat_any(path):
    try:
        return scipy.io.loadmat(path, simplify_cells=True)
    except NotImplementedError:          # v7.3
        import mat73
        return mat73.loadmat(path)


def load_sleepscore_metrics(basepath):
    """t_clus / broadbandSlowWave (1 s grid) + SW channel id from
    <basename>.SleepState.states.mat. Returns (t_clus, bb, sw_chan, meta)."""
    path = _find_file(basepath, ".SleepState.states.mat")
    if path is None:
        raise FileNotFoundError(f"no SleepState.states.mat in {basepath}")
    ss = _loadmat_any(path)["SleepState"]
    parms = ss["detectorinfo"]["detectionparms"]
    msm = parms["SleepScoreMetrics"]
    t_clus = np.asarray(msm["t_clus"], float).ravel()
    bb = np.asarray(msm["broadbandSlowWave"], float).ravel()
    if t_clus.size != bb.size:
        raise ValueError(f"t_clus/broadbandSlowWave length mismatch in {path}")

    sw_chan = None
    for container, key in ((msm, "SWchanID"), (parms, "SWchannum"),
                           (ss["detectorinfo"], "SWchannum")):
        try:
            sw_chan = int(np.ravel(container[key])[0])
            break
        except Exception:
            continue

    lfpp = msm.get("LFPparams", {}) if isinstance(msm, dict) else {}
    meta = {
        "sw_source": os.path.basename(path),
        "sw_grid_step_s": float(np.median(np.diff(t_clus))) if t_clus.size > 1 else np.nan,
        "sw_weights": str(lfpp.get("SWweights", "?")),
        "sw_irasa": int(np.ravel(lfpp["IRASA"])[0]) if isinstance(lfpp, dict) and "IRASA" in lfpp else None,
    }
    return t_clus, bb, sw_chan, meta


def load_sleepscore_lfp(basepath):
    """Full-length SW channel (int16, 1250 Hz) from <basename>.SleepScoreLFP.LFP.mat.
    Returns (swLFP int16, sf, t0, sw_chan_lfp, path)."""
    path = _find_file(basepath, ".SleepScoreLFP.LFP.mat")
    if path is None:
        raise FileNotFoundError(f"no SleepScoreLFP.LFP.mat in {basepath}")
    d = _loadmat_any(path)
    s = d.get("SleepScoreLFP", d)
    sw = np.asarray(s["swLFP"]).ravel()            # keep int16
    sf = float(np.ravel(s["sf"])[0])
    t0 = float(np.ravel(s["t"])[0]) if "t" in s else 0.0
    chan = int(np.ravel(s["SWchanID"])[0]) if "SWchanID" in s else None
    return sw, sf, t0, chan, path


def verify_channel(basepath, swLFP, sw_chan, n_check=1000, offset=100_000):
    """Prove which .lfp column swLFP came from (read-only memmap, ~0.5 MB read).
    Returns '1-based', '0-based', 'MISMATCH', or None if unverifiable."""
    if sw_chan is None:
        return None
    ss_path = _find_file(basepath, ".SleepScoreLFP.LFP.mat")
    basename = os.path.basename(ss_path).replace(".SleepScoreLFP.LFP.mat", "")
    lfp_path = os.path.join(basepath, basename + ".lfp")
    if not os.path.exists(lfp_path):
        lfp_path = os.path.join(basepath, basename + ".eeg")
        if not os.path.exists(lfp_path):
            return None
    try:
        from neuro_py.io import loading
        n_channels = int(loading.loadXML(basepath)[0])
    except Exception:
        return None
    mm = np.memmap(lfp_path, dtype=np.int16, mode="r")
    n_samp = mm.size // n_channels
    if offset + n_check > min(n_samp, swLFP.size):
        offset = 0
        n_check = min(n_check, n_samp, swLFP.size)
    view = mm[: n_samp * n_channels].reshape(n_samp, n_channels)
    ref = np.asarray(swLFP[offset:offset + n_check])
    verdict = "MISMATCH"
    for col, label in ((sw_chan - 1, "1-based"), (sw_chan, "0-based")):
        if 0 <= col < n_channels and np.array_equal(np.asarray(view[offset:offset + n_check, col]), ref):
            verdict = label
            break
    del view, mm
    return verdict


# ----------------------------------------------------------------------------
# Per-window metrics
# ----------------------------------------------------------------------------
def sw_broadband_window(t_clus, bb, t0, t1):
    lo, hi = np.searchsorted(t_clus, (t0, t1))
    if hi <= lo:
        return np.nan, 0
    return float(np.mean(bb[lo:hi])), int(hi - lo)


def psd_slopes(x, sf, nperseg_s=NPERSEG_S, fit_ranges=FIT_RANGES, notch=NOTCH_HZ):
    """Welch (median-averaged) PSD + log-log linregress slope per fit range."""
    out = {}
    for name in fit_ranges:
        out[f"psd_slope_{name}"] = np.nan
        out[f"psd_r2_{name}"] = np.nan
    nps = int(round(nperseg_s * sf))
    if x.size < 2 * nps:
        return out
    f, p = welch(x, fs=sf, nperseg=nps, noverlap=nps // 2,
                 average="median", detrend="constant")
    for name, (lo, hi) in fit_ranges.items():
        m = (f >= lo) & (f <= hi) & ~((f >= notch[0]) & (f <= notch[1])) & (p > 0)
        if m.sum() < 5:
            continue
        res = scipy.stats.linregress(np.log10(f[m]), np.log10(p[m]))
        out[f"psd_slope_{name}"] = float(res.slope)
        out[f"psd_r2_{name}"] = float(res.rvalue ** 2)
    return out


# ----------------------------------------------------------------------------
# Session driver
# ----------------------------------------------------------------------------
def load_grid(tag):
    """The window grid IS the existing pickle's — guarantees joinability."""
    path = os.path.join(OUTDIR, f"{tag}_fig1_metrics_nonoverlap.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — run run_fig1_metrics.py first; the drop-in grid "
            "is defined by that pickle")
    with open(path, "rb") as f:
        d = pickle.load(f)
    grid = d["windows"][["state", "iv_idx", "w_idx", "win_start", "win_stop",
                         "center_s", "nonoverlap"]].copy()
    return grid, d["meta"], path


def check_grid_reproducible(grid, state_dict):
    """Rebuild the grid from SleepState and report drift (non-fatal: the pickle
    grid is authoritative for joining either way)."""
    state_iv = build_state_intervals(state_dict, GUARD_S)
    parts = [make_windows(state_iv[s], s) for s in STATES if len(state_iv[s])]
    if not parts:
        return 0.0
    win = pd.concat(parts, ignore_index=True)
    win = apply_state_caps(win[win.nonoverlap].reset_index(drop=True))
    a = set(zip(grid.state, grid.center_s.round(3)))
    b = set(zip(win.state, win.center_s.round(3)))
    return len(a & b) / max(len(a), 1)


def process_session(basepath, region="CA1", seed=0, skip_lfp=False,
                    skip_spikes=False, do_verify_channel=True, verbose=True):
    t_start = time.time()
    animal, day = session_id(basepath)
    tag = f"{animal}_{day}_{region}"
    if verbose:
        print(f"\n=== {tag} ===", flush=True)

    grid, base_meta, base_path = load_grid(tag)
    if verbose:
        print(f"  grid: {len(grid)} windows from {os.path.basename(base_path)}",
              flush=True)

    meta = {
        "animal_name": animal, "day": day, "region": region, "basepath": basepath,
        "source_pickle": os.path.basename(base_path),
        "source_pickle_created": base_meta.get("created"),
        "n_windows": len(grid),
        "sync_target_fixed": SYNC_TARGET_FIXED,
        "sync_draws": SYNC_SUBSAMPLE_DRAWS, "seed": seed,
        "psd_nperseg_s": NPERSEG_S, "psd_average": "median",
        "fit_ranges": dict(FIT_RANGES), "notch_excluded_hz": list(NOTCH_HZ),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    cols = {}

    # --- spikes: fixed-target synchrony -------------------------------------
    if not skip_spikes:
        unit_spikes, is_e, is_i, state_dict = load_session(basepath, region)
        n_e = int(is_e.sum())
        if n_e != base_meta.get("n_E_units"):
            print(f"  [warn] n_E now {n_e} vs {base_meta.get('n_E_units')} in the "
                  "base pickle — spike sorting changed since the base run?")
        grid_match = check_grid_reproducible(grid, state_dict)
        meta["grid_reproducible_frac"] = grid_match
        if verbose:
            print(f"  grid reproducible from SleepState: {grid_match:.1%}", flush=True)
        if grid_match < 0.999:
            print("  [warn] grid drift vs SleepState — pickle grid still used")

        e_spikes = [unit_spikes[i] for i in np.where(is_e)[0]]
        rng = np.random.default_rng(seed)
        sub_mean = np.full(len(grid), np.nan)
        sub_std = np.full(len(grid), np.nan)
        for i, r in enumerate(grid.itertuples(index=False)):
            act = active_matrix(e_spikes, r.win_start, r.win_stop)
            sub_mean[i], sub_std[i] = synchrony_subsampled_from_matrix(
                act, SYNC_TARGET_FIXED, rng)
        cols["synchrony_sub81_mean"] = sub_mean
        cols["synchrony_sub81_std"] = sub_std
        meta["n_E_units"] = n_e
        if verbose:
            print(f"  synchrony_sub81 done ({n_e} E units)", flush=True)

    # --- ripple rate: event START times, searchsorted per window -- one count
    #     per event (avoids double-counting an event at a zero-width epoch edge) --
    rip = rip_amp = None
    try:
        from neuro_py.io import loading as _loading
        ev = _loading.load_ripples_events(basepath=basepath)
        if ev is not None and len(ev):
            order = np.argsort(np.asarray(ev["start"], float))
            rip = np.asarray(ev["start"], float)[order]
            if "amplitude" in ev.columns:
                rip_amp = np.asarray(ev["amplitude"], float)[order]
    except Exception as e:                                       # noqa: BLE001
        print(f"  [warn] ripple load failed ({type(e).__name__}: {e})")
    if rip is not None:
        n_rip = np.zeros(len(grid), int)
        amp_mean = np.full(len(grid), np.nan)
        for i, r in enumerate(grid.itertuples(index=False)):
            i0 = np.searchsorted(rip, r.win_start, side="left")
            i1 = np.searchsorted(rip, r.win_stop, side="right")
            n_rip[i] = i1 - i0
            if rip_amp is not None and i1 > i0:
                amp_mean[i] = float(np.nanmean(rip_amp[i0:i1]))
        cols["n_ripples"] = n_rip
        cols["ripple_rate"] = n_rip / (grid.win_stop - grid.win_start).to_numpy()
        cols["ripple_amp_mean"] = amp_mean     # NaN when a window has no ripples;
        # raw detector units — session-relative, z-score per session before use
        meta["ripple_source"] = "load_ripples_events, start times + amplitude"
        meta["n_ripples_total"] = int(rip.size)
        if verbose:
            print(f"  ripple rate/amplitude done ({rip.size} events)", flush=True)
    else:
        print("  [warn] ripple_rate columns absent for this session")

    # --- LFP-side: sw_broadband + psd slopes --------------------------------
    if not skip_lfp:
        t_clus, bb, sw_chan_states, ss_meta = load_sleepscore_metrics(basepath)
        meta.update(ss_meta)
        swLFP, sf, lfp_t0, sw_chan_lfp, lfp_src = load_sleepscore_lfp(basepath)
        sw_chan = sw_chan_lfp if sw_chan_lfp is not None else sw_chan_states
        if (sw_chan_lfp is not None and sw_chan_states is not None
                and sw_chan_lfp != sw_chan_states):
            print(f"  [warn] SWchanID disagrees: LFP.mat {sw_chan_lfp} vs "
                  f"states.mat {sw_chan_states}")
        meta.update({"lfp_source": os.path.basename(lfp_src), "lfp_fs": sf,
                     "lfp_t0": lfp_t0, "sw_chan_matlab": sw_chan,
                     "sw_chan_0based": (sw_chan - 1) if sw_chan is not None else None})

        if do_verify_channel:
            verdict = verify_channel(basepath, swLFP, sw_chan)
            meta["channel_verified"] = verdict
            if verbose:
                print(f"  channel identity vs .lfp: {verdict}", flush=True)
            if verdict == "MISMATCH":
                print("  [warn] swLFP matches neither SWchanID-1 nor SWchanID "
                      "in the .lfp — slopes still computed from swLFP as stored")

        sw_mean = np.full(len(grid), np.nan)
        sw_n = np.zeros(len(grid), int)
        slope_cols = {k: np.full(len(grid), np.nan)
                      for name in FIT_RANGES
                      for k in (f"psd_slope_{name}", f"psd_r2_{name}")}
        n_lfp = swLFP.size
        for i, r in enumerate(grid.itertuples(index=False)):
            sw_mean[i], sw_n[i] = sw_broadband_window(t_clus, bb, r.win_start, r.win_stop)
            i0 = int(round((r.win_start - lfp_t0) * sf))
            i1 = int(round((r.win_stop - lfp_t0) * sf))
            if i0 < 0 or i1 > n_lfp:
                continue
            res = psd_slopes(swLFP[i0:i1].astype(np.float64), sf)
            for k, v in res.items():
                slope_cols[k][i] = v
        cols["sw_broadband"] = sw_mean
        cols["sw_n"] = sw_n
        cols.update(slope_cols)
        if verbose:
            print("  sw_broadband + psd slopes done", flush=True)

    wdf = grid.copy()
    for k, v in cols.items():
        wdf[k] = v

    out = {"meta": meta, "windows": wdf}
    assert_local_path(OUTDIR)
    os.makedirs(OUTDIR, exist_ok=True)
    path = assert_local_path(os.path.join(OUTDIR, f"{tag}_fig1_dropin_nonoverlap.pkl"))
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"  saved -> {path}   [{time.time()-t_start:.1f}s]", flush=True)
    return out


def summarize(out):
    w = out["windows"]
    cols = [c for c in ("sw_broadband", "psd_slope_30_45", "psd_r2_30_45",
                        "psd_slope_65_90", "synchrony_sub81_mean") if c in w.columns]
    if cols:
        print("\n--- per-state summary (median over windows) ---")
        print(w.groupby("state")[cols].median().round(4).to_string())
        print("\n--- n windows with finite values ---")
        print(w.groupby("state")[cols].apply(
            lambda g: g.notna().sum()).to_string())


def write_log_csv():
    """Concatenate every drop-in pickle into one convenience CSV."""
    rows = []
    for f in sorted(glob.glob(os.path.join(OUTDIR, "*_fig1_dropin_nonoverlap.pkl"))):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        m = d["meta"]
        w = d["windows"].copy()
        w["session"] = f"{m['animal_name']}_{m['day']}_{m['region']}"
        w["animal"] = m["animal_name"]
        rows.append(w)
    if not rows:
        return
    allw = pd.concat(rows, ignore_index=True)
    assert_local_path(LOGDIR)
    os.makedirs(LOGDIR, exist_ok=True)
    out = assert_local_path(os.path.join(LOGDIR, "fig1_dropin_windows.csv"))
    allw.to_csv(out, index=False)
    print(f"\nlog csv -> {out}  ({len(allw)} windows, "
          f"{allw.session.nunique()} sessions)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="PPP20 day4 only")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--region", default="CA1")
    ap.add_argument("--skip-lfp", action="store_true")
    ap.add_argument("--skip-spikes", action="store_true")
    ap.add_argument("--no-verify-channel", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompute sessions whose drop-in pickle already exists")
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
        tag = f"{'_'.join(session_id(p))}_{args.region}"
        done = os.path.join(OUTDIR, f"{tag}_fig1_dropin_nonoverlap.pkl")
        if os.path.exists(done) and not args.force and not args.test:
            print(f"\n=== {tag} === [skip] exists (use --force to redo)")
            ok += 1
            continue
        try:
            out = process_session(p, args.region, skip_lfp=args.skip_lfp,
                                  skip_spikes=args.skip_spikes,
                                  do_verify_channel=not args.no_verify_channel)
            if out is not None:
                ok += 1
                if args.test:
                    summarize(out)
        except Exception as e:
            import traceback
            print(f"  [FAIL] {p}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    print(f"\ndone: {ok}/{len(paths)}")
    write_log_csv()


if __name__ == "__main__":
    main()
