"""
r35_swr_property_control.py -- the replay/non-replay SWR-rate dissociation (submitted
Fig. 2e) under four controls for SWR duration, amplitude, participating-cell count, and
decoded trajectory length.

    $PY r35_swr_property_control.py --test     # PPP20_day4 only, with the reproduction check
    $PY r35_swr_property_control.py --all

Within NREM, d2 (2024 Euclidean, p=20) is correlated against the rate of SWRs that carry
significant replay (`rate_in`) and the rate of SWRs that do not (`rate_out`). Event
definition, matched to `generate_replay_non_replay.ipynb`: ripples (`load_ripples_events`)
intersected with NREM (nelpy `&`); a candidate replay event is significant if
min(p-value over the two template directions) < 0.05; a ripple is replay-carrying if it
overlaps any significant interval, else non-replay; rates are counted per 30 s d2 window,
stepped 2 s.

Four controls, in order of stringency: (a) the replay vs non-replay comparison of SWR
duration, amplitude, participating-cell count and decoded trajectory span; (b) a 1:1
propensity-matched non-replay subset (matched on log duration, amplitude, n_cells),
re-correlated with d2, carrying the replay set's properties but not its content;
(c) partial correlation of d2 with rate_in controlling for the window's mean SWR
duration, amplitude, cell count and total SWR rate; (d) within the replay set only, the
correlation of d2 with mean decoded trajectory span, and of rate_in with d2 controlling
for it.

Session is the unit of analysis. Within-session correlations are computed on the
non-overlapping window subset (every 15th window, 30 s apart); the all-window value
(published 2 s-step grid) is reported alongside. Across sessions, correlations are
averaged with Fisher z and compared with a paired Wilcoxon on z.

Inputs: SESSIONS (session -> raw data path); SESSION_TEMPLATE / NOTEBOOK_EPOCHS (below,
per-session replay-event selection); data/ripple_barr/replay_non_replay/<session>_CA1.pkl
(published d2 and window grid).

Reads the shares read-only; writes only to D:.
    -> data/revision/r35/<session>_swr_events.pkl
    -> data/revision/log/r35_event_properties.csv
    -> data/revision/log/r35_matched_control.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import scipy.io
from scipy import stats

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
RES = r"D:\Documents\GitHub\R_Project\PPP\codebook\Criticality_analysis\replay_results"
PUB = os.path.join(REPO, "data", "ripple_barr", "replay_non_replay")
OUTDIR = os.path.join(REPO, "data", "revision", "r35")
LOGDIR = os.path.join(REPO, "data", "revision", "log")

WIN_S = 30.0          # d2 window, as in the producer notebook
STEP_S = 2.0          # published grid step
NONOVL = int(round(WIN_S / STEP_S))   # 15 -> every 15th window is 30 s apart
SIG_P = 0.05
PROPS = ["duration", "amplitude", "frequency", "peakNormedPower", "n_cells", "n_spikes"]
MATCH_ON = ["log_duration", "amplitude", "n_cells"]

# The Fig. 2e session set is whatever is in data/ripple_barr/replay_non_replay/ --
# the figure notebook globs the directory. This differs from the 9 sessions in
# Fig. 2a/g/h: it swaps PPP4_day11 -> PPP4_day10 and PPP7_day10 -> PPP7_day8, and it
# pools one rat (HP18_day16) into an otherwise mouse panel. See the docstring of
# report_session_set().
SESSIONS = {
    # PPP4_day11, not day10 -- day10 has only one epoch decoded and is retired to
    # replay_non_replay/_superseded/ by r35_make_ppp4day11_pkl.py.
    "PPP4_day11":  r"W:\data\PPP\PPP4\day11",
    "PPP7_day8":   r"W:\data\PPP\PPP7\day8",
    "PPP8_day8":   r"W:\data\PPP\PPP8\day8",
    "PPP20_day4":  r"W:\data\PPP\PPP20\day4",
    "PPP20_day5":  r"W:\data\PPP\PPP20\day5",
    "PPP21_day4":  r"W:\data\PPP\PPP21\day4",
    "PPP21_day5":  r"W:\data\PPP\PPP21\day5",
    "PPP23_day7":  r"W:\data\PPP\PPP23\day7",
    "PPP23_day8":  r"W:\data\PPP\PPP23\day8",
}


def assert_local_path(path):
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        raise PermissionError(f"refusing to write to UNC path: {p}")
    if os.path.splitdrive(p)[0].rstrip(":").upper() not in {"C", "D"}:
        raise PermissionError(f"refusing to write outside C:/D:: {p}")
    return p


# ---------------------------------------------------------------------------
# per-session event table
# ---------------------------------------------------------------------------
# Per-session template actually used by `generate_replay_non_replay.ipynb`'s cell 9, which
# hardcodes `epochs = ['epoch1','epoch3','epoch5']` and a single `template` set by hand per
# session; the notebook itself preserves only the last one used, so this mapping records it
# per session and is not recoverable from the notebook code.
#
# Using every epoch/template combination present in the .mat file instead changes which
# events the per-event property measures are computed over (the extra combinations exist
# only for PPP4_day11), so the per-session template below is used throughout rather than
# the full combination set.
SESSION_TEMPLATE = {
    "PPP20_day4": "template04", "PPP20_day5": "template04",
    "PPP21_day4": "template04", "PPP21_day5": "template04",
    "PPP23_day7": "template04", "PPP23_day8": "template04",
    "PPP8_day8":  "template02", "PPP4_day11": "template02",
    "PPP7_day8":  "template02",
}
NOTEBOOK_EPOCHS = ("epoch1", "epoch3", "epoch5")

# Which sleep epochs each session ACTUALLY contributes. The d2 timeline covers only part
# of most recordings, so events outside it are inert and the epoch list is mostly a no-op
# -- epochs 1/3/5, with the time range doing the filtering.
#
#   session      d2 NREM span (s)   sig events by epoch (1/3/5)   effective epochs
#   PPP20_day4        624-29933          619 /  828 / 1225            1,3,5
#   PPP20_day5        897-29144          714 /  934 / 1198            1,3,5
#   PPP21_day4        230-17218          575 / 1080 /   --            1,3
#   PPP21_day5        853-28338          872 /  847 /  908            1,3,5
#   PPP23_day7      20253-26928            0 /    0 / 1015            5      <- post-task only
#   PPP23_day8      18036-25717            0 /    0 / 1328            5      <- post-task only
#   PPP8_day8       10314-26220           -- /  855 / 1041            3,5
#   PPP4_day11       9455-23157            0 /  953 /  445            3,5
#   PPP7_day8        9982-25578           -- /  556 /  560            3,5
#
# PPP23_day7 and PPP23_day8 contribute ONLY the final sleep epoch, so they are post-task
# sleep alone; any pre/post-learning contrast must account for that.


def load_replay_events(sess, strict=True):
    """-> DataFrame of every decoded CANDIDATE event with its significance and
    decoded-trajectory geometry. The posterior is (nPos, 2*nTime): the two template
    directions concatenated along time.

    strict=True restricts to the session's own (epoch, template) selection above, which
    is what the published rates were built from. Sessions with no recorded template use
    every combo in the file."""
    f = os.path.join(RES, f"{sess}_replay_sequences.mat")
    d = scipy.io.loadmat(f)
    present = sorted({re.match(r"(epoch\d+_template\d+)", k).group(1)
                      for k in d if re.match(r"epoch\d+_template\d+", k)})
    if strict and sess in SESSION_TEMPLATE:
        t = SESSION_TEMPLATE[sess]
        keys = [f"{e}_{t}" for e in NOTEBOOK_EPOCHS if f"{e}_{t}_eventtimes" in d]
        dropped = [k for k in present if k not in keys]
        if dropped:
            print(f"    ({sess}: using {t}, epochs 1/3/5 -- ignoring "
                  f"{len(dropped)} other combo(s): {', '.join(dropped)})", flush=True)
    else:
        keys = present
        if strict:
            print(f"    [!] {sess}: no recorded template, falling back to all "
                  f"{len(keys)} combos", flush=True)
    rows = []
    for k in keys:
        ev = np.squeeze(d[k + "_eventtimes"]).reshape(-1, 2)
        n = len(ev)
        pv = np.squeeze(d[k + "_pvalue"]).reshape(n, -1)[:, :2]
        rv = np.squeeze(d[k + "_rvalues"]).reshape(n, -1)[:, :2]
        # Goodfit / slope are absent in some sessions (e.g. PPP4_day11 stores only
        # eventtimes / pvalue / rvalues / posterior); recorded as NaN.
        if k + "_Goodfit" in d:
            gof = np.squeeze(d[k + "_Goodfit"]).reshape(n, -1)[:, :2]
        else:
            gof = np.full((n, 2), np.nan)
        post = d[k][0] if k in d else None
        slope_raw = d.get(k + "_slope", [[None] * n])[0]
        best = np.nanargmin(np.where(np.isfinite(pv), pv, np.inf), axis=1)
        for i in range(n):
            b = int(best[i])
            span = net = np.nan
            if post is not None:
                s = np.squeeze(post[i])
                traj = None
                if s.ndim == 2 and s.shape[1] >= 2:
                    # full posterior (nPos, 2*nTime): the two template directions
                    # concatenated along time -- take the selected direction, then argmax
                    h = s.shape[1] // 2
                    half = s[:, :h] if b == 0 else s[:, h:]
                    if half.size and half.shape[1] > 1:
                        traj = np.argmax(half, axis=0)
                elif s.ndim == 1 and s.size >= 4:
                    # some sessions (PPP4_day11) store the already-decoded position
                    # sequence instead of the posterior, still 2*nTime long
                    h = s.size // 2
                    traj = np.asarray(s[:h] if b == 0 else s[h:], float)
                if traj is not None and traj.size > 1 and np.isfinite(traj).all():
                    span = float(np.max(traj) - np.min(traj))
                    net = float(abs(traj[-1] - traj[0]))
            sl = np.nan
            try:
                sv = np.squeeze(slope_raw[i])
                sl = float(np.atleast_1d(sv).ravel()[b])
            except Exception:                                   # noqa: BLE001
                pass
            rows.append(dict(ep=k, start=ev[i, 0], stop=ev[i, 1],
                             pval=float(pv[i, b]), gof=float(gof[i, b]),
                             rval=float(rv[i, b]), slope=sl,
                             traj_span=span, traj_net=net))
    E = pd.DataFrame(rows)
    # Some sessions decode the SAME events against more than one template within an
    # epoch (PPP4_day11: template02 and template04 for epochs 1/3/5), so an event
    # appears once per template. Epochs are disjoint in time, so identical (start, stop)
    # can only mean the same event decoded twice -- keep the best (smallest p) decode so
    # per-event property means are not double-counted. No-op for single-template
    # sessions, which is every session in the Fig. 2e set.
    n0 = len(E)
    E = (E.sort_values("pval", kind="mergesort")
           .drop_duplicates(subset=["start", "stop"], keep="first")
           .sort_values("start")
           .reset_index(drop=True))
    if len(E) < n0:
        print(f"    ({n0 - len(E)} duplicate multi-template decodes merged, "
              f"{len(E)} unique events)", flush=True)
    E["is_sig"] = E["pval"] < SIG_P
    return E


def build_events(sess, basepath, verbose=True):
    import nelpy as nel
    from neuro_py.io import loading

    rip = loading.load_ripples_events(basepath)
    sd = loading.load_SleepState_states(basepath)
    nrem = nel.EpochArray(sd["NREMstate"])
    rip_nrem = (nrem & nel.EpochArray(np.array(rip[["start", "stop"]]))).data

    # map each NREM-clipped ripple back to its source row (unclipped properties)
    src = np.searchsorted(rip["start"].to_numpy(), rip_nrem[:, 0], side="right") - 1
    src = np.clip(src, 0, len(rip) - 1)
    R = rip.iloc[src].reset_index(drop=True).copy()
    R["start"] = rip_nrem[:, 0]
    R["stop"] = rip_nrem[:, 1]
    R["duration"] = R["stop"] - R["start"]
    R["mid"] = 0.5 * (R["start"] + R["stop"])

    # participating cells / spikes -- CA1, the population d2 is fitted on
    st, cm = loading.load_spikes(basepath, brainRegion="CA1")
    units = [np.asarray(u, float) for u in np.asarray(st.data, dtype=object)]
    ncell = np.zeros(len(R), int)
    nspk = np.zeros(len(R), int)
    a, b = R["start"].to_numpy(), R["stop"].to_numpy()
    for u in units:
        c = np.searchsorted(u, b) - np.searchsorted(u, a)
        ncell += (c > 0)
        nspk += c
    R["n_cells"] = ncell
    R["n_spikes"] = nspk
    R["n_units_total"] = len(units)

    # replay label + trajectory geometry of the overlapping candidate event
    E = load_replay_events(sess)
    S = E[E.is_sig][["start", "stop"]].to_numpy()

    def first_overlap(iv, others):
        j = np.where((iv[0] < others[:, 1]) & (iv[1] > others[:, 0]))[0]
        return int(j[0]) if j.size else -1

    riv = R[["start", "stop"]].to_numpy()
    R["is_replay"] = np.fromiter(
        (np.any((r[0] < S[:, 1]) & (r[1] > S[:, 0])) if len(S) else False
         for r in riv), bool, len(R))
    cand = E[["start", "stop"]].to_numpy()
    ci = np.fromiter((first_overlap(r, cand) for r in riv), int, len(R))
    R["cand_idx"] = ci
    for c in ["pval", "gof", "rval", "slope", "traj_span", "traj_net"]:
        v = E[c].to_numpy()
        R[c] = np.where(ci >= 0, v[np.clip(ci, 0, None)], np.nan)
    R["session"] = sess
    R["animal"] = sess.split("_")[0]
    R["log_duration"] = np.log10(R["duration"].clip(lower=1e-4))

    if verbose:
        print(f"  {sess}: {len(rip)} ripples, {len(R)} in NREM | "
              f"{len(E)} candidates, {int(E.is_sig.sum())} sig | "
              f"replay SWR {int(R.is_replay.sum())} "
              f"({R.is_replay.mean():.1%}), {len(units)} CA1 units", flush=True)
    return R


# ---------------------------------------------------------------------------
# window rates
# ---------------------------------------------------------------------------
def window_rate(iv, centers, win=WIN_S):
    w0, w1 = centers - win / 2, centers + win / 2
    if iv.size == 0:
        return np.zeros(len(centers))
    return np.array([np.sum((iv[:, 0] < b) & (iv[:, 1] > a))
                     for a, b in zip(w0, w1)]) / win


def window_mean(iv, vals, centers, win=WIN_S):
    """Mean of `vals` over the events falling in each window (NaN if none)."""
    w0, w1 = centers - win / 2, centers + win / 2
    out = np.full(len(centers), np.nan)
    for i, (a, b) in enumerate(zip(w0, w1)):
        m = (iv[:, 0] < b) & (iv[:, 1] > a)
        if m.any():
            v = vals[m]
            v = v[np.isfinite(v)]
            if v.size:
                out[i] = v.mean()
    return out


def match_nonreplay(R, seed=0):
    """Greedy 1:1 nearest-neighbour match of non-replay to replay SWRs on MATCH_ON,
    standardised over the pooled set, without replacement. Returns the matched
    non-replay intervals plus a balance table."""
    rep = R[R.is_replay]
    non = R[~R.is_replay]
    if len(rep) == 0 or len(non) == 0:
        return np.empty((0, 2)), pd.DataFrame()
    mu = R[MATCH_ON].mean()
    sd = R[MATCH_ON].std().replace(0, 1.0)
    A = ((rep[MATCH_ON] - mu) / sd).to_numpy()
    B = ((non[MATCH_ON] - mu) / sd).to_numpy()
    ok = np.isfinite(A).all(1)
    A = A[ok]
    rep_ok = rep[ok]
    used = np.zeros(len(B), bool)
    # match in a fixed, data-determined order (rarest replay events first) so the
    # result does not depend on row order; no RNG needed
    order = np.argsort(-np.linalg.norm(A - B.mean(0), axis=1))
    pick = np.full(len(A), -1, int)
    for i in order:
        d = np.linalg.norm(B - A[i], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if np.isfinite(d[j]):
            pick[i] = j
            used[j] = True
    pick = pick[pick >= 0]
    M = non.iloc[pick]
    bal = pd.DataFrame({
        "replay": rep_ok[MATCH_ON + ["frequency", "n_spikes"]].median(),
        "nonreplay_all": non[MATCH_ON + ["frequency", "n_spikes"]].median(),
        "nonreplay_matched": M[MATCH_ON + ["frequency", "n_spikes"]].median(),
    })
    return M[["start", "stop"]].to_numpy(), bal


def partial_r(y, x, Z):
    """Spearman partial correlation of y and x given covariates Z (rank-based)."""
    keep = np.isfinite(y) & np.isfinite(x) & np.isfinite(Z).all(1)
    if keep.sum() < 10:
        return np.nan, keep.sum()
    ry = stats.rankdata(y[keep]); rx = stats.rankdata(x[keep])
    RZ = np.column_stack([stats.rankdata(Z[keep, j]) for j in range(Z.shape[1])])
    RZ = np.column_stack([np.ones(len(RZ)), RZ])
    ey = ry - RZ @ np.linalg.lstsq(RZ, ry, rcond=None)[0]
    ex = rx - RZ @ np.linalg.lstsq(RZ, rx, rcond=None)[0]
    if ey.std() == 0 or ex.std() == 0:
        return np.nan, keep.sum()
    return float(np.corrcoef(ey, ex)[0, 1]), int(keep.sum())


def fisher_mean(rs):
    r = np.asarray([v for v in rs if np.isfinite(v)], float)
    if r.size == 0:
        return np.nan, np.nan, 0
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    m, se = z.mean(), z.std(ddof=1) / np.sqrt(z.size) if z.size > 1 else np.nan
    return float(np.tanh(m)), float(se), z.size


# ---------------------------------------------------------------------------
def analyse(R, pub, sess):
    """All four tests for one session. Returns (prop_rows, corr_row, balance)."""
    c_all = pub.center_time.to_numpy()
    c_no = c_all[::NONOVL]                      # 30 s apart -> non-overlapping
    d2_all = pub.d2_value.to_numpy()
    d2_no = d2_all[::NONOVL]

    rep = R[R.is_replay][["start", "stop"]].to_numpy()
    non = R[~R.is_replay][["start", "stop"]].to_numpy()
    allr = R[["start", "stop"]].to_numpy()
    mat, bal = match_nonreplay(R)

    # --- (a) property comparison ------------------------------------------
    prows = []
    for p in PROPS + ["traj_span", "traj_net", "gof", "rval"]:
        if p not in R.columns:
            continue
        a = R.loc[R.is_replay, p].to_numpy(float)
        b = R.loc[~R.is_replay, p].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 5 or b.size < 5:
            continue
        U, pv = stats.mannwhitneyu(a, b, alternative="two-sided")
        prows.append(dict(session=sess, prop=p, n_replay=a.size, n_non=b.size,
                          med_replay=np.median(a), med_non=np.median(b),
                          # rank-biserial: 0 = no difference, +1 = replay always larger
                          cliffs_delta=2 * U / (a.size * b.size) - 1, p=pv))

    # --- (b)/(c)/(d) window-level correlations ----------------------------
    out = dict(session=sess, n_win_all=len(c_all), n_win_nonovl=len(c_no),
               n_replay=int(R.is_replay.sum()), n_non=int((~R.is_replay).sum()),
               n_matched=len(mat))
    for tag, cc, dd in [("all", c_all, d2_all), ("no", c_no, d2_no)]:
        r_in = window_rate(rep, cc)
        r_out = window_rate(non, cc)
        r_mat = window_rate(mat, cc)
        r_tot = window_rate(allr, cc)
        for nm, v in [("in", r_in), ("out", r_out), ("matched", r_mat),
                      ("all", r_tot)]:
            out[f"r_{nm}_{tag}"] = (stats.spearmanr(dd, v).statistic
                                    if np.std(v) > 0 else np.nan)
        # (c) partial: d2 ~ rate_in | mean duration, amplitude, n_cells, total rate
        Z = np.column_stack([
            window_mean(allr, R["duration"].to_numpy(), cc),
            window_mean(allr, R["amplitude"].to_numpy(), cc),
            window_mean(allr, R["n_cells"].to_numpy(float), cc),
            r_tot])
        out[f"r_in_partial_{tag}"], out[f"n_partial_{tag}"] = partial_r(dd, r_in, Z)
        # (d) trajectory: correlation of d2 with decoded span, and of rate_in with d2
        # controlling for span
        span = window_mean(rep, R.loc[R.is_replay, "traj_span"].to_numpy(), cc)
        ok = np.isfinite(span)
        out[f"r_span_{tag}"] = (stats.spearmanr(dd[ok], span[ok]).statistic
                                if ok.sum() >= 10 else np.nan)
        out[f"r_in_given_span_{tag}"], _ = partial_r(dd, r_in, span[:, None])
    return prows, out, bal


def report_session_set():
    """Fig. 2e globs data/ripple_barr/replay_non_replay/, so its N is the file count
    there -- 9 mouse + 1 rat. That set differs from the Fig. 2a/g/h set (PPP4_day11 and
    PPP7_day10 in place of PPP4_day10 and PPP7_day8, and no rat); the difference is
    printed below."""
    have = sorted(os.path.basename(f)[:-4].replace("_CA1", "")
                  for f in os.listdir(PUB) if f.endswith(".pkl"))
    fig2agh = {"PPP4_day11", "PPP7_day10", "PPP8_day8", "PPP20_day4", "PPP20_day5",
               "PPP21_day4", "PPP21_day5", "PPP23_day7", "PPP23_day8"}
    print("\n=== session set actually behind Fig. 2e ===")
    print(f"  {len(have)} pkl files: {', '.join(have)}")
    only2e = set(have) - fig2agh
    only_agh = fig2agh - set(have)
    print(f"  in Fig.2e but not the documented Fig.2a/g/h list: "
          f"{', '.join(sorted(only2e)) or 'none'}")
    print(f"  in Fig.2a/g/h but not Fig.2e:                     "
          f"{', '.join(sorted(only_agh)) or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="PPP20_day4 only")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sessions", nargs="*")
    a = ap.parse_args()

    names = (["PPP20_day4"] if a.test else
             a.sessions if a.sessions else list(SESSIONS))
    report_session_set()

    assert_local_path(OUTDIR); os.makedirs(OUTDIR, exist_ok=True)
    assert_local_path(LOGDIR); os.makedirs(LOGDIR, exist_ok=True)

    t0 = time.time()
    prop_rows, corr_rows, bals, tables = [], [], {}, []
    print(f"\n=== building per-SWR tables ({len(names)} sessions) ===")
    for s in names:
        bp = SESSIONS[s]
        pubf = os.path.join(PUB, f"{s}_CA1.pkl")
        if not os.path.exists(pubf):
            print(f"  [SKIP] {s}: no published window file")
            continue
        try:
            R = build_events(s, bp)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [SKIP] {s}: {type(e).__name__}: {e}", flush=True)
            continue
        R.to_pickle(assert_local_path(os.path.join(OUTDIR, f"{s}_swr_events.pkl")))
        pub = pd.read_pickle(pubf)
        pr, cr, bal = analyse(R, pub, s)
        prop_rows += pr; corr_rows.append(cr); bals[s] = bal
        tables.append(R)

    P = pd.DataFrame(prop_rows)
    C = pd.DataFrame(corr_rows)
    P.to_csv(assert_local_path(os.path.join(LOGDIR, "r35_event_properties.csv")),
             index=False)
    C.to_csv(assert_local_path(os.path.join(LOGDIR, "r35_matched_control.csv")),
             index=False)

    # ---- (a) ------------------------------------------------------------
    print("\n=== (a) do replay and non-replay SWRs differ? ===")
    print("    per-session medians, then paired Wilcoxon across sessions")
    print(f"{'property':>16}{'med replay':>12}{'med non':>12}{'ratio':>8}"
          f"{'delta':>8}{'n':>4}{'p_paired':>10}")
    for p in P["prop"].unique():
        s = P[P["prop"] == p]
        if len(s) < 3:
            continue
        try:
            w = stats.wilcoxon(s.med_replay, s.med_non).pvalue
        except Exception:                                        # noqa: BLE001
            w = np.nan
        print(f"{p:>16}{s.med_replay.median():12.4f}{s.med_non.median():12.4f}"
              f"{s.med_replay.median()/max(s.med_non.median(),1e-9):8.2f}"
              f"{s.cliffs_delta.median():8.3f}{len(s):4d}{w:10.4f}")

    # ---- balance --------------------------------------------------------
    if bals:
        print("\n=== matching balance (median over sessions) ===")
        B = pd.concat(bals.values(), keys=bals.keys())
        print(B.groupby(level=1).median().round(4).to_string())

    # ---- (b) ------------------------------------------------------------
    if len(C):
        print("\n=== (b) d2 (2024 Euclidean, p=20) ~ SWR rate, Spearman ===")
        print("    NON-OVERLAPPING windows (30 s apart) are the inferential set;")
        print("    'all' reproduces the published 2 s-step grid.")
        print(f"{'rate':>12}{'r_nonovl':>12}{'se':>8}{'r_allwin':>12}{'n':>4}")
        for nm in ["in", "out", "matched", "all"]:
            rn, se, k = fisher_mean(C[f"r_{nm}_no"])
            ra, _, _ = fisher_mean(C[f"r_{nm}_all"])
            print(f"{nm:>12}{rn:12.3f}{se:8.3f}{ra:12.3f}{k:4d}")
        for nm in ["out", "matched"]:
            zi = np.arctanh(C["r_in_no"].clip(-0.999, 0.999))
            zo = np.arctanh(C[f"r_{nm}_no"].clip(-0.999, 0.999))
            ok = np.isfinite(zi) & np.isfinite(zo)
            if ok.sum() >= 3:
                w = stats.wilcoxon(zi[ok], zo[ok]).pvalue
                print(f"    paired z(in) vs z({nm}): dz={np.median(zi[ok]-zo[ok]):+.3f}"
                      f"  Wilcoxon p={w:.4f}  n={int(ok.sum())}")

        print("\n=== (c)/(d) controls, non-overlapping windows ===")
        for nm, lbl in [("r_in_no", "d2 ~ rate_in (raw)"),
                        ("r_in_partial_no",
                         "d2 ~ rate_in | dur, amp, n_cells, total rate"),
                        ("r_span_no", "d2 ~ mean decoded trajectory span"),
                        ("r_in_given_span_no", "d2 ~ rate_in | mean span")]:
            r, se, k = fisher_mean(C[nm])
            print(f"    {lbl:48} r={r:+.3f} +/- {se:.3f}  (n={k})")

    print(f"\nwrote -> {OUTDIR}\n         {LOGDIR}\\r35_*.csv   "
          f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
