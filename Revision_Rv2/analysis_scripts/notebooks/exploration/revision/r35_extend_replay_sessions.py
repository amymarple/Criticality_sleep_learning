"""
r35_extend_replay_sessions.py -- rebuilds the replay/SWR window table over every session
with decoded replay available (10 sessions), not only the subset already carried in the
binomial ladder, each with one recipe so provenance does not differ by session; the
sessions that already have a window table are validated against their stored columns.

Windows: 30 s stepped 2 s inside guarded NREM intervals (`make_windows`), the
non-overlapping subset, restricted to the time span of the session's decoded epochs (a
window outside that span has no replay label because it was never decoded, not because no
replay occurred there). Events: ripple_count / replay_count by interval overlap; an event
is replay if either decoded direction has p < 0.05 (the `_new` replay file where it
exists, else the original). Covariates: FR_std (across-neuron SD of the per-neuron rate in
the window, ddof=0) and MUA_std (temporal SD of the 40 ms population rate over a centred
30 s rolling window) are two different quantities carried under similar names; each has a
`_bg` counterpart computed over the ripple-excised part of the window, since otherwise
MUA_std is largely a readout of the ripples themselves. MUA is the
rolling population rate; EI_ratio is pyramidal over interneuron rate, per unit. d2:
`d2_published` from the stored `*_D2_time_resolved_error_bar.mat`, joined on window
centre; `py_full` and `py_excised` computed in Python (2024 Euclidean, p=20, OLS),
excision dropping 40 ms bins overlapping a raw ripple and gluing the remainder
(>= 1 s segments).

PPP7_day12 (two maze epochs, non-standard epoch numbering) is kept in the table but excluded
from the ladder by name (`EXCLUDE`).

Reads the shares read-only; writes only to D:.
    -> data/revision/log/r35_extended_windows.csv
"""
from __future__ import annotations

import os
import sys
import time
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
from fig1_metrics import (GUARD_S, MIN_FIRING_RATE, binned_counts,      # noqa: E402
                          build_state_intervals, make_windows, slice_sorted)
from r35_swr_property_control import RES                                # noqa: E402
from r35_background_vs_ripple import load_d2_mat                        # noqa: E402
from r35_sham_excision import bin_mask, glue, d2_of                     # noqa: E402
from fig2f_backtrace_model import (population_mua,                       # noqa: E402
                                   fr_std_across_neurons)
from fig2f_background_state_glm import window_pop_stats, window_fr_std  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
PROB_DIR = os.path.join(REPO, "data", "replay", "replay_probality")
OUT = os.path.join(LOGDIR, "r35_extended_windows.csv")
BIN = 0.040
P = 20
WIN_S = 30.0
NONOVL = 15

# basepath, replay template, and which replay file variant exists.
SESSIONS = {
    "PPP4_day11":  (r"W:\data\PPP\PPP4\day11",  "template02"),
    "PPP7_day8":   (r"W:\data\PPP\PPP7\day8",   "template02"),
    "PPP7_day12":  (r"W:\data\PPP\PPP7\day12",  "template04"),
    "PPP8_day8":   (r"W:\data\PPP\PPP8\day8",   "template02"),
    "PPP20_day4":  (r"W:\data\PPP\PPP20\day4",  "template04"),
    "PPP20_day5":  (r"W:\data\PPP\PPP20\day5",  "template04"),
    "PPP21_day4":  (r"W:\data\PPP\PPP21\day4",  "template04"),
    "PPP21_day5":  (r"W:\data\PPP\PPP21\day5",  "template04"),
    "PPP23_day7":  (r"W:\data\PPP\PPP23\day7",  "template04"),
    "PPP23_day8":  (r"W:\data\PPP\PPP23\day8",  "template04"),
}
NEW = ("PPP7_day12",)

# Structural exclusion. Every other session
# runs sleep / maze / sleep / maze / sleep, so the replay file's epoch1/3/5 are the sleep
# epochs. PPP7_day12 runs presleep / presleep / T_maze2 / midsleep / T_maze1 / postsleep
# -- six epochs, two mazes, and its "epoch3" is T_maze2, a running epoch, not sleep. Its
# replay file also carries only template04 although two different mazes were run, and it
# is the only session with no regenerated `_new` file, so its decoded labels do not mean
# what they mean elsewhere. Kept in the window table and excluded from the
# ladder by name.
EXCLUDE = {"PPP7_day12": "epoch numbering differs: its epoch3 is T_maze2 (a running "
                         "epoch), two mazes share one template, no _new replay file"}


def replay_events(sess, template):
    """Significant decoded events and the decoded time span. Prefers the `_new` file."""
    import re
    for suf in ("_new", ""):
        f = os.path.join(RES, f"{sess}_replay_sequences{suf}.mat")
        if os.path.exists(f):
            break
    else:
        return None, None, None
    d = scipy.io.loadmat(f)
    combos = sorted({re.match(r"(epoch\d+_template\d+)", k).group(1)
                     for k in d if re.match(r"epoch\d+_template\d+_eventtimes$", k)})
    combos = [c for c in combos if c.endswith(template)] or combos
    sig, spans = [], []
    for k in combos:
        ev = np.squeeze(d[k + "_eventtimes"]).reshape(-1, 2)
        n = len(ev)
        pv = np.squeeze(d[k + "_pvalue"]).reshape(n, -1)[:, :2]
        sig.append(ev[(pv < .05).any(1)])
        spans.append((float(ev[:, 0].min()), float(ev[:, 1].max())))
    return (np.concatenate(sig) if sig else np.zeros((0, 2))), spans, os.path.basename(f)


def counts(iv, centers, mask=None):
    w0, w1 = centers - WIN_S / 2, centers + WIN_S / 2
    a, b = iv[:, 0], iv[:, 1]
    out = np.zeros(len(centers), int)
    for i in range(len(centers)):
        inw = (a < w1[i]) & (b > w0[i])
        out[i] = int((inw & mask).sum() if mask is not None else inw.sum())
    return out


def one_session(sess):
    from neuro_py.io import loading
    bp, template = SESSIONS[sess]
    st, cm = loading.load_spikes(bp, brainRegion="CA1")
    sd = loading.load_SleepState_states(bp)
    rip = loading.load_ripples_events(basepath=bp)
    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    units = [np.sort(np.asarray(u, float)) for u in np.asarray(st.data, dtype=object)[keep]]
    sp = np.sort(np.concatenate(units))
    ct = np.asarray(cm.putativeCellType.astype(str))[keep]
    e_sp = np.sort(np.concatenate([u for u, t in zip(units, ct) if t == "Pyramidal Cell"] or [np.zeros(0)]))
    i_sp = np.sort(np.concatenate([u for u, t in zip(units, ct) if "Interneuron" in t] or [np.zeros(0)]))
    n_e = max(int((ct == "Pyramidal Cell").sum()), 1)
    n_i = max(int(sum("Interneuron" in t for t in ct)), 1)
    riv = np.sort(np.array(rip[["start", "stop"]], float), axis=0)
    nrem = build_state_intervals(sd, GUARD_S)["NREM"]
    riv_nrem = riv[np.array([(s >= nrem[:, 0]) & (s < nrem[:, 1]) for s in riv[:, 0]]).any(1)]

    sig, spans, fname = replay_events(sess, template)
    if sig is None:
        return None, f"no replay file"
    win = make_windows(nrem, "NREM")
    win = win[win.nonoverlap].reset_index(drop=True)
    c = win.center_s.to_numpy(float)
    inside = np.zeros(len(c), bool)
    for a, b in spans:
        inside |= (c >= a - WIN_S / 2) & (c <= b + WIN_S / 2)
    win, c = win[inside].reset_index(drop=True), c[inside]

    n_swr = counts(riv_nrem, c)
    has = np.fromiter((np.any((r[0] < sig[:, 1]) & (r[1] > sig[:, 0])) if len(sig) else False
                       for r in riv_nrem), bool, len(riv_nrem))
    n_rep = counts(riv_nrem, c, has)

    mua, mua_std = population_mua(units, c)
    # background versions: the same two moments over the ripple-excised part of the
    # window, since otherwise MUA_std is largely a readout of the ripples themselves.
    st_ = window_pop_stats(units, riv, c)
    mua_rc, mua_std_rc, mua_bg, mua_std_bg, frac_kept = st_.T
    # Fig 2F's third feature is the ACROSS-NEURON SD, not the temporal one. Both are
    # carried so the ablation can be run on the Fig-2F-matched set and on the ladder's.
    fr_std, fr_std_bg, fr_mean_bg = window_fr_std(units, riv, c).T
    # cross-check the grid-based full-window FR_std against the direct cell-4 routine
    fr_ref = fr_std_across_neurons(units, c)
    fr_err = float(np.nanmax(np.abs(fr_std - fr_ref)))
    nb = int(round(WIN_S / BIN))
    py_full = np.full(len(c), np.nan)
    py_exc = np.full(len(c), np.nan)
    ei = np.full(len(c), np.nan)
    for i, cc in enumerate(c):
        edges = cc - WIN_S / 2 + BIN * np.arange(nb + 1)
        lo, hi = np.searchsorted(sp, (edges[0], edges[-1]))
        x = np.histogram(sp[lo:hi], bins=edges)[0].astype(float)
        py_full[i] = d2_of(x)
        inw = riv[(riv[:, 1] > edges[0]) & (riv[:, 0] < edges[-1])]
        py_exc[i] = d2_of(glue(x, bin_mask(edges, inw)))
        ne = slice_sorted(e_sp, edges[0], edges[-1]).size / n_e
        ni = slice_sorted(i_sp, edges[0], edges[-1]).size / n_i
        ei[i] = (ne + 1e-3) / (ni + 1e-3)

    W = pd.DataFrame(dict(session=sess, center_s=c, n_swr=n_swr, n_replay=np.minimum(n_rep, n_swr),
                          MUA=mua, MUA_std=mua_std, MUA_bg=mua_bg, MUA_std_bg=mua_std_bg,
                          FR_std=fr_std, FR_std_bg=fr_std_bg, FR_mean_bg=fr_mean_bg,
                          frac_kept=frac_kept, EI_ratio=ei, py_full=py_full, py_excised=py_exc,
                          n_units=int(keep.sum())))
    m = load_d2_mat(bp, "_error_bar")
    if m is not None and len(m):
        mc = m["center"].to_numpy(float)
        idx = np.clip(np.searchsorted(mc, c), 0, len(mc) - 1)
        v = m["d2_x"].to_numpy(float)[idx]
        v[np.abs(mc[idx] - c) > 1.0] = np.nan
        W["d2_published"] = v
    else:
        W["d2_published"] = np.nan
    m2 = load_d2_mat(bp, "_rippleablate")
    if m2 is not None and len(m2):
        mc = m2["center"].to_numpy(float)
        idx = np.clip(np.searchsorted(mc, c), 0, len(mc) - 1)
        v = m2["d2_x"].to_numpy(float)[idx]
        v[np.abs(mc[idx] - c) > 1.0] = np.nan
        W["d2_mat_excised"] = v
    else:
        W["d2_mat_excised"] = np.nan
    W = W[W.n_swr > 0].reset_index(drop=True)
    note = (f"{len(W):4d} win | {int(W.n_swr.sum()):6d} SWR, {W.n_replay.sum() / max(W.n_swr.sum(), 1):.1%} replay "
            f"| decoded spans {len(spans)} | {fname}"
            f" | FR_std vs cell 4 max|diff| {fr_err:.2e}"
            f" | mat published {'yes' if W.d2_published.notna().any() else 'NO'}"
            f", mat excised {'yes' if W.d2_mat_excised.notna().any() else 'NO'}")
    return W, note


def main():
    t0 = time.time()
    out = []
    for sess in SESSIONS:
        W, note = one_session(sess)
        tag = "  NEW  " if sess in NEW else "       "
        print(f"{tag}{sess:12s} {note}", flush=True)
        if W is not None:
            out.append(W)
    D = pd.concat(out, ignore_index=True)
    assert_local_path(LOGDIR)
    D.to_csv(assert_local_path(OUT), index=False)
    print(f"\nwrote {OUT}  ({len(D)} windows, {D.session.nunique()} sessions, {time.time() - t0:.0f} s)")
    validate(D)


def validate(D):
    """Against the stored per_center tables for the 9 sessions that have them."""
    print("\n" + "=" * 96)
    print("Validation against the stored replay_probality tables (nearest centre, 1 s)")
    print("=" * 96)
    print(f"  {'session':12s}{'n matched':>10s}{'r(n_swr)':>10s}{'r(n_replay)':>12s}{'r(MUA)':>9s}{'r(MUA_std)':>11s}{'r(d2)':>8s}")
    for sess, g in D.groupby("session"):
        f = os.path.join(PROB_DIR, f"{sess}.pkl")
        if not os.path.exists(f):
            print(f"  {sess:12s}      (no stored table -- this is one of the new sessions)")
            continue
        p = pd.read_pickle(f)["per_center"]
        p = p[p.state.astype(str).str.upper().eq("NREM")]
        j = pd.merge_asof(g.sort_values("center_s"), p.sort_values("center_s"),
                          on="center_s", direction="nearest", tolerance=1.0, suffixes=("", "_ref")).dropna(subset=["ripple_count"])
        def r(a, b):
            v = j[[a, b]].dropna()
            return v.corr().iloc[0, 1] if len(v) > 5 and v[a].std() > 0 and v[b].std() > 0 else np.nan
        print(f"  {sess:12s}{len(j):>10d}{r('n_swr', 'ripple_count'):>10.3f}{r('n_replay', 'replay_count'):>12.3f}"
              f"{r('MUA', 'MUA_ref') if 'MUA_ref' in j else np.nan:>9.3f}"
              f"{r('MUA_std', 'MUA_std_ref') if 'MUA_std_ref' in j else np.nan:>11.3f}"
              f"{r('d2_published', 'd2'):>8.3f}")
    print("\n  Python d2 vs the .mat arms, median within-session r:")
    for a, b, lab in (("py_full", "d2_published", "full"), ("py_excised", "d2_mat_excised", "excised")):
        rs = [g[[a, b]].dropna().corr().iloc[0, 1] for _, g in D.groupby("session")
              if g[[a, b]].dropna().shape[0] > 20]
        print(f"    {lab:8s} r = {np.median(rs):.3f}  ({len(rs)} sessions with both)")


if __name__ == "__main__":
    main()
