"""fig3g_background.py -- Figure 3g (CCK silencing, Sncg sessions) with the submitted pipeline and
one change: each window's d2 replaced by its background d2.

Fig. 3g compares presleep and postsleep d2 in the seven Sncg sessions. Its d2 is the full-window
value, so ripples and barrages inside a window enter it. The background d2 (every ripple and
barrage interval excised) had not been computed on these sessions.

Everything else is the submitted pipeline:
  * Rows: the stored pickles `data/ripple_barr/manipulation/sncg/*.pkl`. `fig3g_reproduce.py`
    asserts that each pickle's rows are the first N NREM windows of each epoch of its source
    `.mat` (CA1+CA2, 80 ms bins, 60 s windows, order 20, for six sessions; the 40 ms / 30 s `.mat`
    for SNCG06_day27), so row k of an epoch is the k-th `.mat` window of that epoch; its centre is
    taken from the `.mat`.
  * Units: those the `.mat` pooled (`params.brain_region`, matched by `contains` as the MATLAB
    worker does, no firing-rate filter; the selection matched on params.n_cells / total_spikes,
    since labels were re-curated after some files were written); all units pooled into one train.
  * Intact d2 recomputed per window with `d2_yw_matlab_match.d2_yw_segments` on the whole window at
    the `.mat`'s bin and order -- checked against the `.mat` d2 before the background arm is trusted.
  * Background d2: the same estimator on the window minus every ripple and barrage interval
    (`load_ripples_events`, `load_barrage_events`), event-free runs >= 1 s
    (`r16_glue_control.free_segments`, MIN_FREE_S), autocovariance within segments.
  * Statistics: `fig3g_reproduce.resample` (1 000 rows per session x epoch, seed 0) and
    `fig3g_reproduce.lmm` (d2 ~ C(epoch) + C(animal), random intercept per session), unchanged;
    plus the session-level paired comparison (`session_level`).

    -> data/revision/log/fig3g_background_windows.csv   (per pickle row: epoch, centre, d2 arms)
    -> data/revision/log/fig3g_background.csv           (LMM and session statistics per arm)
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                          # noqa: E402
from fig3g_reproduce import SN_DIR, ORDER, load_plot_df, resample, lmm, session_level  # noqa: E402
from opto_sncg_metrics import SNCG_SESSIONS, load_epochs                # noqa: E402
import opto_cck_metrics as ocm                                          # noqa: E402
from d2_yw_matlab_match import d2_yw_segments                           # noqa: E402
from r16_glue_control import free_segments                              # noqa: E402
from r16_event_excised_windows import intervals                        # noqa: E402

LOG = os.path.join(os.path.abspath(os.path.join(_HERE, "..", "..", "..")), "data", "revision", "log")


def mat_source(bp, tag):
    """The .mat each pickle was cut from: NREM windows (centre, d2) + bin, window, order, regions."""
    import scipy.io
    day = os.path.basename(bp.rstrip("\\/"))
    name = (f"{day}_CA1_D2_time_resolved.mat" if tag == "SNCG06_day27"
            else f"{day}_CA1_CA2_D2_time_resolved_error_bar_80ms_60second.mat")
    m = scipy.io.loadmat(os.path.join(bp, "complex_system", name), simplify_cells=True)
    res = m["results"] if isinstance(m["results"], (list, np.ndarray)) else [m["results"]]
    v = pd.concat([pd.DataFrame({"center_s": np.atleast_1d(r["center"]).astype(float),
                                 "d2": np.atleast_1d(r["d2"]).astype(float)})
                   for r in res if "NREM" in str(r["state"]).upper()]).sort_values("center_s").reset_index(drop=True)
    p = m["params"]
    dt, win, order = float(p["deltaT"]), float(p["window_size"]), int(p["order"])
    regions = [r for r in str(p["brain_region"]).split("|") if r]
    return v, dict(file=name, dt=dt, win=win, order=order, regions=regions,
                   n_cells=int(p["n_cells"]), total_spikes=int(p["total_spikes"]))


def pooled_spikes(bp, prm):
    """The units the .mat pooled. Labels were re-curated after some .mat files were written, so the
    candidate selections are tried in order and the one whose cell count and spike total match
    params.n_cells / total_spikes is used: the .mat's regions, CA1 + CA2, all units."""
    from neuro_py.io import loading
    st, cm = loading.load_spikes(bp)
    lab = cm.brainRegion.astype(str).str.upper()
    trains = np.asarray(st.data, dtype=object)
    cands = [("+".join(prm["regions"]), prm["regions"]), ("CA1+CA2", ["CA1", "CA2"]), ("all", [])]
    best = None
    for name, regs in cands:
        keep = np.ones(len(cm), bool) if not regs else             np.any([lab.str.contains(r.upper(), regex=False).to_numpy() for r in regs], axis=0)
        n, tot = int(keep.sum()), int(sum(len(u) for u in trains[keep]))
        score = abs(n - prm["n_cells"]) + abs(tot - prm["total_spikes"]) / max(prm["total_spikes"], 1)
        if best is None or score < best[0]:
            best = (score, name, keep, n, tot)
    _, name, keep, n, tot = best
    sp = np.sort(np.concatenate([np.asarray(u, float) for u in trains[keep]]))
    return sp, f"{name}: {n} units / {tot} spikes (.mat {prm['n_cells']} / {prm['total_spikes']})"


def session(bp, pk):
    from neuro_py.io import loading
    tag = ocm.session_tag(bp)
    v, prm = mat_source(bp, tag)
    ep = load_epochs(bp)
    sp, units = pooled_spikes(bp, prm)
    ev = np.vstack([intervals(loading, bp, "load_ripples_events"), intervals(loading, bp, "load_barrage_events")])
    ev = ev[np.argsort(ev[:, 0])] if len(ev) else np.zeros((0, 2))
    rows = []
    for key in ORDER:
        iv = ep[key]
        w = v[(v.center_s >= iv[0, 0]) & (v.center_s < iv[-1, 1])].reset_index(drop=True)
        pv = pk[pk.epoch == key].d2.to_numpy()
        assert np.allclose(w.d2.to_numpy()[:len(pv)], pv), f"{tag} {key}: pkl != first-N of {prm['file']}"
        for k in range(len(pv)):
            c = float(w.center_s[k]); t0, t1 = c - prm["win"] / 2, c + prm["win"] / 2
            segs = free_segments(t0, t1, ev)
            rows.append(dict(animal_day=tag, animal=tag.split("_")[0], epoch=key, center_s=c, d2_pkl=pv[k],
                             d2_intact=d2_yw_segments(sp, [(t0, t1)], p=prm["order"], dt=prm["dt"]),
                             d2_bg=d2_yw_segments(sp, segs, p=prm["order"], dt=prm["dt"]) if segs else np.nan,
                             free_frac=sum(b - a for a, b in segs) / prm["win"],
                             n_events=int(((ev[:, 1] > t0) & (ev[:, 0] < t1)).sum()) if len(ev) else 0))
    R = pd.DataFrame(rows)
    ok = R[["d2_pkl", "d2_intact"]].dropna()
    print(f"  {tag:12s} {prm['file'][:60]:60s} dt {prm['dt']:.2f} win {prm['win']:.0f} p {prm['order']} "
          f"| {units} | rows {len(R)} | r(intact, .mat) "
          f"{np.corrcoef(ok.d2_pkl, ok.d2_intact)[0, 1]:.4f}, max |diff| {np.abs(ok.d2_pkl - ok.d2_intact).max():.2e} | "
          f"events/window {R.n_events.median():.0f}, kept {100 * R.free_frac.median():.0f} %", flush=True)
    return R


def main():
    pkls = sorted(glob.glob(os.path.join(SN_DIR, "*.pkl")))
    pooled = pd.concat([load_plot_df(p) for p in pkls], ignore_index=True)
    plot_df = pooled[pooled["epoch"].isin(ORDER)].copy()
    t0 = time.time()
    W = pd.concat([session(bp, plot_df[plot_df.animal_day == ocm.session_tag(bp)]) for bp in SNCG_SESSIONS],
                  ignore_index=True)
    W.to_csv(assert_local_path(os.path.join(LOG, "fig3g_background_windows.csv")), index=False)
    print(f"  ({time.time() - t0:.0f} s)")

    out = []
    # the published rows, in the pickles' own order, so resample() draws the same indices
    base = plot_df.reset_index(drop=True)
    assert len(base) == len(W) and np.allclose(base.d2.to_numpy(), W.d2_pkl.to_numpy()), "row order"
    for arm, lab in (("d2_pkl", "published d2 (.mat)"), ("d2_intact", "intact, recomputed"),
                     ("d2_bg", "background d2 (ripples + barrages excised)")):
        df = base.assign(d2=W[arm].to_numpy())
        df["epoch"] = pd.Categorical(df["epoch"], categories=ORDER, ordered=True)
        A = lmm(resample(df, seed=0))
        S = session_level(df.dropna(subset=["d2"]))
        print(f"\n{lab}\n  LMM (resampled 1000 / session x epoch): coef {A['coef']:+.4f} "
              f"CI [{A['ci_lo']:+.4f}, {A['ci_hi']:+.4f}]  p = {A['p']:.3g}\n"
              f"  sessions (median per epoch, paired): delta {S['median_delta']:+.4f}, {S['n_up']}/{S['n_sessions']} up, "
              f"Wilcoxon p = {S['p_wilcoxon']:.3f}\n  {S['per_session']}")
        out.append(dict(arm=arm, label=lab, **A, **S))
    pd.DataFrame(out).to_csv(assert_local_path(os.path.join(LOG, "fig3g_background.csv")), index=False)
    print("\n-> fig3g_background_windows.csv, fig3g_background.csv")


if __name__ == "__main__":
    main()
