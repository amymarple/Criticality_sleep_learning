"""
r16_event_type_classifier_br.py -- does the branching ratio classify SWR- vs barrage-dominated time
the way d2 does? Also the writer of r16_event_excised_windows_br.csv (BR and BR_bg per window), which
r24_residual_test.py reads for the barrage cohort (Extended Data Fig. 4g).

The panel follows the design of `r16_event_type_classifier_session.png` -- one binomial classifier
per session, barrages against SWRs, predictors z-scored within session, coefficient per
within-session SD, signed-rank across sessions (`r16_block_level_dissociation.classify_sessions`)
-- with the branching ratio m added as a sixth predictor, drawn for both arms side by side:
  excised : every predictor on the time overlapping neither a ripple nor a barrage (the
            panel's convention; d2 (2024 Euclidean, p = 20), MUA, Fano, synchrony and E/I
            from r16_event_excised_windows.csv, BR recomputed here on the free segments)
  intact  : every predictor on the whole 30 s window (circular; comparison only)

BR recipe = `excised_br_entropy.py` (validated there against the Fig-1 pickle, r ~ 1):
MR estimator on 4 ms active-unit counts of the PYRAMIDAL CA1 population, lag regressions
pooled over segments with `state_level_br.rk_curve_multi` (no regression crosses a gap),
`fit_m` at k_max = 200 (800 ms, d2's kernel). A window is skipped when < 1 s is event-free;
segments < 0.1 s are dropped. Units: load_spikes(brainRegion="CA1"), firingRate >
MIN_FIRING_RATE, putativeCellType "Pyramidal Cell" -- the same CA1 units as the other
predictors, pyramidal subset as in Fig 1.

Per arm: the six-predictor classifier (the figure); BR alone, d2 alone, d2 + BR (footnote),
so the answer covers "does BR classify at all", "does d2 survive BR" and "does BR add to
d2". Zero-trial windows are skipped in every likelihood.

Reads the shares read-only; writes only to D:. RECOMPUTE=1 forces the BR pass (else the
cached per-window CSV is used).
    -> data/revision/log/r16_event_excised_windows_br.csv          (BR, BR_bg per window)
    -> data/revision/log/r16_event_type_classifier_session_br.csv  (per session x arm x model)
    -> reports/revision_figs/r16_event_type_classifier_session_br.png/.svg
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
from scipy import stats                                               # noqa: E402

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                        # noqa: E402
from fig1_metrics import MIN_FIRING_RATE, active_unit_counts, fit_m, BIN_BR_S  # noqa: E402
from state_level_br import rk_curve_multi                             # noqa: E402
from barrage_order_kernel_sweep import BARRAGE_SESSIONS               # noqa: E402
from r16_event_excised_windows import intervals, WIN_S                # noqa: E402
import r16_block_level_dissociation as rbd                            # noqa: E402
from r16_block_level_dissociation import load, LOGDIR, FIGDIR, _BASECOLOR  # noqa: E402

K_MAX = 200
MIN_FREE_S = 1.0          # as excised_br_entropy.py / the MATLAB producer
MIN_SEG_S = 0.1
BR_CSV = os.path.join(LOGDIR, "r16_event_excised_windows_br.csv")
OUT = os.path.join(LOGDIR, "r16_event_type_classifier_session_br.csv")
BR_COLOR = "#1b9e77"
ARMS = [("excised", "_bg"), ("intact", "")]


def free_segments(t0, t1, r0, r1):
    """[t0, t1) minus the event intervals (sorted by start) -- excised_br_entropy.free_segments."""
    lo = np.searchsorted(r1, t0, "right")
    hi = np.searchsorted(r0, t1, "left")
    segs, cur = [], t0
    for a, b in zip(r0[lo:hi], r1[lo:hi]):
        if a > cur:
            segs.append((cur, min(a, t1)))
        cur = max(cur, b)
    if cur < t1:
        segs.append((cur, t1))
    return [(a, b) for a, b in segs if b - a >= MIN_SEG_S]


def br_of(e_spikes, segs):
    acts = [active_unit_counts(e_spikes, a, b, BIN_BR_S) for a, b in segs]
    K, r_k, se, _ = rk_curve_multi(acts, K_MAX + 1)
    return fit_m(K, r_k, se, K_MAX)


def compute_br(W):
    from neuro_py.io import loading
    rows, t_start = [], time.time()
    for sess, bp in BARRAGE_SESSIONS.items():
        g = W[W.session == sess]
        if not len(g):
            continue
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        units = [np.sort(np.asarray(u, float)) for u in np.asarray(st.data, dtype=object)[keep]]
        ct = np.asarray(cm.putativeCellType.astype(str))[keep]
        e_spikes = [u for u, t in zip(units, ct) if t == "Pyramidal Cell"]
        ev = np.vstack([intervals(loading, bp, "load_ripples_events"),
                        intervals(loading, bp, "load_barrage_events")])
        ev = ev[np.argsort(ev[:, 0])] if len(ev) else np.zeros((0, 2))
        r0, r1 = ev[:, 0], ev[:, 1]
        n_ok = 0
        for w in g.w.to_numpy(float):
            t0, t1 = w, w + WIN_S
            br = br_of(e_spikes, [(t0, t1)])
            segs = free_segments(t0, t1, r0, r1)
            free = float(sum(b - a for a, b in segs))
            br_bg = br_of(e_spikes, segs) if free >= MIN_FREE_S else np.nan
            n_ok += int(np.isfinite(br_bg))
            rows.append(dict(session=sess, w=w, BR=br, BR_bg=br_bg, free_s=free, n_pyr=len(e_spikes)))
        print(f"  {sess:12s} {len(g):4d} windows, {len(e_spikes):3d} pyr | BR_bg finite {n_ok:4d} | "
              f"{time.time() - t_start:5.0f} s", flush=True)
    return pd.DataFrame(rows)


def session_fit(W, preds):
    """`classify_sessions` on an explicit predictor list (module globals swapped in)."""
    rbd.PRED = preds
    return rbd.classify_sessions(W)


def summarise(C, preds):
    out = {}
    for c, lab in preds:
        v = C[C.predictor == c].coef.to_numpy()
        p = float(stats.wilcoxon(v).pvalue) if v.size > 1 else np.nan
        out[c] = dict(label=lab, median=float(np.median(v)), n=int(v.size),
                      same=max(int((v < 0).sum()), int((v > 0).sum())), p=p)
    return out


def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def plot(C, S, R):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 12, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4), sharey=False)
    for ax, (arm, suf) in zip(axes, ARMS):
        preds = [(f"BR{suf}", "branching\nratio"), (f"d2{suf}", "d2"), (f"MUA{suf}", "MUA"),
                 (f"fano{suf}", "MUA Fano"), (f"sync{suf}", "synchrony"), (f"EI{suf}", "E/I")]
        order = [c for c, _ in preds]
        pal = {f"BR{suf}": BR_COLOR, **{f"{k}{suf}": v for k, v in _BASECOLOR.items()}}
        Ca = C[(C.arm == arm) & (C.model == "six")]
        sns.boxplot(data=Ca, x="predictor", y="coef", order=order, ax=ax, width=0.6,
                    showfliers=False, linewidth=2, palette=pal, hue="predictor", legend=False)
        lo, hi = Ca.coef.quantile(0.02), Ca.coef.quantile(0.98)
        lo, hi = min(lo, -0.6), max(hi, 0.6)
        pad = 0.12 * (hi - lo)
        rng = np.random.default_rng(1)
        for i, c in enumerate(order):
            v = Ca[Ca.predictor == c].coef.to_numpy()
            ax.plot(i + rng.uniform(-0.13, 0.13, v.size), np.clip(v, lo, hi), "o", ms=4,
                    color="k", alpha=0.55, mew=0)
            r = S[arm]["six"][c]
            ax.text(i, hi + 1.1 * pad, f"{r['same']}/{r['n']}\n{star(r['p'])}", ha="center",
                    va="bottom", fontsize=10, fontweight="bold" if r["p"] < .05 else "normal")
        n_clip = int(((Ca.coef < lo) | (Ca.coef > hi)).sum())
        if n_clip:
            ax.text(0.99, 0.015, f"{n_clip} clipped to axis", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="0.4")
        ax.axhline(0, ls="--", lw=1.2, color="k")
        ax.set_ylim(lo - pad, hi + 3.8 * pad)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([l for _, l in preds], rotation=30, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel("log-odds an event is a BARRAGE\nrather than a SWR, per within-session SD"
                      if arm == "excised" else "")
        dev = Ca.groupby("session").dev_expl.first().median()
        state = ("predictors measured BETWEEN events (ripples and barrages excised)"
                 if arm == "excised" else
                 "predictors measured on the INTACT window (circular; comparison only)")
        n_s = Ca.session.nunique()
        ax.set_title(f"{'a' if arm == 'excised' else 'b'}  {arm}: which kind of event is this?\n"
                     "barrage vs SWR, one classifier per session, CA1 units only\n"
                     f"{state}\n"
                     f"{n_s} sessions, {dev:.0f}% of deviance explained (median)", fontsize=9.5, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    foot = ("Branching ratio m: MR estimator on 4 ms active-unit counts of the pyramidal CA1 population, "
            "k_max = 200 (800 ms); in the excised arm the lag regressions are pooled over the event-free\n"
            "segments (nothing crosses a gap). Predictors z-scored within session; numbers are sessions "
            "sharing the sign; stars are the paired signed-rank across sessions. Negative = where this\n"
            "variable is high, the events are SWRs. Reduced models: ")
    bits = []
    for arm, _ in ARMS:
        r = R[arm]
        bits.append(f"{arm} — BR alone {r['br_alone']}, d2 alone {r['d2_alone']}, "
                    f"d2 + BR: d2 {r['d2_with_br']}, BR {r['br_with_d2']}")
    foot += ";\n".join(bits) + "."
    fig.text(0.01, 0.005, foot, fontsize=8, color="0.3", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    assert_local_path(FIGDIR)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGDIR, f"r16_event_type_classifier_session_br.{ext}"), dpi=300)
    plt.close(fig)
    print(f"figure -> {os.path.join(FIGDIR, 'r16_event_type_classifier_session_br')}.png/.svg")


def fmt(r):
    return f"{r['median']:+.3f}, {r['same']}/{r['n']}, p = {r['p']:.3f}"


def main():
    W = load()
    if os.environ.get("RECOMPUTE", "") == "1" or not os.path.exists(BR_CSV):
        B = compute_br(W)
        assert_local_path(LOGDIR)
        B.to_csv(assert_local_path(BR_CSV), index=False)
        print(f"-> {BR_CSV}")
    else:
        B = pd.read_csv(BR_CSV)
        print(f"[cached BR table: {BR_CSV}; set RECOMPUTE=1 to redo the share pass]")
    W = W.merge(B[["session", "w", "BR", "BR_bg", "free_s"]], on=["session", "w"], how="left")
    ok = W[["BR", "BR_bg"]].dropna()
    print(f"BR: {W.BR.notna().sum()} intact, {W.BR_bg.notna().sum()} excised windows; "
          f"r(BR_bg, BR) = {np.corrcoef(ok.T.to_numpy())[0, 1]:.3f}; "
          f"median m intact {W.BR.median():.3f}, excised {W.BR_bg.median():.3f}; "
          f"within-session r(BR_bg, d2_bg) median "
          f"{W.groupby('session').apply(lambda h: h.BR_bg.corr(h.d2_bg), include_groups=False).median():+.2f}")

    rows, S, R = [], {}, {}
    for arm, suf in ARMS:
        six = [(f"BR{suf}", "branching ratio"), (f"d2{suf}", "d2"), (f"MUA{suf}", "MUA"),
               (f"fano{suf}", "MUA Fano"), (f"sync{suf}", "synchrony"), (f"EI{suf}", "E/I")]
        models = {"six": six, "br_alone": six[:1], "d2_alone": six[1:2], "d2_br": six[:2]}
        S[arm] = {}
        print(f"\n{'=' * 84}\n{arm.upper()} arm, session-level classifier\n{'=' * 84}")
        for name, preds in models.items():
            C = session_fit(W, preds)
            C["arm"], C["model"] = arm, name
            rows.append(C)
            S[arm][name] = summarise(C, preds)
            print(f"  [{name}]")
            for c, lab in preds:
                r = S[arm][name][c]
                print(f"    {lab:16s} median {r['median']:+.3f}  same sign {r['same']}/{r['n']}  "
                      f"signed-rank p = {r['p']:.4f}")
        R[arm] = dict(br_alone=fmt(S[arm]["br_alone"][f"BR{suf}"]),
                      d2_alone=fmt(S[arm]["d2_alone"][f"d2{suf}"]),
                      d2_with_br=fmt(S[arm]["d2_br"][f"d2{suf}"]),
                      br_with_d2=fmt(S[arm]["d2_br"][f"BR{suf}"]))
    C = pd.concat(rows, ignore_index=True)
    C.to_csv(assert_local_path(OUT), index=False)
    print(f"\n-> {OUT}")
    plot(C, S, R)


if __name__ == "__main__":
    main()
