"""
r16_glue_control.py -- gluing control for the event-excised d2.

Excising ripple/barrage bins and gluing the surviving runs before fitting one AR(20) across
the joins means lag regressions cross the glue points. This computes d2 vs ripple rate, d2 vs
barrage rate (barrage cohort), and d2 vs replay (replay set) under four arms per window: intact
(whole window); glued (the production estimator: event bins dropped, runs >= 1 s glued, one
AR(20) across the joins); segments (autocovariance accumulated within each event-free segment
only, no lag product crosses a gap); and sham (the same number and duration of intervals placed
at event-free positions, removed and glued the same way). Segments shorter than MIN_FREE_S (1 s)
are discarded in every arm. Units: CA1, firingRate > MIN_FIRING_RATE, pooled.

Barrage side (12 sessions, this script's own pass): within-session Spearman of each arm with
ripple rate and with barrage rate, each partialled on the other rate; signed-rank across
sessions. Replay side (8 sessions, read from existing tables): the same Spearman with replay
probability, replay count and ripple rate, plus the sham binomial ladder already on record.

Inputs: r16_event_excised_windows.csv, d2_yw_matlab_match.csv, r35_sham_excision_windows.csv,
r35_sham_excision.csv.

Reads the shares read-only; writes only to D:. RECOMPUTE=1 forces the barrage share pass.
    -> data/revision/log/r16_glue_control_windows.csv   (barrage cohort, per window, 4 arms)
    -> data/revision/log/r16_glue_control.csv           (both sides, per arm x target)
    -> reports/revision_figs/r16_glue_control.png/.svg   (a: ripple rate, b: barrage rate, c: replay)
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
from fig1_metrics import MIN_FIRING_RATE                              # noqa: E402
from barrage_order_kernel_sweep import BARRAGE_SESSIONS               # noqa: E402
from r16_event_excised_windows import intervals, WIN_S, BIN, OUT as EXCISED_CSV  # noqa: E402
from r16_event_dissociation import partial_spearman                   # noqa: E402
from r35_sham_excision import sham_intervals, bin_mask, glue, d2_of, MIN_FREE_S, N_SHAM  # noqa: E402
from d2_yw_matlab_match import d2_yw_segments, d2_ols_glued           # noqa: E402
from r16_block_level_dissociation import LOGDIR                       # noqa: E402

OUT_WIN = os.path.join(LOGDIR, "r16_glue_control_windows.csv")
OUT = os.path.join(LOGDIR, "r16_glue_control.csv")
YW_CSV = os.path.join(LOGDIR, "d2_yw_matlab_match.csv")
SHAM_WIN = os.path.join(LOGDIR, "r35_sham_excision_windows.csv")
SHAM_CSV = os.path.join(LOGDIR, "r35_sham_excision.csv")
SEED = 20260924


def free_segments(t0, t1, ev):
    """[t0, t1) minus the (sorted, possibly overlapping) event intervals; runs >= MIN_FREE_S."""
    segs, cur = [], t0
    for a, b in ev:
        if b <= t0 or a >= t1:
            continue
        if a > cur:
            segs.append((cur, min(a, t1)))
        cur = max(cur, b)
    if cur < t1:
        segs.append((cur, t1))
    return [(a, b) for a, b in segs if b - a >= MIN_FREE_S]


def d2_glued_bins(sp, t0, t1, iv):
    """The production recipe (`metrics`): 40 ms bins overlapping any interval dropped, glued."""
    nb = int(round(WIN_S / BIN))
    edges = t0 + BIN * np.arange(nb + 1)
    lo, hi = np.searchsorted(sp, (t0, t1))
    cnt = np.histogram(sp[lo:hi], bins=edges)[0].astype(float)
    return d2_of(glue(cnt, bin_mask(edges, iv)))


def barrage_pass(W):
    from neuro_py.io import loading
    rng = np.random.default_rng(SEED)
    rows, t_start = [], time.time()
    for sess, bp in BARRAGE_SESSIONS.items():
        g = W[W.session == sess]
        if not len(g):
            continue
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        units = [np.sort(np.asarray(u, float)) for u in np.asarray(st.data, dtype=object)[keep]]
        sp = np.sort(np.concatenate(units))
        ev = np.vstack([intervals(loading, bp, "load_ripples_events"),
                        intervals(loading, bp, "load_barrage_events")])
        ev = ev[np.argsort(ev[:, 0])] if len(ev) else np.zeros((0, 2))
        n_sh, n_ev = 0, 0
        for r in g.itertuples(index=False):
            t0, t1 = float(r.w), float(r.w) + WIN_S
            segs = free_segments(t0, t1, ev)
            ev_in = ev[(ev[:, 1] > t0) & (ev[:, 0] < t1)]
            ev_in = np.column_stack([np.clip(ev_in[:, 0], t0, t1), np.clip(ev_in[:, 1], t0, t1)]) \
                if len(ev_in) else np.zeros((0, 2))
            d_glued_re = d2_ols_glued(sp, segs) if segs else np.nan
            d_seg = d2_yw_segments(sp, segs) if segs else np.nan
            shams = []
            for _ in range(N_SHAM):
                siv = sham_intervals(ev_in, t0, t1, rng)
                n_sh += len(siv)
                n_ev += len(ev_in)
                shams.append(d2_glued_bins(sp, t0, t1, siv))
            rows.append(dict(session=sess, w=r.w, ripple_rate=r.ripple_rate,
                             barrage_rate=r.barrage_rate, d2_intact=r.d2, d2_glued=r.d2_bg,
                             d2_glued_recomp=d_glued_re, d2_segments=d_seg,
                             d2_sham=float(np.nanmean(shams)) if np.isfinite(shams).any() else np.nan,
                             free_s=float(sum(b - a for a, b in segs)), n_events=len(ev_in)))
        B = pd.DataFrame([x for x in rows if x["session"] == sess])
        ok = B[["d2_glued", "d2_glued_recomp"]].dropna()
        print(f"  {sess:12s} {len(B):4d} windows | r(glued, recomputed from segments) "
              f"{np.corrcoef(ok.T.to_numpy())[0, 1]:.3f} | r(glued, segments) "
              f"{B[['d2_glued', 'd2_segments']].corr().iloc[0, 1]:.3f} | r(glued, sham) "
              f"{B[['d2_glued', 'd2_sham']].corr().iloc[0, 1]:.3f} | sham intervals placed "
              f"{n_sh / max(n_ev, 1):.0%} | {time.time() - t_start:4.0f} s", flush=True)
    return pd.DataFrame(rows)


def signed(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return dict(median=float(np.median(v)), n=int(v.size),
                pos=int((v > 0).sum()), neg=int((v < 0).sum()),
                p=float(stats.wilcoxon(v).pvalue) if v.size > 1 else np.nan)


PER_SESSION = []      # (side, arm, target, session, rho) -- filled by corr_table, read by plot()


def corr_table(D, arms, targets, side, min_n=30):
    """Per arm x target: within-session Spearman (and partial on the other target when two)."""
    out = []
    for arm, lab in arms:
        for tgt, tlab in targets:
            rho, prho = [], []
            other = [t for t, _ in targets if t != tgt]
            for s, h in D.groupby("session"):
                h = h.dropna(subset=[arm] + [t for t, _ in targets])
                if len(h) < min_n or h[arm].std() == 0:
                    continue
                rho.append(stats.spearmanr(h[arm], h[tgt]).correlation)
                PER_SESSION.append(dict(side=side, arm=arm, arm_label=lab, target=tgt,
                                        target_label=tlab, session=s, rho=rho[-1]))
                if len(other) == 1:
                    prho.append(partial_spearman(h[arm].to_numpy(), h[tgt].to_numpy(),
                                                 h[other].to_numpy()))
            r = signed(rho)
            pr = signed(prho) if prho else {}
            out.append(dict(side=side, arm=arm, arm_label=lab, target=tgt, target_label=tlab,
                            rho_median=r["median"], n_sessions=r["n"], pos=r["pos"], neg=r["neg"],
                            p=r["p"], partial_median=pr.get("median", np.nan),
                            partial_pos=pr.get("pos", np.nan), partial_neg=pr.get("neg", np.nan),
                            partial_p=pr.get("p", np.nan)))
    return pd.DataFrame(out)


def show(T, title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(f"  {'arm':22s}{'target':18s}{'rho med':>9s}{'+/-/n':>10s}{'p':>9s}"
          f"{'partial':>9s}{'+/-':>8s}{'p':>9s}")
    for _, r in T.iterrows():
        pp = (f"{r.partial_median:9.3f}{int(r.partial_pos):4d}/{int(r.partial_neg):<3d}{r.partial_p:9.4f}"
              if np.isfinite(r.partial_median) else "")
        print(f"  {r.arm_label:22s}{r.target_label:18s}{r.rho_median:+9.3f}"
              f"{r.pos:4d}/{r.neg:<2d}/{r.n_sessions:<3d}{r.p:9.4f}{pp}")


def plot(T, with_sham=False):
    """Three panels, per-session Spearman (dots), box over sessions, signed-rank stars, one
    box per arm: a: d2 vs ripple rate; b: d2 vs barrage rate (barrage cohort, 12 sessions);
    c: d2 vs replay probability (replay set, 8 sessions; the sham arm from the sham table).
    The sham arm sits in backup/; with_sham=True redraws the four-arm version into
    backup/r16_glue_control_with_sham."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from r16_block_level_dissociation import FIGDIR
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 11, "axes.linewidth": 1.4,
                         "xtick.major.width": 1.4, "ytick.major.width": 1.4})
    P = pd.DataFrame(PER_SESSION)
    COL = {"intact": "0.75", "glued": "#7b3294", "segments": "#1b9e77", "sham": "#fdb863"}
    KIND = {"d2_intact": "intact", "d2_glued": "glued", "d2_segments": "segments", "d2_sham": "sham",
            "d2_published": "intact", "ols_exc_glued_thr": "glued", "yw_exc_thr": "segments",
            "py_sham": "sham"}
    LAB = {"intact": "intact\nwindow", "glued": "excised,\nglued\n(current)",
           "segments": "excised,\nwithin-\nsegment", "sham": "sham-\nexcised,\nglued"}
    panels = [
        ("a  d2 vs ripple rate", "barrage cohort", "ripple_rate",
         ["d2_intact", "d2_glued", "d2_segments", "d2_sham"]),
        ("b  d2 vs barrage rate", "barrage cohort", "barrage_rate",
         ["d2_intact", "d2_glued", "d2_segments", "d2_sham"]),
        ("c  d2 vs replay probability", None, None, None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6 if with_sham else 10.2, 4.9))
    for ax, (title, side, tgt, arms) in zip(axes, panels):
        if side is None:   # replay: three arms from yw_match, sham from the sham table
            Q = pd.concat([P[(P.side == "replay set (yw_match)") & (P.target == "replay_prob")
                             & P.arm.isin(["d2_published", "ols_exc_glued_thr", "yw_exc_thr"])],
                           P[(P.side == "replay set (sham)") & (P.target == "replay_frac")
                             & (P.arm == "py_sham")]], ignore_index=True)
            arms = ["d2_published", "ols_exc_glued_thr", "yw_exc_thr", "py_sham"]
            n_lab = "8 sessions"
        else:
            Q = P[(P.side == side) & (P.target == tgt) & P.arm.isin(arms)]
            n_lab = "12 sessions"
        if not with_sham:
            arms = [a for a in arms if KIND[a] != "sham"]
            Q = Q[Q.arm.isin(arms)]
        Q = Q.assign(kind=Q.arm.map(KIND))
        order = [KIND[a] for a in arms]
        sns.boxplot(data=Q, x="kind", y="rho", order=order, ax=ax, width=0.6, showfliers=False,
                    linewidth=1.8, palette=COL, hue="kind", legend=False)
        rng = np.random.default_rng(2)
        ymax = max(0.05, Q.rho.max()); ymin = min(-0.05, Q.rho.min())
        pad = 0.12 * (ymax - ymin)
        for i, k in enumerate(order):
            v = Q[Q.kind == k].rho.to_numpy()
            ax.plot(i + rng.uniform(-0.13, 0.13, v.size), v, "o", ms=4, color="k", alpha=0.6, mew=0)
            pw = stats.wilcoxon(v).pvalue if v.size > 1 else np.nan
            same = max(int((v > 0).sum()), int((v < 0).sum()))
            st = "***" if pw < .001 else "**" if pw < .01 else "*" if pw < .05 else "n.s."
            ax.text(i, ymax + 1.1 * pad, f"{same}/{v.size}\n{st}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold" if pw < .05 else "normal")
        ax.axhline(0, ls="--", lw=1.2, color="k")
        ax.set_ylim(ymin - pad, ymax + 3.6 * pad)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([LAB[k] for k in order], fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("within-session Spearman \u03c1" if ax is axes[0] else "")
        ax.set_title(f"{title}\n{n_lab}", loc="left", fontsize=10.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    note = ("Gluing control for the event-excised d2 (2024 Euclidean, p = 20). glued = the production estimator "
            "(event bins dropped, runs >= 1 s glued,\none AR(20) across the joins); within-segment = autocovariance "
            "accumulated inside each event-free segment only, no lag product crosses a join (Yule-Walker).\n"
            "Dots: sessions; numbers: sessions sharing the sign; stars: signed-rank across sessions. "
            "a-b: barrage cohort, ripples and barrages excised. c: replay set, ripples excised.")
    if not with_sham:
        fig.text(0.01, 0.006, note, fontsize=7.8, color="0.3", ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        assert_local_path(FIGDIR)
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(FIGDIR, f"r16_glue_control.{ext}"), dpi=300)
        plt.close(fig)
        print(f"figure -> {os.path.join(FIGDIR, 'r16_glue_control')}.png/.svg")
        return
    fig.text(0.01, 0.006,
             "Gluing control for the event-excised d2 (2024 Euclidean, p = 20). glued = the production estimator "
             "(event bins dropped, runs >= 1 s glued, one AR(20) across the joins); within-segment = autocovariance "
             "accumulated inside each\nevent-free segment only, no lag product crosses a join (Yule-Walker); sham = the "
             "same number and durations of intervals at random event-free positions, removed and glued the same way "
             "(5 draws averaged).\nDots: sessions; numbers: sessions sharing the sign; stars: signed-rank across sessions. "
             "a-b: barrage cohort, ripples and barrages excised. c: replay set, ripples excised (sham arm from the "
             "replay sham table, replay per SWR).",
             fontsize=7.8, color="0.3", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = os.path.join(FIGDIR, "backup", "r16_glue_control_with_sham")
    os.makedirs(assert_local_path(os.path.dirname(out)), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(f"{out}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"figure -> {out}.png/.svg")


def main():
    W = pd.read_csv(EXCISED_CSV)
    if os.environ.get("RECOMPUTE", "") == "1" or not os.path.exists(OUT_WIN):
        B = barrage_pass(W)
        assert_local_path(LOGDIR)
        B.to_csv(assert_local_path(OUT_WIN), index=False)
        print(f"-> {OUT_WIN}")
    else:
        B = pd.read_csv(OUT_WIN)
        print(f"[cached barrage arms: {OUT_WIN}; RECOMPUTE=1 to redo the share pass]")
    print(f"barrage cohort: {len(B)} windows, {B.session.nunique()} sessions; median d2 intact "
          f"{B.d2_intact.median():.4f}, glued {B.d2_glued.median():.4f}, segments "
          f"{B.d2_segments.median():.4f}, sham {B.d2_sham.median():.4f}; pooled r(glued, segments) "
          f"{B[['d2_glued', 'd2_segments']].corr().iloc[0, 1]:.3f}, r(glued, sham) "
          f"{B[['d2_glued', 'd2_sham']].corr().iloc[0, 1]:.3f}, r(glued, recomputed) "
          f"{B[['d2_glued', 'd2_glued_recomp']].corr().iloc[0, 1]:.4f}")
    arms_b = [("d2_intact", "intact"), ("d2_glued", "excised, glued (current)"),
              ("d2_segments", "excised, within-segment"), ("d2_sham", "sham-excised, glued")]
    tb = corr_table(B, arms_b, [("ripple_rate", "ripple rate"), ("barrage_rate", "barrage rate")],
                    "barrage cohort")
    show(tb, "BARRAGE COHORT: within-session Spearman of d2 with the two event rates (partial = on the other rate)")

    # replay side, from the stored arms
    Y = pd.read_csv(YW_CSV)
    arms_y = [("d2_published", "intact (published)"), ("ols_exc_glued_thr", "excised, glued OLS"),
              ("yw_exc_thr", "excised, within-segment YW"), ("d2_excised_mat", "excised, MATLAB .mat")]
    ty = corr_table(Y, arms_y, [("replay_prob", "replay probability"), ("ripple_rate", "ripple rate")],
                    "replay set (yw_match)")
    show(ty, "REPLAY SET (d2_yw_matlab_match.csv, 8 sessions): Spearman with replay probability and ripple rate")
    S = pd.read_csv(SHAM_WIN)
    S["replay_frac"] = S.n_replay / S.n_swr.replace(0, np.nan)
    S["replay_count"] = S.n_replay
    arms_s = [("py_full", "intact"), ("py_excised", "excised, glued"), ("py_sham", "sham-excised, glued")]
    ts = corr_table(S, arms_s, [("replay_frac", "replay per SWR"), ("replay_count", "replay count"),
                                ("rate", "SWR rate")], "replay set (sham)")
    show(ts, "REPLAY SET (r35_sham_excision_windows.csv, 8 sessions): Spearman with replay per SWR, replay count, SWR rate")
    R = pd.read_csv(SHAM_CSV)
    print("\n  sham binomial ladder already on record (r35_sham_excision.csv), d2 alone (M1), coefficient per session:")
    rows_r = []
    for arm, g in R[R.model == "M1"].groupby("arm"):
        r = signed(g.b1)
        rows_r.append(dict(side="replay set (sham ladder M1)", arm=arm, arm_label=arm, target="replay_per_swr",
                           target_label="binomial b1", rho_median=r["median"], n_sessions=r["n"],
                           pos=r["pos"], neg=r["neg"], p=r["p"]))
        print(f"    {arm:14s} median b1 {r['median']:+.3f}  positive {r['pos']}/{r['n']}  signed-rank p = {r['p']:.4f}")
    T = pd.concat([tb, ty, ts, pd.DataFrame(rows_r)], ignore_index=True)
    T.to_csv(assert_local_path(OUT), index=False)
    print(f"\n-> {OUT}")
    plot(T)
    plot(T, with_sham=True)


if __name__ == "__main__":
    main()
