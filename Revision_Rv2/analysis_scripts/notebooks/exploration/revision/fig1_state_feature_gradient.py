"""
fig1_state_feature_gradient.py -- organisation of d2 (2024 Euclidean, 40 ms, AR p = 20) within EACH
state in the window-feature space, and in total.

fig1_nrem_feature_gradient.py tests NREM only, per session. This script gives every state, and the total,
the same time-preserving test.

Bouts here are state EPISODES: the scored interval a window was cut from (`iv_idx` in the per-session
pickles; for NREM identical to the 30 s-adjacency bouts of fig1_nrem_feature_gradient). Episode length
differs sharply by state: NREM episodes are long and few per session, while nonTHETA, THETA and REM
episodes are short and more numerous, with THETA and REM windows also far fewer overall (nonTHETA is
further thinned evenly per session by run_fig1_metrics.STATE_WINDOW_CAP). The NREM per-session episode
and shift thresholds would therefore delete most wake and REM windows, and THETA and REM are too few per
session for a per-session statistic. Hence:
  per session  NREM (episodes >= 6, shift 2..L-2; reproduces fig1_nrem_feature_gradient) and nonTHETA (episodes
               >= 3, shift 1..L-1), neighbours within the session, as fig1_nrem_feature_gradient.
  pooled       every state (episodes >= 3, shift 1..L-1): the 16 sessions in one feature space (features z-scored
               per session), d2 z-scored within session over all its windows (raw session medians track unit
               yield), neighbours searched across sessions WITHIN the state, excluding the same session within
               EXCL_S. The null shifts d2 within each episode, so each session's and each episode's own d2 values
               and their order are kept: between-session and between-episode offsets are in the null too.
  total        all four states together, each window's neighbours from its own state (so the separation BETWEEN
               states cannot contribute): smoothness over all windows; SI = window-weighted mean of the state SIs;
               the same surrogate draw for every state.
Statistics as fig1_nrem_feature_gradient: smoothness = 1 - mean neighbour |d2 difference| (K nearest, Euclidean)
/ its null median; SI = PridaLab structure index (10 bins, 15 neighbours) minus its null median; one-sided
p = (1 + #null at least as extreme) / (1 + N_NULL). Pooled results are also split by animal (the windows of each
mouse, neighbours from any mouse) for replication. Feature sets: 10 and 8 (no ripple rate / amplitude).
Cohort and missing-value rule: fig1_nrem_feature_gradient.cohort_sessions / prepare (16 sessions).

THETA and REM keep few windows after the episode >= 3 rule, so their SI (a handful of windows per d2 bin) is
close to uninformative and their smoothness rests on few windows; shift 1 leaves the lag-1 autocorrelation
partly in the null.

Writes only to D:.
    -> data/revision/log/fig1_state_feature_gradient.csv          (pooled: state x feature set, + total)
    -> data/revision/log/fig1_state_feature_gradient_animal.csv   (pooled, split by animal)
    -> data/revision/log/fig1_state_feature_gradient_session.csv  (per session: NREM, nonTHETA)
    -> reports/revision_figs/fig1_state_feature_gradient.png / .svg
"""
from __future__ import annotations

import os
import sys
import zlib

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                        # noqa: E402
from fig1_state_space_umap import FEATURES, LOGDIR, FIGDIR, STATE_COLOR, STATE_LABEL   # noqa: E402
from fig1_nrem_feature_gradient import SETS, cohort_sessions, prepare, si, K, EXCL_S   # noqa: E402
from m25_tmaze_box import MOUSE_COL, MOUSE_LABEL      # noqa: E402  the manuscript's Mouse 1-6 (ID order) and colours

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
STATES = ["NREM", "nonTHETA", "THETA", "REM"]
FSETS = ["10", "8"]
POOLED = (3, 1)                                   # (min episode windows, min shift)
PER_SESSION = {"NREM": (6, 2), "nonTHETA": (3, 1)}
N_NULL = int(os.environ.get("N_NULL", "200"))
OUT = os.path.join(LOGDIR, "fig1_state_feature_gradient.csv")


def load():
    F = pd.read_csv(os.path.join(LOGDIR, "fig1_state_space_features.csv"))
    coh = cohort_sessions(F)
    Z = prepare(F, coh)
    iv = pd.concat([pd.read_pickle(os.path.join(REPO, "data", "revision", f"{s}_fig1_metrics_nonoverlap.pkl"))["windows"]
                    [["state", "center_s", "iv_idx"]].assign(session=s) for s in coh])
    Z = Z.merge(iv, on=["session", "state", "center_s"], how="left")
    assert Z.iv_idx.notna().all(), "a window without its state interval"
    Z["d2_z"] = Z.groupby("session").d2.transform(lambda v: (v - v.mean()) / v.std(ddof=0))
    Z["ep"] = Z.session + "|" + Z.state + "|" + Z.iv_idx.astype(int).astype(str)
    Z["L"] = Z.groupby("ep").ep.transform("size")
    return Z.sort_values(["session", "center_s"]).reset_index(drop=True)


def surrogates(y, ep, offset, rng):
    """N_NULL copies of y, each episode circularly shifted by offset..L-offset"""
    groups = [np.flatnonzero(ep == u) for u in pd.unique(ep)]
    out = np.empty((N_NULL, len(y)))
    for k in range(N_NULL):
        yn = y.copy()
        for idx in groups:
            yn[idx] = np.roll(y[idx], rng.integers(offset, len(idx) - offset + 1))
        out[k] = yn
    return out


def nbrs(X, t, sess):
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1).astype(np.float32)
    d[(sess[:, None] == sess[None, :]) & (np.abs(t[:, None] - t[None, :]) <= EXCL_S)] = np.inf
    return np.argsort(d, axis=1)[:, :K]


def pvalue(obs, null, larger=True):
    return (1 + np.sum(null >= obs if larger else null <= obs)) / (1 + len(null))


def pooled(Z):
    """per state and total, both feature sets; returns (state rows, animal rows)"""
    min_ep, off = POOLED
    P = Z[Z.L >= min_ep].reset_index(drop=True)
    rng = np.random.default_rng(zlib.crc32(b"pooled"))
    Y = surrogates(P.d2_z.to_numpy(float), P.ep.to_numpy(), off, rng)       # one draw for every state
    y = P.d2_z.to_numpy(float)
    rows, arows = [], []
    for fs in FSETS:
        feats = SETS[fs]
        D_obs = np.zeros(len(P)); D_null = np.zeros((N_NULL, len(P)))     # per-window neighbour |dd2|
        si_parts = {}
        for st in STATES:
            m = np.flatnonzero(P.state.to_numpy() == st)
            X = P.loc[m, feats].to_numpy(float)
            nb = nbrs(X, P.center_s.to_numpy(float)[m], P.session.to_numpy()[m])
            D_obs[m] = np.abs(y[m][:, None] - y[m][nb]).mean(1)
            for k in range(N_NULL):
                D_null[k, m] = np.abs(Y[k, m][:, None] - Y[k, m][nb]).mean(1)
            s_obs = si(X, y[m])[0]
            s_null = np.array([si(X, Y[k, m])[0] for k in range(N_NULL)], float)
            si_parts[st] = (len(m), s_obs, s_null)
            print(f"  {fs} features, {st}: n {len(m)}, SI {s_obs:.3f}", flush=True)
        scopes = [(st, P.state.to_numpy() == st) for st in STATES] + [("within-state total", np.ones(len(P), bool))]
        for scope, m in scopes:
            S_obs, S_null = D_obs[m].mean(), D_null[:, m].mean(1)
            if scope in si_parts:
                n, s_obs, s_null = si_parts[scope]
            else:                                                         # window-weighted mean of the state SIs
                w = np.array([si_parts[s][0] for s in STATES], float)
                s_obs = np.average([si_parts[s][1] for s in STATES], weights=w)
                s_null = np.average(np.vstack([np.nan_to_num(si_parts[s][2]) for s in STATES]), axis=0, weights=w)
                n = int(w.sum())
            rows.append(dict(scope=scope, feature_set=fs, n_windows=int(m.sum()), n_sessions=P[m].session.nunique(),
                             n_episodes=P[m].ep.nunique(),
                             frac_state_used=m.sum() / (Z.state == scope).sum() if scope in STATES else m.sum() / len(Z),
                             smooth_effect=1 - S_obs / np.median(S_null), smooth_null_lo=1 - np.percentile(S_null, 97.5) / np.median(S_null),
                             smooth_null_hi=1 - np.percentile(S_null, 2.5) / np.median(S_null), p_smooth=pvalue(S_obs, S_null, False),
                             SI=s_obs, SI_null_median=np.nanmedian(s_null), SI_null_p95=np.nanpercentile(s_null, 95),
                             SI_excess=s_obs - np.nanmedian(s_null), p_SI=pvalue(s_obs, np.nan_to_num(s_null))))
            for a in sorted(P[m].session.str.split("_").str[0].unique()):
                ma = m & (P.session.str.split("_").str[0] == a).to_numpy()
                Sa, Sna = D_obs[ma].mean(), D_null[:, ma].mean(1)
                arows.append(dict(scope=scope, feature_set=fs, animal=a, n_windows=int(ma.sum()),
                                  smooth_effect=1 - Sa / np.median(Sna), p_smooth=pvalue(Sa, Sna, False)))
    return pd.DataFrame(rows), pd.DataFrame(arows)


def per_session(Z):
    rows = []
    for st, (min_ep, off) in PER_SESSION.items():
        for s, g in Z[Z.state == st].groupby("session"):
            g = g[g.L >= min_ep].reset_index(drop=True)
            if len(g) <= K + 5:
                continue
            rng = np.random.default_rng(zlib.crc32(f"{s}|{st}".encode()))
            y = g.d2.to_numpy(float)
            Y = surrogates(y, g.ep.to_numpy(), off, rng)
            for fs in FSETS:
                X = g[SETS[fs]].to_numpy(float)
                nb = nbrs(X, g.center_s.to_numpy(float), g.session.to_numpy())
                S_obs = np.abs(y[:, None] - y[nb]).mean()
                S_null = np.array([np.abs(yn[:, None] - yn[nb]).mean() for yn in Y])
                s_obs = si(X, y)[0]
                s_null = np.nan_to_num(np.array([si(X, yn)[0] for yn in Y], float))
                rows.append(dict(state=st, session=s, animal=s.split("_")[0], feature_set=fs, min_episode=min_ep,
                                 min_shift=off, n_windows=len(g), n_episodes=g.ep.nunique(),
                                 smooth_effect=1 - S_obs / np.median(S_null), p_smooth=pvalue(S_obs, S_null, False),
                                 SI=s_obs, SI_excess=s_obs - np.median(s_null), p_SI=pvalue(s_obs, s_null)))
        print(f"  per session {st}: done", flush=True)
    return pd.DataFrame(rows)


def plot(R, A, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    scopes = STATES + ["within-state total"]
    lab = {**STATE_LABEL, "within-state total": "within-state\ntotal"}
    col = {**STATE_COLOR, "within-state total": "#444444"}
    P, Aa = R[R.feature_set == "10"].set_index("scope"), A[A.feature_set == "10"]
    animals = sorted(Aa.animal.unique(), key=lambda a: int(a[3:]))
    acol = MOUSE_COL
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    ax = axes[0]
    for i, sc in enumerate(scopes):
        r = P.loc[sc]
        ax.add_patch(plt.Rectangle((i - 0.3, r.smooth_null_lo), 0.6, r.smooth_null_hi - r.smooth_null_lo, color="0.85", lw=0))
        g = Aa[Aa.scope == sc]
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(g))
        ax.scatter(i + jit, g.smooth_effect, s=12, c=[acol[a] for a in g.animal], zorder=3, lw=0)
        ax.scatter(i, r.smooth_effect, marker="D", s=36, color=col[sc], edgecolor="k", zorder=4)
        ax.text(i, 0.47, f"n {int(r.n_windows)}\np {r.p_smooth:.3f}\n{int((g.smooth_effect > 0).sum())}/{len(g)} mice",
                ha="center", va="top", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xticks(range(len(scopes))); ax.set_xticklabels([lab[s] for s in scopes]); ax.set_ylim(-0.25, 0.5)
    ax.set_ylabel("1 - neighbour |Δd2| / null"); ax.set_title("a  local smoothness, pooled 16 sessions", loc="left", fontweight="bold")
    ax = axes[1]
    for i, sc in enumerate(scopes):
        r = P.loc[sc]
        ax.bar(i, r.SI_null_p95, width=0.6, color="0.85", lw=0)
        ax.scatter(i, r.SI, marker="D", s=36, color=col[sc], edgecolor="k", zorder=4)
        uninformative = r.SI == 0 and r.SI_null_p95 == 0          # observed and every surrogate at the floor
        ax.text(i, 0.34, f"n {int(r.n_windows)}\n" + ("SI = 0, null = 0:\ntoo few windows" if uninformative else f"p {r.p_SI:.3f}"),
                ha="center", va="top", fontsize=6.5)
    ax.set_xticks(range(len(scopes))); ax.set_xticklabels([lab[s] for s in scopes]); ax.set_ylim(0, 0.35)
    ax.set_ylabel("structure index of d2"); ax.set_title("b  structure index, pooled (grey: null 95th)", loc="left", fontweight="bold")
    ax = axes[2]
    Ss = S[S.feature_set == "10"]
    for i, st in enumerate(PER_SESSION):
        g = Ss[Ss.state == st]
        jit = np.random.default_rng(i).uniform(-0.15, 0.15, len(g))
        for (_, r), j in zip(g.iterrows(), jit):
            ax.scatter(i + j, r.smooth_effect, s=14, edgecolor=acol[r.animal], lw=0.8, zorder=3,
                       facecolor=acol[r.animal] if r.p_smooth < .05 else "none")
        am = g.groupby("animal").smooth_effect.mean()
        ax.scatter(np.full(len(am), i + 0.3), am, marker="D", s=12, color="k", zorder=4)
        ax.text(i, 0.47, f"{int((g.p_smooth < .05).sum())}/{len(g)} sessions p<.05\n{int((am > 0).sum())}/{len(am)} mice > 0",
                ha="center", va="top", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xticks(range(len(PER_SESSION))); ax.set_xticklabels([STATE_LABEL[s] for s in PER_SESSION])
    ax.set_xlim(-0.6, len(PER_SESSION) - 0.4); ax.set_ylim(-0.25, 0.5)
    ax.set_ylabel("1 - neighbour |Δd2| / null"); ax.set_title("c  local smoothness, per session", loc="left", fontweight="bold")
    h = [plt.Line2D([], [], marker="o", ls="", color=acol[a], label=MOUSE_LABEL[a]) for a in animals]
    ax.legend(handles=h, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("d2 (2024 Euclidean, 40 ms, p = 20) within each state, in the z-scored 10-feature space, against d2 "
                 f"circularly shifted within each state episode ({N_NULL} surrogates)", fontsize=8.5)
    fig.text(0.01, -0.04,
             f"a, b: 16 Fig-1 sessions with SWR detection in one space; windows in episodes of >= {POOLED[0]} windows, shift "
             f"{POOLED[1]}..L-{POOLED[1]}; d2 z-scored within session; neighbours ({K}) across sessions within the state, "
             f"excluding the same session within {EXCL_S:.0f} s. Diamond: pooled effect; grey: null 2.5-97.5 % (a) / 95th (b); "
             "dots: the windows of each mouse.\nWithin-state total: every window's neighbours from its own state, so the "
             "separation between states does not enter; SI = window-weighted mean. Wake theta and REM are small (see n): "
             f"their SI is close to uninformative. c: per session, NREM episodes >= {PER_SESSION['NREM'][0]} (shift "
             f"{PER_SESSION['NREM'][1]}), wake non-theta >= {PER_SESSION['nonTHETA'][0]} (shift {PER_SESSION['nonTHETA'][1]}); "
             "filled: p < 0.05.", fontsize=6.3, va="top")
    fig.tight_layout()
    o = os.path.join(FIGDIR, "fig1_state_feature_gradient")
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(f"{o}.{ext}"), dpi=200, bbox_inches="tight")
    print("->", o + ".png")


def main():
    names = (OUT, OUT.replace(".csv", "_animal.csv"), OUT.replace(".csv", "_session.csv"))
    if os.environ.get("PLOT_ONLY"):
        return plot(*[pd.read_csv(f, dtype={"feature_set": str}) for f in names])
    Z = load()
    print(f"{Z.session.nunique()} sessions, {len(Z)} windows; {N_NULL} surrogates")
    S = per_session(Z)
    R, A = pooled(Z)
    for df, f in zip((R, A, S), names):
        df.to_csv(assert_local_path(f), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
    print(R[["scope", "feature_set", "n_windows", "frac_state_used", "smooth_effect", "p_smooth", "SI", "SI_null_p95",
             "SI_excess", "p_SI"]].round(4).to_string(index=False))
    print(A.groupby(["feature_set", "scope"]).apply(lambda g: f"{int((g.smooth_effect > 0).sum())}/{len(g)} mice > 0, "
                                                             f"{int((g.p_smooth < .05).sum())} p<.05").to_string())
    print(S.groupby(["state", "feature_set"]).apply(lambda g: f"{int((g.p_smooth < .05).sum())}/{len(g)} smooth p<.05, "
                                                            f"{int((g.p_SI < .05).sum())}/{len(g)} SI p<.05, "
                                                            f"median effect {g.smooth_effect.median():.3f}").to_string())
    plot(R, A, S)


if __name__ == "__main__":
    main()
