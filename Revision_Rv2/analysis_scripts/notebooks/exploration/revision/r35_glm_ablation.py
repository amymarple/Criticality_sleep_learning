"""
r35_glm_ablation.py -- contribution of replay to d2 beyond SWR
properties, by GLM ablation (following Chang et al. 2025, Nature 637:1161, Fig. 2e).

Because rate_in + rate_out == rate_all identically, the replay-carrying and non-replay
SWR rates cannot be compared directly as two arms of one total. Instead every candidate
SWR property and replay are put into one model together; each predictor is ablated in
turn (its train-fold coefficient zeroed -- predictors are z-scored, so this is equivalent
to holding that variable at its mean), and the loss in held-out prediction is the
variable's contribution.

    predictors : SWR rate, mean ripple amplitude, mean participating cells, MUA,
                 replay probability (replay_count / ripple_count, rate-normalised by
                 construction)
    target     : d2 (2024 Euclidean, p=20)
    gain       : cross-validated R^2, full model minus each ablated model
    floor      : the same pipeline with the target permuted (N_SHUF draws)

Windows are non-overlapping (every 15th of the 30 s / 2 s grid), NREM only. Two
cross-validation schemes (--scheme): `block`, 5 contiguous time blocks (every fold
extrapolates across session drift); `interleaved`, every 5th window (measures fit rather
than extrapolation). Session is the unit of analysis: per-session CV R^2, then
a signed-rank test across sessions.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/r35_glm_ablation.csv
    -> reports/revision_figs/fig2_glm_ablation.png / .svg
"""
from __future__ import annotations

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
PROB_DIR = os.path.join(REPO, "data", "replay", "replay_probality")
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
WIN_S = 30.0
NONOVL = 15
N_FOLD = 5
N_SHUF = 200
SEED = 20260819

BASEPATH = {
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

PREDICTORS = [
    ("ripple_rate",  "SWR rate"),
    ("ripple_amp",   "ripple amplitude"),
    ("n_cells",      "participating cells"),
    ("MUA",          "MUA"),
    ("replay_prob",  "replay probability"),
]


def assert_local_path(path):
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        raise PermissionError(f"refusing to write to UNC path: {p}")
    if os.path.splitdrive(p)[0].rstrip(":").upper() not in {"C", "D"}:
        raise PermissionError(f"refusing to write outside C:/D:: {p}")
    return p


def window_mean(ev, vals, centers, win=WIN_S):
    out = np.full(len(centers), np.nan)
    if not len(ev):
        return out
    a, b = ev[:, 0], ev[:, 1]
    for i, c in enumerate(centers):
        m = (a < c + win / 2) & (b > c - win / 2)
        if m.any():
            v = vals[m][np.isfinite(vals[m])]
            if v.size:
                out[i] = v.mean()
    return out


def build(session, extra_centers=None):
    """One row per non-overlapping NREM window.

    `extra_centers` additionally keeps the Fig. 2d rows nearest to those times (within
    1 s). Used to bring the **d2 (2025 KL)** window grid into the same table: KL windows
    come from the revision pipeline's own tiling and do not coincide with every 15th
    window of the Fig. 2d grid, so without this they would be dropped. Default None
    reproduces the original behaviour exactly."""
    import nelpy as nel
    from neuro_py.io import loading

    with open(os.path.join(PROB_DIR, f"{session}.pkl"), "rb") as fh:
        o = pickle.load(fh)
    d = o["per_center"]
    d = d[d["state"].astype(str).str.upper().eq("NREM")]
    d = d[d["ripple_count"].astype(float) > 0]
    d = d.dropna(subset=["d2", "replay_prob"]).sort_values("center_s")
    keep = np.zeros(len(d), bool)
    keep[::NONOVL] = True
    if extra_centers is not None and len(extra_centers):
        cs = d["center_s"].to_numpy(float)
        ec = np.sort(np.asarray(extra_centers, float))
        idx = np.searchsorted(cs, ec)
        cand = np.clip(np.unique(np.concatenate([idx - 1, idx])), 0, len(cs) - 1)
        near = np.abs(cs[cand][:, None] - ec[None, :]).min(axis=1) <= 1.0
        keep[cand[near]] = True
    d = d[keep].reset_index(drop=True)
    c = d["center_s"].to_numpy(float)

    bp = BASEPATH[session]
    rip = loading.load_ripples_events(bp)
    sd = loading.load_SleepState_states(bp)
    nrem = nel.EpochArray(sd["NREMstate"])
    riv = (nrem & nel.EpochArray(np.array(rip[["start", "stop"]]))).data
    # map NREM-clipped ripples back to their source rows for the unclipped amplitude
    src = np.clip(np.searchsorted(rip["start"].to_numpy(), riv[:, 0], "right") - 1,
                  0, len(rip) - 1)
    amp = rip["amplitude"].to_numpy(float)[src]

    st, cm = loading.load_spikes(bp, brainRegion="CA1")
    units = [np.asarray(u, float) for u in np.asarray(st.data, dtype=object)]
    ncell = np.zeros(len(riv), int)
    for u in units:
        ncell += (np.searchsorted(u, riv[:, 1]) - np.searchsorted(u, riv[:, 0])) > 0

    # MUA is computed here rather than read from the pickle: PPP23_day8's
    # replay_probality pickle has no MUA column, and taking it from the pickle would
    # silently drop that session from an n=9 analysis. Population rate over all CA1
    # units in the window.
    allsp = np.sort(np.concatenate(units)) if units else np.array([])
    mua = ((np.searchsorted(allsp, c + WIN_S / 2)
            - np.searchsorted(allsp, c - WIN_S / 2)) / WIN_S
           if allsp.size else np.full(len(c), np.nan))

    W = pd.DataFrame({
        "session": session, "center_s": c,
        "d2": d["d2"].to_numpy(float),
        "ripple_rate": d["ripple_count"].to_numpy(float) / WIN_S,
        "replay_prob": d["replay_prob"].to_numpy(float),
        "MUA": mua,
        "ripple_amp": window_mean(riv, amp, c),
        "n_cells": window_mean(riv, ncell.astype(float), c),
    })
    if "MUA" in d.columns and d["MUA"].notna().any():
        r = np.corrcoef(mua, d["MUA"].to_numpy(float))[0, 1]
        print(f"    (MUA recomputed; r={r:.3f} vs the pickle's own column)", flush=True)
    print(f"  {session:12} {len(W):4d} windows, {len(riv):6d} NREM ripples, "
          f"{len(units):3d} CA1 units", flush=True)
    return W


def cv_r2(X, y, ablate=None, n_fold=N_FOLD, scheme="block"):
    """Cross-validated R^2. `ablate` = column index whose FULL-model coefficient is
    zeroed before predicting held-out data (Chang 2025's ablation).

    scheme='block'      : 5 CONTIGUOUS time blocks. Every fold is an extrapolation
                          across session drift, which understates fit.
    scheme='interleaved': every 5th window. Windows are already 30 s apart and
                          non-overlapping, so leakage is limited, and this measures
                          whether the model captures structure at all rather than
                          whether it extrapolates across drift."""
    n = len(y)
    if n < n_fold * 4:
        return np.nan
    bounds = np.linspace(0, n, n_fold + 1).astype(int)
    pred = np.full(n, np.nan)
    for k in range(n_fold):
        te = np.zeros(n, bool)
        if scheme == "block":
            te[bounds[k]:bounds[k + 1]] = True
        else:
            te[k::n_fold] = True
        tr = ~te
        if tr.sum() < 10 or te.sum() < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        A = np.column_stack([np.ones(Xtr.shape[0]), Xtr])
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        if ablate is not None:
            beta = beta.copy()
            beta[1 + ablate] = 0.0          # z-scored -> == holding it at its mean
        pred[te] = np.column_stack([np.ones(Xte.shape[0]), Xte]) @ beta
    ok = np.isfinite(pred)
    if ok.sum() < 10:
        return np.nan
    ss_res = np.sum((y[ok] - pred[ok]) ** 2)
    ss_tot = np.sum((y[ok] - y[ok].mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def run_session(W, rng, scheme="block"):
    cols = [c for c, _ in PREDICTORS]
    sub = W[["d2"] + cols].dropna()
    if len(sub) < 40:
        return None
    y = sub["d2"].to_numpy(float)
    X = sub[cols].to_numpy(float)
    out = {"n": len(sub), "scheme": scheme, "r2_full": cv_r2(X, y, scheme=scheme)}
    for j, (c, _) in enumerate(PREDICTORS):
        out[f"r2_abl_{c}"] = cv_r2(X, y, ablate=j, scheme=scheme)
        out[f"gain_{c}"] = out["r2_full"] - out[f"r2_abl_{c}"]
    shuf = [cv_r2(X, rng.permutation(y), scheme=scheme) for _ in range(N_SHUF)]
    shuf = np.asarray([s for s in shuf if np.isfinite(s)])
    out["r2_shuffle"] = float(np.mean(shuf)) if shuf.size else np.nan
    out["r2_shuffle_p95"] = float(np.percentile(shuf, 95)) if shuf.size else np.nan
    out["p_vs_shuffle"] = (float((np.sum(shuf >= out["r2_full"]) + 1) / (shuf.size + 1))
                           if shuf.size else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--scheme", choices=("block", "interleaved"), default="block",
                    help="block = 5 contiguous time blocks (conservative, extrapolates "
                         "across session drift); interleaved = every 5th window "
                         "(measures fit rather than extrapolation)")
    a = ap.parse_args()
    names = a.sessions or list(BASEPATH)
    rng = np.random.default_rng(SEED)
    print(f"CV scheme: {a.scheme}")

    print(f"building windows ({len(names)} sessions, non-overlapping, NREM)")
    rows, frames = [], []
    for s in names:
        try:
            W = build(s)
        except Exception as e:                                # noqa: BLE001
            print(f"  [SKIP] {s}: {type(e).__name__}: {e}", flush=True)
            continue
        frames.append(W)
        r = run_session(W, rng, scheme=a.scheme)
        if r:
            rows.append(dict(session=s, **r))
    D = pd.DataFrame(rows)
    assert_local_path(LOGDIR); os.makedirs(LOGDIR, exist_ok=True)
    D.to_csv(assert_local_path(os.path.join(LOGDIR, f"r35_glm_ablation_{a.scheme}.csv")),
             index=False)
    pd.concat(frames, ignore_index=True).to_csv(assert_local_path(
        os.path.join(LOGDIR, "r35_glm_ablation_windows.csv")), index=False)

    print(f"\n{'='*76}\nGLM ablation — target: d2 (2024 Euclidean, p=20), "
          f"n={len(D)} sessions\n{'='*76}")
    print(f"\n  full-model CV R2 : {D.r2_full.median():+.4f}  "
          f"(shuffle {D.r2_shuffle.median():+.4f}, "
          f"sessions beating their own 95th pct: "
          f"{int((D.r2_full > D.r2_shuffle_p95).sum())}/{len(D)})")

    print(f"\n{'ablated variable':24}{'dR2':>10}{'IQR':>20}{'p':>9}{'pos':>7}")
    res = []
    for c, lbl in PREDICTORS:
        v = D[f"gain_{c}"].dropna()
        p = stats.wilcoxon(v).pvalue if len(v) >= 3 else np.nan
        q1, q3 = np.percentile(v, [25, 75])
        print(f"{lbl:24}{np.median(v):+10.4f}   [{q1:+.4f},{q3:+.4f}]"
              f"{p:9.4f}{int((v>0).sum())}/{len(v):<3}")
        res.append(dict(var=c, label=lbl, dr2=np.median(v), q1=q1, q3=q3, p=p,
                        n_pos=int((v > 0).sum()), n=len(v)))
    R = pd.DataFrame(res).sort_values("dr2", ascending=False)
    print(f"\n  ranking by contribution: "
          + " > ".join(R.label.tolist()))
    print("\n  A variable's dR2 is how much held-out prediction is LOST when its"
          "\n  coefficient is zeroed. replay_prob is rate-normalised by construction,"
          "\n  so its contribution cannot be a restatement of SWR rate.")
    plot(D, R, a.scheme)


def plot(D, R, scheme='block'):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax = axes[0]
    x = np.arange(len(D))
    ax.bar(x - 0.2, D.r2_full, 0.4, color="#691c1b", label="full model")
    ax.bar(x + 0.2, D.r2_shuffle, 0.4, color="0.72", label="target shuffled")
    ax.set_xticks(x)
    ax.set_xticklabels(D.session, rotation=90, fontsize=7)
    ax.set_ylabel("cross-validated $R^2$")
    ax.set_title("full model vs shuffle, per session", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(0, color="0.3", lw=0.9)

    ax = axes[1]
    y = np.arange(len(R))
    col = ["#c0392b" if v == "replay_prob" else "#2c3e50" for v in R["var"]]
    ax.barh(y, R.dr2, xerr=[R.dr2 - R.q1, R.q3 - R.dr2], color=col,
            error_kw=dict(lw=1.1))
    ax.set_yticks(y)
    ax.set_yticklabels(R.label, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_xlabel("$\\Delta R^2$ when ablated  (median, IQR)")
    ax.set_title("contribution to predicting d2", fontsize=9)
    for i, r in R.reset_index(drop=True).iterrows():
        ax.text(0.98, i, f"p={r.p:.3f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7.5,
                fontweight="bold" if r.p < 0.05 else "normal",
                color="#111" if r.p < 0.05 else "0.45")
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    assert_local_path(FIGDIR); os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(
            FIGDIR, f"fig2_glm_ablation_{scheme}.{ext}")), dpi=200, bbox_inches="tight")
    print(f"\nfigure -> {FIGDIR}\\fig2_glm_ablation.png/.svg")
    print(f"tables -> {LOGDIR}\\r35_glm_ablation*.csv")


if __name__ == "__main__":
    main()
