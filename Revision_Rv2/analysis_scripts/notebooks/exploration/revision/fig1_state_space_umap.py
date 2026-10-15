"""
fig1_state_space_umap.py — stage 2 of the Chang-2025-style state-space figure.

RUN WITH A UMAP-CAPABLE ENV, NOT `criticality` (umap-learn is LAPACK-dead there):

    C:/Users/Cornell/.conda/envs/scientific-skills/python.exe fig1_state_space_umap.py

Standalone on purpose: numpy/pandas/matplotlib/umap only, no repo imports.

Design mirrors the lab's own published paper (Chang et al., Nature 637, 2025, Fig. 1d-e):
  - UMAP is VISUALIZATION ONLY. Params follow the repo's in-house convention
    (low_high_ripples_controlwoavalanche.ipynb, Previous_Rv1/analysis_scripts): n_neighbors=20, metric='cosine',
    random_state=42.
  - The statistic lives in the ORIGINAL feature space: per-session kNN state
    purity (fraction of a window's k=15 nearest neighbours, within session,
    sharing its state) against a within-session label-shuffle null — the same
    embed-for-the-eye / test-in-native-space logic as Chang et al.'s
    graph-based structure index.

Features are z-scored PER SESSION before pooling, so the pooled embedding
shows state structure rather than animal/unit-count batch structure. A
session-colored diagnostic panel is included to verify that.

Inputs : data/revision/log/fig1_state_space_features.csv  (stage 1)
Outputs: reports/revision_figs/fig1_state_umap.png/.svg
         data/revision/log/fig1_state_space_embedding.csv
         data/revision/log/fig1_state_space_knn.csv
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")

# CURATED one-per-family set (see fig1_state_space_export.py for what was
# dropped as redundant and why) — no family dominates the embedding distance.
FEATURES = [
    "MUA_mean", "MUA_fano",
    "EI_logratio",
    "synchrony_sub81_mean",
    "isi_entropy_pooled",
    "psd_slope_20_45", "sw_broadband",
    "theta_ratio",                     # buzcode theta above 1/f
    "ripple_rate", "ripple_amp_mean",  # drop the 2 sessions without ripple files
]
STATES = ["THETA", "nonTHETA", "REM", "NREM"]
STATE_LABEL = {"THETA": "Wake \u03b8", "nonTHETA": "Wake non-\u03b8",
               "REM": "REM", "NREM": "NREM"}
# the manuscript's canonical state palette
# (wake family = oranges, sleep family = purples; from Fig1_D_histogram_S1_E.ipynb)
STATE_COLOR = {"THETA": "#d95f0e", "nonTHETA": "#fdb863",
               "REM": "#d7b5d8", "NREM": "#7b3294"}

KNN_K = 15
N_SHUFFLE = 500
UMAP_KW = dict(n_components=2, n_neighbors=20, metric="cosine", random_state=42)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none",
})


def assert_local_path(path):
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or os.path.splitdrive(p)[0].rstrip(":").upper() not in {"C", "D"}:
        raise PermissionError(f"refusing to write outside C:/D:: {p}")
    return p


def zscore_per_session(df, cols):
    z = df.copy()
    for s, idx in df.groupby("session").groups.items():
        block = df.loc[idx, cols]
        z.loc[idx, cols] = (block - block.mean()) / block.std(ddof=0).replace(0, 1)
    return z


def knn_purity(X, labels, k=KNN_K):
    """Mean fraction of each point's k nearest neighbours sharing its label,
    plus the fixed neighbour index matrix for shuffle reuse."""
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nbr = np.argsort(d2, axis=1)[:, :k]
    purity = (labels[nbr] == labels[:, None]).mean()
    return float(purity), nbr


def main():
    df = pd.read_csv(os.path.join(LOGDIR, "fig1_state_space_features.csv"))
    feats = [c for c in FEATURES if c in df.columns]

    # ripple_amp_mean is undefined (NaN) in windows with zero ripples. For
    # sessions that HAVE ripple data, impute those windows with the session
    # mean (=> z-score 0: neutral, keeps the window in the map). Sessions
    # without ripple files stay NaN and drop with ripple_rate.
    if "ripple_amp_mean" in feats and "ripple_rate" in df.columns:
        n_imp = 0
        for s, idx in df.groupby("session").groups.items():
            sub = df.loc[idx]
            m = sub["ripple_rate"].notna() & sub["ripple_amp_mean"].isna()
            if m.any():
                df.loc[sub.index[m], "ripple_amp_mean"] = sub["ripple_amp_mean"].mean()
                n_imp += int(m.sum())
        print(f"ripple_amp_mean: {n_imp} zero-ripple windows imputed to session mean")
    # a feature that is missing for a whole session (e.g. a failed ripple load)
    # must not silently delete that session via the row-wise NaN drop below
    bad = [c for c in feats if df[c].isna().mean() > 0.2]
    if bad:
        print(f"[!] excluding mostly-NaN features from the embedding: {bad}")
        feats = [c for c in feats if c not in bad]
    ok = df[feats].notna().all(axis=1)
    print("windows dropped for NaN features, by state:")
    print(df.loc[~ok].groupby("state").size().to_string() if (~ok).any() else "  none")
    df = df.loc[ok].reset_index(drop=True)

    dfz = zscore_per_session(df, feats)
    X = dfz[feats].to_numpy(float)

    # --- statistic first, in the original feature space --------------------
    rng = np.random.default_rng(0)
    rows = []
    for s, sub in dfz.groupby("session"):
        Xs = sub[feats].to_numpy(float)
        lab = sub["state"].to_numpy()
        if len(sub) <= KNN_K or len(np.unique(lab)) < 2:
            continue
        purity, nbr = knn_purity(Xs, lab)
        null = np.empty(N_SHUFFLE)
        for i in range(N_SHUFFLE):
            perm = rng.permutation(lab)
            null[i] = (perm[nbr] == perm[:, None]).mean()
        rows.append({"session": s, "n_windows": len(sub),
                     "knn_purity": purity,
                     "null_median": float(np.median(null)),
                     "null_p95": float(np.percentile(null, 95)),
                     "p_perm": float((np.sum(null >= purity) + 1) / (N_SHUFFLE + 1))})
    knn = pd.DataFrame(rows)
    knn.to_csv(assert_local_path(os.path.join(LOGDIR, "fig1_state_space_knn.csv")),
               index=False)
    n_beat = int((knn.knn_purity > knn.null_p95).sum())
    print(f"\nkNN state purity (original {len(feats)}-D space, k={KNN_K}): "
          f"median {knn.knn_purity.median():.3f} vs null {knn.null_median.median():.3f}; "
          f"{n_beat}/{len(knn)} sessions beat their 95th shuffle pct; "
          f"all p_perm<={knn.p_perm.max():.4f}")

    # --- UMAP for the eye ---------------------------------------------------
    import umap
    emb = umap.UMAP(**UMAP_KW).fit_transform(X)
    df["UMAP1"], df["UMAP2"] = emb[:, 0], emb[:, 1]
    df.to_csv(assert_local_path(
        os.path.join(LOGDIR, "fig1_state_space_embedding.csv")), index=False)

    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.0))

    ax = axes[0]
    # draw big states first so the scarce ones (REM 227, THETA 231) stay visible
    order = sorted(STATES, key=lambda s: -(df.state == s).sum())
    for s in order:
        m = df.state == s
        ax.scatter(df.UMAP1[m], df.UMAP2[m], s=3, c=STATE_COLOR[s],
                   label=STATE_LABEL[s], alpha=0.6, linewidths=0)
    handles, labels_ = ax.get_legend_handles_labels()
    idx = [labels_.index(STATE_LABEL[s]) for s in STATES]
    ax.legend([handles[i] for i in idx], [labels_[i] for i in idx],
              markerscale=3, fontsize=7, frameon=False)
    ax.set_title("Fingerprint space by brain state", loc="left", fontsize=9)

    ax = axes[1]
    if "d2" in df.columns and df.d2.notna().any():
        v = df.d2.to_numpy(float)
        vmin, vmax = np.nanpercentile(v, [5, 95])
        sc = ax.scatter(df.UMAP1, df.UMAP2, s=3, c=v, cmap="viridis",
                        vmin=vmin, vmax=vmax, alpha=0.8, linewidths=0)
        fig.colorbar(sc, ax=ax, shrink=0.8, label="d2 (2024 Euclidean)")
    ax.set_title("Same map, colored by d2 (2024 Euclidean)", loc="left",
                 fontsize=9)

    ax = axes[2]
    if "ripple_rate" in df.columns and df.ripple_rate.notna().any():
        v = df.ripple_rate.to_numpy(float)
        fin = np.isfinite(v)
        vmin, vmax = np.nanpercentile(v[fin], [5, 95])
        ax.scatter(df.UMAP1[~fin], df.UMAP2[~fin], s=3, c="0.85",
                   alpha=0.5, linewidths=0)     # sessions without ripple files
        sc = ax.scatter(df.UMAP1[fin], df.UMAP2[fin], s=3, c=v[fin],
                        cmap="magma", vmin=vmin, vmax=vmax, alpha=0.8,
                        linewidths=0)
        fig.colorbar(sc, ax=ax, shrink=0.8, label="SWR rate (Hz)")
        n_miss = df.loc[~fin, "session"].nunique()
        ttl = ("Same map, colored by SWR rate"
               + (f"\n(grey: {n_miss} sessions without ripple files)" if n_miss else ""))
    else:
        ttl = "Same map, colored by SWR rate (no data)"
    ax.set_title(ttl, loc="left", fontsize=9)

    ax = axes[3]
    sess_ids = {s: i for i, s in enumerate(sorted(df.session.unique()))}
    ax.scatter(df.UMAP1, df.UMAP2, s=3,
               c=[sess_ids[s] for s in df.session], cmap="tab20",
               alpha=0.6, linewidths=0)
    ax.set_title("Diagnostic: colored by session\n(structure here = batch effect)",
                 loc="left", fontsize=9)

    for ax in axes:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
    txt = (f"{len(df)} non-overlapping 30 s windows, {df.session.nunique()} sessions; "
           f"{len(feats)} fingerprint features, z-scored per session; "
           f"UMAP n_neighbors={UMAP_KW['n_neighbors']}, {UMAP_KW['metric']}. "
           f"Stats in native space: kNN purity "
           f"{knn.knn_purity.median():.2f} vs shuffle {knn.null_median.median():.2f}, "
           f"{n_beat}/{len(knn)} sessions > 95th shuffle pct.")
    fig.text(0.01, 0.005, txt, fontsize=6.5, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(
            os.path.join(FIGDIR, f"fig1_state_umap.{ext}")), dpi=250)
    print(f"wrote -> {FIGDIR}\\fig1_state_umap.png/.svg")


if __name__ == "__main__":
    main()
