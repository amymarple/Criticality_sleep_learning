"""fig2f_producer_glm.py -- Figure 2f: the GLM model function, extracted unchanged from its
producer notebook.

Source: the cell of the first submission's replay-GLM notebook that defines
slope_divergence_bootstrap, _build_design and cv_predictive_gain_logistic; the same functions are in
notebooks/exploration/generate_GLM.ipynb (cell 15), where the default feature set is (d2, MUA, FR_std).
They are reproduced without modification, so downstream scripts run the panel's own model rather than a
reimplementation.

Model as the code specifies it: L2 logistic regression (C=1, liblinear), predictors
z-scored, within-fold StandardScaler, state = quantile tiles of the tile_col passed in,
reduced = state + main effects, full = + feature x state interactions, 5-fold CV
(GroupKFold if a group column is given, else shuffled KFold), AUC and log-loss, then each
interaction ablated one at a time.

The producer bins the tile on ripple_count (SWR rate), not on d2; this rebuild follows the
code.
"""
# --- imports used by the functions below ---
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

def slope_divergence_bootstrap(
    d: pd.DataFrame,
    features=("d2", "MUA", "MUA_std", "EI_ratio"),
    tile_col="tile",
    replay_col="replay_present",
    n_bins=6,
    n_boot=1000,
    random_state=0,
    strict_require_both=True,   # require both replay & non-replay per bin
):
    """
    Computes slope of Δ(b) across bins where Δ(b) = mean(feature|replay=1,bin) - mean(feature|replay=0,bin),
    with bootstrap over rows. Returns:
      summ_df: per-feature mean slope + bootstrap SE
      raw_df:  per-bootstrap slopes
      df_work: processed dataframe with robust_tile + z-scored features
    """
    rng = np.random.default_rng(random_state)

    features = [f for f in features if f in d.columns]
    if len(features) == 0:
        raise ValueError("No requested features found in dataframe.")

    df = d.dropna(subset=[tile_col, replay_col] + features).copy()

    # robust bins
    if df[tile_col].nunique() > n_bins or df[tile_col].max() > n_bins:
        df["robust_tile"] = pd.qcut(df[tile_col], q=n_bins, labels=False, duplicates="drop")
    else:
        df["robust_tile"] = df[tile_col].astype(int)

    # z-score features globally; to z-score within session, do so before calling
    for f in features:
        mu = df[f].mean()
        sd = df[f].std(ddof=1)
        df[f"{f}_z"] = (df[f] - mu) / (sd if sd > 0 else 1.0)

    bins = np.sort(df["robust_tile"].unique())
    boot_slopes = {f: [] for f in features}

    for bi in range(n_boot):
        sample = df.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 1e9)))
        grouped = sample.groupby(["robust_tile", replay_col])[[f"{f}_z" for f in features]].mean()

        for f in features:
            deltas, x = [], []
            for b in bins:
                # require replay=1
                if (b, 1) not in grouped.index:
                    continue
                # require replay=0 if strict
                if strict_require_both and (b, 0) not in grouped.index:
                    continue

                val_in = grouped.loc[(b, 1), f"{f}_z"]
                if (b, 0) in grouped.index:
                    val_out = grouped.loc[(b, 0), f"{f}_z"]
                else:
                    # fallback anchor
                    val_out = 0.0

                deltas.append(val_in - val_out)
                x.append(b)

            if len(deltas) >= 2:
                slope, *_ = stats.linregress(x, deltas)
                boot_slopes[f].append(slope)
            else:
                boot_slopes[f].append(np.nan)

    raw_rows = []
    summ_rows = []
    for f in features:
        slopes = np.asarray(boot_slopes[f], float)
        slopes = slopes[np.isfinite(slopes)]
        raw_rows.extend([{"Feature": f, "Slope": s} for s in slopes])

        if len(slopes) == 0:
            mean_slope, se = np.nan, np.nan
            n_ok = 0
        else:
            mean_slope = slopes.mean()
            se = slopes.std(ddof=1)  # bootstrap SE
            n_ok = len(slopes)

        summ_rows.append({"Feature": f, "Mean_Slope": mean_slope, "Boot_SE": se, "N_Valid_Boot": n_ok})

    summ_df = pd.DataFrame(summ_rows).sort_values("Feature")
    raw_df = pd.DataFrame(raw_rows)

    return summ_df, raw_df, df


# ============================================================
# 2) Predictive GLM (logistic regression) + ablation gain
#    Predict replay_present from features + (feature × state-bin) interactions
# ============================================================
def _build_design(df, features, state_col="robust_tile", add_interactions=True):
    """
    Returns X matrix and column names.
    Interactions are feature_z * state (state treated numeric).
    """
    cols = []
    X_list = []

    # state
    s = df[state_col].astype(float).to_numpy().reshape(-1, 1)
    X_list.append(s)
    cols.append(state_col)

    # main effects
    for f in features:
        z = df[f"{f}_z"].to_numpy().reshape(-1, 1)
        X_list.append(z)
        cols.append(f"{f}_z")

    # interactions
    if add_interactions:
        for f in features:
            z = df[f"{f}_z"].to_numpy().reshape(-1, 1)
            X_list.append(z * s)
            cols.append(f"{f}_z:X{state_col}")

    X = np.hstack(X_list)
    return X, cols


def cv_predictive_gain_logistic(
    d: pd.DataFrame,
    features=("d2", "MUA", "MUA_std", "EI_ratio"),
    tile_col="tile",
    replay_col="replay_present",
    n_bins=6,
    group_col=None,       # e.g. "session" or "file"
    n_splits=5,
    C=1.0,                # inverse regularization strength (bigger = less reg)
    random_state=0,
):
    """
    Cross-validated predictive performance for:
      - reduced model: main effects only (state + features)
      - full model:    main + interactions (feature × state)
    Plus ablation: drop each interaction term one-at-a-time from the full model.

    Returns:
      results_df (summary), details (dict with fold-wise metrics)
    """
    # ---------- preprocess (same robust tiles + zscores) ----------
    summ_df, raw_df, df = slope_divergence_bootstrap(
        d, features=features, tile_col=tile_col, replay_col=replay_col, n_bins=n_bins,
        n_boot=50,  # quick sanity check only; not used for prediction
        random_state=random_state,
        strict_require_both=False
    )
    # keep only usable rows
    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=[replay_col, "robust_tile"] + [f"{f}_z" for f in features]).copy()

    y = df[replay_col].astype(int).to_numpy()

    # grouping / splits
    if group_col is not None and group_col in df.columns:
        groups = df[group_col].to_numpy()
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(df, y, groups=groups))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(df, y))

    # models
    # Use liblinear for smaller datasets; saga works too.
    def fit_eval(X, y, splits):
        aucs, lls = [], []
        for tr, te in splits:
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]

            # standardize predictors *within fold* for clean CV
            sc = StandardScaler(with_mean=True, with_std=True)
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)

            clf = LogisticRegression(
                penalty="l2",
                C=C,
                solver="liblinear",
                max_iter=2000,
                random_state=random_state
            )
            clf.fit(Xtr_s, ytr)
            p = clf.predict_proba(Xte_s)[:, 1]

            aucs.append(roc_auc_score(yte, p))
            lls.append(log_loss(yte, p, labels=[0, 1]))

        return np.array(aucs), np.array(lls)

    # ---------- Reduced model (no interactions) ----------
    X_red, red_cols = _build_design(df, features, state_col="robust_tile", add_interactions=False)
    auc_red, ll_red = fit_eval(X_red, y, splits)

    # ---------- Full model (with interactions) ----------
    X_full, full_cols = _build_design(df, features, state_col="robust_tile", add_interactions=True)
    auc_full, ll_full = fit_eval(X_full, y, splits)

    # predictive gain (higher AUC better; lower logloss better)
    gain_auc = auc_full.mean() - auc_red.mean()
    gain_ll  = ll_red.mean() - ll_full.mean()

    rows = [{
        "Model": "Reduced (main effects)",
        "AUC_mean": auc_red.mean(), "AUC_sd": auc_red.std(ddof=1),
        "LogLoss_mean": ll_red.mean(), "LogLoss_sd": ll_red.std(ddof=1),
        "n": len(df)
    },{
        "Model": "Full (main + interactions)",
        "AUC_mean": auc_full.mean(), "AUC_sd": auc_full.std(ddof=1),
        "LogLoss_mean": ll_full.mean(), "LogLoss_sd": ll_full.std(ddof=1),
        "n": len(df)
    },{
        "Model": "Gain (Full - Reduced)",
        "AUC_mean": gain_auc, "AUC_sd": np.nan,
        "LogLoss_mean": gain_ll, "LogLoss_sd": np.nan,
        "n": len(df)
    }]

    # ---------- Ablate each interaction one-at-a-time ----------
    # Identify interaction columns in full design
    inter_idx = [i for i, c in enumerate(full_cols) if ":Xrobust_tile" in c]
    base_auc, base_ll = auc_full.mean(), ll_full.mean()

    for idx in inter_idx:
        keep = [i for i in range(X_full.shape[1]) if i != idx]
        X_ab = X_full[:, keep]
        auc_ab, ll_ab = fit_eval(X_ab, y, splits)

        rows.append({
            "Model": f"Ablate {full_cols[idx]}",
            "AUC_mean": auc_ab.mean(), "AUC_sd": auc_ab.std(ddof=1),
            "LogLoss_mean": ll_ab.mean(), "LogLoss_sd": ll_ab.std(ddof=1),
            "ΔAUC_vs_Full": auc_ab.mean() - base_auc,
            "ΔLogLoss_vs_Full": ll_ab.mean() - base_ll,
            "n": len(df)
        })

    results_df = pd.DataFrame(rows)

    details = {
        "slope_divergence_summary_quick": summ_df,   # quick check only
        "reduced": {"cols": red_cols, "auc": auc_red, "logloss": ll_red},
        "full":    {"cols": full_cols, "auc": auc_full, "logloss": ll_full},
        "splits": splits,
        "df_used": df
    }
    return results_df, details
