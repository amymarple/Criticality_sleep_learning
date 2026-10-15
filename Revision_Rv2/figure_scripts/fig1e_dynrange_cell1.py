# Code cell 1 of analysis_scripts/notebooks/figures/Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb (up to 'df_base = all_pairs.copy()'), executed by FigS2_AF_robustness.ipynb.
# assume all_pairs already has:
# - 'd2'
# - 'MUA'  (or rename from 'mua30' if needed)
# - 'group'         (NREM, REM, Wake θ, etc.)
# - 'epoch_name'    (NaN for non-task, string for T-maze epochs)
# - 'animal', 'day' (for per-file ID)
# --- Tile by d2_z per state (6 tiles) ---
# ---------- Summary stats ----------
import matplotlib as mpl
# Global font: prefer Arial, with safe fallbacks
mpl.rcParams['svg.fonttype'] = 'none' 
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = [
    'Helvetica Neue', 'Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans'
]

# Mathtext to match (works if those faces exist on your system)
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Helvetica Neue'   # or 'Helvetica'
mpl.rcParams['mathtext.it'] = 'Helvetica Neue:italic'
mpl.rcParams['mathtext.bf'] = 'Helvetica Neue:bold'

# (optional) sizes + minus sign
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.unicode_minus'] = False

# Global sizes (tweak to taste)
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams["axes.linewidth"] = 1.5       # spine width
mpl.rcParams["xtick.major.width"] =1.5
mpl.rcParams["ytick.major.width"] = 1.5
mpl.rcParams["xtick.minor.width"] = 1.5
mpl.rcParams["ytick.minor.width"] = 1.5
mpl.rcParams["axes.linewidth"] = 1.5       # spine width
mpl.rcParams["xtick.major.width"] =1.5
mpl.rcParams["ytick.major.width"] = 1.5
mpl.rcParams["xtick.minor.width"] = 1.5
mpl.rcParams["ytick.minor.width"] = 1.5
# -------- settings --------
N_BINS = 6
MIN_PER_BIN = 2
BASELINE_Q = 0.25
STATE_LIST = ["NREM", "WAKE"]

# ---------- utils ----------
def coerce_to_df(dfs):
    if isinstance(dfs, pd.DataFrame):
        df = dfs.copy()
        if "__file__" not in df.columns:
            df["__file__"] = "file1"
        return df
    if isinstance(dfs, dict):
        parts = []
        for name, d in dfs.items():
            if isinstance(d, pd.DataFrame):
                t = d.copy()
                t["__file__"] = str(name)
                parts.append(t)
        return pd.concat(parts, ignore_index=True)
    parts = []
    for i, d in enumerate(dfs, start=1):
        if isinstance(d, pd.DataFrame):
            t = d.copy()
            if "__file__" not in t.columns:
                t["__file__"] = f"file{i}"
            parts.append(t)
    return pd.concat(parts, ignore_index=True)

def plot_bars(dfB, ycol, xlabel, title):
    if dfB.empty:
        print(f"[warn] empty results for: {title}")
        return
    x = np.arange(len(dfB))
    plt.figure(figsize=(6.5,4))
    plt.bar(x, dfB[ycol].values)
    plt.xticks(x, [f"{v:.2g}" for v in dfB["d2_center"].values], rotation=45, ha="right")
    plt.ylabel(ycol)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ---------- per-file d2 normalization ----------
def add_perfile_d2_normalized(df, method="percentile"):
    """
    Adds a normalized d2 column 'd2n' computed per file.
    method: 'percentile' (0..1), 'z', or 'minmax'
    """
    df = df.copy()
    def _percentile_col(g):
        x = g["d2"].to_numpy(float)
        # ranks in [0,1]
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(x.size, dtype=float)
        q = ranks / max(x.size - 1, 1)  # avoid div/0 for size=1
        g["d2n"] = q
        return g

    def _z_col(g):
        # clip to 1–99% to compute robust-ish mean/std
        lo, hi = g["d2"].quantile([0.01, 0.99])
        xc = g["d2"].clip(lo, hi)
        mu, sd = xc.mean(), xc.std(ddof=0)
        sd = sd if sd > 0 else np.nan
        g["d2n"] = (g["d2"] - mu) / sd
        return g

    def _minmax_col(g):
        lo, hi = g["d2"].min(), g["d2"].max()
        span = hi - lo
        g["d2n"] = (g["d2"] - lo) / (span if span > 0 else np.nan)
        return g

    if method == "percentile":
        df = df.groupby("__file__", group_keys=False).apply(_percentile_col)
    elif method == "z":
        df = df.groupby("__file__", group_keys=False).apply(_z_col)
    elif method == "minmax":
        df = df.groupby("__file__", group_keys=False).apply(_minmax_col)
    else:
        raise ValueError("method must be 'percentile', 'z', or 'minmax'")
    return df

def perfile_dyn_by_bin(dstate, method="percentile", bin_mode="equalwidth"):
    """
    Returns one row per (__file__, d2_bin) with per-file dynamic-range metric.
    
    method:
        'percentile', 'z', 'minmax'  -> how to normalize d2 per file into 'd2n'
    bin_mode:
        'equalwidth' -> fixed-width bins in normalized d2n
        'tile'       -> quantile / equal-count bins (forced tiles)
    """
    if dstate.empty:
        return pd.DataFrame(columns=["__file__", "d2_bin", "d2_center", "dyn", "n_in_bin"])

    # 1) normalize d2 per file
    dstate = add_perfile_d2_normalized(dstate, method=method)

    # keep only finite d2n
    dstate = dstate[np.isfinite(dstate["d2n"])].copy()
    if dstate.empty:
        return pd.DataFrame(columns=["__file__", "d2_bin", "d2_center", "dyn", "n_in_bin"])

    # 2) define bins
    if bin_mode == "equalwidth":
        # fixed-width bins across the normalized d2n range
        if method in ("percentile", "minmax"):
            bins = np.linspace(0, 1, N_BINS+1)
            centers = (bins[:-1] + bins[1:]) / 2.0
        else:  # z
            zmin, zmax = dstate["d2n"].quantile([0.01, 0.99])
            if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
                return pd.DataFrame(columns=["__file__", "d2_bin", "d2_center", "dyn", "n_in_bin"])
            bins = np.linspace(zmin, zmax, N_BINS+1)
            centers = (bins[:-1] + bins[1:]) / 2.0

        dstate["d2_bin"] = pd.cut(dstate["d2n"], bins=bins, labels=False, include_lowest=True)
        # lookup centers by integer bin index
        center_lookup = {i: float(c) for i, c in enumerate(centers)}

    elif bin_mode == "tile":
        # forced tiles: equal-count quantile bins across all rows
        # labels are 0..k-1; duplicates="drop" may give < N_BINS if data are degenerate
        dstate["d2_bin"] = pd.qcut(
            dstate["d2n"],
            q=N_BINS,
            labels=False,
            duplicates="drop"
        )
        # compute center as the median d2n within each tile
        centers_ser = (
            dstate
            .groupby("d2_bin")["d2n"]
            .median()
            .dropna()
        )
        center_lookup = centers_ser.to_dict()
    else:
        raise ValueError("bin_mode must be 'equalwidth' or 'tile'")

    # 3) per-(file, bin) mean of raw MUA
    fb_mean = dstate.groupby(["__file__", "d2_bin"])["MUA"].transform("mean")

    # normalize: mean within each (file, bin) becomes 1
    eps = 1e-12
    dstate["MUA_norm"] = dstate["MUA"] / (fb_mean.replace(0, np.nan) + eps)

    # keep positive finite values
    dstate = dstate[np.isfinite(dstate["MUA_norm"]) & (dstate["MUA_norm"] > 0)]

    rows = []
    for (f, b), g in dstate.groupby(["__file__", "d2_bin"], sort=True):
        if pd.isna(b):
            continue
        x = g["MUA_norm"].to_numpy(float)
        x = x[np.isfinite(x) & (x > 0)]
        if x.size < MIN_PER_BIN:
            continue

        p05, p95 = np.percentile(x, [5, 95])
        b_int = int(b)
        if b_int not in center_lookup:
            continue

        rows.append({
            "__file__": f,
            "d2_bin": b_int,
            "d2_center": float(center_lookup[b_int]),
            "dyn": float(10 * np.log10(p95 / p05)),
            "n_in_bin": int(x.size),
        })
    return pd.DataFrame(rows)


def summarize_mean_sem(df_perfile):
    """
    Aggregates over files: mean and SEM per d2_bin.
    """
    if df_perfile.empty:
        return pd.DataFrame(columns=["d2_bin", "d2_center", "mean", "sem", "n_files"])
    g = df_perfile.groupby("d2_bin", as_index=False)
    out = g.agg(
        d2_center=("d2_center", "median"),
        mean=("dyn", "mean"),
        std=("dyn", "std"),
        n_files=("__file__", "nunique"),
    )
    out["sem"] = out["std"] / np.sqrt(out["n_files"].clip(lower=1))
    return out[["d2_bin", "d2_center", "mean", "sem", "n_files"]].sort_values("d2_center")

def plot_mean_sem(dfS, xlabel, title, color, ylabel="dynamic range (p95–p05)", save_path=None):
    if dfS.empty:
        print(f"[warn] empty results for: {title}")
        return
    
    # Plot horizontally: d2 bin center on the y-axis, dynamic range (mean +/- SEM) on the x-axis
    y_data = dfS["d2_center"].values
    x_data = dfS["mean"].values
    x_error = dfS["sem"].values

    plt.figure(figsize=(4, 6))

    plt.errorbar(x_data, y_data, xerr=x_error, color=color,fmt='-o', capsize=4, linewidth=2)

    plt.xlabel(ylabel)  # dynamic range axis
    plt.ylabel(xlabel)  # d2 bin center axis
    
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
    plt.show()

method = "minmax"  # or 'percentile' or 'z'
xlab = ("per-file d² percentile (bin center)" if method in ("percentile","minmax")
        else "per-file d² z-score (bin center)")

from scipy.stats import spearmanr, norm
import numpy as np, pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

def spearman_stouffer(T, alternative="increasing", min_unique=3, weight_mode="sqrt(df)"):
    """
    Per-file Spearman correlation between d2_bin and dyn, then Stouffer combine signed z.

    Numerical safeguards:
      - clip p2 away from {0,1} before norm.isf
      - skip nan rho/p
      - skip degenerate groups (too few unique bins / constant arrays)
      - ensure combined z uses only finite z and positive weights

    alternative:
      - "increasing": one-sided p = P(Z >= z_comb) = sf(z_comb)
      - "decreasing": one-sided p = P(Z <= z_comb) = cdf(z_comb)
    """
    rows = []

    # smallest positive float64 (subnormal); avoids p==0 issues
    pmin = np.nextafter(0.0, 1.0)
    pmax = 1.0 - pmin

    for f, g in T.groupby("__file__"):
        n_unique = g["d2_bin"].nunique(dropna=True)
        if n_unique < min_unique:
            continue

        x = np.asarray(g["d2_bin"], float)
        y = np.asarray(g["dyn"], float)

        # drop NaNs pairwise
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        if x.size < 3:
            continue
        # if either is constant, spearmanr can return nan
        if np.all(x == x[0]) or np.all(y == y[0]):
            continue

        rho, p2 = spearmanr(x, y)  # two-sided p
        if (rho is None) or (p2 is None) or (not np.isfinite(rho)) or (not np.isfinite(p2)):
            continue

        # Clip p2 to avoid p2/2 == 0 -> isf(0)=inf
        p2 = float(np.clip(p2, pmin, pmax))

        # Convert two-sided p to |z| then apply sign of rho
        # p(|Z| >= |z|) = p2  =>  p(Z >= |z|) = p2/2  =>  |z| = isf(p2/2)
        abs_z = norm.isf(0.5 * p2)  # stable because p2 clipped
        z = np.copysign(abs_z, rho)

        # weight choice
        if weight_mode == "sqrt(df)":
            # df ~ n_unique - 2
            w = np.sqrt(max(n_unique - 2, 1))
        elif weight_mode == "sqrt(n)":
            w = np.sqrt(x.size)
        else:
            w = 1.0

        if np.isfinite(z) and np.isfinite(w) and w > 0:
            rows.append({"__file__": f, "rho": rho, "p2": p2, "z": z, "w": w, "n_unique": n_unique, "n": x.size})

    R = pd.DataFrame(rows)
    if R.empty:
        return None

    # Combine only finite entries (extra safety)
    mask = np.isfinite(R["z"].to_numpy()) & np.isfinite(R["w"].to_numpy()) & (R["w"].to_numpy() > 0)
    R = R.loc[mask].copy()
    if R.empty:
        return None

    w = R["w"].to_numpy()
    z = R["z"].to_numpy()

    denom = np.sqrt(np.sum(w**2))
    if denom == 0 or not np.isfinite(denom):
        return None

    z_comb = float(np.sum(w * z) / denom)

    # Stable tail probabilities (avoid 1 - cdf)
    alt = alternative.lower()
    if alt in ("increasing", "greater", "pos", "+"):
        p1 = float(norm.sf(z_comb))   # P(Z >= z_comb)
    elif alt in ("decreasing", "less", "neg", "-"):
        p1 = float(norm.cdf(z_comb))  # P(Z <= z_comb)
    else:
        raise ValueError(f"Unknown alternative={alternative!r}. Use 'increasing' or 'decreasing'.")
        # Summaries of per-file Spearman r (rho)
    rho = R["rho"].to_numpy(float)
    w   = R["w"].to_numpy(float)

    rho_clip = np.clip(rho, -0.999999, 0.999999)
    z_rho = np.arctanh(rho_clip)                 # Fisher transform
    z_bar = np.sum(w * z_rho) / np.sum(w)        # weighted mean in z-space
    rho_wmean_fisher = float(np.tanh(z_bar))     # back-transform
    rho_median = float(np.median(rho))


    # Optional: also return a reporting-safe string floor (doesn't change p1)
    return {
        "z_comb": z_comb,
        "p_one_sided": p1,
        "n_files": len(R),
        "rho_wmean_fisher": rho_wmean_fisher,
        "rho_median": rho_median,
        "per_file": R
    }


# build a tidy table: one row per (__file__, d2_bin) with dyn and bin center
def perfile_dyn_by_bin_no_baseline(dstate, method="minmax"):
    dstate = add_perfile_d2_normalized(dstate, method=method).copy()
    # bins in normalized d2 space
    if method in ("percentile","minmax"):
        bins = np.linspace(0, 1, N_BINS+1)
    else:
        zmin, zmax = dstate["d2n"].quantile([0.01, 0.99])
        bins = np.linspace(zmin, zmax, N_BINS+1)
    centers = (bins[:-1] + bins[1:]) / 2.0

    dstate = dstate[np.isfinite(dstate["d2n"])].copy()
    dstate["d2_bin"] = pd.cut(dstate["d2n"], bins=bins, labels=False, include_lowest=True)

    # per-(file,bin) mean-normalized MUA
    fb_mean = dstate.groupby(["__file__", "d2_bin"])["MUA"].transform("mean")
    dstate["MUA_norm"] = dstate["MUA"] / fb_mean.clip(lower=1e-12)

    rows = []
    for (f, b), g in dstate.groupby(["__file__", "d2_bin"], sort=True):
        if pd.isna(b): 
            continue
        x = g["MUA_norm"].to_numpy(float)
        x = x[np.isfinite(x) & (x > 0)]
        if x.size < MIN_PER_BIN:
            continue
        p05, p95 = np.percentile(x, [5, 95])
        rows.append({
            "__file__": f,
            "d2_bin": int(b),
            "d2_center": float(centers[int(b)]),
            "dyn": float(10 * np.log10(p95 / p05)),
            "n_in_bin": int(x.size),
        })
    return pd.DataFrame(rows)


