# Method notes

Details of the revision analyses beyond each script's header: formulas, selection rules, statistical models and
per-panel steps, grouped by topic. Each entry links to its code and names the panels it serves; the full list of
scripts is in [README.md](README.md).

d2 is d2 (2024 Euclidean) of Sooter et al. 2024 ([https://doi.org/10.6084/m9.figshare.25927081](https://doi.org/10.6084/m9.figshare.25927081)), with 40 ms bins, AR order 20
and 30 s windows, unless stated as d2 (2025 KL) (Sooter et al. 2025, [https://github.com/sam-sooter/prox_crit_toolkit](https://github.com/sam-sooter/prox_crit_toolkit)). `ED<n>` / "Extended Data n" refer to
the figures of the first submission ([`../../Previous_Rv1`](../../Previous_Rv1)).

## Contents

- [d2: what it is and how it is computed](#d2-what-it-is-and-how-it-is-computed): [r38_hyperplane_explainer.py](#r38_hyperplane_explainerpy), [d2_yw_matlab_match.py](#d2_yw_matlab_matchpy), [r35_kl_rippleablate.py](#r35_kl_rippleablatepy)
- [Non-overlapping windows](#non-overlapping-windows): [ed_nonoverlap.py](#ed_nonoverlappy), [ed_nonoverlap_d2err.py](#ed_nonoverlap_d2errpy), [ed_nonoverlap_ed2a_ed1f.py](#ed_nonoverlap_ed2a_ed1fpy), [ed_nonoverlap_rats_tiles.py](#ed_nonoverlap_rats_tilespy), [fig1de_nonoverlap.py](#fig1de_nonoverlappy)
- [Figure 1: window measures and the state space](#figure-1-window-measures-and-the-state-space): [fig1_metrics.py](#fig1_metricspy), [fig1_dropin_metrics.py](#fig1_dropin_metricspy), [fig1_nrem_feature_gradient.py](#fig1_nrem_feature_gradientpy), [fig1_state_feature_gradient.py](#fig1_state_feature_gradientpy), [fig1e_scatter_d2raw.py](#fig1e_scatter_d2rawpy-meta_r), [fig1e_dynrange_cell1.py](#figure_scriptsfig1e_dynrange_cell1py)
- [Removing SWRs and barrages from the window](#removing-swrs-and-barrages-from-the-window): [r35_background_vs_ripple.py](#r35_background_vs_ripplepy), [r35_sham_excision.py](#r35_sham_excisionpy), [r16_event_excised_windows.py](#r16_event_excised_windowspy), [opto_cck_excised.py](#opto_cck_excisedpy)
- [Replay and SWRs (Figure 2)](#replay-and-swrs-figure-2): [r35_binomial_replay.py](#r35_binomial_replaypy), [r35_binomial_fig2fvars.py](#r35_binomial_fig2fvarspy), [r35_swr_property_control.py](#r35_swr_property_controlpy), [r35_extend_replay_sessions.py](#r35_extend_replay_sessionspy), [r35_make_ppp4day11_pkl.py](#r35_make_ppp4day11_pklpy), [r35_glm_ablation.py](#r35_glm_ablationpy), [r35_replay_ablation_lrt.py](#r35_replay_ablation_lrtpy), [r35_replay_lmm.py](#r35_replay_lmmpy), [fig2f_backtrace_model.py](#fig2f_backtrace_modelpy), [fig2g_psth_excised.py](#fig2g_psth_excisedpy)
- [Barrages and CCK silencing (Figure 3, Supplementary Figure 6)](#barrages-and-cck-silencing-figure-3-supplementary-figure-6): [fig3_background_repro.py](#fig3_background_repropy), [r16_event_dissociation.py](#r16_event_dissociationpy), [r16_block_level_dissociation.py](#r16_block_level_dissociationpy), [r16_predictor_heterogeneity.py](#r16_predictor_heterogeneitypy), [r16_robustness_index.py](#r16_robustness_indexpy), [FigS6_AC_regime_indicator.ipynb](#figure_scriptsfigs6_ac_regime_indicatoripynb)
- [Drift and stationarity](#drift-and-stationarity): [r25_drift_control_rates.py](#r25_drift_control_ratespy), [r25_stationarity_battery.py](#r25_stationarity_batterypy)
- [Avalanches](#avalanches): [avalanche_ripple.py](#avalanche_ripplepy)

## d2: what it is and how it is computed

What d2 measures on real data, the Python reproduction of the MATLAB fit, and the 2025 KL version.

### r38_hyperplane_explainer.py

Code: [`r38_hyperplane_explainer.py`](notebooks/exploration/revision/r38_hyperplane_explainer.py) · Used for: Supplementary Fig 1a-d

The panels of the explainer:

- **a. The data.** One 30 s NREM window (the one nearest the centre of the cloud in panel d), as 40 ms population
  counts.
- **b. What is fitted.** The history kernel phi_1 ... phi_20 (lags 40-800 ms), with sum(phi) and d2 annotated.
- **c. What beta = 2 means** (Sooter et al. 2025, Eq. 3-4: g(w) = |1 - phi~(w)|^2 / 2 pi s^2 ~ w^beta). The power
  spectrum of the fitted model (= 1/g), with the window's own Welch spectrum as points, and the nearest model on the
  hyperplane (phi + (1 - sum phi) / p, dashed), whose spectrum keeps rising as 1/f^2 at low frequency; off the plane
  the spectrum flattens (beta = 0).
- **d. The geometry, every window of the session.** The vertical axis is the signed distance to the hyperplane,
  (sum(phi) - 1) / sqrt(p), so the plane is the horizontal sheet at 0 and d2 is each point's depth below it. The
  horizontal axes are the two principal directions of the kernels within the plane (the components that do not change
  d2). Exact for p = 20.
- **e. The ruler.** sum(phi) of every window by state, with d2 = |1 - sum(phi)| / sqrt(20) on a second axis.

### d2_yw_matlab_match.py

Code: [`d2_yw_matlab_match.py`](notebooks/exploration/revision/d2_yw_matlab_match.py) · Used for: shared code (background d2 of Fig 3e, g)

How the MATLAB Yule-Walker computation is reproduced in Python (`d2_yw_segments`):

- `getFR` bins spike counts with `histc` on `t0:deltaT:t1` per sub-window and drops the trailing (edge) bin.
- Every sub-window is z-scored with the mean and SD pooled over all sub-windows of the window.
- The autocorrelation at each lag sums products only within a sub-window, so no product crosses a removed gap, and
  is normalised by the total pair count over all sub-windows.
- The Yule-Walker solve builds the Toeplitz matrix from the autocorrelation at lags 0 ... p-1 and solves for phi; d2
  is then computed from phi as in Sooter et al. 2024.
- All CA1 cells are pooled, with no firing-rate threshold.

`d2_ols_glued` is the alternative that joins the remaining sub-windows end to end and fits OLS across the joined
series, so its lag regressions cross the removed gaps. The fourth output stored by the MATLAB ripple-removal script
(`d_beta_err`) is `getDbetas`' explained variance, not a standard error.

### r35_kl_rippleablate.py

Code: [`r35_kl_rippleablate.py`](notebooks/exploration/revision/r35_kl_rippleablate.py) · Used for: shared code

Full-window and ripple-excised d2 exist for both definitions: d2 (2024 Euclidean) from the MATLAB workers (full window
and `_rippleablate`), d2 (2025 KL) on the full window in `kl_error_bars.csv`, and d2 (2025 KL) on ripple-excised
windows from this script, so that full vs excised can be compared within one definition.

## Non-overlapping windows

How the first-submission panels are redrawn on windows that share no data.

### ed_nonoverlap.py

Code: [`ed_nonoverlap.py`](notebooks/exploration/revision/ed_nonoverlap.py) · Used for: Fig 1e, 2a; Supplementary Figs 2-4, 7, 8

How the first-submission tables are read when they are thinned to non-overlapping windows:

- Extended Data 3e drops the "PPP7_day7" rows of `d2_barrage_ablated_windows.csv`, which repeat PPP7_day12's rows
  under another label.
- `rat_isi_entropy_by_swr_rate.csv` has no time column but is row-aligned with `rat_d2_windows_by_state.csv` (their
  `ripple_rate` columns are identical row by row, which is asserted), so its rows take that table's window centre and
  session before thinning.
- Extended Data 10 (`branching_ratio_windows.csv`) is thinned at its own window length (20 s).
- Saved as submitted (not window statistics, or no window times exist): Extended Data 1d, 2d-f, 5a, 6b, 6e.
- Drawn by other scripts: the main barrage panels (3e-g) and Fig 1d, e with Extended Data 1g (`fig1de_nonoverlap.py`).

### ed_nonoverlap_d2err.py

Code: [`ed_nonoverlap_d2err.py`](notebooks/exploration/revision/ed_nonoverlap_d2err.py) · Used for: Supplementary Figs 2b, 3d

- Both tables are built by the producer functions of the figure notebook itself
  (`load_d2_and_err_30s_60s_from_session` for Extended Data 1c; `load_pool_rate_binned` with `_force_sextiles` for
  2c). `_force_sextiles` is not a plain `qcut`: when `qcut(..., duplicates='drop')` returns fewer than 6 bins, it uses
  `qcut` on `rank(method='first')`.
- Each is run twice: with the inputs as they are (the result must equal the shipped CSV in row count, strings and
  floats within 1e-9, or the script stops before writing), and with the non-overlapping rule applied when the data are
  loaded, so that the producer's own state filter, z-score and tiling run on the kept windows.
- The rule: within one session and one window length, windows of every state in time order, a window kept only if
  its centre is at least L seconds after the last kept one (L = 30 for 30 s windows, 60 for 60 s windows), the same
  as `fig1de_nonoverlap.select_disjoint` (asserted). Extended Data 2c's tables include wake windows, so these take part
  in the thinning before the NREM filter.

### ed_nonoverlap_ed2a_ed1f.py

Code: [`ed_nonoverlap_ed2a_ed1f.py`](notebooks/exploration/revision/ed_nonoverlap_ed2a_ed1f.py) · Used for: Supplementary Figs 2e, 3b

**Extended Data 2a (novel vs familiar).** Built from `notebooks/exploration/MUA_sleepstage.ipynb` cell 8
(`load_d2_timeline_complete`: every d2 window of the session, all states, from the newest
`*_CA1_D2_time_resolved_error_bar.mat`) and cell 15 (`annotate_epochs`: NREM rows of the three sleep epochs, the
first 1200 rows per epoch, `m_pre` = mean presleep d2, `d2_rel` = (d2 - m_pre) / (d2 + m_pre)), then
`FigS2_A_novel_familar` cell 1 (stacks the mid and post rows).

For the non-overlapping table, the session's whole d2 timeline (every state) is thinned in time order at L = 30 s
before cell 15 runs. The 1200-row cap is applied as a time span (the windows the producer selects on all windows), so
the thinned table is a subset of the shipped rows, and `m_pre` is recomputed from the kept presleep windows.
PPP7_day14 and PPP8_day14 are left out of the non-overlapping table (their stored tables carry day 12's d2 timeline);
they still enter the all-window equality check. PPP7_day14's epoch name "midsleep" is not read by the loader, so it
has post rows only.

**Extended Data 1f (branching ratio by state).** Built from
`notebooks/manuscript_code/FigS1_F_branchingratio_histogram.ipynb` cell 0, which reads the per-session tables
`branching_ratio_allstates_win10s_step2s_guard5s_*.pkl` (10 s windows on a 2 s grid; NREM, REM, THETA, nonTHETA),
weighted by the table's `step_size` (2 s for every window). The producer drops THETA / nonTHETA windows whose centre
lies in a sleep epoch; the all-wake version keeps every wake window. Each session's windows of all four states are
thinned in time order (L = 10 s) before the producer runs. The weights are constant, and the notebook uses them only
as normalised `gaussian_kde` weights, so they do not change the panel.

### ed_nonoverlap_rats_tiles.py

Code: [`ed_nonoverlap_rats_tiles.py`](notebooks/exploration/revision/ed_nonoverlap_rats_tiles.py) · Used for: Supplementary Fig 8c, f

Each table is rebuilt with the producers' own functions:

- **Extended Data 6f.** `Previous_Rv1/figure_scripts/FigS5_S6_rats.ipynb` cell 11: `_force_sextiles`,
  `load_pool_rate_binned` (NREM, rate `barrage_rate`, 6 tiles, at least 2 tiles), from
  `data/rats/ripple_barr/isi/*.pkl` (field `per_center`).
- **Extended Data 6g.** The same notebook, cell 13: `load_pool_rate_binned_isi` (at least 3 tiles). 6f and 6g z-score
  the value per file (ddof 0) over the file's NREM rows, and set tile = `qcut(rate, 6, duplicates='drop') + 1` when that
  gives 6 bins, else `qcut` of `rank(method='first')`. The loaders read a folder of pickles, so the thinned pickles are
  written to a temporary folder and the loader is run on it.
- **Extended Data 6c.** `MUA_sleepstage_rat_replay.ipynb` cells 7 and 8 (`run_replay_d2_for_session`: replay SWRs in
  forced sextiles of replay-SWR rate, d2 mean and SEM; non-replay SWRs from 500 bootstraps, seed 0, of non-replay SWRs
  matched in number to the replay SWRs, d2 mean and the 2.5 / 97.5th percentiles of the per-tile means), then
  `FigS5_S6_rats.ipynb` cell 28 `load_all_tile_pkls` (non-replay SE = (hi - lo) / 3.92). Windows: the stored
  `data/rats/replay_nonreplay/*_CA1.pkl` (NREM windows of sleep epochs without a manipulation). SWRs: the producer's
  cell 5, read from the session; replay or not by overlap with `data/replay/<session>_sig_replay.csv`.
- Bin sizes are those of the first submission (40, 60 or 80 ms by session, in `ED6C_SESSIONS` / `ED6FG_MAT`); only the
  window set changes.

### fig1de_nonoverlap.py

Code: [`fig1de_nonoverlap.py`](notebooks/exploration/revision/fig1de_nonoverlap.py) · Used for: Supplementary Fig 2d, f

The panels use the non-overlapping rule (`select_disjoint`: a window is kept only if its centre is at least one window
length after the last kept window of any state in the session). A second selection, kept in the output for comparison
only: every 15th window within each run of the same state on the 2 s grid (`fig1_nonoverlap_contrasts.select_nonoverlap`).
Wake states are labelled at the window centre, so theta and non-theta runs are short and each run restarts the count;
windows of neighbouring runs can then lie closer than one window length. NREM and REM runs are long, so the two
selections agree there.

## Figure 1: window measures and the state space

Per-window measures, the branching-ratio lag range and the structure index of d2.

### fig1_metrics.py

Code: [`fig1_metrics.py`](notebooks/exploration/revision/fig1_metrics.py) · Used for: shared window and metric library

**Lag range of the branching ratio (K_MAX_SWEEP, K_MAX_REF).** k_max = 200 matches d2's kernel (200 lags of 4 ms, 796 ms;
d2's p x dt is 800 ms). k_max = 2500 is Wilting & Priesemann's rat value (10 s), the range at which the state-level estimate
passes their H_offset / H_tau tests; a 30 s window (7500 bins) reaches it. `fit_m` / `fit_m_offset` recover the m of
simulated branching processes across this range of k_max.

**Cost.** `fit_m`'s grid scan is O(n_grid x k_max) with n_grid = 1560, so k_max = 2500 costs about ten times
k_max = 200 per window and population; over all sessions the sweep takes several hours. To run it on fewer windows,
lower the per-state cap (`STATE_WINDOW_CAP` in `run_fig1_metrics.py`) rather than `--max-windows`, which removes
windows in proportion and so takes most of THETA and REM.

### fig1_dropin_metrics.py

Code: [`fig1_dropin_metrics.py`](notebooks/exploration/revision/fig1_dropin_metrics.py) · Used for: Figure 1 window tables

**Aperiodic-slope fit range (FIT_RANGES).** The primary range is 20-45 Hz: of the ranges swept on a subset of
sessions, it gave the best split-half reliability and agreement with the direction of the E/I relation of Gao et al.
(2017). The literature default, 30-45 Hz (Gao 2017; Lendner 2020), contains slow gamma in CA1 and is kept as a second
version. No fit includes < 20 Hz (theta harmonics) or > 90 Hz (CA1 ripple band, about 100-250 Hz); 55-65 Hz is always
excluded.

**synchrony_sub81_\*.** Subsampling to a fixed 81 excitatory units (the smallest number across the 18 Figure 1
sessions) makes the column comparable across sessions; the `synchrony_sub_*` columns subsample 0.8 x n_E per session.

### fig1_nrem_feature_gradient.py

Code: [`fig1_nrem_feature_gradient.py`](notebooks/exploration/revision/fig1_nrem_feature_gradient.py) · Used for: shared code (Fig 1c, d)

The structure index (SI) of d2 is computed within NREM only, so it does not reflect the separation between states.
It is tested against a within-bout circular-shift null: d2 is shifted in time within each NREM bout, which keeps each
bout's temporal structure and autocorrelation and breaks only the pairing of d2 with the window features.

**MIN_BOUT and OFFSET.** NREM bouts (runs of window centres 30 s apart) vary in length; most are short, and a large
fraction of NREM windows lie in bouts long enough for several independent shifts. MIN_BOUT and OFFSET are set from
this bout-length distribution and from the window-level d2 autocorrelation (both in the `_bouts` output table). The
"long_bouts" setting (MIN_BOUT = 10, OFFSET = 3) is reported beside the primary one.

### fig1_state_feature_gradient.py

Code: [`fig1_state_feature_gradient.py`](notebooks/exploration/revision/fig1_state_feature_gradient.py) · Used for: shared code (Fig 1d)

Two designs:

- **Per session**, NREM and nonTHETA only, each with its own episode and shift thresholds. NREM episodes are long and
  few per session, which leaves room for circular shifts within a bout.
- **Pooled across sessions**, every state and all states together. The circular-shift null keeps each window within
  its own episode and session, so between-session and between-episode offsets stay part of the null.

nonTHETA, THETA and REM episodes are short and numerous, and THETA and REM have few windows (nonTHETA is also capped
per session by `run_fig1_metrics.STATE_WINDOW_CAP`), hence the separate thresholds. The shortest shift for the
short-episode states is 1 window.

### fig1e_scatter_d2raw.py (meta_r)

Code: [`fig1e_scatter_d2raw.py`](notebooks/exploration/revision/fig1e_scatter_d2raw.py) · Used for: Supplementary Figs 2f, 7c

`meta_r()`, also in [`figure_scripts/fig1e_scatter_d2raw_drawing.py`](../figure_scripts/fig1e_scatter_d2raw_drawing.py):
the per-session Pearson correlations between MUA (z-score) and d2 are pooled with inverse-variance weights (a session
with few windows gives a less reliable correlation). Fisher z_i = arctanh(r_i), variance 1 / (n_i - 3). The
fixed-effect estimate is the precision-weighted mean of z_i; the random-effects estimate (DerSimonian-Laird, the
primary one) reweights by 1 / (v_i + tau^2), so a few large sessions do not dominate when sessions differ. Both use
two-sided z-tests and report r = tanh(pooled z).

### figure_scripts/fig1e_dynrange_cell1.py

Code: [`fig1e_dynrange_cell1.py`](../figure_scripts/fig1e_dynrange_cell1.py) · Used for: Supplementary Fig 2f

`spearman_stouffer()`: per-file Spearman correlation between the d2 bin index and the MUA dynamic range, combined across
files by the signed Stouffer method (each two-sided p converted to a z with the sign of rho; the weighted sum divided
by the square root of the summed squared weights). p is clipped away from 0 and 1 before `norm.isf`, and non-finite or
degenerate per-file fits are skipped.

## Removing SWRs and barrages from the window

d2 and the other measures computed on what is left of a window after the events are cut out, and the sham control.

### r35_background_vs_ripple.py

Code: [`r35_background_vs_ripple.py`](notebooks/exploration/revision/r35_background_vs_ripple.py) · Used for: shared code

**The ripple-excised d2.** Made by
[`time_resolved_d2_across_state_ripple_ablate.m`](notebooks/helper/time_resolved_d2_across_state_ripple_ablate.m):
ripple intervals are subtracted from each 30 s window (`fast_subtract_intervals`), the remaining sub-windows are
joined, and d2 (40 ms bins, AR 20, `getDbetas`) is fitted on the joined series; a window with less than 1 s of
ripple-free time (`MIN_FREE_SEC`) gives NaN. Only the NREM rows differ from the full-window file (REM and wake rows are
identical); every other parameter (30 s windows, 2 s step, 5 s guard) is the same.

**Inputs.** Both d2 versions are read from the `.mat` files. `frac_window_removed` is the fraction of each 30 s window
covered by ripple intervals (not clipped to NREM), from the same `fast_subtract_intervals` call. The d2 (2025 KL) pair
comes from `r35_kl_rippleablate.py` (same Yule-Walker estimator, same windows).

**Comparison.** For each version, the Spearman correlation with each GLM predictor and a cross-validated GLM ablation
(`r35_glm_ablation.build` / `cv_r2`, the same predictors and folds for both), on identical windows.

### r35_sham_excision.py

Code: [`r35_sham_excision.py`](notebooks/exploration/revision/r35_sham_excision.py) · Used for: shared code

Three versions of d2 per window:

- **full**: the window as it is;
- **excised**: the 40 ms bins overlapping a ripple are removed and the rest is joined;
- **sham**: the same number and lengths of intervals as the ripples, placed at random where they overlap neither a
  ripple nor each other, removed and joined the same way, averaged over `N_SHAM` draws per window.

Each version enters the same binomial models: d2 alone; d2 + MUA; d2 + MUA + MUA_std; d2 + MUA + MUA_std + SWR rate
(all z-scored within session). Each model is fitted per session (`fit_binomial`) and summarised by the median d2
coefficient and a Wilcoxon test of the per-session coefficients.

### r16_event_excised_windows.py

Code: [`r16_event_excised_windows.py`](notebooks/exploration/revision/r16_event_excised_windows.py) · Used for: Supplementary Fig 6a-c

For each 30 s window of the barrage sessions, every 40 ms bin overlapping a ripple or a barrage is removed, the
remaining bins are joined (segments shorter than `MIN_FREE_S` = 1 s are dropped), and every predictor (d2, MUA,
MUA_std, Fano, synchrony, E/I) is recomputed on the joined series: the `_bg` (background) columns. The same code is run
with an empty event list for the full-window columns, so the two differ only by the removal. E/I is a ratio of rates
over the kept time, so it does not depend on how much time is removed.

`audit()` reports, per predictor, the within-session correlation between the full and background versions, the
fraction of each window kept, and the agreement of this script's full-window d2 with `d2_real` of
`r25_drift_control_rates.py`. `rate_correlations()` gives the within-session Spearman correlation of each predictor
with each event rate, alone and partialled on the other four predictors.

### opto_cck_excised.py

Code: [`opto_cck_excised.py`](notebooks/exploration/revision/opto_cck_excised.py) · Used for: shared code

The d2 versions: the stored background pickle (`d2_pkl`) and the ripple-excised `.mat` (`d2_excised_mat`) were
computed together and are the matched full-window and excised pair. The plain `.mat` on the share (`d2_plain_mat`) was
computed separately for some sessions and is printed for reference. `params` of both `.mat` files are printed per
session, so a change of unit set is visible.

## Replay and SWRs (Figure 2)

The binomial replay models, the SWR-property controls, the replay tables and the GLM ablations.

### r35_binomial_replay.py

Code: [`r35_binomial_replay.py`](notebooks/exploration/revision/r35_binomial_replay.py) · Used for: shared code (Fig 2d)

**Model.** replay_count_w ~ Binomial(N_SWR_w, p_w), logit p_w = b0 + b1 d2_w (+ covariates), fitted per session
(`fit_binomial`) and pooled with session fixed effects and session-clustered robust SE (`pooled_fit`). d2, MUA and SWR
rate are z-scored within session, so b1 is the change in log-odds per within-session SD and is comparable across
sessions. Overdispersion is reported as Pearson chi2 / residual df, with a quasi-binomial SE (se x
sqrt(max(overdispersion, 1))) beside the GLM SE.

**d2 sources** (`load()`): `euclid`, the d2 (2024 Euclidean, p = 20) column of the submitted Fig 2d tables, every 15th
window (non-overlapping); `kl`, d2 (2025 KL) at a chosen AR order on the revision's own non-overlapping windows, joined
to the Fig 2d grid by nearest centre within 1 s (the 18 Figure 1 sessions only).

**Third panel.** Replay counts pooled into within-session d2 sextiles, each session's per-sextile replay probability
divided by its own mean, beside the per-session coefficients.

### r35_binomial_fig2fvars.py

Code: [`r35_binomial_fig2fvars.py`](notebooks/exploration/revision/r35_binomial_fig2fvars.py) · Used for: shared code

Uses `fit_binomial`, `pooled_fit` and `zs` of `r35_binomial_replay.py`, so the fitting is identical; only the
covariates and the model sequence differ. The covariates (MUA, MUA_std, EI_ratio) are taken from the submitted Fig 2f
table, so the variables and windows are exactly that panel's. The models add one covariate at a time (d2 alone up to
d2 + MUA + MUA_std + EI_ratio). Models with SWR rate are reported separately, because SWR rate already enters the
binomial model as the number of trials. The full-window and ripple-excised d2 are joined from the `.mat` files
(`r35_background_vs_ripple.load_d2_mat`, nearest centre within 1 s). A separate fit of d2 and EI_ratio, each alone
and each with the other, separates their shared variance from their own.

### r35_swr_property_control.py

Code: [`r35_swr_property_control.py`](notebooks/exploration/revision/r35_swr_property_control.py) · Used for: shared code

**Sessions.** The submitted Fig 2e notebook reads every file in `data/ripple_barr/replay_non_replay/` (9 mouse sessions
and 1 rat session); its session set differs from that of Fig 2a, g, h (PPP4_day11 and PPP7_day10 instead of
PPP4_day10 and PPP7_day8, plus the rat). `report_session_set()` prints both sets and their difference.

**Replay events.** For each session, the decoded events of one template and epochs 1, 3 and 5, as in
`generate_replay_non_replay.ipynb` cell 9 (recorded per session in `SESSION_TEMPLATE`). Using every epoch and template
in the `.mat` file gives the same window rates (the extra combinations exist only for PPP4_day11 and duplicate
intervals do not change an overlap) but other events for the per-event measures, so the per-session selection is used
throughout. Epochs are also restricted to those with significant events inside the session's d2 NREM windows; this
changes only PPP23_day7 and PPP23_day8, which contribute the final sleep epoch only.

**Tests.** (a) Cliff's delta (0 = no difference, +1 = replay SWRs always larger). (b) A greedy 1:1 nearest-neighbour
match without replacement, in a fixed order (rarest replay events first, by distance from the non-replay centroid), so
the result does not depend on row order. (c) Spearman partial correlations: both variables rank-regressed on the
covariate ranks, residuals correlated.

### r35_extend_replay_sessions.py

Code: [`r35_extend_replay_sessions.py`](notebooks/exploration/revision/r35_extend_replay_sessions.py) · Used for: Fig 2e windows

Ten sessions have decoded replay. PPP7_day12 has no stored window table: 598 decoded candidates (304 significant,
epochs 1 and 3, template04), 89 CA1 units, 3.09 h of NREM and a d2 `_error_bar` file; its epoch numbering differs
from the other sessions (see `EXCLUDE`), so it is in the window table but not in the replay models. PPP23_day8's stored
table has no MUA, MUA_std or EI_ratio, which this script computes from the spikes.

`FR_std` is cell 4 of the covariate notebook and the third feature of the submitted Fig 2f; `MUA_std` (cells 5-7) is
the column stored in the replay-probability tables; `MUA` is the rolling population rate (cell 7 replaces cell 4's
MUA). Both SD covariates are checked against a second routine (`fr_std_across_neurons`), and the largest difference
is printed per session.

### r35_make_ppp4day11_pkl.py

Code: [`r35_make_ppp4day11_pkl.py`](notebooks/exploration/revision/r35_make_ppp4day11_pkl.py) · Used for: Fig 2 replay table

Writes the PPP4_day11 table into the replay / non-replay folder read by the submitted Fig 2e notebook, in place of
PPP4_day10 (one decoded epoch, `epoch5_template02`).

Recipe, with the schema of `generate_replay_non_replay.ipynb`:

1. d2 timeline: the NREM rows of `data/replay/replay_probality/PPP4_day11.pkl`.
2. Ripples: `load_ripples_events` intersected with NREM.
3. Replay intervals: one template per session (`SESSION_TEMPLATE`), epochs 1, 3, 5 (`NOTEBOOK_EPOCHS`), left- and
   right-significant events (p < 0.05) appended separately, so `replay_rate` counts each direction, as in the rest of
   the replay pipeline.
4. A ripple carries replay if it overlaps a significant replay interval.

Before writing, the rebuilt `ripple_rate_all * 30` and `ripple_rate_in * 30` must equal the `ripple_count` and
`replay_count` of the stored table, window by window. The replaced file is moved to `_superseded/` with a short README.

### r35_glm_ablation.py

Code: [`r35_glm_ablation.py`](notebooks/exploration/revision/r35_glm_ablation.py) · Used for: shared code

GLM ablation: predictors z-scored, one coefficient set to zero per ablation, and the loss in cross-validated R^2 as
that predictor's contribution. This is the design of Chang et al. 2025 (Nature 637:1161, Fig. 2e), with d2 as the
outcome in place of pupil size.

### r35_replay_ablation_lrt.py

Code: [`r35_replay_ablation_lrt.py`](notebooks/exploration/revision/r35_replay_ablation_lrt.py) · Used for: shared code (Fig 2e)

**Panels.** (a) Effect size: the pooled, cluster-robust coefficient per within-session SD of the predictor, with its
95% CI. (b) Contribution to the fit as deviance explained, two bars per predictor: hatched, the predictor alone;
filled, its unique share in the joint model (full deviance explained minus that with the predictor dropped).

MUA_bg and FR_std_bg are correlated, so their filled bars reflect what the two do not share; d2 is close to
orthogonal to both. The panel's note gives the within-session correlations among the three.

**Confidence intervals.** Session-cluster bootstrap (`cluster_bootstrap`): sessions resampled with replacement,
duplicates relabelled so that each draw keeps one fixed effect per session, 2.5 / 97.5th percentiles over the refits.

### r35_replay_lmm.py

Code: [`r35_replay_lmm.py`](notebooks/exploration/revision/r35_replay_lmm.py) · Used for: Fig 2e statistics

The count and fraction outcomes are modelled with a Gaussian linear mixed model on log-transformed values (REML). The
Python route to a Poisson or binomial mixed model without an R fitter (statsmodels' variational-Bayes BayesMixedGLM)
does not separate the fixed-effect posterior from the random-slope spread in this design.

### fig2f_backtrace_model.py

Code: [`fig2f_backtrace_model.py`](notebooks/exploration/revision/fig2f_backtrace_model.py) · Used for: shared code

The steps of `generate_GLM.ipynb`, cell by cell:

| cell | step |
|---|---|
| 2 | d2 timeline: every state in the `.mat`, sorted by `center_s` |
| 4 | FR_std: across-neuron SD (ddof 0) of per-neuron rates in [c - 15, c + 15) |
| 5-7 | MUA: 40 ms population rate (all CA1 units), 30 s centred rolling mean (750 bins, min_periods 187), interpolated to the window centres; MUA_std, the rolling SD (not used by the model). Cell 7 replaces cell 4's MUA. |
| 9 | ripples = `load_ripples_events` & NREM |
| 11 | replay events: `<sess>_replay_sequences_new.mat`; an event is replay if either direction has p < .05 and r < .05 |
| 13 | per window: ripple_count and replay_count by interval overlap |
| 14 | NREM rows; replay_present = replay_count > 0; tile = qcut(ripple_count, 6) |
| 15 | `cv_predictive_gain_logistic(features=(d2, MUA, FR_std), n_bins=6, group_col="session", n_splits=5, C=1, rs=0)`, the function in `fig2f_producer_glm.py` |

**MATCHED.** For each session, the epoch / template combination(s) that reproduce its stored ablation results (fold
AUCs and ablation deltas) exactly, found by the SEARCH mode (every non-empty subset of the session's decoded
combinations, both replay-sequence files, p-only significance).

**Coverage.** `candidate_coverage()` / `coverage()` restrict the windows to the time spans decoded in MATCHED
(outside them, replay_present is 0 by definition), refit the model on all windows and on the decoded spans only, and
recompute the panel's statistics (median importance per feature, signed-rank tests, Holm-corrected pairwise
contrasts) on both.

### fig2g_psth_excised.py

Code: [`fig2g_psth_excised.py`](notebooks/exploration/revision/fig2g_psth_excised.py) · Used for: Fig 2f

**Shuffled times** (`random_times()`). By default the shuffled times are drawn from all of NREM, as in the figure.
`EXCLUDE_RIPPLES=1` draws them at least 1 s away from every ripple.

## Barrages and CCK silencing (Figure 3, Supplementary Figure 6)

The barrage panels, and which background measures predict the type of NREM event.

### fig3_background_repro.py

Code: [`fig3_background_repro.py`](notebooks/exploration/revision/fig3_background_repro.py) · Used for: Fig 3e-h; Supplementary Fig 8d

**Panel e.** d2 z-scored per session (or raw, `z=False`), binned in within-session sextiles of barrage rate (rank
sextiles by default; `BINS=width` gives 6 equal-width bins of each session's barrage-rate range [0, max], the default
for panel h), mean per
session and sextile, then mean +- SEM over sessions, and the Pearson r of the six means (`sextile_trend`). Beside it,
the per-session Spearman correlation of d2 with barrage rate, signed-rank across sessions (`session_rho`).

**Panel f.** Mean of the `barrage_br_changes` / `_shuffle` values per file from the stored branching-ratio tables; a
mixed model (value ~ C(condition) + C(animal), random intercept per animal_day) as in the figure notebook, and a
paired Wilcoxon on the file means. Files are keyed by animal and day (the first two fields of the file name), so
NN2_day10 and NN2_day10_3 form one file mean (n = 11).

**Panel g.** The background-d2 rows of `fig3g_background.py`, one mean per session and epoch (presleep, postsleep),
paired Wilcoxon over the 7 Sncg sessions.

**Panel h.** The figure notebook's ISI-entropy pipeline (`load_pooled_per_center`, `add_session_tiles`,
`add_session_zscores`, `aggregate_from_precomputed_tiles`) on rank sextiles per session (every session fills all six)
and, for the session-level statistic, non-overlapping windows (>= 30 s apart). Its star is the mixed model of
`fig3h_entropy_diagnostics.py` (ISI entropy ~ barrage rate + C(animal), random intercept and slope per session).

### r16_event_dissociation.py

Code: [`r16_event_dissociation.py`](notebooks/exploration/revision/r16_event_dissociation.py) · Used for: shared code

**Figure 1 sessions.** Within-session Spearman correlation of each measure with SWR rate, the same partialled on the
other measures, and a Shapley decomposition of held-out linear R^2 (5-fold block cross-validation) among d2, MUA, MUA
Fano, E/I and synchrony (`shapley_shares` averages each predictor's marginal contribution over all orderings; `cv_r2`
is the block-CV R^2).

**Barrage sessions.** MUA, MUA Fano, MUA_std and population synchrony (SD across 50 ms bins of the fraction of active
units) from the CA1 spikes of each window of `r25_drift_control_rates.py`, related to SWR rate and barrage rate
(within-session Spearman, sign agreement across sessions, partial r of d2 given the other four). `partial_spearman`
regresses both variables' ranks on the covariate ranks and correlates the residuals.

### r16_block_level_dissociation.py

Code: [`r16_block_level_dissociation.py`](notebooks/exploration/revision/r16_block_level_dissociation.py) · Used for: Supplementary Fig 6a

**Blocks.** The Poisson models of SWR count and of barrage count on five predictors (d2, MUA, MUA Fano, synchrony,
E/I) are fitted inside contiguous NREM blocks (at least `MINW` windows, cut at bout boundaries),
and the per-block coefficients are treated as repeated measurements of the session's effect. Tests are at the session
level: a one-sample t-test of the session-mean block coefficients against zero (df = sessions - 1).

**Event-type model** (`classify_blocks`, `classify_sessions`). One binomial GLM per block or session: successes =
barrage count, failures = SWR count, so the outcome is P(event is a barrage) and the coefficients describe how the
predictors shift the mix of the two event types. A block or session enters if it has at least `MIN_EVENTS` of each
type. Windows with no events are left out of the fit (they add nothing to the likelihood); predictors are z-scored
over all windows.

**Predictors.** `PRED_SET=excised` (default) computes every predictor only on the bins that overlap neither a ripple
nor a barrage; `PRED_SET=intact` uses the whole window. `PREDS` selects a subset (for example without synchrony, which
is correlated with MUA and Fano). `ORTHO=1` keeps all five but residualises MUA, E/I, Fano and synchrony on one another
in a fixed order (rate, then balance, then burstiness), so each coefficient is what that measure adds beyond the ones
before it; d2 is unchanged and the joint fit is the same.

**Helpers.** `block_vif`: variance-inflation factor per column (1 / (1 - R^2) of that column on the others).
`pairwise_holm`: the `perform_pairwise_comparisons` of `Fig2_F_GLM.ipynb` (paired Wilcoxon between predictors, paired
by session, Holm-corrected), run on |coefficient| because the coefficient is signed; sign consistency across sessions
is reported separately.

**Outputs.** The session-level event-type figure is Supplementary Fig 6a. The block-level and Poisson figures are
written to `reports/revision_figs/backup/`.

### r16_predictor_heterogeneity.py

Code: [`r16_predictor_heterogeneity.py`](notebooks/exploration/revision/r16_predictor_heterogeneity.py) · Used for: Supplementary Fig 6b, c

- `meta()`: DerSimonian-Laird random-effects meta-analysis over per-block coefficients and their SEs: Cochran's Q,
  I^2, the between-block SD tau, and the fixed- and random-effects pooled estimates with their z-statistics.
- `good_blocks`: the blocks of `r16_block_level_dissociation.py` with at least `MIN_EVENTS` of each event type.
- `within_block`: each predictor regressed on a cubic in time per block, giving the coefficient of variation, the R^2
  of the trend and the lag-1 autocorrelation.
- Cell types: MUA_bg and EI_bg are split into per-cell-type rates (r_E, r_I) via MUA_bg = n_e r_E + n_i r_I and
  EI_bg = log(r_E / r_I) (asserted: E_tot + I_tot = MUA_bg, every unit is E or I), and the block model is refitted with
  r_E and r_I in place of MUA and E/I (together and each alone), relating MUA's coefficient sign to the interneuron
  fraction of the recorded population.

### r16_robustness_index.py

Code: [`r16_robustness_index.py`](notebooks/exploration/revision/r16_robustness_index.py) · Used for: Supplementary Fig 6b, c

Two kinds of index, each with a session-level confidence interval.

**Consistency across blocks** (meta-analytic, Supplementary Fig 6b). `r16_predictor_heterogeneity.meta` on the
per-block coefficients (predictors orthogonalised, `ORTHO`) gives, per predictor, the pooled effect mu and the
between-block SD tau. The indices are |mu| / tau (a standardised consistency, like a Cohen's d across blocks),
Phi(|mu| / tau) (the expected fraction of blocks whose effect has the pooled sign), and the 95% prediction interval for a
new block, mu +- 1.96 sqrt(tau^2 + SE_mu^2) (IntHout et al. 2016).

**Transfer to a new session** (Supplementary Fig 6c). `loso()` fits one binomial GLM per leave-one-session-out split,
with an intercept per block and a slope shared by all blocks of the training sessions. Each held-out block is scored
with that slope and only its own intercept refitted. The transfer ratio is the held-out deviance explained divided by
the deviance explained when the same block is fitted with its own intercept and slope (median over blocks).

**Confidence intervals.** `cluster_boot` resamples sessions with replacement, each with all its blocks, and gives the
95% interval of each index.

### figure_scripts/FigS6_AC_regime_indicator.ipynb

Code: [`FigS6_AC_regime_indicator.ipynb`](../figure_scripts/FigS6_AC_regime_indicator.ipynb) · Used for: Supplementary Fig 6

**Data.** 11 barrage sessions (NN8_day6, with 6 barrages, is not fitted), CA1 units, SWRs and barrages in
non-overlapping 30 s NREM windows. Every 40 ms bin overlapping a ripple or a barrage is removed before the five
background measures are computed (d2, MUA, MUA Fano, E/I ratio, population synchrony), so each describes the state
between events.

**Panel a.** One binomial GLM per session predicts the barrage share of a window's events (barrages / (barrages +
SWRs)) from the five measures together, each z-scored within session. The coefficient is the effect of one
within-session SD on the log-odds of the barrage share, adjusted for the other four. Session coefficients are tested
against 0 with a one-sample signed-rank test.

**Panel b.** NREM is split into 30-minute blocks (45 blocks over the 11 sessions). Per-block versions of panel a's
coefficients (E/I, MUA Fano and synchrony residualised on the measures before them: E/I | MUA, Fano | MUA + E/I,
synchrony | MUA + E/I + Fano) are pooled with a random-effects meta-analysis. The stability index |mu| / tau is the
pooled effect over the between-block SD; Phi(|mu| / tau), in the legend, is the expected fraction of blocks whose
coefficient has the pooled sign. The brackets compare d2 with each other measure by a paired session bootstrap: each
of 2 000 resamples draws one set of 11 sessions and computes both indices on it; p = 2 x the smaller tail of the
paired difference around 0, two-sided, Holm-corrected, floored at 1/2000 (an infinite |mu| / tau, tau = 0, counts as
the larger).

**Panel c.** Leave-one-session-out transfer for four models (d2 alone, MUA alone, d2 + MUA, all five): the slope is
fitted on the other 10 sessions and the intercept refitted on the held-out session; the transfer ratio is the
held-out deviance explained over the in-sample deviance explained. The line at 1 marks equal held-out and in-sample
fit.

**Boxes of b and c.** Session-level cluster bootstrap: 2 000 resamples of the 11 sessions (blocks stay with their
session); box = IQR, whiskers = 2.5-97.5th percentiles, diamond = full-data estimate.

## Drift and stationarity

d2 within windows with trends removed or added, and over window lengths.

### r25_drift_control_rates.py

Code: [`r25_drift_control_rates.py`](notebooks/exploration/revision/r25_drift_control_rates.py) · Used for: Supplementary Fig 6 windows

Four versions of each relationship (d2 vs SWR rate, d2 vs barrage rate):

- **real**: d2 as in the figures;
- **detrended**: the same window with its linear trend removed before the AR fit (`detrend()`);
- **partial**: Spearman r of d2 and rate after both are residualised on rank(|trend|) (`_partial_spearman`);
- **low-drift**: only windows whose end-to-end trend is below 0.5 SD.

A fifth check is the correlation between trend size and event rate, r(|trend|, rate). Per-session Spearman
correlations are pooled by a Fisher-z mean and SE (`fisher()`), with the number of sessions of each sign and a
Wilcoxon signed-rank test. Panel d (`window_length_panel`) draws the SWR-rate relationship at other window lengths.

### r25_stationarity_battery.py

Code: [`r25_stationarity_battery.py`](notebooks/exploration/revision/r25_stationarity_battery.py) · Used for: shared code

**Versions of each window.** Per session and state, d2 on non-overlapping windows of six lengths (10-120 s), for the
real windows and five variants:

- **detrended**: linear trend removed before z-scoring;
- **stationary**: an AR(20) fitted on the longest window of that state, simulated forward with new Gaussian noise
  (`simulate_ar`) and cut into every shorter length, giving the finite-sample baseline of a stationary process;
- **ramp0.5 / ramp1 / ramp2**: a linear ramp of 0.5, 1 or 2 SD added end to end to the real z-scored window;
- **rw**: an integrated random walk scaled to 1 SD added to the window.

**Tests per window.** ADF (unit-root null) and KPSS (stationarity null) at lag / order 20. `drift_metrics` gives, for
each real window, the linear trend of the binned counts in SD units (end to end) and the difference between the two
half-window means; these are the per-window covariate (panel C and the low-drift tercile) and show the range covered
by the ramp and random-walk variants.

**Statistics.** Per-session slope of d2 against log2(window length), real minus stationary, signed-rank across
sessions; within-session Spearman of d2 with |trend|; the paired state contrasts in the lowest-drift tercile of windows
per session and state. lambda = 1 - mean(sd^2) / var(d2), the share of across-window d2 variance not due to
estimation noise, per state and window length.

## Avalanches

Avalanche detection and power-law fits by SWR rate.

### avalanche_ripple.py

Code: [`avalanche_ripple.py`](notebooks/exploration/revision/avalanche_ripple.py) · Used for: shared code (Supplementary Fig 3e-g)

**Avalanche detection.** Bins at the population mean inter-event interval of each session (a few ms; the submitted
analysis used a fixed 50 ms, tens of times larger); xmin chosen by the `powerlaw` package's KS procedure and reported;
the activity threshold at the 20th, 30th and 40th percentiles.

**NREM only.** SWRs are about ten times sparser in wake than in NREM, so a smoothing window that resolves the NREM
rate leaves most wake windows at zero or one SWR, and wake SWR-rate terciles collapse to ties at zero. Restricting
the rate analysis (tercile fit and mixture test) to NREM also keeps the terciles from following the sleep state.
