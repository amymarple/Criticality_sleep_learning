# Methods

Methods of the revision analyses, by topic; each entry links to its code. The script list is in [README.md](README.md).
d2 is d2 (2024 Euclidean) of Sooter et al. 2024 ([https://doi.org/10.6084/m9.figshare.25927081](https://doi.org/10.6084/m9.figshare.25927081)), 40 ms bins, AR order 20, 30 s windows, unless stated
as d2 (2025 KL) (Sooter et al. 2025, [https://github.com/sam-sooter/prox_crit_toolkit](https://github.com/sam-sooter/prox_crit_toolkit)).

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

### r38_hyperplane_explainer.py

[`r38_hyperplane_explainer.py`](notebooks/exploration/revision/r38_hyperplane_explainer.py) · Supplementary Fig 1a-d

- **a.** One 30 s NREM window as 40 ms population counts.
- **b.** The fitted history kernel phi_1 ... phi_20 (lags 40-800 ms), with sum(phi) and d2.
- **c.** The power spectrum of the fitted model, with the window's Welch spectrum, and the nearest model on the beta = 2
  hyperplane (phi + (1 - sum phi) / p), whose spectrum rises as 1/f^2 at low frequency (Sooter et al. 2025, Eq. 3-4).
- **d.** Every window of the session: signed distance to the hyperplane, (sum(phi) - 1) / sqrt(p), against the two
  principal directions of the kernels within the plane.
- **e.** sum(phi) of every window by state, with d2 = |1 - sum(phi)| / sqrt(20) on a second axis.

### d2_yw_matlab_match.py

[`d2_yw_matlab_match.py`](notebooks/exploration/revision/d2_yw_matlab_match.py) · background d2 (Fig 3e, g)

`d2_yw_segments` computes d2 of a window with segments removed: spike counts binned per remaining sub-window, every
sub-window z-scored with the mean and SD pooled over the window, autocorrelation summed within sub-windows only and
normalised by the total number of pairs, Yule-Walker solve on lags 0 ... p-1, and d2 from phi as in Sooter et al. 2024.
All CA1 cells are pooled.

### r35_kl_rippleablate.py

[`r35_kl_rippleablate.py`](notebooks/exploration/revision/r35_kl_rippleablate.py) · shared code

d2 (2025 KL) on ripple-excised windows and on the matching full windows, Yule-Walker fit, so that full and excised
windows are compared within each d2 definition.

## Non-overlapping windows

### ed_nonoverlap.py

[`ed_nonoverlap.py`](notebooks/exploration/revision/ed_nonoverlap.py) · Fig 1e, 2a; Supplementary Figs 2-4, 7, 8

Runs the figure notebooks of the first submission with every window table thinned to non-overlapping windows as it is
read: per session, all states in time order, a window is kept only if its centre is at least one window length after
the last kept one. Tables without a time column take the window centres of the table they are row-aligned with.

### ed_nonoverlap_d2err.py

[`ed_nonoverlap_d2err.py`](notebooks/exploration/revision/ed_nonoverlap_d2err.py) · Supplementary Figs 2b, 3d

Builds the two tables with the producer functions of the first-submission notebook
(`load_d2_and_err_30s_60s_from_session` for Extended Data 1c; `load_pool_rate_binned` for 2c), with the windows
thinned when they are loaded (L = 30 s for 30 s windows, 60 s for 60 s windows), so the notebook's state filter,
z-score and tiling run on the kept windows.

### ed_nonoverlap_ed2a_ed1f.py

[`ed_nonoverlap_ed2a_ed1f.py`](notebooks/exploration/revision/ed_nonoverlap_ed2a_ed1f.py) · Supplementary Figs 2e, 3b

**Extended Data 2a (novel vs familiar).** From `MUA_sleepstage.ipynb` cell 8 (every d2 window of the session) and cell
15 (NREM windows of the three sleep epochs, the first 1200 per epoch; d2_rel = (d2 - m_pre) / (d2 + m_pre), m_pre =
mean presleep d2), then `FigS2_A_novel_familar` cell 1. Each session's windows are thinned at L = 30 s before cell 15.

**Extended Data 1f (branching ratio by state).** From `FigS1_F_branchingratio_histogram.ipynb` cell 0, on 10 s windows
(2 s step; NREM, REM, THETA, nonTHETA) thinned at L = 10 s; the all-wake version keeps every wake window.

### ed_nonoverlap_rats_tiles.py

[`ed_nonoverlap_rats_tiles.py`](notebooks/exploration/revision/ed_nonoverlap_rats_tiles.py) · Supplementary Fig 8c, f

Builds Extended Data 6c, 6f and 6g with the producers' functions on non-overlapping windows: 6f and 6g from
`FigS5_S6_rats.ipynb` cells 11 and 13 (per-file z-score over NREM windows, six tiles of barrage rate); 6c from
`MUA_sleepstage_rat_replay.ipynb` cells 7 and 8 (replay SWRs in sextiles of replay-SWR rate; non-replay SWRs from 500
bootstraps matched in number, seed 0) and `FigS5_S6_rats.ipynb` cell 28. Each session keeps its bin size (40, 60 or
80 ms).

### fig1de_nonoverlap.py

[`fig1de_nonoverlap.py`](notebooks/exploration/revision/fig1de_nonoverlap.py) · Supplementary Fig 2d, f

Runs the producer cells of the Figure 1d, e notebooks on non-overlapping windows (`select_disjoint`) and computes the
state contrasts and the d2-MUA pairs.

## Figure 1: window measures and the state space

### fig1_metrics.py

[`fig1_metrics.py`](notebooks/exploration/revision/fig1_metrics.py) · window and metric library

Windows of 30 s on a 2 s grid inside guarded state intervals (5 s guard); units above `MIN_FIRING_RATE`. Branching
ratio m by multistep regression on 4 ms bins: k_max = 200 (796 ms, matching d2's 800 ms kernel) and k_max = 2500
(10 s, as in Wilting & Priesemann 2018); tau = -bin / ln(m).

### fig1_dropin_metrics.py

[`fig1_dropin_metrics.py`](notebooks/exploration/revision/fig1_dropin_metrics.py) · Figure 1 window tables

**Aperiodic slope** fitted over 20-45 Hz, excluding < 20 Hz, > 90 Hz and 55-65 Hz; 30-45 Hz (Gao et al. 2017) as a
second version. **Synchrony** subsampled to 81 excitatory units in every session (`synchrony_sub81_*`).

### fig1_nrem_feature_gradient.py

[`fig1_nrem_feature_gradient.py`](notebooks/exploration/revision/fig1_nrem_feature_gradient.py) · Fig 1c, d

Structure index of d2 within NREM in the window-feature space, tested against a circular shift of d2 within each NREM
bout (bouts of at least `MIN_BOUT` windows, shift of at least `OFFSET` windows); a second setting uses MIN_BOUT = 10,
OFFSET = 3.

### fig1_state_feature_gradient.py

[`fig1_state_feature_gradient.py`](notebooks/exploration/revision/fig1_state_feature_gradient.py) · Fig 1d

Structure index of d2 per session (NREM and nonTHETA) and pooled across sessions (every state and all states together),
against circular shifts that keep each window within its episode and session.

### fig1e_scatter_d2raw.py (meta_r)

[`fig1e_scatter_d2raw.py`](notebooks/exploration/revision/fig1e_scatter_d2raw.py) · Supplementary Figs 2f, 7c

`meta_r()`: per-session Pearson correlations between MUA (z-score) and d2, pooled through Fisher z (variance
1 / (n - 3)): fixed-effect and DerSimonian-Laird random-effects estimates (the primary), two-sided z-tests, r = tanh(z).

### figure_scripts/fig1e_dynrange_cell1.py

[`fig1e_dynrange_cell1.py`](../figure_scripts/fig1e_dynrange_cell1.py) · Supplementary Fig 2f

`spearman_stouffer()`: per-session Spearman correlation between the d2 bin and the MUA dynamic range, combined across
sessions by the signed Stouffer method.

## Removing SWRs and barrages from the window

### r35_background_vs_ripple.py

[`r35_background_vs_ripple.py`](notebooks/exploration/revision/r35_background_vs_ripple.py) · shared code

The ripple-excised d2 of
[`time_resolved_d2_across_state_ripple_ablate.m`](notebooks/helper/time_resolved_d2_across_state_ripple_ablate.m):
ripple intervals are removed from each 30 s NREM window, the remaining sub-windows are joined and d2 (40 ms, AR 20) is
fitted on the joined series; windows with less than 1 s left are excluded. For the full-window and the ripple-excised
d2 (both definitions), the Spearman correlation with each GLM predictor and a cross-validated GLM ablation
(`r35_glm_ablation.py`), on identical windows.

### r35_sham_excision.py

[`r35_sham_excision.py`](notebooks/exploration/revision/r35_sham_excision.py) · shared code

Three versions of d2 per window: full; ripple-excised (40 ms bins overlapping a ripple removed, the rest joined); sham
(the same number and lengths of intervals removed at random ripple-free positions, averaged over `N_SHAM` draws). Each
enters the binomial replay models (d2; + MUA; + MUA_std; + SWR rate), fitted per session, summarised by the median d2
coefficient and a Wilcoxon test across sessions.

### r16_event_excised_windows.py

[`r16_event_excised_windows.py`](notebooks/exploration/revision/r16_event_excised_windows.py) · Supplementary Fig 6a-c

For each 30 s window of the barrage sessions, every 40 ms bin overlapping a ripple or a barrage is removed, the rest is
joined (segments shorter than 1 s dropped), and d2, MUA, MUA_std, Fano factor, synchrony and E/I (a ratio of rates over
the kept time) are computed on the joined series (`_bg` columns), and on the whole window (full-window columns).

### opto_cck_excised.py

[`opto_cck_excised.py`](notebooks/exploration/revision/opto_cck_excised.py) · shared code

The CCK-silencing contrast (baseline vs stimulation blocks) with ripple time removed from every window: MUA, ISI
entropy and branching ratio recomputed on the ripple-free segments; d2 from the ripple-excised `.mat` files.

## Replay and SWRs (Figure 2)

### r35_binomial_replay.py

[`r35_binomial_replay.py`](notebooks/exploration/revision/r35_binomial_replay.py) · Fig 2d

replay_count ~ Binomial(N_SWR, p), logit p = b0 + b1 d2 (+ covariates), per window. Fitted per session and pooled with
session fixed effects and session-clustered robust SE. d2, MUA and SWR rate are z-scored within session, so b1 is the
change in log-odds per within-session SD. Overdispersion: Pearson chi2 / residual df, with a quasi-binomial SE. Windows:
every 15th window of the 2 s grid (non-overlapping), NREM, at least one SWR.

### r35_binomial_fig2fvars.py

[`r35_binomial_fig2fvars.py`](notebooks/exploration/revision/r35_binomial_fig2fvars.py) · shared code

The binomial replay models with the covariates of the submitted Fig 2f (MUA, MUA_std, EI_ratio), added one at a time,
for the full-window and the ripple-excised d2; models with SWR rate are reported separately (SWR count is the number
of trials).

### r35_swr_property_control.py

[`r35_swr_property_control.py`](notebooks/exploration/revision/r35_swr_property_control.py) · shared code

Replay events: for each session, the decoded events of one template and epochs 1, 3 and 5 (`SESSION_TEMPLATE`). Tests:
(a) Cliff's delta between replay and non-replay SWRs; (b) 1:1 nearest-neighbour matching on SWR properties without
replacement; (c) Spearman partial correlations controlling for SWR duration, amplitude, participating cells and
trajectory length.

### r35_extend_replay_sessions.py

[`r35_extend_replay_sessions.py`](notebooks/exploration/revision/r35_extend_replay_sessions.py) · Fig 2e windows

Replay / SWR counts, MUA, MUA_std, FR_std (across-neuron SD of firing rates) and EI_ratio per non-overlapping 30 s
NREM window, for the ten sessions with decoded replay; MUA_std and FR_std also on the ripple-excised window.
PPP7_day12 is excluded from the replay models (`EXCLUDE`).

### r35_make_ppp4day11_pkl.py

[`r35_make_ppp4day11_pkl.py`](notebooks/exploration/revision/r35_make_ppp4day11_pkl.py) · Fig 2 replay table

Builds the PPP4_day11 replay / non-replay table with the schema of `generate_replay_non_replay.ipynb`: NREM d2 windows,
ripples in NREM, replay intervals of one template and epochs 1, 3, 5 (left- and right-significant events, p < 0.05), a
ripple counted as replay if it overlaps a replay interval.

### r35_glm_ablation.py

[`r35_glm_ablation.py`](notebooks/exploration/revision/r35_glm_ablation.py) · shared code

GLM ablation as in Chang et al. 2025 (Nature 637:1161, Fig. 2e), with d2 as the outcome: predictors z-scored, one
coefficient set to zero per ablation, and the loss in cross-validated R^2 as that predictor's contribution.

### r35_replay_ablation_lrt.py

[`r35_replay_ablation_lrt.py`](notebooks/exploration/revision/r35_replay_ablation_lrt.py) · Fig 2e

One pooled binomial GLM with session fixed effects. (a) Coefficient per within-session SD, cluster-robust 95% CI.
(b) Deviance explained by each predictor alone (hatched) and its unique share in the joint model (filled). CIs:
session-cluster bootstrap (sessions resampled with replacement, 2.5 / 97.5th percentiles).

### r35_replay_lmm.py

[`r35_replay_lmm.py`](notebooks/exploration/revision/r35_replay_lmm.py) · Fig 2e

Linear mixed model on log-transformed replay counts and fractions (REML): windows within sessions within animals, a
random slope per session for each predictor.

### fig2f_backtrace_model.py

[`fig2f_backtrace_model.py`](notebooks/exploration/revision/fig2f_backtrace_model.py) · shared code

The input table of the submitted Fig 2f GLM (`generate_GLM.ipynb`): d2 of every window; FR_std, the across-neuron SD
of firing rates in [c - 15, c + 15) s; MUA, the 40 ms population rate averaged over 30 s; ripples in NREM; replay
events (either direction with p < .05 and r < .05); per window, the ripple and replay counts; NREM windows, replay
present = replay count > 0, tile = sextile of ripple count; the model `cv_predictive_gain_logistic(features = (d2, MUA,
FR_std), n_bins = 6, n_splits = 5, C = 1)`.

### fig2g_psth_excised.py

[`fig2g_psth_excised.py`](notebooks/exploration/revision/fig2g_psth_excised.py) · Fig 2f

Branching ratio around each SWR and at shuffled NREM times, full window and ripple-excised;
`EXCLUDE_RIPPLES=1` keeps the shuffled times at least 1 s from every ripple.

## Barrages and CCK silencing (Figure 3, Supplementary Figure 6)

### fig3_background_repro.py

[`fig3_background_repro.py`](notebooks/exploration/revision/fig3_background_repro.py) · Fig 3e-h; Supplementary Fig 8d

**e.** d2 (full window and background) per session, in within-session sextiles of barrage rate (rank sextiles;
`BINS=width`: 6 equal-width bins of [0, max], the default for h); mean +- SEM over sessions, Pearson r of the six means;
per-session Spearman correlation, signed-rank across sessions.

**f.** Branching-ratio change around barrages vs shuffled times, one mean per file; mixed model (value ~ condition +
animal, random intercept per session) and paired Wilcoxon.

**g.** Background d2, one mean per session and epoch (presleep, postsleep), paired Wilcoxon over the 7 Sncg sessions.

**h.** ISI entropy by barrage-rate bin per session; the star is the mixed model of `fig3h_entropy_diagnostics.py` (ISI
entropy ~ barrage rate + animal, random intercept and slope per session).

### r16_event_dissociation.py

[`r16_event_dissociation.py`](notebooks/exploration/revision/r16_event_dissociation.py) · shared code

Within-session Spearman correlation of d2, MUA, MUA Fano, E/I and synchrony with SWR rate and with barrage rate, the
same partialled on the other measures (`partial_spearman`: ranks regressed on the covariate ranks), and a Shapley
decomposition of held-out R^2 (5-fold block cross-validation).

### r16_block_level_dissociation.py

[`r16_block_level_dissociation.py`](notebooks/exploration/revision/r16_block_level_dissociation.py) · Supplementary Fig 6a

**Event type.** One binomial GLM per session (and per contiguous NREM block of at least `MINW` windows): successes =
barrage count, failures = SWR count, so the outcome is the probability that an event is a barrage. Predictors: d2, MUA,
MUA Fano, E/I and synchrony, computed on the event-free bins (`PRED_SET=excised`) or the whole window
(`PRED_SET=intact`), z-scored within session. Sessions or blocks with at least `MIN_EVENTS` of each type.

**Orthogonalised predictors** (`ORTHO=1`): MUA, E/I, Fano and synchrony residualised on one another in that order; d2
is not residualised.

**Tests.** Session coefficients against zero (one-sample test across sessions); pairwise comparisons of |coefficient|
between predictors (paired Wilcoxon, Holm).

### r16_predictor_heterogeneity.py

[`r16_predictor_heterogeneity.py`](notebooks/exploration/revision/r16_predictor_heterogeneity.py) · Supplementary Fig 6b, c

DerSimonian-Laird random-effects meta-analysis of the per-block coefficients (`meta()`): Cochran's Q, I^2, the
between-block SD tau and the pooled estimates. Per block, each predictor's trend (cubic in time), coefficient of
variation and lag-1 autocorrelation. MUA and E/I are also split into excitatory and inhibitory rates (r_E, r_I) and
refitted in their place.

### r16_robustness_index.py

[`r16_robustness_index.py`](notebooks/exploration/revision/r16_robustness_index.py) · Supplementary Fig 6b, c

**Consistency across blocks** (Supplementary Fig 6b). From the meta-analysis of the per-block coefficients
(orthogonalised predictors): the pooled effect mu and the between-block SD tau; the indices are |mu| / tau,
Phi(|mu| / tau), the expected fraction of blocks with the pooled sign, and the 95% prediction interval
mu +- 1.96 sqrt(tau^2 + SE_mu^2) (IntHout et al. 2016).

**Transfer to a new session** (Supplementary Fig 6c). Leave-one-session-out: a binomial GLM with an intercept per block
and a shared slope is fitted on the other sessions; each held-out block is scored with that slope and its own
intercept. Transfer ratio = held-out deviance explained / in-sample deviance explained (median over blocks).

**Confidence intervals.** Session-level cluster bootstrap: sessions resampled with replacement, each with its blocks;
95% interval of each index.

### figure_scripts/FigS6_AC_regime_indicator.ipynb

[`FigS6_AC_regime_indicator.ipynb`](../figure_scripts/FigS6_AC_regime_indicator.ipynb) · Supplementary Fig 6

**Data.** 11 barrage sessions, CA1 units, SWRs and barrages in non-overlapping 30 s NREM windows; the five background
measures (d2, MUA, MUA Fano, E/I, synchrony) computed after removing every 40 ms bin that overlaps a ripple or a
barrage.

**a.** One binomial GLM per session: barrage share of a window's events on the five measures, each z-scored within
session; session coefficients tested against 0 (one-sample signed-rank).

**b.** NREM in 30-minute blocks (45 blocks); per-block coefficients (E/I, Fano and synchrony residualised on the
measures before them) pooled by random-effects meta-analysis; stability index |mu| / tau. Brackets: paired session
bootstrap of d2 against each measure (2 000 resamples; p = 2 x the smaller tail, two-sided, Holm).

**c.** Leave-one-session-out transfer ratio for d2 alone, MUA alone, d2 + MUA and all five.

**Boxes.** Session-level cluster bootstrap (2 000 resamples): box = IQR, whiskers = 2.5-97.5th percentiles, diamond =
full-data estimate.

## Drift and stationarity

### r25_drift_control_rates.py

[`r25_drift_control_rates.py`](notebooks/exploration/revision/r25_drift_control_rates.py) · Supplementary Fig 6 windows

d2 vs SWR rate and d2 vs barrage rate in four versions: as in the figures; with each window's linear trend removed
before the AR fit; partialled on rank(|trend|); and in windows whose end-to-end trend is below 0.5 SD. Per-session
Spearman correlations are pooled by Fisher z, with a Wilcoxon signed-rank test across sessions.

### r25_stationarity_battery.py

[`r25_stationarity_battery.py`](notebooks/exploration/revision/r25_stationarity_battery.py) · shared code

d2 per session and state on non-overlapping windows of 10-120 s, for the real windows and for: detrended windows; a
stationary AR(20) simulation fitted on the longest window; added linear ramps of 0.5, 1 and 2 SD; an added random walk
of 1 SD. Per window: ADF and KPSS tests (lag 20) and the linear trend in SD units. Statistics: per-session slope of d2
against log2(window length), real minus stationary (signed-rank across sessions); within-session Spearman of d2 with
|trend|.

## Avalanches

### avalanche_ripple.py

[`avalanche_ripple.py`](notebooks/exploration/revision/avalanche_ripple.py) · Supplementary Fig 3e-g

Avalanches detected in NREM on bins equal to each session's population mean inter-event interval; activity threshold at
the 20th, 30th and 40th percentiles; xmin by the KS procedure of the `powerlaw` package. Each avalanche is assigned the
local SWR rate (+-5 s) and split at the session median.
