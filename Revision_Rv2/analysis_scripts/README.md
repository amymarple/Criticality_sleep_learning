# analysis_scripts: from the recordings to the figure tables

The scripts that turn the recordings into the tables in [`../csv/`](../csv). They write to `data/` inside this folder.
Paths to the recordings (`W:\data\PPP`, `X:\data\Barrage`, ...) are written in full; change them for another machine.
How the steps connect: [README](../README.md#from-the-recordings-to-the-figures); methods by topic:
[METHODS.md](METHODS.md).

Names: `ed*` / `ED<n>` are the Extended Data figures of the first submission; `fig2f_*` is the replay-model panel
(Fig 2e). Background (ripple-excised) d2: d2 after removing every 40 ms bin that overlaps an SWR (and a barrage)
and joining the rest of the window.

## Contents

- [MATLAB: d2 per session](#matlab-d2-per-session)
- [First-submission producers](#first-submission-producers)
- [Revision analyses](#revision-analyses): [scripts that make the figure tables](#scripts-that-make-the-figure-tables),
  [data preparation](#data-preparation), [shared code](#shared-code)
- [First-submission figure notebooks, raw-data versions](#first-submission-figure-notebooks-raw-data-versions)
- [Other folders](#other-folders)

## MATLAB: d2 per session

In `notebooks/helper/` and `notebooks/exploration/`. Each worker is run inside a session folder and writes `<basepath>/complex_system/<day>_CA1_D2_time_resolved*.mat`. The d2 code itself is that of Sooter et al. 2024 ([https://doi.org/10.6084/m9.figshare.25927081](https://doi.org/10.6084/m9.figshare.25927081)).

| script | what it does |
|---|---|
| [`d2_fast.py`](notebooks/helper/d2_fast.py) | d2 (2025 KL) in Python (Sooter et al. 2025); its Yule-Walker AR fit and KL rate are imported by `euclid_order_sweep.py` and `r35_kl_rippleablate.py`. |
| [`getDbetas_error_bar.m`](notebooks/helper/getDbetas_error_bar.m) | d2 (2024 Euclidean) of one window and its delta-method error (calls the Sooter et al. 2024 functions). |
| [`time_resolved_d2_across_state.m`](notebooks/helper/time_resolved_d2_across_state.m) | (from the lab's R_Project repository) Per session: d2 (2024 Euclidean) in 30 s windows every 2 s, state by state (40 ms, AR 20), without the error bar -> <basepath>/complex_system/*_D2_time_resolved.mat. |
| [`time_resolved_d2_across_state_error_bar.m`](notebooks/helper/time_resolved_d2_across_state_error_bar.m) | Per session: d2 (2024 Euclidean) in 30 s windows every 2 s, state by state (40 ms, AR 20) -> <basepath>/complex_system/*_D2_time_resolved_error_bar*.mat. |
| [`time_resolved_d2_across_state_ripple_ablate.m`](notebooks/helper/time_resolved_d2_across_state_ripple_ablate.m) | As above with every SWR cut out of the NREM windows -> *_rippleablate.mat. |
| [`time_resolved_d2_across_state_barrage_ablate.m`](notebooks/helper/time_resolved_d2_across_state_barrage_ablate.m) | As above with every BARR cut out of the NREM windows -> *_barrageablate.mat. |
| [`time_resolved_d2_state_transition.m`](notebooks/helper/time_resolved_d2_state_transition.m) | As above without the 5 s guard at state edges (transitions) -> *_state_transition.mat. |
| [`Getreplaypmatforpython.m`](notebooks/helper/Getreplaypmatforpython.m) | Exports the replay decoding (.mat) for Python. |
| [`coarse_grain_ddc.m`](notebooks/helper/coarse_grain_ddc.m) | (from the lab's R_Project repository) AR(20) fit and variance explained in sliding NREM windows of 0.5-60 s, every session (Supplementary Fig 2a). |
| [`d2_loopthrough_data.m`](notebooks/exploration/d2_loopthrough_data.m) | Batch driver: runs the d2 workers above over every session of the paper. |
| [`CA1_uLED_d2.m`](notebooks/exploration/CA1_uLED_d2.m) | d2 of the uLED CCK-silencing sessions, CA1 / CA1+CA2, 80 ms bins, 60 s windows. |
| [`CA1_uLED_d2_batch.m`](notebooks/exploration/CA1_uLED_d2_batch.m) | Batch driver for `CA1_uLED_d2.m`. |

## First-submission producers

In `notebooks/exploration/`: the lab notebooks of the first submission that read each session folder (spikes, sleep states, ripples, barrages, replay decoding, the d2 `.mat` files) and write per-session tables to `data/<experiment>/`. Commented-out cells of the `MUA_sleepstage*` notebooks are run by uncommenting them, one session at a time. Producers used only by the first-submission figures are in [`../../Previous_Rv1/analysis_scripts/`](../../Previous_Rv1/analysis_scripts).

| script | what it does |
|---|---|
| [`MUA_sleepstage.ipynb`](notebooks/exploration/MUA_sleepstage.ipynb) | Mouse d2 windows (from the d2 .mat files) with MUA and SWR / BARR rates. Saves the d2-uncertainty windows (`data/ripple_barr/d2_uncertainty`), the d2 and event-rate windows of the barrage sessions (`data/cross_state/mice_d2_barr_ripple`) and the novel / familiar d2 relative to presleep (`data/novel_familiar`). |
| [`MUA_sleepstage_barr_model_supple_ver2.ipynb`](notebooks/exploration/MUA_sleepstage_barr_model_supple_ver2.ipynb) | Rat CA1 / CA2 / CA3 firing rates per d2 window (barrage sessions), saved per session (`data/rats/barr_CA2_CA3`). |
| [`MUA_sleepstage_rat.ipynb`](notebooks/exploration/MUA_sleepstage_rat.ipynb) | Rat d2 windows with MUA and SWR / BARR rate (`data/rats`) and their ISI entropy. |
| [`MUA_sleepstage_rat_replay.ipynb`](notebooks/exploration/MUA_sleepstage_rat_replay.ipynb) | Rat replay / non-replay SWR-rate tiles of d2 (`data/rats/replay_nonreplay`). |
| [`barr_CA2_CA3_pooled_analysis.py`](notebooks/exploration/barr_CA2_CA3_pooled_analysis.py) | Pools the rat CA2 / CA3 firing tables. |
| [`generate_GLM.ipynb`](notebooks/exploration/generate_GLM.ipynb) | Per-session replay GLM ablation (submitted Fig 2f). |
| [`generate_ISI_entropy_ripple.ipynb`](notebooks/exploration/generate_ISI_entropy_ripple.ipynb) | ISI entropy per window with SWR and BARR rate. |
| [`generate_ablate_ripple.ipynb`](notebooks/exploration/generate_ablate_ripple.ipynb) | The ripple-excised d2 windows (*_rippleablate.mat) with ISI entropy, SWR rate and sleep epoch. |
| [`generate_barr_manipulation.ipynb`](notebooks/exploration/generate_barr_manipulation.ipynb) | d2 of the CCK-silencing (Sncg) sessions, pre- and post-task sleep. |
| [`generate_br_change_ripple.ipynb`](notebooks/exploration/generate_br_change_ripple.ipynb) | Branching-ratio change around every SWR / BARR and at shuffled times (mouse). |
| [`generate_branching_ratio_mouse.ipynb`](notebooks/exploration/generate_branching_ratio_mouse.ipynb) | Branching ratio (MR estimator) per 10 s window (2 s step) and state, and its change around BARRs and at shuffled times (barrage cohort); mouse. |
| [`generate_branching_ratio_rats.ipynb`](notebooks/exploration/generate_branching_ratio_rats.ipynb) | Branching ratio per window and its change around SWRs / BARRs, rats. |
| [`generate_pairs_df.ipynb`](notebooks/exploration/generate_pairs_df.ipynb) | d2-MUA window pairs per session (from the d2 .mat files and the spikes). |
| [`generate_pre_post.ipynb`](notebooks/exploration/generate_pre_post.ipynb) | d2 windows of the pre / post-task sleep sessions with sleep epoch, state and SWR rate. |
| [`generate_replay_non_replay.ipynb`](notebooks/exploration/generate_replay_non_replay.ipynb) | Replay-SWR and non-replay-SWR rate per d2 window (decoded sessions). |
| [`generate_replay_probablity_tile.ipynb`](notebooks/exploration/generate_replay_probablity_tile.ipynb) | Replay probability by d2 sextile per session. |

## Revision analyses

In `notebooks/exploration/revision/`: every script the figure tables come from, the data steps before them, and the scripts these import. Most write one table to `data/revision/log/`, named at the end of their header (`-> data/revision/log/<name>.csv`).

### Scripts that make the figure tables

| script | what it does | panels |
|---|---|---|
| [`fig1c_example_traces.py`](notebooks/exploration/revision/fig1c_example_traces.py) | The PPP20 day 5 example: population rate, E/I balance, synchrony, theta power, SWR rate and d2 per window. | Fig 1b |
| [`fig1_state_space_umap_session.py`](notebooks/exploration/revision/fig1_state_space_umap_session.py) | Window-feature UMAP of one session, one animal or all 18 sessions; writes the embedding drawn in Fig 1c. | Fig 1c |
| [`fig1_state_umap_all18.ipynb`](notebooks/exploration/revision/fig1_state_umap_all18.ipynb) | Draws the saved UMAP embedding of all 18 sessions (10 window features). | Fig 1c |
| [`fig1_si_sessions.py`](notebooks/exploration/revision/fig1_si_sessions.py) | Structure index of d2 per session against the session's own null, in the format of the lab's pupil paper (two columns per session: the index and its null). | Fig 1d |
| [`ed_nonoverlap.py`](notebooks/exploration/revision/ed_nonoverlap.py) | Runs the first-submission figure notebooks on non-overlapping windows. | Fig 1e, 2a; Supp 2a-e, 3c, 4a, b, e, f, 7, 8 |
| [`fig2c_example_excised.py`](notebooks/exploration/revision/fig2c_example_excised.py) | The example trace of Fig 2b with the ripple-excised d2. | Fig 2b (curve) |
| [`excised_br_entropy.py`](notebooks/exploration/revision/excised_br_entropy.py) | Ripple-excised branching ratio and ISI entropy per non-overlapping NREM window, on the windows of the ripple-excised d2. | Fig 2c; Supp 4d |
| [`fig2_new.py`](notebooks/exploration/revision/fig2_new.py) | Drawing functions of the Figure 2 panels. | Fig 2c, d, f, g; Supp 3g, 4c, d (drawing) |
| [`fig2f_swr_property_ladder.py`](notebooks/exploration/revision/fig2f_swr_property_ladder.py) | Replay model with d2, adding SWR rate, amplitude and duration one at a time. | Fig 2d |
| [`fig2f_standalone.py`](notebooks/exploration/revision/fig2f_standalone.py) | Replay rate: the explained deviance lost when each background predictor is dropped. | Fig 2e |
| [`r35_replay_rate_model.py`](notebooks/exploration/revision/r35_replay_rate_model.py) | Replay rate (replays per window) as the outcome, with the variables of the submitted Fig 2f. | Fig 2e |
| [`fig2g_psth_excised.py`](notebooks/exploration/revision/fig2g_psth_excised.py) | Branching ratio around SWRs vs shuffled times, full window and ripple-excised, nine decoded sessions. | Fig 2f |
| [`fig2_agh_excised.py`](notebooks/exploration/revision/fig2_agh_excised.py) | d2, branching ratio and ISI entropy by SWR-rate sextile, full window and ripple-excised, nine replay-decoded sessions. | Fig 2g; Supp 4c |
| [`fig3_background_repro.py`](notebooks/exploration/revision/fig3_background_repro.py) | Figure 3e-h: d2 (full window and background), branching ratio and ISI entropy against barrage rate; g as session means of the background d2. | Fig 3e-h; Supp 8d |
| [`r16_glue_control.py`](notebooks/exploration/revision/r16_glue_control.py) | d2 with SWRs and barrages removed, computed four ways: whole window, remaining segments joined, within segments only, and a sham removal of the same length. | Fig 3e |
| [`fig3f_psth_excised.py`](notebooks/exploration/revision/fig3f_psth_excised.py) | Branching ratio around barrages, on the ripple- and barrage-excised background. | Fig 3f |
| [`fig3g_background.py`](notebooks/exploration/revision/fig3g_background.py) | CCK silencing (Sncg sessions) with each window's d2 replaced by its background d2. | Fig 3g |
| [`fig3h_entropy_diagnostics.py`](notebooks/exploration/revision/fig3h_entropy_diagnostics.py) | ISI entropy vs barrage rate: sextile means, per-session correlations and the mixed model drawn as the panel's star. | Fig 3h |
| [`r38_hyperplane_explainer.py`](notebooks/exploration/revision/r38_hyperplane_explainer.py) | d2 (2024 Euclidean) on real data: the beta = 2 hyperplane, a fitted AR(20) kernel and one session's windows relative to it. | Supp 1a-d |
| [`euclid_order_sweep.py`](notebooks/exploration/revision/euclid_order_sweep.py) | d2 (2024 Euclidean) across AR orders in all states; also the beta-manifolds drawn in Supplementary Fig 1c, d. | Supp 1c, d (beta manifolds) |
| [`ed_nonoverlap_ed1b.py`](notebooks/exploration/revision/ed_nonoverlap_ed1b.py) | Variance explained by the AR(20) fit vs window length (submitted Extended Data 1b), on non-overlapping windows. | Supp 2a |
| [`ed_nonoverlap_d2err.py`](notebooks/exploration/revision/ed_nonoverlap_d2err.py) | The two input tables of the d2-uncertainty panels (Extended Data 1c, 2c) on non-overlapping windows. | Supp 2b; 3d |
| [`fig1de_nonoverlap.py`](notebooks/exploration/revision/fig1de_nonoverlap.py) | Figure 1d, e of the first submission on non-overlapping windows. | Supp 2d, f |
| [`ed_nonoverlap_ed2a_ed1f.py`](notebooks/exploration/revision/ed_nonoverlap_ed2a_ed1f.py) | The inputs of Extended Data 2a (novel vs familiar d2) and 1f (branching ratio by state) on non-overlapping windows. | Supp 2e; 3b |
| [`fig1e_scatter_d2raw.py`](notebooks/exploration/revision/fig1e_scatter_d2raw.py) | The d2-MUA scatter (submitted Fig 1e) with d2 in raw units and MUA z-scored, on non-overlapping windows. | Supp 2f; 7c |
| [`m25_task_performance.py`](notebooks/exploration/revision/m25_task_performance.py) | Trials and fraction correct per task block for every Figure 1-3 session, from the behaviour files. | Supp 3a |
| [`m25_tmaze_box.py`](notebooks/exploration/revision/m25_tmaze_box.py) | T-maze alternation performance per session against chance. | Supp 3a |
| [`ed2a_novel_vs_control.py`](notebooks/exploration/revision/ed2a_novel_vs_control.py) | Novel vs familiar sessions (Extended Data 2a) as a difference in differences between the two groups. | Supp 3b |
| [`ed2a_ppp8_day14_rebuild.py`](notebooks/exploration/revision/ed2a_ppp8_day14_rebuild.py) | d2 of PPP8 day 14 relative to pre-task sleep, for the novel vs familiar comparison. | Supp 3b |
| [`mixture_proof.py`](notebooks/exploration/revision/mixture_proof.py) | The NREM avalanche-size distribution as a two-component mixture, split by each avalanche's local ripple rate (median split: low vs high), compared against the pooled single power-law fit. | Supp 3e |
| [`ripple_criticality_batch.ipynb`](notebooks/exploration/revision/ripple_criticality_batch.ipynb) | Avalanche power-law fits in low vs high SWR-rate NREM periods, per session. | Supp 3e-g |
| [`ks_xmin_convention.py`](notebooks/exploration/revision/ks_xmin_convention.py) | KS distance of low vs high SWR-rate avalanches under both xmin conventions (common and per group). | Supp 3f |
| [`m22_dcc_threshold_box.py`](notebooks/exploration/revision/m22_dcc_threshold_box.py) | DCC and KS distance of low vs high SWR-rate NREM avalanches at three activity thresholds (1x IEI bins). | Supp 3f, g |
| [`ed4d_replay_prob_lmm.py`](notebooks/exploration/revision/ed4d_replay_prob_lmm.py) | The mixed model of replay probability by d2 sextile (submitted Fig 2d), on non-overlapping windows. | Supp 4d |
| [`r16_block_level_dissociation.py`](notebooks/exploration/revision/r16_block_level_dissociation.py) | Which background measures predict whether a window's NREM events are barrages or SWRs, per session and per NREM block. | Supp 6a |
| [`r16_event_excised_windows.py`](notebooks/exploration/revision/r16_event_excised_windows.py) | The barrage-cohort window table with every predictor computed after removing SWRs and barrages. | Supp 6a-c |
| [`r16_predictor_heterogeneity.py`](notebooks/exploration/revision/r16_predictor_heterogeneity.py) | How much each predictor's event-type coefficient varies across NREM blocks (predictors orthogonalised in a fixed order). | Supp 6b, c |
| [`r16_robustness_index.py`](notebooks/exploration/revision/r16_robustness_index.py) | Indices of how consistently each predictor relates to event type across NREM blocks and sessions. | Supp 6b, c |
| [`ed_nonoverlap_rats_tiles.py`](notebooks/exploration/revision/ed_nonoverlap_rats_tiles.py) | Three rat tables without window times (Extended Data 6c, 6f, 6g) on non-overlapping windows. | Supp 8c, f |

### Data preparation

| script | what it does | prepares |
|---|---|---|
| [`fig1_dropin_metrics.py`](notebooks/exploration/revision/fig1_dropin_metrics.py) | Adds per-window columns (synchrony, SWR rate) to the Figure 1 window tables. | window columns (Fig 1) |
| [`fig1_emg_lfp.py`](notebooks/exploration/revision/fig1_emg_lfp.py) | LFP-derived EMG per 30 s window, for the state-space UMAP. | LFP features of the Fig 1c, d windows |
| [`fig1_metrics.py`](notebooks/exploration/revision/fig1_metrics.py) | Shared window, state, and metric library for the Figure 1 analyses. | window metrics library |
| [`fig1_state_space_export.py`](notebooks/exploration/revision/fig1_state_space_export.py) | The 10 features of every non-overlapping window: the input of the state-space UMAP. | the Fig 1c, d window features |
| [`fig1_theta_ratio.py`](notebooks/exploration/revision/fig1_theta_ratio.py) | Theta power per 30 s window, for the state-space UMAP and the Fig 1c example. | theta power per window (Fig 1b, c) |
| [`metric_replay_ladder.py`](notebooks/exploration/revision/metric_replay_ladder.py) | Replay models with the branching ratio or ISI entropy in place of d2, on the same windows. | the Fig 2d windows (metric_replay_ladder_windows.csv) |
| [`opto_sncg_metrics.py`](notebooks/exploration/revision/opto_sncg_metrics.py) | Branching ratio, ISI entropy, MUA and d2 in the Sncg CCK-silencing sessions, presleep vs postsleep. | the Sncg windows (Fig 3g) |
| [`r25_drift_control_rates.py`](notebooks/exploration/revision/r25_drift_control_rates.py) | d2 vs SWR rate and d2 vs barrage rate after detrending, after partialling out the within-window trend, and in low-drift windows. | the intact-window d2 of Supp 6 (r25_drift_control_rates_windows.csv) |
| [`r35_extend_replay_sessions.py`](notebooks/exploration/revision/r35_extend_replay_sessions.py) | The replay / SWR window table of the 10 sessions with decoded replay. | the Fig 2e replay windows |
| [`r35_make_ppp4day11_pkl.py`](notebooks/exploration/revision/r35_make_ppp4day11_pkl.py) | Writes the PPP4 day 11 table into the replay / non-replay folder read by the submitted Fig 2e notebook. | the PPP4_day11 replay table of the Fig 2 sessions |
| [`r35_replay_full_covariates.py`](notebooks/exploration/revision/r35_replay_full_covariates.py) | The replay model with all the event-classifier covariates. | the Fig 2e window table |
| [`run_fig1_metrics.py`](notebooks/exploration/revision/run_fig1_metrics.py) | Batch driver: the per-window metrics of every Figure 1 session. | per-session window metrics (batch driver) |

### Shared code

Imported by other scripts.

| script | what it does | imported by |
|---|---|---|
| [`aggregate_fig1_metrics.py`](notebooks/exploration/revision/aggregate_fig1_metrics.py) | Group-level tables from the per-session window metrics of `run_fig1_metrics.py`. | [`fig1_state_space_export.py`](notebooks/exploration/revision/fig1_state_space_export.py) |
| [`avalanche_ripple.py`](notebooks/exploration/revision/avalanche_ripple.py) | Avalanche detection (`detect`) and the local SWR rate of each avalanche (`ripple_rate_at`). | [`ks_xmin_convention.py`](notebooks/exploration/revision/ks_xmin_convention.py), [`mixture_proof.py`](notebooks/exploration/revision/mixture_proof.py), [`ripple_criticality_batch.ipynb`](notebooks/exploration/revision/ripple_criticality_batch.ipynb) |
| [`barrage_order_kernel_sweep.py`](notebooks/exploration/revision/barrage_order_kernel_sweep.py) | The barrage-session list (`BARRAGE_SESSIONS`) and d2 of those sessions over AR orders. | [`fig3f_psth_excised.py`](notebooks/exploration/revision/fig3f_psth_excised.py), [`m25_task_performance.py`](notebooks/exploration/revision/m25_task_performance.py), [`r16_event_dissociation.py`](notebooks/exploration/revision/r16_event_dissociation.py) ... |
| [`br_ripple_replay.py`](notebooks/exploration/revision/br_ripple_replay.py) | Branching ratio by SWR-rate sextile and by replay-probability sextile, on non-overlapping windows. | [`excised_br_entropy.py`](notebooks/exploration/revision/excised_br_entropy.py) |
| [`d2_composition_glm.py`](notebooks/exploration/revision/d2_composition_glm.py) | GLM with d2 as the outcome and the window's state measures, SWR activity and replay content as predictors (Shapley attribution). | [`br_ripple_replay.py`](notebooks/exploration/revision/br_ripple_replay.py), [`d2_correlates_table.py`](notebooks/exploration/revision/d2_correlates_table.py), [`excised_br_entropy.py`](notebooks/exploration/revision/excised_br_entropy.py) |
| [`d2_correlates_table.py`](notebooks/exploration/revision/d2_correlates_table.py) | Per-session correlation table of d2 with window metrics, extended with E/I balance and synchrony. | [`br_ripple_replay.py`](notebooks/exploration/revision/br_ripple_replay.py), [`d2_yw_matlab_match.py`](notebooks/exploration/revision/d2_yw_matlab_match.py) |
| [`d2_yw_matlab_match.py`](notebooks/exploration/revision/d2_yw_matlab_match.py) | Yule-Walker d2 of a window with segments removed (`d2_yw_segments`), as in the MATLAB workers. | [`fig3g_background.py`](notebooks/exploration/revision/fig3g_background.py), [`r16_glue_control.py`](notebooks/exploration/revision/r16_glue_control.py), [`r38_hyperplane_explainer.py`](notebooks/exploration/revision/r38_hyperplane_explainer.py) |
| [`euclid_d2_err_orders.py`](notebooks/exploration/revision/euclid_d2_err_orders.py) | d2 (2024 Euclidean) and its delta-method error bar across AR orders, for every NREM window. | [`d2_yw_matlab_match.py`](notebooks/exploration/revision/d2_yw_matlab_match.py), [`r25_drift_control_rates.py`](notebooks/exploration/revision/r25_drift_control_rates.py), [`r25_stationarity_battery.py`](notebooks/exploration/revision/r25_stationarity_battery.py) ... |
| [`fig1_d2_vs_mua_2x2.py`](notebooks/exploration/revision/fig1_d2_vs_mua_2x2.py) | d2 vs MUA per state, for both d2 definitions (2024 Euclidean, 2025 KL), full window and ripple-excised. | [`fig2_new.py`](notebooks/exploration/revision/fig2_new.py), [`fig2c_example_excised.py`](notebooks/exploration/revision/fig2c_example_excised.py) |
| [`fig1_lfp_manifold.py`](notebooks/exploration/revision/fig1_lfp_manifold.py) | State space built from LFP features (as in Chang, Tang et al., Nature 2025), with d2 shown on it rather than used as a feature. | [`fig1_nrem_feature_gradient.py`](notebooks/exploration/revision/fig1_nrem_feature_gradient.py), [`fig1_si_sessions.py`](notebooks/exploration/revision/fig1_si_sessions.py), [`fig1_state_space_umap_session.py`](notebooks/exploration/revision/fig1_state_space_umap_session.py) ... |
| [`fig1_nonoverlap_contrasts.py`](notebooks/exploration/revision/fig1_nonoverlap_contrasts.py) | Non-overlapping window selection and the Figure 1 state contrasts. | [`fig1de_nonoverlap.py`](notebooks/exploration/revision/fig1de_nonoverlap.py) |
| [`fig1_nrem_feature_gradient.py`](notebooks/exploration/revision/fig1_nrem_feature_gradient.py) | Structure index of d2 within NREM in the window-feature space, against a time-shift null that keeps d2's own time structure. | [`fig1_state_feature_gradient.py`](notebooks/exploration/revision/fig1_state_feature_gradient.py), [`fig1_state_space_umap_session.py`](notebooks/exploration/revision/fig1_state_space_umap_session.py) |
| [`fig1_state_feature_gradient.py`](notebooks/exploration/revision/fig1_state_feature_gradient.py) | Structure index of d2 within each state and across all states, in the window-feature space. | [`fig1_si_sessions.py`](notebooks/exploration/revision/fig1_si_sessions.py) |
| [`fig1_state_space_umap.py`](notebooks/exploration/revision/fig1_state_space_umap.py) | UMAP embedding of the window features, with its colours and labels. | [`fig1_nrem_feature_gradient.py`](notebooks/exploration/revision/fig1_nrem_feature_gradient.py), [`fig1_si_sessions.py`](notebooks/exploration/revision/fig1_si_sessions.py), [`fig1_state_feature_gradient.py`](notebooks/exploration/revision/fig1_state_feature_gradient.py) ... |
| [`fig2_composite.py`](notebooks/exploration/revision/fig2_composite.py) | The quantitative Figure 2 panels on the nine decoded sessions and non-overlapping windows. | [`fig2_new.py`](notebooks/exploration/revision/fig2_new.py), [`fig2c_example_excised.py`](notebooks/exploration/revision/fig2c_example_excised.py) |
| [`fig2f_background_state_glm.py`](notebooks/exploration/revision/fig2f_background_state_glm.py) | The replay GLM of the submitted Fig 2f with background-state covariates, a measurement-error correction and temporal precedence. | [`metric_replay_ladder.py`](notebooks/exploration/revision/metric_replay_ladder.py), [`r35_extend_replay_sessions.py`](notebooks/exploration/revision/r35_extend_replay_sessions.py) |
| [`fig2f_backtrace.py`](notebooks/exploration/revision/fig2f_backtrace.py) | Per-window state table of the replay GLM (`load_all_states`). | [`fig2f_backtrace_model.py`](notebooks/exploration/revision/fig2f_backtrace_model.py) |
| [`fig2f_backtrace_model.py`](notebooks/exploration/revision/fig2f_backtrace_model.py) | Population MUA per window (`population_mua`) and the replay GLM of the submitted Fig 2f. | [`r35_extend_replay_sessions.py`](notebooks/exploration/revision/r35_extend_replay_sessions.py) |
| [`fig2f_producer_glm.py`](notebooks/exploration/revision/fig2f_producer_glm.py) | The replay GLM functions of the submitted Fig 2f (`cv_predictive_gain_logistic`). | [`fig2f_backtrace_model.py`](notebooks/exploration/revision/fig2f_backtrace_model.py) |
| [`fig2f_seat_ladders.py`](notebooks/exploration/revision/fig2f_seat_ladders.py) | Replay models (binomial, per SWR) with d2, MUA and SWR rate each as the main predictor, alone and together. | [`fig2_new.py`](notebooks/exploration/revision/fig2_new.py), [`fig2f_swr_property_ladder.py`](notebooks/exploration/revision/fig2f_swr_property_ladder.py) |
| [`fig3g_reproduce.py`](notebooks/exploration/revision/fig3g_reproduce.py) | Resampling, mixed model and session-level comparison of Fig 3g (`resample`, `lmm`, `session_level`). | [`fig3g_background.py`](notebooks/exploration/revision/fig3g_background.py) |
| [`opto_cck_excised.py`](notebooks/exploration/revision/opto_cck_excised.py) | The CCK-silencing contrast with ripple time removed from every window. | [`fig3f_psth_excised.py`](notebooks/exploration/revision/fig3f_psth_excised.py) |
| [`opto_cck_metrics.py`](notebooks/exploration/revision/opto_cck_metrics.py) | Branching ratio, ISI entropy and MUA in the uLED CCK-silencing sessions, per stimulation block (baseline vs stimulation). | [`fig3g_background.py`](notebooks/exploration/revision/fig3g_background.py), [`fig3g_reproduce.py`](notebooks/exploration/revision/fig3g_reproduce.py), [`opto_cck_excised.py`](notebooks/exploration/revision/opto_cck_excised.py) ... |
| [`r16_event_dissociation.py`](notebooks/exploration/revision/r16_event_dissociation.py) | Burst measures and d2 against SWR rate and barrage rate. | [`r16_event_excised_windows.py`](notebooks/exploration/revision/r16_event_excised_windows.py), [`r16_glue_control.py`](notebooks/exploration/revision/r16_glue_control.py) |
| [`r25_stationarity_battery.py`](notebooks/exploration/revision/r25_stationarity_battery.py) | Stationarity measures per window: window-length invariance, unit-root tests, drift surrogates and the trend in each window. | [`r25_drift_control_rates.py`](notebooks/exploration/revision/r25_drift_control_rates.py) |
| [`r35_background_vs_ripple.py`](notebooks/exploration/revision/r35_background_vs_ripple.py) | d2 with ripple time removed from the fitting window, compared with d2 on the full window. | [`d2_composition_glm.py`](notebooks/exploration/revision/d2_composition_glm.py), [`fig2f_background_state_glm.py`](notebooks/exploration/revision/fig2f_background_state_glm.py), [`fig3g_reproduce.py`](notebooks/exploration/revision/fig3g_reproduce.py) ... |
| [`r35_binomial_fig2fvars.py`](notebooks/exploration/revision/r35_binomial_fig2fvars.py) | The binomial replay model with the variables of the submitted Fig 2f, for the full-window and the ripple-excised d2. | [`fig2f_background_state_glm.py`](notebooks/exploration/revision/fig2f_background_state_glm.py), [`r35_sham_excision.py`](notebooks/exploration/revision/r35_sham_excision.py) |
| [`r35_binomial_replay.py`](notebooks/exploration/revision/r35_binomial_replay.py) | Binomial model of the probability that an SWR carries replay, as a function of d2. | [`fig2f_seat_ladders.py`](notebooks/exploration/revision/fig2f_seat_ladders.py), [`fig2f_swr_property_ladder.py`](notebooks/exploration/revision/fig2f_swr_property_ladder.py), [`metric_replay_ladder.py`](notebooks/exploration/revision/metric_replay_ladder.py) ... |
| [`r35_glm_ablation.py`](notebooks/exploration/revision/r35_glm_ablation.py) | Contribution of replay to d2 beyond SWR properties, by GLM ablation (as in Chang et al., Nature 2025). | [`br_ripple_replay.py`](notebooks/exploration/revision/br_ripple_replay.py), [`d2_composition_glm.py`](notebooks/exploration/revision/d2_composition_glm.py), [`d2_correlates_table.py`](notebooks/exploration/revision/d2_correlates_table.py) ... |
| [`r35_kl_rippleablate.py`](notebooks/exploration/revision/r35_kl_rippleablate.py) | d2 (2025 KL) on ripple-excised windows, with a matched full-window computation. | [`opto_cck_excised.py`](notebooks/exploration/revision/opto_cck_excised.py) |
| [`r35_replay_ablation_lrt.py`](notebooks/exploration/revision/r35_replay_ablation_lrt.py) | Replay ablation by nested likelihood-ratio tests on one pooled GLM, with covariates computed on the ripple-excised background. | [`fig2f_standalone.py`](notebooks/exploration/revision/fig2f_standalone.py), [`r35_replay_full_covariates.py`](notebooks/exploration/revision/r35_replay_full_covariates.py), [`r35_replay_lmm.py`](notebooks/exploration/revision/r35_replay_lmm.py) ... |
| [`r35_replay_lmm.py`](notebooks/exploration/revision/r35_replay_lmm.py) | Replay vs the four background predictors as a linear mixed model (windows within sessions within animals). | [`fig2f_standalone.py`](notebooks/exploration/revision/fig2f_standalone.py) |
| [`r35_sham_excision.py`](notebooks/exploration/revision/r35_sham_excision.py) | Sham control for the ripple excision: the same number and length of intervals removed at ripple-free positions. | [`opto_cck_excised.py`](notebooks/exploration/revision/opto_cck_excised.py), [`r16_event_excised_windows.py`](notebooks/exploration/revision/r16_event_excised_windows.py), [`r16_glue_control.py`](notebooks/exploration/revision/r16_glue_control.py) ... |
| [`r35_swr_property_control.py`](notebooks/exploration/revision/r35_swr_property_control.py) | Replay vs non-replay SWR rate with SWR duration, amplitude, participating-cell count and decoded trajectory length as controls. | [`br_ripple_replay.py`](notebooks/exploration/revision/br_ripple_replay.py), [`d2_composition_glm.py`](notebooks/exploration/revision/d2_composition_glm.py), [`d2_correlates_table.py`](notebooks/exploration/revision/d2_correlates_table.py) ... |
| [`state_level_br.py`](notebooks/exploration/revision/state_level_br.py) | Branching ratio per state, all windows of a state pooled (as in Wilting & Priesemann 2018). | [`excised_br_entropy.py`](notebooks/exploration/revision/excised_br_entropy.py), [`fig2g_psth_excised.py`](notebooks/exploration/revision/fig2g_psth_excised.py) |

## First-submission figure notebooks, raw-data versions

In `notebooks/figures/`. Revision scripts run cells of these notebooks (state corrections, loading and drawing functions). Their CSV-reading copies are in [`../../Previous_Rv1/figure_scripts/`](../../Previous_Rv1/figure_scripts).

| notebook | used by |
|---|---|
| [`Fig1_C_Example_plot_mouse.ipynb`](notebooks/figures/Fig1_C_Example_plot_mouse.ipynb) | [`fig1c_example_traces.py`](notebooks/exploration/revision/fig1c_example_traces.py) |
| [`Fig1_D_histogram_S1_E.ipynb`](notebooks/figures/Fig1_D_histogram_S1_E.ipynb) | [`fig1de_nonoverlap.py`](notebooks/exploration/revision/fig1de_nonoverlap.py) |
| [`Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb`](notebooks/figures/Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb) | [`fig1de_nonoverlap.py`](notebooks/exploration/revision/fig1de_nonoverlap.py); Supplementary Fig 2f (its cells 1, 2) |
| [`Fig2_A_FigS2_B_FigS3_A.ipynb`](notebooks/figures/Fig2_A_FigS2_B_FigS3_A.ipynb) | [`fig2_new.py`](notebooks/exploration/revision/fig2_new.py) |
| [`Fig2_C_Example_replayprobalit.ipynb`](notebooks/figures/Fig2_C_Example_replayprobalit.ipynb) | [`fig2c_example_excised.py`](notebooks/exploration/revision/fig2c_example_excised.py) |
| [`Fig2_E_replay_non_replay.ipynb`](notebooks/figures/Fig2_E_replay_non_replay.ipynb) | [`r35_make_ppp4day11_pkl.py`](notebooks/exploration/revision/r35_make_ppp4day11_pkl.py) |
| [`Fig2_F_GLM.ipynb`](notebooks/figures/Fig2_F_GLM.ipynb) | [`r16_block_level_dissociation.py`](notebooks/exploration/revision/r16_block_level_dissociation.py) |
| [`Fig2_G_branchingratio.ipynb`](notebooks/figures/Fig2_G_branchingratio.ipynb) | [`fig2g_psth_excised.py`](notebooks/exploration/revision/fig2g_psth_excised.py) |
| [`Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar.ipynb`](notebooks/figures/Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar.ipynb) | [`fig3_background_repro.py`](notebooks/exploration/revision/fig3_background_repro.py) |
| [`FigS1_C_SNR_S2C_d2 uncertainty.ipynb`](notebooks/figures/FigS1_C_SNR_S2C_d2%20uncertainty.ipynb) | [`ed_nonoverlap_d2err.py`](notebooks/exploration/revision/ed_nonoverlap_d2err.py) |
| [`FigS2_A_novel_familar.ipynb`](notebooks/figures/FigS2_A_novel_familar.ipynb) | [`ed2a_ppp8_day14_rebuild.py`](notebooks/exploration/revision/ed2a_ppp8_day14_rebuild.py) |
| [`FigS5_S6_rats.ipynb`](notebooks/figures/FigS5_S6_rats.ipynb) | [`ed_nonoverlap_rats_tiles.py`](notebooks/exploration/revision/ed_nonoverlap_rats_tiles.py) |
| [`Plot all figure_supplementary_final.ipynb`](notebooks/figures/Plot%20all%20figure_supplementary_final.ipynb) | [`m25_task_performance.py`](notebooks/exploration/revision/m25_task_performance.py) (its definition of task performance) |

## Other folders

| folder | what is in it |
|---|---|
| [`notebooks/figures_revision/`](notebooks/figures_revision) | The lab versions of the notebooks in [`../figure_scripts/`](../figure_scripts), reading the lab's tables instead of `../csv/`; `final_figs.py` is the lab's version. |
| [`notebooks/manuscript_code/`](notebooks/manuscript_code) | `FigS1_F_branchingratio_histogram.ipynb`, the raw-data version of submitted Extended Data 1f: it pools the per-session branching-ratio windows by state. `ed_nonoverlap_ed2a_ed1f.py` runs its cell 0 to rebuild Supplementary Fig 2e on non-overlapping windows. |
| [`src/criticality/`](src/criticality) | The lab's library: avalanche detection and fits (`av_analysis.py`), the MR branching-ratio estimator (`WiltingPriesemann.py`). |
| [`environment.yml`](environment.yml) | The conda environment (`criticality`). The notebooks also use seaborn, nelpy, neuro_py, mat73, pingouin, scikit-image and mpmath. |
