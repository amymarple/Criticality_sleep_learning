# Previous_Rv1 / sleep: code and data behind the sleep figures of the first submission

Chang, Chen, et al., *Distinct critical network regimes facilitate memory encoding and
consolidation*, the version first submitted to Nature Neuroscience
(`../../112596_0_merged_pdf_1150299_td8mg9 (1).pdf`; 5 main figures, Extended Data 1–10).


## Folders

| folder | what is in it |
|---|---|
| [`figure_scripts/`](figure_scripts) | The notebooks that draw the panels. Each reads `../csv/<notebook name>/` and saves to `../figure_output/`. |
| [`csv/`](csv) | The tables each notebook draws, one sub-folder per notebook. |
| [`analysis_scripts/`](analysis_scripts) | The code that made the tables from the recordings. `matlab/`: d2 of every session. `session_tables/`: notebooks that go through the recordings one session at a time and save one table per session (one row per d2 window with its MUA, SWR and barrage rates, branching ratio and ISI entropy); the figure tables are built from these. |
| `figure_output/` | Where the notebooks save the panels (created when they run). |


## Quick start

Draw Figure 2h from its table.

1. Create the Python environment of the repository ([`environment.yml`](../../environment.yml)), once:

   ```
   conda env create -f ../../environment.yml   # creates the environment "criticality"
   conda activate criticality
   ```

   Some notebooks also use nelpy, neuro_py, mat73, pingouin, scikit-image and mpmath.
   `FigS10_eplipsy.ipynb` also needs the branching-ratio estimator, fetched once from its authors with
   `python ../../methods/criticality/get_wiltingpriesemann.py`.

2. Run the notebook from inside `figure_scripts/` (the notebooks read `../csv/` and write `../figure_output/`):

   ```
   cd figure_scripts
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 Fig2_H_entropy.ipynb
   ```

   or start `jupyter lab` in `figure_scripts/`, open [`Fig2_H_entropy.ipynb`](figure_scripts/Fig2_H_entropy.ipynb) and
   run all cells.

3. After about 10 seconds the panel is in `figure_output/fig2_H_isi_entropy_by_d2_tile_pooled.svg`.

The revised figures are in [`../../Revision_Rv2`](../../Revision_Rv2). The behaviour figures are in
[`../behavior`](../behavior).

d2 everywhere in these figures is the distance to criticality of Sooter et al. 2024 (method and code:
[https://doi.org/10.6084/m9.figshare.25927081](https://doi.org/10.6084/m9.figshare.25927081)), with 40 ms bins, AR order 20 and 30 s windows stepped 2 s, computed in MATLAB by
`analysis_scripts/matlab/` with the d2 functions in [`../../methods/d2`](../../methods/d2). The lab's Python package
for avalanches and the branching ratio, which several notebooks import, is in
[`../../methods/criticality`](../../methods/criticality).

Each notebook in `figure_scripts/` is the lab's figure notebook, reading its tables from `../csv/`.

## Figure 1

| panel | figure notebook | data (`csv/<notebook>/`) | made by ([`analysis_scripts/`](analysis_scripts)) |
|---|---|---|---|
| 1a, 1b | schematics | - | - |
| 1c | [`Fig1_C_Example_plot_mouse.ipynb`](figure_scripts/Fig1_C_Example_plot_mouse.ipynb) | `states_NREM/WAKE/REM.csv`, `epochs.csv`: sleep states and epochs of the example session (PPP20 day 5); `mua_40ms.csv`: multiunit rate, all CA1 units, 40 ms bins; `d2_timeline_CA1.csv`: time-resolved d2 | [`matlab/`](analysis_scripts/matlab) (d2); states, epochs and spikes from the session recording |
| 1d | [`Fig1_D_histogram_S1_E.ipynb`](figure_scripts/Fig1_D_histogram_S1_E.ipynb) | `d2_windows_18sessions.csv`: every d2 window of the 18 sessions (6 mice, CA1) with its state (wake split into theta / non-theta) and task epoch | [`matlab/`](analysis_scripts/matlab) |
| 1e | [`Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb`](figure_scripts/Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb) | `d2_mua_pairs_18sessions.csv`: d2 and MUA per window, 18 sessions, with task-epoch labels | [`generate_pairs_df.ipynb`](analysis_scripts/session_tables/generate_pairs_df.ipynb) (from the [`matlab/`](analysis_scripts/matlab) output and the recordings) |

## Figure 2

| panel | figure notebook | data (`csv/<notebook>/`) | made by ([`analysis_scripts/`](analysis_scripts)) |
|---|---|---|---|
| 2a | [`Fig2_A_FigS2_B_FigS3_A.ipynb`](figure_scripts/Fig2_A_FigS2_B_FigS3_A.ipynb) | `d2_windows_pre_post.csv`: d2 windows of the pre-/post-task sleep sessions, with sleep epoch, state and SWR rate | [`generate_pre_post.ipynb`](analysis_scripts/session_tables/generate_pre_post.ipynb) |
| 2b | [`Fig2_B_Example_replay.ipynb`](figure_scripts/Fig2_B_Example_replay.ipynb) | example session PPP21 day 4: states / epochs; `replay_events_significant.csv`, `replay_events_nonsignificant.csv` (every decoded event: interval, fit, slope or p-values); the decoded posterior of the two plotted events (`replay_matrix_event1731.csv`, `nonreplay_matrix_event934_left.csv`); `lfp_ch255_around_plotted_events.csv` (ripple channel, ±0.3 s around the two events) | replay decoding: `.mat` files of the R_Project replay pipeline (exported by [`matlab/Getreplaypmatforpython.m`](analysis_scripts/matlab/Getreplaypmatforpython.m)); LFP from the recording |
| 2c | [`Fig2_C_Example_replayprobalit.ipynb`](figure_scripts/Fig2_C_Example_replayprobalit.ipynb) | example session PPP8 day 8: states / epochs; `d2_timeline_CA1.csv`; `mua_40ms.csv`, `rate_E_40ms.csv`, `rate_I_40ms.csv` (whole session, 40 ms); `ripple_peaks.csv`; `replay_times_left/right.csv`, `replay_intervals.csv` | [`matlab/`](analysis_scripts/matlab) (d2); replay decoding as for 2b |
| 2d | [`Fig2_D_E_replayprobality_SWR_replaynon_replay.ipynb`](figure_scripts/Fig2_D_E_replayprobality_SWR_replaynon_replay.ipynb) | `replay_prob_by_d2_sextile_per_file.csv` (one curve per session), `replay_prob_files.csv`, `replay_prob_lmm_rows.csv` (rows for the mixed model) | [`generate_replay_probablity_tile.ipynb`](analysis_scripts/session_tables/generate_replay_probablity_tile.ipynb) |
| 2e | [`Fig2_D_E_replayprobality_SWR_replaynon_replay.ipynb`](figure_scripts/Fig2_D_E_replayprobality_SWR_replaynon_replay.ipynb) (also drawn by [`Fig2_E_replay_non_replay.ipynb`](figure_scripts/Fig2_E_replay_non_replay.ipynb)) | `d2_by_replay_nonreplay_swr_rate_tiles.csv`: d2 (z within session) by tile of replay-SWR and non-replay-SWR rate | [`generate_replay_non_replay.ipynb`](analysis_scripts/session_tables/generate_replay_non_replay.ipynb) |
| 2f | [`Fig2_F_GLM.ipynb`](figure_scripts/Fig2_F_GLM.ipynb) | `glm_ablations.csv`, `glm_all_models.csv`, `glm_gain.csv`, `glm_files.csv`: per-session GLM ablation results | [`generate_GLM.ipynb`](analysis_scripts/session_tables/generate_GLM.ipynb) |
| 2g | [`Fig2_G_branchingratio.ipynb`](figure_scripts/Fig2_G_branchingratio.ipynb) | `br_change_per_event.csv`: branching-ratio change around every SWR and barrage and at shuffled times | [`generate_br_change_ripple.ipynb`](analysis_scripts/session_tables/generate_br_change_ripple.ipynb) (uses the MR estimator of [Wilting and Priesemann](https://github.com/jwilting/WiltingPriesemann2018), see [`methods`](../../methods)) |
| 2g inset | [`Fig2_G_Example_branchingratio.ipynb`](figure_scripts/Fig2_G_Example_branchingratio.ipynb) | example session PPP21 day 5: states; `ripple_events.csv`; `branching_ratio_windows_NREM.csv` (branching ratio per 10 s window) | [`generate_br_change_ripple.ipynb`](analysis_scripts/session_tables/generate_br_change_ripple.ipynb) |
| 2h | [`Fig2_H_entropy.ipynb`](figure_scripts/Fig2_H_entropy.ipynb) | `isi_entropy_by_swr_rate_sextile.csv`: ISI entropy per window with its SWR-rate sextile | [`generate_ISI_entropy_ripple.ipynb`](analysis_scripts/session_tables/generate_ISI_entropy_ripple.ipynb) |

## Figure 3

| panel | figure notebook | data ([`csv/Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar/`](csv/Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar)) | made by ([`analysis_scripts/`](analysis_scripts)) |
|---|---|---|---|
| 3e | [`Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar.ipynb`](figure_scripts/Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar.ipynb) (cell "Fig3_E") | `d2_by_barrage_rate_sextile.csv`: d2 windows of the barrage sessions (z within session) with their barrage-rate sextile | [`MUA_sleepstage.ipynb`](analysis_scripts/session_tables/MUA_sleepstage.ipynb) (from the [`matlab/`](analysis_scripts/matlab) d2) |
| 3f | same notebook ("Fig3 F") | `br_change_per_barrage.csv`: branching-ratio change around every barrage and at shuffled times | [`generate_branching_ratio_mouse.ipynb`](analysis_scripts/session_tables/generate_branching_ratio_mouse.ipynb) (uses the MR estimator of [Wilting and Priesemann](https://github.com/jwilting/WiltingPriesemann2018), see [`methods`](../../methods)) |
| 3g | same notebook ("Fig3_G") | `d2_windows_sncg.csv`: d2 windows of the CCK-silencing (Sncg) sessions, pre- and post-task sleep | [`generate_barr_manipulation.ipynb`](analysis_scripts/session_tables/generate_barr_manipulation.ipynb) (d2 from [`matlab/time_resolved_d2_across_state_error_bar.m`](analysis_scripts/matlab/time_resolved_d2_across_state_error_bar.m) run on CA1+CA2 with 80 ms bins and 60 s windows) |
| 3h | same notebook ("Fig3H") | `isi_entropy_windows_barrage.csv`: ISI entropy per window with barrage rate | [`generate_ISI_entropy_ripple.ipynb`](analysis_scripts/session_tables/generate_ISI_entropy_ripple.ipynb) |

## Extended Data

The `FigS*` notebook names follow an older numbering; the table gives the panels each draws.

| ED panel | figure notebook | data (`csv/<notebook>/`) | made by ([`analysis_scripts/`](analysis_scripts)) |
|---|---|---|---|
| 1a | schematic | - | - |
| 1b, 1d | [`FigS1_B_Explain variance_D_statetransition.ipynb`](figure_scripts/FigS1_B_Explain%20variance_D_statetransition.ipynb) | `explained_variance_by_window.csv`: variance explained by the AR fit, per window duration (`win…`) and animal; `d2_windows_state_transition.csv`: d2 windows around state transitions | [`matlab/coarse_grain_ddc.m`](analysis_scripts/matlab/coarse_grain_ddc.m); [`matlab/time_resolved_d2_state_transition.m`](analysis_scripts/matlab/time_resolved_d2_state_transition.m) |
| 1c, 2c | [`FigS1_C_SNR_S2C_d2 uncertainty.ipynb`](figure_scripts/FigS1_C_SNR_S2C_d2%20uncertainty.ipynb) | `d2_err_30s_60s_18sessions.csv`: d2 and its error bar per window, 30 s and 60 s windows, 18 sessions; `d2_err_by_swr_rate_sextile.csv`: d2 and its error by SWR-rate sextile | [`matlab/time_resolved_d2_across_state_error_bar.m`](analysis_scripts/matlab/time_resolved_d2_across_state_error_bar.m) (30 s and 60 s windows) |
| 1e | [`Fig1_D_histogram_S1_E.ipynb`](figure_scripts/Fig1_D_histogram_S1_E.ipynb) | see Figure 1 | see Figure 1 |
| 1f | [`FigS1_F_branchingratio_histogram.ipynb`](figure_scripts/FigS1_F_branchingratio_histogram.ipynb) | `branching_ratio_by_state.csv`: branching ratio of every window with its state, weighted | [`generate_branching_ratio_mouse.ipynb`](analysis_scripts/session_tables/generate_branching_ratio_mouse.ipynb) |
| 1g | [`Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb`](figure_scripts/Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb) | see Figure 1 | see Figure 1 |
| 2a | [`FigS2_A_novel_familar.ipynb`](figure_scripts/FigS2_A_novel_familar.ipynb) | `d2_rel_novel_familiar.csv`: d2 relative to pre-task sleep, novel and control (familiar) sessions | [`MUA_sleepstage.ipynb`](analysis_scripts/session_tables/MUA_sleepstage.ipynb) |
| 2b | [`Fig2_A_FigS2_B_FigS3_A.ipynb`](figure_scripts/Fig2_A_FigS2_B_FigS3_A.ipynb) | see Figure 2 | see Figure 2 |
| 2d–f | [`FigS2_D_E_F_low_avalanche_example_avalanche_sizeduradcc_.ipynb`](figure_scripts/FigS2_D_E_F_low_avalanche_example_avalanche_sizeduradcc_.ipynb) | example session PPP20 day 4: states, epochs, `ripple_events.csv`, SWR rate in 1 s bins (`swr_rate_1s_pre/mid/post.csv`), avalanches in low and high SWR-rate periods (`avalanches_low/high_swr_rate.csv`); group: `av_size_all_sleep.csv`, `av_duration_all_sleep.csv` (power-law fits), `dcc_all_all_sleep.csv` (DCC) | [`low_high_ripple_avalanche.ipynb`](analysis_scripts/session_tables/low_high_ripple_avalanche.ipynb), [`generate_avalanche.ipynb`](analysis_scripts/session_tables/generate_avalanche.ipynb) (use [`criticality/av_analysis.py`](../../methods/criticality/av_analysis.py)), on the low / high SWR-rate periods of [`low_high_ripples_controlwoavalanche.ipynb`](analysis_scripts/session_tables/low_high_ripples_controlwoavalanche.ipynb) (`data/l_h_ripples`; SIA periods and slow-wave slope from [`matlab/SIA__swslope_save.m`](analysis_scripts/matlab/SIA__swslope_save.m), [`matlab/detect_SIA.m`](analysis_scripts/matlab/detect_SIA.m)) |
| 3a | [`Fig2_A_FigS2_B_FigS3_A.ipynb`](figure_scripts/Fig2_A_FigS2_B_FigS3_A.ipynb) | see Figure 2 | see Figure 2 |
| 3b–d | [`FigS3_B_S3C_S3D_ripple_ablate.ipynb`](figure_scripts/FigS3_B_S3C_S3D_ripple_ablate.ipynb) | `d2_ripple_ablated_windows.csv`: d2 windows with the SWRs removed, with ISI entropy, SWR rate and sleep epoch | [`generate_ablate_ripple.ipynb`](analysis_scripts/session_tables/generate_ablate_ripple.ipynb) (from [`matlab/time_resolved_d2_across_state_ripple_ablate.m`](analysis_scripts/matlab/time_resolved_d2_across_state_ripple_ablate.m)) |
| 3e | [`Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar.ipynb`](figure_scripts/Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar.ipynb) | see Figure 3 | [`MUA_sleepstage_ablate_barr.ipynb`](analysis_scripts/session_tables/MUA_sleepstage_ablate_barr.ipynb) |
| 3f, 3g | [`FigS3_G_F_CA3_CA2_pyr_model_surpport_fig.ipynb`](figure_scripts/FigS3_G_F_CA3_CA2_pyr_model_surpport_fig.ipynb) | `rat_CA2_CA3_firing_by_window.csv`: per-unit firing rate of CA1, CA2 and CA3 (all, E, I) per d2 window, rat barrage sessions | [`MUA_sleepstage_barr_model_supple_ver2.ipynb`](analysis_scripts/session_tables/MUA_sleepstage_barr_model_supple_ver2.ipynb), [`barr_CA2_CA3_pooled_analysis.py`](analysis_scripts/session_tables/barr_CA2_CA3_pooled_analysis.py) |
| 5a | [`FigS5_A_Example_plot_rat.ipynb`](figure_scripts/FigS5_A_Example_plot_rat.ipynb) | example rat session HP18 day 36: states, epochs, `mua_40ms.csv`, `d2_timeline_CA1.csv` | [`matlab/`](analysis_scripts/matlab) (d2) |
| 5b–d, 6a–g | [`FigS5_S6_rats.ipynb`](figure_scripts/FigS5_S6_rats.ipynb) | rat versions of the Figure 1–3 tables: `rat_d2_windows.csv`, `rat_d2_windows_by_state.csv`, `rat_d2_mua_pairs.csv`, `rat_br_change_per_swr.csv`, `rat_br_change_per_event_nonoverlap.csv`, `rat_replay_nonreplay_tiles.csv`, `rat_d2_by_barrage_rate.csv`, `rat_isi_entropy_by_swr_rate.csv`, `rat_isi_entropy_by_barrage_rate.csv`; epoch tables of the sessions used (`rat_epoch_tables_ED5c/ED6a.csv`, `rat_sessions_presleep_first.csv`) | [`MUA_sleepstage_rat.ipynb`](analysis_scripts/session_tables/MUA_sleepstage_rat.ipynb) (per-session window tables), [`MUA_sleepstage_rat_replay.ipynb`](analysis_scripts/session_tables/MUA_sleepstage_rat_replay.ipynb) (replay / non-replay), [`generate_branching_ratio_rats.ipynb`](analysis_scripts/session_tables/generate_branching_ratio_rats.ipynb) (branching ratio) |
| 10 | [`FigS10_eplipsy.ipynb`](figure_scripts/FigS10_eplipsy.ipynb) | epilepsy session AYA9 day 28: `unit_groups.csv` (unit indices per region and cell type); `avalanches_epileptic_control.csv`; `branching_ratio_windows.csv`: branching ratio (MR estimator; CA1 population counts in 10 ms bins, k_max = 100, i.e. lags up to 1 s) of every 20 s window, stepped 2 s, in the epileptic (108–118 min) and control (14–24 min) epochs | the notebook itself ([`criticality/`](../../methods/criticality)) |
