# Previous_Rv1 — code and data behind the figures of the first submission

Chang, Chen, et al., *Distinct critical network regimes facilitate memory encoding and
consolidation* — the version first submitted to Nature Neuroscience
(`../112596_0_merged_pdf_1150299_td8mg9 (1).pdf`; 5 main figures, Extended Data 1–10).

Everything here reproduces a figure of that PDF. Nothing has been re-analysed.

## Folders

| folder | what is in it |
|---|---|
| `figure_scripts/` | The notebooks that draw the panels, one per original notebook, same names. Each reads its data from `../csv/<notebook name>/` and saves to `../figure_output/`. |
| `csv/` | The data each figure notebook reads, one sub-folder per notebook. |
| `csv_full/` | (not in git) Everything each loading block returned, before slimming; kept locally. |
| `analysis_scripts/` | The scripts that made those data from the recordings (they need the raw data on the lab servers). |
| `figure_output/` | Where the figure notebooks save their panels when you run them. |

## How the figure notebooks were made

Each notebook in `figure_scripts/` is the author's original notebook with one kind of change:
the code that loaded data from the lab servers (pickles, MATLAB `.mat` files, raw recordings
on `W:`, `V:`, `X:`, `Z:`) is replaced by reading a CSV. The original loading code is still in
the notebook, commented out, directly below the CSV read. All processing, statistics and
plotting code is unchanged. Save paths point at `../figure_output/`.

`analysis_scripts/make_figure_csvs.py` did this. For every notebook it ran the original from the
lab data, wrote the CSVs at the end of each loading block, wrote the copy, ran the copy, and
checked that (1) every line the original prints after loading (statistics, n, p-values) is
printed identically by the copy and (2) every figure file the copy saves is identical to the
original's (vector paths, and pixel by pixel for rasterised parts). The results are listed in
`analysis_scripts/make_figure_csvs_check.txt` (one line per build; the last line for a notebook is
the current one).

`csv/` is slimmed: bulk data that no panel uses is left out (the full exports stay in `csv_full/`,
which is not in git). What was removed:

- `Fig2_C_Example_replayprobalit`: the 40 ms rate series (MUA, pyramidal, interneuron) are cut to
  14 200–14 700 s around the plotted 3-min window (14 350.6–14 530.6 s, hard-coded in the notebook);
  they are smoothed over 30 s before being sampled there, so the panel is unchanged. One printout of
  that notebook (a whole-session mean E/I ratio, not in any panel) therefore differs and is not
  compared.
- `Fig2_G_Example_branchingratio`: `ripple_events.csv` keeps only `start`, `stop`, `peaks`.
- `Fig2_E_replay_non_replay` reads the same table as `Fig2_D_E_...` (panel 2e) instead of a copy.

- `FigS10_eplipsy`: no spike times (see Extended Data 10 below).

No CSV holds spike times. Kept whole on purpose: the window- and event-level tables (every d2 window,
every SWR), because the statistics (mixed models, paired tests, histograms) run on them; and the
session-wide 40 ms population rate of the two example sessions (`Fig1_C`, `FigS5_A`; binned counts,
not spikes), because the panel clips the trace at the 99.5th percentile of the whole session and the
notebook interpolates the session's 30 s-smoothed rate at every d2 window.

## Running

Python 3.11 with the lab environment (`criticality` conda env: numpy, pandas, matplotlib,
seaborn, scipy, statsmodels, nelpy, neuro_py). Open a notebook in `figure_scripts/` and run all
cells; the working directory must be `figure_scripts/` (Jupyter's default).

d2 everywhere in these figures is the distance to criticality of Sooter et al. 2024 (Euclidean
distance from the fitted AR(20) model to the β = 2 hyperplane), 40 ms bins, 30 s windows stepped
2 s, computed in MATLAB by `analysis_scripts/matlab/`. Those MATLAB scripts call
`myYuleWalker`, `getDbetas`, `getFixedPointDistance` and `myAutocorr` from the published code of
Sooter et al. 2024, which is not included here. `analysis_scripts/criticality/` is the lab's small
Python library (avalanches, branching ratio) that several analysis notebooks import.

Each copied notebook starts with one added cell: the standard imports, the CSV folder and a
`read_csv(name)` helper that reads the CSVs back exactly. A few exploration cells draw random
numbers (e.g. "randomly select an epoch" in `Fig2_C`); the windows actually plotted are hard-coded
by the author, so the panels do not depend on them.

## Figure 1

| panel | figure notebook | data (`csv/<notebook>/`) | made by (`analysis_scripts/`) |
|---|---|---|---|
| 1a, 1b | schematics | — | — |
| 1c | `Fig1_C_Example_plot_mouse.ipynb` | `states_NREM/WAKE/REM.csv`, `epochs.csv`: sleep states and epochs of the example session (PPP20 day 5); `mua_40ms.csv`: multiunit rate, all CA1 units, 40 ms bins; `d2_timeline_CA1.csv`: time-resolved d2 | `matlab/` (d2); states, epochs and spikes from the session recording |
| 1d | `Fig1_D_histogram_S1_E.ipynb` | `d2_windows_18sessions.csv`: every d2 window of the 18 sessions (6 mice, CA1) with its state (wake split into theta / non-theta) and task epoch | `matlab/` |
| 1e | `Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb` | `d2_mua_pairs_18sessions.csv`: d2 and MUA per window, 18 sessions, with task-epoch labels | `generate_pairs_df.ipynb` (from the `matlab/` output and the recordings) |

The same two notebooks also draw Extended Data 1 panels: `Fig1_D_histogram_S1_E` saves
`figS1_B_wake_sleep.svg` (wake vs sleep d2), and `Fig1_E_..._dynrange` saves
`figS1_D_MUA_d2_scatter.svg` and the dynamic-range panels `fig1_E_*_dyn_by_d2_minmax_norm.svg`.
These are listed again under Extended Data 1.

Figure 1 output files: `PPP20_day5_MUA_d2_example.svg`, `fig1_D_d2_density_sleep_wake.svg`,
`fig1_D_d2_density_sleep_wake_boxplots.svg`, `figS1_B_wake_sleep.svg`,
`fig1_E_NREM_vs_REM_dyn_by_d2_minmax_norm.svg`, `fig1_E_THETA_vs_nonTHETA_dyn_by_d2_minmax_norm.svg`,
`figS1_D_MUA_d2_scatter.svg`.

## Figure 2

| panel | figure notebook | data (`csv/<notebook>/`) | made by (`analysis_scripts/`) |
|---|---|---|---|
| 2a | `Fig2_A_FigS2_B_FigS3_A.ipynb` | `d2_windows_pre_post.csv`: d2 windows of the pre-/post-task sleep sessions, with sleep epoch, state and SWR rate | `generate_pre_post.ipynb` |
| 2b | `Fig2_B_Example_replay.ipynb` | example session PPP21 day 4: states / epochs; `replay_events_significant.csv`, `replay_events_nonsignificant.csv` (every decoded event: interval, fit, slope or p-values); the decoded posterior of the two plotted events (`replay_matrix_event1731.csv`, `nonreplay_matrix_event934_left.csv`); `lfp_ch255_around_plotted_events.csv` (ripple channel, ±0.3 s around the two events) | replay decoding: `.mat` files of the R_Project replay pipeline (exported by `matlab/Getreplaypmatforpython.m`); LFP from the recording |
| 2c | `Fig2_C_Example_replayprobalit.ipynb` | example session PPP8 day 8: states / epochs; `d2_timeline_CA1.csv`; `mua_40ms.csv`, `rate_E_40ms.csv`, `rate_I_40ms.csv` (whole session, 40 ms); `ripple_peaks.csv`; `replay_times_left/right.csv`, `replay_intervals.csv` | `matlab/` (d2); replay decoding as for 2b |
| 2d | `Fig2_D_E_replayprobality_SWR_replaynon_replay.ipynb` | `replay_prob_by_d2_sextile_per_file.csv` (one curve per session), `replay_prob_files.csv`, `replay_prob_lmm_rows.csv` (rows for the mixed model) | `generate_replay_probablity_tile.ipynb` |
| 2e | `Fig2_D_E_replayprobality_SWR_replaynon_replay.ipynb` (also drawn by `Fig2_E_replay_non_replay.ipynb`) | `d2_by_replay_nonreplay_swr_rate_tiles.csv`: d2 (z within session) by tile of replay-SWR and non-replay-SWR rate | `generate_replay_non_replay.ipynb` |
| 2f | `Fig2_F_GLM.ipynb` | `glm_ablations.csv`, `glm_all_models.csv`, `glm_gain.csv`, `glm_files.csv`: per-session GLM ablation results | `generate_GLM.ipynb` (`MUA_sleepstage_mouse_replay_GLM.ipynb` builds its input table) |
| 2g | `Fig2_G_branchingratio.ipynb` | `br_change_per_event.csv`: branching-ratio change around every SWR and barrage and at shuffled times | `generate_br_change_ripple.ipynb` (uses `criticality/WiltingPriesemann.py`) |
| 2g inset | `Fig2_G_Example_branchingratio.ipynb` | example session PPP21 day 5: states; `ripple_events.csv`; `branching_ratio_windows_NREM.csv` (branching ratio per 10 s window) | `generate_br_change_ripple.ipynb` |
| 2h | `Fig2_H_entropy.ipynb` | `isi_entropy_by_swr_rate_sextile.csv`: ISI entropy per window with its SWR-rate sextile | `generate_ISI_entropy_ripple.ipynb` |

`Fig2_A_FigS2_B_FigS3_A` also draws Extended Data panels: `fig2A_S2B_pre_vs_postsleep_perfile_paired.svg`
and the SWR-rate-matched pre/post comparison `fig3S_A_d2_pre_post_matched_ripple.svg`.

For the example panels (2b, 2c, 2g inset) the raw recordings are far too large to ship (the LFP of
one session is tens of GB). The CSVs therefore hold everything the notebook uses except bulk data
it does not draw: for 2b, every decoded event's interval and statistics, but the posterior matrix and
the LFP only for the two plotted events, arranged so that the author's selection code
(`all_sig_replays[1731]`, `non_sig_matrices[934]`) runs unchanged.

## Figure 3

| panel | figure notebook | data (`csv/Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar/`) | made by (`analysis_scripts/`) |
|---|---|---|---|
| 3a–d | network model | — | not in this repository yet |
| 3e | `Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar.ipynb` (cell "Fig3_E") | `d2_by_barrage_rate_sextile.csv`: d2 windows of the barrage sessions (z within session) with their barrage-rate sextile | `MUA_sleepstage.ipynb` (from the `matlab/` d2) |
| 3f | same notebook ("Fig3 F") | `br_change_per_barrage.csv`: branching-ratio change around every barrage and at shuffled times | `branching_ratio.ipynb` (uses `criticality/WiltingPriesemann.py`) |
| 3g | same notebook ("Fig3_G") | `d2_windows_sncg.csv`: d2 windows of the CCK-silencing (Sncg) sessions, pre- and post-task sleep | `generate_barr_manipulation.ipynb` (d2 from `matlab/time_resolved_d2_across_state_error_bar.m` run on CA1+CA2 with 80 ms bins and 60 s windows) |
| 3h | same notebook ("Fig3H") | `isi_entropy_windows_barrage.csv`: ISI entropy per window with barrage rate | `generate_ISI_entropy_ripple.ipynb` |

The same notebook also draws an Extended Data panel, d2 with the barrages removed against barrage
rate (`S3_E_pooled_D2Z_vs_barrage_tile_equalwidth.svg`; data `d2_barrage_ablated_windows.csv`, made by
`MUA_sleepstage_ablate_barr.ipynb` from `matlab/time_resolved_d2_across_state_barrage_ablate.m`).
Panel 3g is saved as `fig5_F_barrage_manipulation_box_d2_pooled.svg` (its name from an earlier
figure numbering).

The panel-h function `plot_tiles_line` returns a figure `fig_rip` that the notebook no longer draws;
the original ran only because `fig_rip` was left in the Jupyter kernel from an earlier run. The copy
defines `fig_rip = None` in its first cell, with a comment.

## Extended Data

Panel letters as in the PDF. The `FigS*` notebook names follow an older numbering (`FigS3_B` draws
ED 3b–d but saves `S2_B_...`), so go by this table, not by the file names.

| ED panel | figure notebook | data (`csv/<notebook>/`) | made by (`analysis_scripts/`) |
|---|---|---|---|
| 1a | schematic | — | — |
| 1b, 1d | `FigS1_B_Explain variance_D_statetransition.ipynb` | `explained_variance_by_window.csv`: variance explained by the AR fit, per window duration (`win…`) and animal; `d2_windows_state_transition.csv`: d2 windows around state transitions | `matlab/coarse_grain_ddc.m` (from the lab's R_Project repository; writes `20250324timebin_grainedmouse.mat`); `matlab/time_resolved_d2_state_transition.m` |
| 1c, 2c | `FigS1_C_SNR_S2C_d2 uncertainty.ipynb` | `d2_err_30s_60s_18sessions.csv`: d2 and its error bar per window, 30 s and 60 s windows, 18 sessions; `d2_err_by_swr_rate_sextile.csv`: d2 and its error by SWR-rate sextile | `matlab/time_resolved_d2_across_state_error_bar.m` (30 s). The 60 s files (`*_error_bar_60sec.mat`) are presumably the same worker with `window_size = 60`; no script in the source repository writes that file name |
| 1e | `Fig1_D_histogram_S1_E.ipynb` | see Figure 1 | see Figure 1 |
| 1f | `FigS1_F_branchingratio_histogram.ipynb` | `branching_ratio_by_state.csv`: branching ratio of every window with its state, weighted | `generate_branching_ratio_mouse.ipynb` (writes `branching_ratio_allstates_win10s_step2s_guard5s_*.pkl`; the notebook read them from `data/cross_state/branching_ratio_1`) |
| 1g | `Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb` | see Figure 1 | see Figure 1 |
| 2a | `FigS2_A_novel_familar.ipynb` | `d2_rel_novel_familiar.csv`: d2 relative to pre-task sleep, novel and control (familiar) sessions | `MUA_sleepstage.ipynb`: a commented-out cell saves `d2_rel` per session to `data/novel_familiar/novel` (folder edited by hand for the control sessions) |
| 2b | `Fig2_A_FigS2_B_FigS3_A.ipynb` | see Figure 2 | see Figure 2 |
| 2d–f | `FigS2_D_E_F_low_avalanche_example_avalanche_sizeduradcc_.ipynb` | example session PPP20 day 4: states, epochs, `ripple_events.csv`, SWR rate in 1 s bins (`swr_rate_1s_pre/mid/post.csv`), avalanches in low and high SWR-rate periods (`avalanches_low/high_swr_rate.csv`); group: `av_size_all_sleep.csv`, `av_duration_all_sleep.csv` (power-law fits), `dcc_all_all_sleep.csv` (DCC) | `low_high_ripple_avalanche.ipynb`, `generate_avalanche.ipynb` (use `criticality/av_analysis.py`) |
| 3a | `Fig2_A_FigS2_B_FigS3_A.ipynb` | see Figure 2 | see Figure 2 |
| 3b–d | `FigS3_B_S3C_S3D_ripple_ablate.ipynb` | `d2_ripple_ablated_windows.csv`: d2 windows with the SWRs removed, with ISI entropy, SWR rate and sleep epoch | `generate_ablate_ripple.ipynb` (from `matlab/time_resolved_d2_across_state_ripple_ablate.m`) |
| 3e | `Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar.ipynb` | see Figure 3 | `MUA_sleepstage_ablate_barr.ipynb` |
| 3f, 3g | `FigS3_G_F_CA3_CA2_pyr_model_surpport_fig.ipynb` | `rat_CA2_CA3_firing_by_window.csv`: per-unit firing rate of CA1, CA2 and CA3 (all, E, I) per d2 window, rat barrage sessions | `MUA_sleepstage_barr_model_supple_ver2.ipynb`, `barr_CA2_CA3_pooled_analysis.py` |
| 5a | `FigS5_A_Example_plot_rat.ipynb` | example rat session HP18 day 36: states, epochs, `mua_40ms.csv`, `d2_timeline_CA1.csv` | `matlab/` (d2) |
| 5b–d, 6a–g | `FigS5_S6_rats.ipynb` | rat versions of the Figure 1–3 tables: `rat_d2_windows.csv`, `rat_d2_windows_by_state.csv`, `rat_d2_mua_pairs.csv`, `rat_br_change_per_swr.csv`, `rat_br_change_per_event_nonoverlap.csv`, `rat_replay_nonreplay_tiles.csv`, `rat_d2_by_barrage_rate.csv`, `rat_isi_entropy_by_swr_rate.csv`, `rat_isi_entropy_by_barrage_rate.csv`; epoch tables of the sessions used (`rat_epoch_tables_ED5c/ED6a.csv`, `rat_sessions_presleep_first.csv`) | `MUA_sleepstage_rat.ipynb` (per-session window tables; its save cells are commented out), `MUA_sleepstage_rat_replay.ipynb` (replay / non-replay), `generate_branching_ratio_rats.ipynb` (branching ratio) |
| 10 | `FigS10_eplipsy.ipynb` | epilepsy session AYA9 day 28: `unit_groups.csv` (unit indices per region and cell type); `avalanches_epileptic_control.csv`; `branching_ratio_windows.csv`: branching ratio (MR estimator; CA1 population counts in 10 ms bins, k_max = 100, i.e. lags up to 1 s) of every 20 s window, stepped 2 s, in the epileptic (108–118 min) and control (14–24 min) epochs | computed in the notebook itself from the spike times (`criticality/`) |

Output files:

- ED 1: `figS1B_explain_variance_violin_diff_window.svg`, `S1_D_nrem_d2_transition_snippets.svg`,
  `S2_C_CA1_snr_box_30s_vs_60s.svg`, `S1_F_branching_ratio_distribution_split.svg`.
- ED 2: `figS2_A_novel_NREM_...svg`, `figS2_A_control_NREM_...svg`, `S2_C_ripple_rate_by_d2_tile_pooled_std.svg`,
  `figS2_E_*_PPP20_day4_CA1_*.svg`, `S2_E_low_high_ripple_avalanche_fit_bar_equalsample.svg`, `S2_F_low_high_ripple_dcc.svg`.
- ED 3: `S2_B_ripple_ablate_d2z_vs_sleepepoch_paired.svg`, `S3_C_..._sextile.svg`, `S3_D_..._sextile.svg`,
  `S3_F_pooled_CA3_E_FR_per_unit_vs_ripple_tiles.svg`, `S3_G_model_pooled_CA2_E_FR_per_unit_vs_barrage_tiles.svg`.
- ED 5, 6: `HP18_hp18_day36_20250510_MUA_d2_example.svg`, `figS5_*.svg`, `figS6*.svg`, `fig2_A_pre_vs_postsleep_perfile_paired.svg` (rats).
- ED 10: `branching_ratio_comparison_epilepsy.svg`.

**ED 10 ships no spike times.** The notebook starts from the spike times of the CA1 units. The CSVs
instead hold what it computes from them: avalanche sizes and durations, and the branching ratio of
every window. The cells that read and bin the spikes are commented out in the copy, as for every
loading block. The epochs are hand-set in the notebook and stay as written. The avalanche plots in
that notebook are drawn on screen only; the saved panel and the statistics use the branching ratios.

## Not here yet

Main Figure 3a–d and Extended Data 4 (network model), main Figures 4 and 5, and Extended Data 7–9
(behaviour, cross-regional coordination) were made outside the source repository; their code is
coming from the colleagues who made them.
