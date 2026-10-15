# Revision_Rv2: code and data behind the revised figures

Chang, Chen et al., *Distinct critical network regimes facilitate memory encoding and consolidation* (Nature
Neuroscience, revision). This folder holds every panel of the revised **Figures 1-3** and **Supplementary Figures
1-4 and 6-8**: the notebook that draws each panel, the tables it is drawn from (CSV), and the analysis code that
made those tables from the recordings. The figures of the first submission are in
[`../Previous_Rv1`](../Previous_Rv1); several revised panels are those figures redrawn on non-overlapping windows.

**d2** is the distance to criticality of Sooter et al. 2024 (method and code: [https://doi.org/10.6084/m9.figshare.25927081](https://doi.org/10.6084/m9.figshare.25927081)),
computed with 40 ms bins, AR order 20 and 30 s windows. The figures label it just "d2". **Non-overlapping windows**:
within a session, all states in time order, a window is kept only if its centre is at least one window length after
the last kept window.

Abbreviations: SWR, sharp-wave ripple; BARR, barrage; BR, branching ratio; MUA, multi-unit activity
(population firing rate); ISI, inter-spike interval; E/I, excitation / inhibition balance.

## Contents

- [Quick start](#quick-start): draw one panel in a few minutes
- [What each figure notebook needs](#what-each-figure-notebook-needs)
- [Folders](#folders)
- [From the recordings to the figures](#from-the-recordings-to-the-figures)
- Panel by panel: [Figure 1](#figure-1), [Figure 2](#figure-2), [Figure 3](#figure-3),
  [Supplementary Figures 1](#supplementary-figure-1-the-three-measures), [2](#supplementary-figure-2-d2-robustness-and-the-measures-by-state),
  [3](#supplementary-figure-3-task-novelty-and-avalanches), [4](#supplementary-figure-4-swr-controls),
  [6](#supplementary-figure-6-d2-as-an-indicator-of-the-nrem-ca1-regime), [7](#supplementary-figure-7-rats),
  [8](#supplementary-figure-8-rats-swrs-replay-and-barrs)
- Every analysis script: [analysis_scripts/README.md](analysis_scripts/README.md); method details by topic:
  [analysis_scripts/METHODS.md](analysis_scripts/METHODS.md)

## Quick start

Draw Figure 1e from its table.

1. Create the Python environment (once):

   ```
   conda env create -f analysis_scripts/environment.yml      # creates the environment "criticality"
   conda activate criticality
   pip install seaborn
   ```

2. Run the notebook from inside `figure_scripts/` (the notebooks read `../csv/` and write `../figure_output/`):

   ```
   cd figure_scripts
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 Fig1_E_d2_histogram_nonoverlap.ipynb
   ```

   or start `jupyter lab` in `figure_scripts/`, open
   [`Fig1_E_d2_histogram_nonoverlap.ipynb`](figure_scripts/Fig1_E_d2_histogram_nonoverlap.ipynb) and run all cells.

3. After about 10 seconds the panel is in `figure_output/Fig1/Fig1_E_d2_histogram_nonoverlap.svg` and `.png`: the
   d2 distribution of every non-overlapping window by state. The notebook prints the statistics given in the figure
   legend. The panels shipped in [`figure_output/`](figure_output) are what every notebook produces.

## What each figure notebook needs

Every notebook in [`figure_scripts/`](figure_scripts) reads only its own folder `csv/<notebook>/`, plus the items
below.

| notebook | panels | also needs |
|---|---|---|
| [`Fig1_AB_hypothesis_example`](figure_scripts/Fig1_AB_hypothesis_example.ipynb) | Fig 1a, b | - |
| [`Fig1_CD_manifold_structure_index`](figure_scripts/Fig1_CD_manifold_structure_index.ipynb) | Fig 1c, d | - |
| [`Fig1_E_d2_histogram_nonoverlap`](figure_scripts/Fig1_E_d2_histogram_nonoverlap.ipynb) | Fig 1e | - |
| [`Fig2_A_prepost`](figure_scripts/Fig2_A_prepost.ipynb) | Fig 2a | - |
| [`Fig2_B_replay_examples`](figure_scripts/Fig2_B_replay_examples.ipynb) | Fig 2b | `../Previous_Rv1` |
| [`Fig2_CDE_replay_bgd2`](figure_scripts/Fig2_CDE_replay_bgd2.ipynb) | Fig 2c-e | - |
| [`Fig2_FG_br_isi`](figure_scripts/Fig2_FG_br_isi.ipynb) | Fig 2f, g | - |
| [`Fig3_ADG_from_submitted`](figure_scripts/Fig3_ADG_from_submitted.ipynb) | Fig 3a-d, g (example) | Inkscape |
| [`Fig3_EFGH_barrage_bgd2`](figure_scripts/Fig3_EFGH_barrage_bgd2.ipynb) | Fig 3e-h | - |
| [`FigS1_AF_methods`](figure_scripts/FigS1_AF_methods.ipynb) | Supp 1 | Inkscape |
| [`FigS2_AF_robustness`](figure_scripts/FigS2_AF_robustness.ipynb) | Supp 2 | `../Previous_Rv1`, first-submission packages |
| [`FigS3_AG_task_novelty_avalanches`](figure_scripts/FigS3_AG_task_novelty_avalanches.ipynb) | Supp 3 | Inkscape |
| [`FigS4_AF_swr_controls`](figure_scripts/FigS4_AF_swr_controls.ipynb) | Supp 4 | `../Previous_Rv1`, first-submission packages |
| [`FigS6_AC_regime_indicator`](figure_scripts/FigS6_AC_regime_indicator.ipynb) | Supp 6 | - |
| [`FigS7_AD_rats`](figure_scripts/FigS7_AD_rats.ipynb) | Supp 7 | `../Previous_Rv1`, first-submission packages, Inkscape |
| [`FigS8_AF_rats`](figure_scripts/FigS8_AF_rats.ipynb) | Supp 8 | Inkscape |

- `../Previous_Rv1`: the first-submission folder of this repository, next to this one (some tables are read from it).
- First-submission packages: nelpy, neuro_py, mat73, mpmath, scikit-image and pingouin. These notebooks run
  first-submission notebooks from [`figure_scripts/rv1_disjoint/`](figure_scripts/rv1_disjoint).
- [Inkscape](https://inkscape.org) 1.x. Set the environment variable `INKSCAPE` if it is not at
  `C:\Program Files\Inkscape\bin\inkscape.exe`.

## Folders

| folder | what is in it |
|---|---|
| [`figure_scripts/`](figure_scripts) | The notebooks that draw the panels, one per figure theme. `*_drawing.py`: drawing functions taken from analysis scripts, so that the notebooks import no analysis code (each names its source). [`rv1_disjoint/`](figure_scripts/rv1_disjoint): first-submission notebooks run on non-overlapping windows for some supplementary panels. [`final_figs.py`](figure_scripts/final_figs.py): shared helpers (print styles, saving, stars, crops). |
| [`csv/<notebook>/`](csv) | The tables each notebook draws: one row per session, window, bin or tile, with the statistics printed on the panels. `assets/`: schematics and source figures. |
| [`figure_output/`](figure_output) | The panels, as the notebooks save them. |
| [`analysis_scripts/`](analysis_scripts) | The pipeline from the recordings to the tables. Index: [analysis_scripts/README.md](analysis_scripts/README.md); methods: [METHODS.md](analysis_scripts/METHODS.md). |

The notebooks in `figure_scripts/` are the lab's figure notebooks
([`analysis_scripts/notebooks/figures_revision/`](analysis_scripts/notebooks/figures_revision)), reading their
tables from `../csv/`.

## From the recordings to the figures

Rebuilding the tables needs the raw sessions. The scripts expect them at their paths on the
lab servers: `W:\data\PPP` (mice), `X:\data\Barrage` (barrage and CCK-silencing mice), `V:\data\hpc_ctx_project`,
`S:\data\latentseq`, `U:\data\HMC` and `Z:\Data\AYAold` (rats). It also needs MATLAB with the d2 code of Sooter et
al. 2024 ([https://doi.org/10.6084/m9.figshare.25927081](https://doi.org/10.6084/m9.figshare.25927081)), the replay decoding of the lab's R_Project repository, the environment above plus umap-learn,
[structure_index](https://github.com/PridaLab/structure_index) (Fig 1c, d) and the Sooter et al. code at
[https://github.com/sam-sooter/prox_crit_toolkit](https://github.com/sam-sooter/prox_crit_toolkit) (d2 order sweeps).

1. **d2 per session (MATLAB).**
   [`d2_loopthrough_data.m`](analysis_scripts/notebooks/exploration/d2_loopthrough_data.m) runs the d2 programs of
   [`notebooks/helper/`](analysis_scripts/notebooks/helper) in every session folder. Each writes
   `<basepath>\complex_system\<day>_CA1_D2_time_resolved*.mat`: d2 in 30 s windows every 2 s, by state, for the
   full window, with ripples removed, with barrages removed, and across state transitions.
2. **Per-session tables (first-submission notebooks).** The notebooks in
   [`notebooks/exploration/`](analysis_scripts/notebooks/exploration) read each session folder (spikes, sleep states,
   ripples, barrages, replay decoding, the d2 `.mat` files) and write per-session tables to `data/<experiment>/`.
3. **Revision analyses.** The scripts in
   [`notebooks/exploration/revision/`](analysis_scripts/notebooks/exploration/revision) read the recordings and the
   step 2 tables and write `data/revision/log/*.csv`: non-overlapping windows, the ripple- and barrage-excised
   measures, the mixed models and the state-space features.
   [`ed_nonoverlap.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap.py) runs the first-submission
   figure notebooks on non-overlapping windows.
4. **Figure notebooks.** The lab's figure notebooks draw the panels from those tables; the tables they draw are the
   CSVs in `csv/`.

Each script states its inputs and outputs in its header. The tables below give each panel's chain; the cell number
after a notebook is the cell that draws the panel.

## Figure 1

| panel | figure notebook (cell) | data (`csv/<notebook>/`) | made by (`analysis_scripts/notebooks/exploration/revision/` unless stated) |
|---|---|---|---|
| a | [`Fig1_AB_hypothesis_example`](figure_scripts/Fig1_AB_hypothesis_example.ipynb) (3) | `assets/fig1_A_Hypothesis_ver7r.svg` | schematic |
| b | [`Fig1_AB_hypothesis_example`](figure_scripts/Fig1_AB_hypothesis_example.ipynb) (5); drawing: [`fig1c_example_traces_drawing.draw_figure`](figure_scripts/fig1c_example_traces_drawing.py) | `fig1b_windows.csv`: per 30 s window of PPP20 day 5 (5 200–10 000 s): d2, E/I balance, synchrony, theta power, SWR rate, state. `fig1b_states.csv`: corrected sleep states. `fig1b_mua_40ms.csv`, `fig1b_mua_cap.csv`: population rate (all CA1 units, 40 ms) and its session 99.5th percentile | [`fig1c_example_traces.py`](analysis_scripts/notebooks/exploration/revision/fig1c_example_traces.py) ← W:\data\PPP\PPP20\day5 (spikes, sleep states, ripples), [`fig1_theta_ratio.py`](analysis_scripts/notebooks/exploration/revision/fig1_theta_ratio.py), the d2 .mat (`*_error_bar_forplot.mat`), the state corrections of [`notebooks/figures/Fig1_C_Example_plot_mouse.ipynb`](analysis_scripts/notebooks/figures/Fig1_C_Example_plot_mouse.ipynb) |
| c | [`Fig1_CD_manifold_structure_index`](figure_scripts/Fig1_CD_manifold_structure_index.ipynb) (3) → [`figure_scripts/fig1_state_umap_all18.ipynb`](figure_scripts/fig1_state_umap_all18.ipynb) | `fig1_state_umap_all18_10f.csv`: UMAP coordinates of every non-overlapping 30 s window wholly in one state (18 sessions), with state, d2 (z within session) and SWR rate | [`fig1_state_space_umap_session.py`](analysis_scripts/notebooks/exploration/revision/fig1_state_space_umap_session.py) ← [`fig1_state_space_export.py`](analysis_scripts/notebooks/exploration/revision/fig1_state_space_export.py) (10 window features) ← [`run_fig1_metrics.py`](analysis_scripts/notebooks/exploration/revision/run_fig1_metrics.py) / [`fig1_metrics.py`](analysis_scripts/notebooks/exploration/revision/fig1_metrics.py), [`fig1_emg_lfp.py`](analysis_scripts/notebooks/exploration/revision/fig1_emg_lfp.py), [`fig1_theta_ratio.py`](analysis_scripts/notebooks/exploration/revision/fig1_theta_ratio.py) ← W: spikes, LFP, states, ripples; d2 .mat |
| d | [`Fig1_CD_manifold_structure_index`](figure_scripts/Fig1_CD_manifold_structure_index.ipynb) (5) | `fig1_si_sessions.csv`: structure index of d2 per session and its time-shift null | [`fig1_si_sessions.py`](analysis_scripts/notebooks/exploration/revision/fig1_si_sessions.py) (structure_index) ← same features as c |
| e | [`Fig1_E_d2_histogram_nonoverlap`](figure_scripts/Fig1_E_d2_histogram_nonoverlap.ipynb) (3) | `fig1e_d2_windows_disjoint.csv`: every non-overlapping d2 window of the 18 sessions with its state | the lab's `Fig1_E` notebook: `Previous_Rv1` [`Fig1_D_histogram_S1_E`](../Previous_Rv1/figure_scripts/Fig1_D_histogram_S1_E.ipynb) run by [`ed_nonoverlap.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap.py) (its `d2_windows_18sessions.csv` thinned) ← d2 .mat |

## Figure 2

| panel | figure notebook (cell) | data | made by |
|---|---|---|---|
| a (top) | [`Fig2_A_prepost`](figure_scripts/Fig2_A_prepost.ipynb) (3) | `assets/fig2a_prepost_schematic_source.png` | illustration |
| a | [`Fig2_A_prepost`](figure_scripts/Fig2_A_prepost.ipynb) (5) | `fig2a_prepost_sessions_disjoint.csv`: d2 per session, presleep vs postsleep, NREM, non-overlapping windows | the lab's `Fig2_A` notebook: `Previous_Rv1` [`Fig2_A_FigS2_B_FigS3_A`](../Previous_Rv1/figure_scripts/Fig2_A_FigS2_B_FigS3_A.ipynb) run by [`ed_nonoverlap.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap.py) ← [`notebooks/exploration/generate_pre_post.ipynb`](analysis_scripts/notebooks/exploration/generate_pre_post.ipynb) ← d2 .mat, ripples, states |
| b (events) | [`Fig2_B_replay_examples`](figure_scripts/Fig2_B_replay_examples.ipynb) (3); ridge fit: `fig2b_replay_fit_drawing` | `replay_1731_matrix.csv`, `nonreplay_934_left_matrix.csv`: decoded posteriors; `event_intervals.csv`; `lfp_around_events.csv` (ripple channel, ±0.15 s) | the replay example of the first submission (`Previous_Rv1` [`Fig2_B_Example_replay`](../Previous_Rv1/figure_scripts/Fig2_B_Example_replay.ipynb)) ← replay decoding ([`notebooks/helper/Getreplaypmatforpython.m`](analysis_scripts/notebooks/helper/Getreplaypmatforpython.m)), LFP |
| b (curve) | [`Fig2_B_replay_examples`](figure_scripts/Fig2_B_replay_examples.ipynb) (5) | `fig2c_example_excised.csv`: PPP8 day 8, 180 s: full and ripple-excised d2, replay probability | [`fig2c_example_excised.py`](analysis_scripts/notebooks/exploration/revision/fig2c_example_excised.py) ← `*_D2_time_resolved.mat`, `*_rippleablate.mat`, [`notebooks/figures/Fig2_C_Example_replayprobalit.ipynb`](analysis_scripts/notebooks/figures/Fig2_C_Example_replayprobalit.ipynb) |
| c | [`Fig2_CDE_replay_bgd2`](figure_scripts/Fig2_CDE_replay_bgd2.ipynb) (3); [`fig2_new_drawing.panel_replay_rates`](figure_scripts/fig2_new_drawing.py) | `excised_br_entropy_windows.csv`: per non-overlapping 30 s NREM window of the 9 decoded sessions: ripple-excised d2, SWR rate, replay probability | [`excised_br_entropy.py`](analysis_scripts/notebooks/exploration/revision/excised_br_entropy.py) ← W: spikes, ripples, the replay decoding |
| d | [`Fig2_CDE_replay_bgd2`](figure_scripts/Fig2_CDE_replay_bgd2.ipynb) (5); `panel_ladder` | `fig2f_swr_property_ladder.csv`: d2's replay coefficient per session, alone and with SWR rate, amplitude, duration | [`fig2f_swr_property_ladder.py`](analysis_scripts/notebooks/exploration/revision/fig2f_swr_property_ladder.py) ← [`r35_binomial_replay.py`](analysis_scripts/notebooks/exploration/revision/r35_binomial_replay.py), [`r35_swr_property_control.py`](analysis_scripts/notebooks/exploration/revision/r35_swr_property_control.py) |
| e | [`Fig2_CDE_replay_bgd2`](figure_scripts/Fig2_CDE_replay_bgd2.ipynb) (7); [`fig2f_standalone_drawing.draw`](figure_scripts/fig2f_standalone_drawing.py) | `fig2e_pooled.csv`, `fig2e_per_session.csv`: deviance explained lost on dropping each predictor (pooled with CI, per session); `fig2e_lmm_pvalues.csv`, `fig2e_contrasts.csv`: the mixed model's tests | [`fig2f_standalone.compute()`](analysis_scripts/notebooks/exploration/revision/fig2f_standalone.py) ← [`r35_replay_rate_model.py`](analysis_scripts/notebooks/exploration/revision/r35_replay_rate_model.py), [`r35_replay_lmm.py`](analysis_scripts/notebooks/exploration/revision/r35_replay_lmm.py) ← [`r35_extend_replay_sessions.py`](analysis_scripts/notebooks/exploration/revision/r35_extend_replay_sessions.py), [`r35_replay_full_covariates.py`](analysis_scripts/notebooks/exploration/revision/r35_replay_full_covariates.py) |
| f | [`Fig2_FG_br_isi`](figure_scripts/Fig2_FG_br_isi.ipynb) (3); `panel_psth` | `fig2g_psth_curves_PPP21_day5.csv`: ripple-excised BR around SWRs vs shuffle; `fig2g_psth_nine_sessions.csv`: per-session ΔBR | [`fig2g_psth_excised.py`](analysis_scripts/notebooks/exploration/revision/fig2g_psth_excised.py) ← [`notebooks/exploration/generate_br_change_ripple.ipynb`](analysis_scripts/notebooks/exploration/generate_br_change_ripple.ipynb) (MR estimator), W: spikes, ripples |
| g | [`Fig2_FG_br_isi`](figure_scripts/Fig2_FG_br_isi.ipynb) (5); `sextile_panel` | `fig2_agh_nine_sessions.csv`: ISI entropy (full and ripple-excised) by SWR-rate sextile, per session | [`fig2_agh_excised.py`](analysis_scripts/notebooks/exploration/revision/fig2_agh_excised.py) |

## Figure 3

| panel | figure notebook (cell) | data | made by |
|---|---|---|---|
| a–d, g (example) | [`Fig3_ADG_from_submitted`](figure_scripts/Fig3_ADG_from_submitted.ipynb) (2) | `assets/Figure3_submitted_2026-03-12.svg` | network model (schematic) |
| e | [`Fig3_EFGH_barrage_bgd2`](figure_scripts/Fig3_EFGH_barrage_bgd2.ipynb) (2); [`fig3_background_repro_drawing.draw_main`](figure_scripts/fig3_background_repro_drawing.py) | `fig3e_sextiles.csv`: d2 (full window and ripples + BARRs excised) by BARR-rate sextile, mean ± SEM over sessions; `fig3e_session_rho.csv`, `fig3e_stats.csv` | [`fig3_background_repro.py`](analysis_scripts/notebooks/exploration/revision/fig3_background_repro.py) (`NONOVERLAP=1`) ← [`r16_glue_control.py`](analysis_scripts/notebooks/exploration/revision/r16_glue_control.py) (per-window d2, full and event-excised, barrage cohort) ← X:\data\Barrage, W:\data\PPP |
| f | same | `fig3f_psth_curves_PPP7_day7.csv`: BR around BARR starts, ripples + BARRs excised; `fig3f_inset_sessions.csv`, `fig3f_inset_lmm.csv`: per-session ΔBR (full window) | [`fig3f_psth_excised.py`](analysis_scripts/notebooks/exploration/revision/fig3f_psth_excised.py) ← W: spikes; inset: [`fig3_background_repro.panel_f`](analysis_scripts/notebooks/exploration/revision/fig3_background_repro.py) ← [`notebooks/exploration/generate_branching_ratio_mouse.ipynb`](analysis_scripts/notebooks/exploration/generate_branching_ratio_mouse.ipynb) |
| g | same | `fig3g_sessions.csv`: background d2 per session, baseline vs BARR disruption | [`fig3_background_repro.py`](analysis_scripts/notebooks/exploration/revision/fig3_background_repro.py) ← [`fig3g_background.py`](analysis_scripts/notebooks/exploration/revision/fig3g_background.py) ← [`opto_sncg_metrics.py`](analysis_scripts/notebooks/exploration/revision/opto_sncg_metrics.py) ← X:\data\Barrage\SNCG* |
| h | same | `fig3h_bins.csv`: ISI entropy by BARR-rate bin; `fig3h_entropy_diagnostics_lmm.csv`: the mixed model (its star) | [`fig3_background_repro.panel_h`](analysis_scripts/notebooks/exploration/revision/fig3_background_repro.py) ← [`notebooks/exploration/generate_ISI_entropy_ripple.ipynb`](analysis_scripts/notebooks/exploration/generate_ISI_entropy_ripple.ipynb); [`fig3h_entropy_diagnostics.py`](analysis_scripts/notebooks/exploration/revision/fig3h_entropy_diagnostics.py) |

`fig3_main_scalars.csv` holds the remaining statistics of e-h (the f inset's session test and LMM, g's session test, h's correlation of the bin means).

## Supplementary Figure 1: the three measures

| panel | figure notebook (cell) | data | made by |
|---|---|---|---|
| a, b, d | [`FigS1_AF_methods`](figure_scripts/FigS1_AF_methods.ipynb) (3) | `figs1_example_window_counts_40ms.csv`: population counts of the example 30 s NREM window (all CA1 units, 40 ms); `figs1_example_kernel.csv`, `figs1_example_s2.csv`: its AR(20) fit | [`r38_hyperplane_explainer.py`](analysis_scripts/notebooks/exploration/revision/r38_hyperplane_explainer.py) ← W:\data\PPP\PPP20\day5 |
| c | [`FigS1_AF_methods`](figure_scripts/FigS1_AF_methods.ipynb) (3, 9) | `r38_hyperplane_windows.csv`: every non-overlapping NREM window of PPP20 day 5 in kernel space; `figs1_window_kernels.csv`: their kernels | [`r38_hyperplane_explainer.py`](analysis_scripts/notebooks/exploration/revision/r38_hyperplane_explainer.py); β-manifolds: [`euclid_order_sweep.py`](analysis_scripts/notebooks/exploration/revision/euclid_order_sweep.py) (the authors' 2024 code, ported) |
| e, f | [`FigS1_AF_methods`](figure_scripts/FigS1_AF_methods.ipynb) (5) | `assets/Extended Data Fig1.svg` | Extended Data Fig. 1a of the first submission |

## Supplementary Figure 2: d2 robustness and the measures by state

Extended Data Fig. 1b–g of the first submission on non-overlapping windows, drawn by the notebooks in
`figure_scripts/rv1_disjoint/`.

| panel | figure notebook (cell) | data | made by |
|---|---|---|---|
| a | [`FigS2_AF_robustness`](figure_scripts/FigS2_AF_robustness.ipynb) (3) ← [`rv1_disjoint/FigS1_B_Explain variance_D_statetransition`](figure_scripts/rv1_disjoint/FigS1_B_Explain%20variance_D_statetransition.ipynb) | `rv1_FigS1_B_.../explained_variance_by_window.csv`: AR(20) variance explained per window, each window length thinned at its own length | [`ed_nonoverlap_ed1b.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap_ed1b.py) ← `20250324timebin_grainedmouse.mat` ← [`notebooks/helper/coarse_grain_ddc.m`](analysis_scripts/notebooks/helper/coarse_grain_ddc.m) (from the lab's R_Project repository) |
| b | same ← [`rv1_disjoint/FigS1_C_SNR_S2C_d2 uncertainty`](figure_scripts/rv1_disjoint/FigS1_C_SNR_S2C_d2%20uncertainty.ipynb) | `rv1_FigS1_C_.../d2_err_30s_60s_18sessions.csv`: d2 and its uncertainty, 30 and 60 s windows | [`ed_nonoverlap_d2err.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap_d2err.py) ← `*_error_bar.mat`, `*_error_bar_60sec.mat` |
| c | same ← `rv1_disjoint/FigS1_B_...` | `Previous_Rv1/csv/.../d2_windows_state_transition.csv` (all windows) | [`notebooks/helper/time_resolved_d2_state_transition.m`](analysis_scripts/notebooks/helper/time_resolved_d2_state_transition.m) |
| d | same ← [`rv1_disjoint/Fig1_D_histogram_S1_E`](figure_scripts/rv1_disjoint/Fig1_D_histogram_S1_E.ipynb) | `rv1_Fig1_D_.../d2_windows_18sessions.csv` (thinned); `fig1de_nonoverlap_stats.csv` (its bracket) | [`ed_nonoverlap.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap.py); [`fig1de_nonoverlap.py`](analysis_scripts/notebooks/exploration/revision/fig1de_nonoverlap.py) |
| e | same ← [`rv1_disjoint/FigS1_F_branchingratio_histogram`](figure_scripts/rv1_disjoint/FigS1_F_branchingratio_histogram.ipynb) | `rv1_FigS1_F_.../branching_ratio_by_state.csv`: BR of every 10 s window thinned at 10 s, every wake window | [`ed_nonoverlap_ed2a_ed1f.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap_ed2a_ed1f.py) (runs cell 0 of [`notebooks/manuscript_code/FigS1_F_branchingratio_histogram.ipynb`](analysis_scripts/notebooks/manuscript_code/FigS1_F_branchingratio_histogram.ipynb)) ← [`notebooks/exploration/generate_branching_ratio_mouse.ipynb`](analysis_scripts/notebooks/exploration/generate_branching_ratio_mouse.ipynb) |
| f (scatter) | [`FigS2_AF_robustness`](figure_scripts/FigS2_AF_robustness.ipynb) (5); [`fig1e_scatter_d2raw_drawing.draw_main`](figure_scripts/fig1e_scatter_d2raw_drawing.py) | `figs2f_scatter_windows.csv`: disjoint windows of the recomputed d2-MUA pairs, task epochs tagged, MUA z per session × Wake/Sleep | [`fig1e_scatter_d2raw.py`](analysis_scripts/notebooks/exploration/revision/fig1e_scatter_d2raw.py) ← `fig1e_pairs_recomputed.csv` ← [`fig1de_nonoverlap.py`](analysis_scripts/notebooks/exploration/revision/fig1de_nonoverlap.py) (each session's own d2) |
| f (range) | same; [`fig1e_dynrange_cell1.py`](figure_scripts/fig1e_dynrange_cell1.py), [`fig1e_dynrange_cell2.py`](figure_scripts/fig1e_dynrange_cell2.py) | `fig1e_pairs_recomputed.csv` (all windows); `fig1de_nonoverlap_stats.csv` (the Stouffer tests) | the submitted Fig 1e notebook's cells ([`notebooks/figures/Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb`](analysis_scripts/notebooks/figures/Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb)), executed on the recomputed pairs |

## Supplementary Figure 3: task, novelty and avalanches

`stats.csv`: the statistics printed on the panels.

| panel | figure notebook (cell) | data | made by |
|---|---|---|---|
| a | [`FigS3_AG_task_novelty_avalanches`](figure_scripts/FigS3_AG_task_novelty_avalanches.ipynb) (3) | `a_tmaze_sessions.csv`: T-maze fraction correct per session | [`m25_tmaze_box.py`](analysis_scripts/notebooks/exploration/revision/m25_tmaze_box.py) ← [`m25_task_performance.py`](analysis_scripts/notebooks/exploration/revision/m25_task_performance.py) ← W:\...\<basename>.animal.behavior.mat (`behavior.trialcorrect`) and the epoch tables |
| b | same (5) | `assets/Extended Data Fig2.svg` (schematic); `b_sessions.csv`: one row per session, control vs novel. Columns `mid` and `post` hold the median d2 relative to presleep in each phase, and `delta` = post − mid. The star is the mixed model S2 × novel (`stats.csv`). | [`ed2a_novel_vs_control.py`](analysis_scripts/notebooks/exploration/revision/ed2a_novel_vs_control.py) ← [`ed_nonoverlap_ed2a_ed1f.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap_ed2a_ed1f.py) (runs [`notebooks/exploration/MUA_sleepstage.ipynb`](analysis_scripts/notebooks/exploration/MUA_sleepstage.ipynb) cells 8, 15), [`ed2a_ppp8_day14_rebuild.py`](analysis_scripts/notebooks/exploration/revision/ed2a_ppp8_day14_rebuild.py) ← `*_CA1_D2_time_resolved_error_bar.mat` |
| c | same (7) | `c_swr_rate_sessions.csv`: SWR rate, presleep vs postsleep, per session; the star is that notebook's mixed model on the session means (`stats.csv`) | as Fig 2a |
| d | same (7) | `d_sextiles.csv`: d2 (z) by SWR-rate sextile: mean, SD and the uncertainty band; R² and p in `stats.csv` | [`ed_nonoverlap_d2err.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap_d2err.py) ← `data/ripple_barr/d2_uncertainty/*_CA1.pkl` ← [`notebooks/exploration/MUA_sleepstage.ipynb`](analysis_scripts/notebooks/exploration/MUA_sleepstage.ipynb) ← `*_D2_time_resolved_error_bar.mat` |
| e | same (9) | `e_points.csv`: PPP20 day 4, low vs high SWR rate. Holds the drawn CCDF points, the fit lines and the mean size by duration. | the lab's `FigS3` notebook (cell 9) runs `run_session` of [`ripple_criticality_batch.ipynb`](analysis_scripts/notebooks/exploration/revision/ripple_criticality_batch.ipynb) (its cells 1, 3; fits by [`mixture_proof.py`](analysis_scripts/notebooks/exploration/revision/mixture_proof.py)) on W:\data\PPP\PPP20\day4. |
| f | same (9) | `f_dcc_sessions.csv`, `f_ks_sessions.csv`: DCC and KS distance per session, low vs high SWR rate | [`ripple_criticality_batch.ipynb`](analysis_scripts/notebooks/exploration/revision/ripple_criticality_batch.ipynb) (`ripple_criticality_all_sessions.csv`), [`ks_xmin_convention.py`](analysis_scripts/notebooks/exploration/revision/ks_xmin_convention.py) ([`avalanche_ripple.py`](analysis_scripts/notebooks/exploration/revision/avalanche_ripple.py)) ← 16 sessions on W: |
| g | same (9) | `g_dcc_terciles.csv`: DCC by SWR-rate tercile, per session | [`ripple_criticality_batch.ipynb`](analysis_scripts/notebooks/exploration/revision/ripple_criticality_batch.ipynb) |

## Supplementary Figure 4: SWR controls

| panel | figure notebook (cell) | data | made by |
|---|---|---|---|
| a | [`FigS4_AF_swr_controls`](figure_scripts/FigS4_AF_swr_controls.ipynb) (5) | `figs4a_prepost_matched_ripple_disjoint.csv`: d2 pre vs post, SWR rate matched, per session | the lab's `FigS4` notebook: `Previous_Rv1` [`Fig2_A_FigS2_B_FigS3_A`](../Previous_Rv1/figure_scripts/Fig2_A_FigS2_B_FigS3_A.ipynb) (its `matched_df`) run by [`ed_nonoverlap.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap.py) |
| b | same | `figs4b_prepost_ripple_excised_disjoint.csv`: ripple-excised d2 pre vs post | `Previous_Rv1` [`FigS3_B_S3C_S3D_ripple_ablate`](../Previous_Rv1/figure_scripts/FigS3_B_S3C_S3D_ripple_ablate.ipynb) by [`ed_nonoverlap.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap.py) ← [`notebooks/exploration/generate_ablate_ripple.ipynb`](analysis_scripts/notebooks/exploration/generate_ablate_ripple.ipynb) ← `*_rippleablate.mat` |
| c | same; `sextile_panel` | `fig2_agh_nine_sessions.csv` | [`fig2_agh_excised.py`](analysis_scripts/notebooks/exploration/revision/fig2_agh_excised.py) |
| d | same; `panel_replay_prob` | `excised_br_entropy_windows.csv`; `ed4d_replay_prob_lmm.csv` (its stars) | [`excised_br_entropy.py`](analysis_scripts/notebooks/exploration/revision/excised_br_entropy.py); [`ed4d_replay_prob_lmm.py`](analysis_scripts/notebooks/exploration/revision/ed4d_replay_prob_lmm.py) |
| e, f | same ← [`rv1_disjoint/FigS3_G_F_CA3_CA2_pyr_model_surpport_fig`](figure_scripts/rv1_disjoint/FigS3_G_F_CA3_CA2_pyr_model_surpport_fig.ipynb) | `rv1_FigS3_G_F_.../rat_CA2_CA3_firing_by_window.csv` (thinned); `figs4e_..._disjoint.csv`, `figs4f_..._disjoint.csv`: per-session bin means | [`notebooks/exploration/MUA_sleepstage_barr_model_supple_ver2.ipynb`](analysis_scripts/notebooks/exploration/MUA_sleepstage_barr_model_supple_ver2.ipynb), [`barr_CA2_CA3_pooled_analysis.py`](analysis_scripts/notebooks/exploration/barr_CA2_CA3_pooled_analysis.py) ← rat barrage sessions (Z:\Data\AYAold) |

## Supplementary Figure 6: d2 as an indicator of the NREM CA1 regime

| panel | figure notebook (cell) | data | made by |
|---|---|---|---|
| a | [`FigS6_AC_regime_indicator`](figure_scripts/FigS6_AC_regime_indicator.ipynb) (3) | `a_session_coefficients.csv`, `a_summary.csv`: per-session coefficient of each measure on the BARR share of the window's events | [`r16_block_level_dissociation.py`](analysis_scripts/notebooks/exploration/revision/r16_block_level_dissociation.py) ← [`r16_event_excised_windows.py`](analysis_scripts/notebooks/exploration/revision/r16_event_excised_windows.py) ← barrage cohort (spikes, SWRs, BARRs) |
| b, c | same (3) | `b_estimates.csv`, `c_estimates.csv`: point estimates. `b_bootstrap.csv`, `c_bootstrap.csv`: the 2 000 session bootstraps per box. `bracket_tests.csv`: the paired session bootstrap of the lab's FigS6 notebook (2 000 resamples, Holm). | [`r16_robustness_index.py`](analysis_scripts/notebooks/exploration/revision/r16_robustness_index.py) (boot_A / boot_B), [`r16_predictor_heterogeneity.py`](analysis_scripts/notebooks/exploration/revision/r16_predictor_heterogeneity.py) (`r16_event_type_classifier_ortho.csv`) ← same chain as a |

## Supplementary Figure 7: rats

| panel | figure notebook (cell) | data | made by |
|---|---|---|---|
| a | [`FigS7_AD_rats`](figure_scripts/FigS7_AD_rats.ipynb) (5) | `assets/Extended Data Fig5.svg` | Extended Data Fig. 5 of the first submission |
| b | [`FigS7_AD_rats`](figure_scripts/FigS7_AD_rats.ipynb) (3, 5) | `figs7b_rat_d2_windows_30s_disjoint.csv`: every non-overlapping 30 s rat d2 window with state, environment and manipulation | the lab's `FigS7` notebook: `Previous_Rv1` [`FigS5_S6_rats`](../Previous_Rv1/figure_scripts/FigS5_S6_rats.ipynb) by [`ed_nonoverlap.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap.py) ← [`notebooks/exploration/MUA_sleepstage_rat.ipynb`](analysis_scripts/notebooks/exploration/MUA_sleepstage_rat.ipynb) ← rat d2 .mat (V:, S:, U:) |
| c | same ← [`rv1_disjoint/FigS5_S6_rats_runA`](figure_scripts/rv1_disjoint/FigS5_S6_rats_runA.ipynb) | `figs7c_rat_scatter_pairs_disjoint.csv` (the pairs the pooled r is computed from); `rv1_FigS5_S6_rats_runA/rat_d2_mua_pairs.csv` (thinned; the drawing) | as b; [`fig1e_scatter_d2raw.meta_r`](analysis_scripts/notebooks/exploration/revision/fig1e_scatter_d2raw.py) ([`fig1e_scatter_d2raw_drawing.py`](figure_scripts/fig1e_scatter_d2raw_drawing.py)) |
| d | same ← [`rv1_disjoint/FigS5_S6_rats_runB`](figure_scripts/rv1_disjoint/FigS5_S6_rats_runB.ipynb) | the notebook's pairs, all windows (`Previous_Rv1/csv`); its Stouffer test (stars) | as b |

## Supplementary Figure 8: rats, SWRs, replay and BARRs

`stats.csv`: the statistics printed on the panels.

| panel | data | made by |
|---|---|---|
| a | `assets/Extended Data Fig6.svg` (schematic); `a_prepost_sessions.csv`: d2 presleep vs postsleep per session | `Previous_Rv1` [`FigS5_S6_rats`](../Previous_Rv1/figure_scripts/FigS5_S6_rats.ipynb) (its `summary_df`) by [`ed_nonoverlap.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap.py) ← `Previous_Rv1/csv/FigS5_S6_rats/rat_d2_windows_by_state.csv` ← [`notebooks/exploration/MUA_sleepstage_rat.ipynb`](analysis_scripts/notebooks/exploration/MUA_sleepstage_rat.ipynb) |
| b, e | `assets/Extended Data Fig6.svg`; `be_inset_sessions.csv`: per-session mean ΔBR around SWRs / BARRs vs shuffle (inset stars, paired signed-rank) | [`notebooks/exploration/generate_branching_ratio_rats.ipynb`](analysis_scripts/notebooks/exploration/generate_branching_ratio_rats.ipynb) |
| c | `c_sextiles.csv`: d2 by replay / non-replay SWR-rate sextile; slopes in `stats.csv` | [`ed_nonoverlap_rats_tiles.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap_rats_tiles.py) (runs [`notebooks/exploration/MUA_sleepstage_rat_replay.ipynb`](analysis_scripts/notebooks/exploration/MUA_sleepstage_rat_replay.ipynb) cells 7, 8) ← `data/rats/replay_nonreplay/*_CA1.pkl`, the sessions' SWR and state files (`ED6C_SESSIONS`), `data/replay/<session>_sig_replay.csv` |
| d | `d_sextiles.csv`: ISI entropy by SWR-rate sextile; `d_session_rho.csv`: per-session ρ (the star) | [`ed_nonoverlap.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap.py) (thinned, row-aligned with `rat_d2_windows_by_state.csv`) ← [`notebooks/exploration/MUA_sleepstage_rat.ipynb`](analysis_scripts/notebooks/exploration/MUA_sleepstage_rat.ipynb) |
| f | `f_sextiles.csv`: d2 and ISI entropy by BARR-rate sextile | [`ed_nonoverlap_rats_tiles.py`](analysis_scripts/notebooks/exploration/revision/ed_nonoverlap_rats_tiles.py) ← `data/rats/ripple_barr/isi/*.pkl` (per window) ← rat barrage sessions (AB4, AYA6, AYA7, AYA10) |




