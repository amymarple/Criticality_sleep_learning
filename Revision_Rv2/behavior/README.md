# Revision_Rv2 / behavior: behaviour and the network model

Chang, Chen et al., *Distinct critical network regimes facilitate memory encoding and consolidation* (Nature
Neuroscience, revision). This folder holds the behaviour and network-model panels of the revision: Figure 3c, d
(network model), Figures 4 and 5 (cheeseboard learning and CA1 context separation) and Extended Data Figs 8 and
12-15. Each notebook draws its panels from the tables in `csv/`. The sleep figures are in [`../sleep`](../sleep).

Notebook names follow the figures: `FigS<N>` is Extended Data Fig. N.

## Quick start

1. Create the Python environment of the repository ([`environment.yml`](../../environment.yml)), once:

   ```
   conda env create -f ../../environment.yml      # creates the environment "criticality"
   conda activate criticality
   ```

2. Run a notebook from inside `figure_scripts/`:

   ```
   cd figure_scripts
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 Fig4_AK_behavior_d2.ipynb
   ```

3. After about 20 seconds the panels are in `figure_output/Fig4/` (`Fig4_AB.pdf` … `Fig4_HK.pdf`). The notebook prints
   the statistics given in the figure legend.

## Notebooks

| notebook | panels | data (`csv/<notebook>/`) |
|---|---|---|
| [`Fig3_CD_network_model`](figure_scripts/Fig3_CD_network_model.ipynb) | Fig 3c, d: CA1 d2 vs CA3 input and vs CA2 input (BARRs) in the network model | `fig3_runs.csv`: one row per run (CA3 drive, BARR rate, seed, d2, CA1 MUA) |
| [`Fig4_AK_behavior_d2`](figure_scripts/Fig4_AK_behavior_d2.ipynb) | Fig 4a-k: path length over trials, CA1 d2 at the beginning and end of learning, d2 vs path length, GLM drop-one ΔR² | `fig4_trials_norm_path.csv` (per trial), `fig4_d2_begin_end.csv`, `fig4_sleep_vs_maze.csv`, `fig4_d2_vs_path.csv`, `fig4_glm_sessions.csv` (per session) |
| [`Fig5_AF_separation`](figure_scripts/Fig5_AF_separation.ipynb) | Fig 5a-f: CA1 context separation (CEBRA embedding, SVM classification) vs shuffles, vs d2 and vs path length | `sep_sessions.csv`: one row per session (30 sessions) |
| [`FigS8_AG_network_model`](figure_scripts/FigS8_AG_network_model.ipynb) | Extended Data Fig. 8a-g: the network model, CA3 drive × BARR rate, window length and CCK decay | `figS8_runs.csv` (per run), `figS8_windows.csv` (per d2 window) |
| [`FigS12_covariates_performance`](figure_scripts/FigS12_covariates_performance.ipynb) | Extended Data Fig. 12: speed, CA1 firing rate and theta vs performance, and d2 vs these covariates | `figS12_sessions.csv`: one row per session |
| [`FigS13_cca_d2`](figure_scripts/FigS13_cca_d2.ipynb) | Extended Data Fig. 13: CA1 d2 vs cross-region coupling (CCA with CA2, CA3, MEC) at the beginning of learning | `cca_d2_report_data.csv`: one row per window |
| [`FigS14_d2_br_performance`](figure_scripts/FigS14_d2_br_performance.ipynb) | Extended Data Fig. 14: CA1 d2 and branching ratio vs maze performance, first acquisition epoch | `behavior_report_data.csv`: one row per epoch |
| [`FigS15_CI_separation_vars`](figure_scripts/FigS15_CI_separation_vars.ipynb) | Extended Data Fig. 15c-i: CA1 latent separation vs the number of latent dimensions, MUA rate and d2 | `figS15_panelc.csv`, `figS15_scatter.csv` |

## Folders

| folder | what is in it |
|---|---|
| [`figure_scripts/`](figure_scripts) | The notebooks. [`fonts/`](figure_scripts/fonts): Nimbus Sans, the figure font. |
| [`csv/<notebook>/`](csv) | The tables each notebook draws. |
| `figure_output/` | Where the notebooks save the panels as PDF, with the statistics tables of Extended Data Figs 8, 12 and 15 (created when they run). |
