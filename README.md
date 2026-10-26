# Criticality of hippocampal network dynamics during sleep and learning

Code and data behind the figures of Chang, Chen et al., *Distinct critical network regimes facilitate memory encoding
and consolidation* (Nature Neuroscience).

The paper measures how close the population activity of the hippocampus (CA1, mice and rats) is to criticality across
sleep and wake, and relates it to sharp-wave ripples (SWRs), replay and barrages, and to learning. Closeness to
criticality is measured with d2, the distance to criticality of Sooter et al. 2024
([method and code](https://doi.org/10.6084/m9.figshare.25927081)), together with neuronal avalanches, the branching
ratio and the entropy of inter-spike intervals.

## Where to start

The repository holds two versions of the figures. Start with [Revision_Rv2](Revision_Rv2), the revised
manuscript; the [quick start](Revision_Rv2/sleep/README.md#quick-start) of its sleep folder draws a panel in a few
seconds.

| folder | figures |
|---|---|
| [`Revision_Rv2/sleep/`](Revision_Rv2/sleep) | Revised Figures 1-3 and Extended Data Figs 1-7 and 9-11. Some notebooks read tables from `Previous_Rv1/sleep`; keep the two version folders side by side. |
| [`Revision_Rv2/behavior/`](Revision_Rv2/behavior) | Figure 3c, d and Extended Data Fig. 8 (network model), Figures 4 and 5 and Extended Data Figs 12-15 (learning). |
| [`Previous_Rv1/sleep/`](Previous_Rv1/sleep) | First submission: Figures 1-3, Extended Data 1-3, 5, 6 and 10, with their tables and analysis code. |
| [`Previous_Rv1/behavior/`](Previous_Rv1/behavior) | The same behaviour figures as the revision. |
| [`methods/`](methods) | The code of the criticality measures, shared by all folders: d2 (Sooter et al. 2024, with its error bar), the branching ratio (MR estimator of Wilting and Priesemann 2018) and neuronal avalanches. |

One environment serves the whole repository: `conda env create -f environment.yml`, then `pip install -e methods`
and `python methods/criticality/get_wiltingpriesemann.py` (the branching-ratio estimator, from its authors) for the
analysis code.

Every figure folder has the same layout:

| folder | what is in it |
|---|---|
| `figure_scripts/` | The notebooks that draw the panels. Run them from inside this folder. |
| `csv/` | The tables each notebook draws, one sub-folder per notebook. No spike times. |
| `figure_output/` | Where the notebooks save the panels (created when they run). |
| `analysis_scripts/` | (sleep folders) The code that made the tables from the recordings (MATLAB and Python). |

In the revision, notebook names follow the figures: `FigS<N>` is Extended Data Fig. N. The analysis scripts read the
recordings at their paths on the lab servers; the figure notebooks need only the tables in `csv/`.
