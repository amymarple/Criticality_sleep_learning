# Criticality of hippocampal network dynamics during sleep and learning

Code and data behind the figures of Chang, Chen et al., *Distinct critical network regimes facilitate memory encoding
and consolidation* (Nature Neuroscience).

The paper measures how close the population activity of the hippocampus (CA1, mice and rats) is to criticality across
sleep and wake, and relates it to sharp-wave ripples (SWRs), replay and barrages. Closeness to criticality is measured
with d2, the distance to criticality of Sooter et al. 2024 ([method and code](https://doi.org/10.6084/m9.figshare.25927081)),
together with neuronal avalanches, the branching ratio and the entropy of inter-spike intervals.

## Where to start

The repository holds two versions of the figures. Start with **[Revision_Rv2](Revision_Rv2)**, the revised
manuscript; its [quick start](Revision_Rv2/README.md#quick-start) draws a panel in a few seconds.

| folder | figures | relation |
|---|---|---|
| [`Revision_Rv2/`](Revision_Rv2) | Revised Figures 1-3 and Supplementary Figures 1-4, 6-8 | The current figures. Several supplementary panels are first-submission panels redrawn on non-overlapping windows, so some notebooks here read tables from, or run notebooks of, `Previous_Rv1`. Keep the two folders side by side. |
| [`Previous_Rv1/`](Previous_Rv1) | First submission: Figures 1-3, Extended Data 1-3, 5, 6 and 10 | The figures as first submitted, with their own tables and analysis code. |

Both folders have the same layout:

| folder | what is in it |
|---|---|
| `figure_scripts/` | The notebooks that draw the panels. Run them from inside this folder. |
| `csv/` | The tables each notebook draws, one sub-folder per notebook. No spike times. |
| `figure_output/` | The panels, as the notebooks save them. |
| `analysis_scripts/` | The code that made the tables from the recordings (MATLAB and Python). |

The analysis scripts read the recordings at their paths on the lab servers; the figure notebooks need only the tables
in `csv/`.
