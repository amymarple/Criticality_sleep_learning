# methods: d2 and the branching ratio

The criticality measures used by every figure folder of this repository (`Revision_Rv2/sleep`,
`Revision_Rv2/behavior`, `Previous_Rv1/sleep`). Each file is used as listed below; the third-party code is unchanged
from its source.

## d2, distance to criticality

| file | what it is |
|---|---|
| [`d2/sooter_2024/`](d2/sooter_2024) | The d_beta code of Sooter, Shew and Fontenele (2024), unchanged from [https://doi.org/10.6084/m9.figshare.25927081](https://doi.org/10.6084/m9.figshare.25927081) (CC BY 4.0). [`getDbetas.m`](d2/sooter_2024/distance%20calculation/getDbetas.m) fits an autoregressive model to the binned population activity ([`myYuleWalker.m`](d2/sooter_2024/AR%20fitting%20and%20simulation/myYuleWalker.m)) and returns its distance to each β manifold ([`getFixedPointDistance.m`](d2/sooter_2024/distance%20calculation/getFixedPointDistance.m)); d2 is the distance for β = 2. The example dataset of `example case/test.m` (`YMV17.mat`) is on the Figshare page. |
| [`d2/getDbetas_error_bar.m`](d2/getDbetas_error_bar.m) | `getDbetas` with the uncertainty of d2: the delta-method error from the covariance of the AR coefficients (observed Hessian of the fit). Used for every d2 with an error bar. |
| [`d2/d2_fast.py`](d2/d2_fast.py) | d2 in its 2025 definition, the KL-divergence rate to the β = 2 manifold in bits per second (Sooter et al. 2025, [prox_crit_toolkit](https://github.com/sam-sooter/prox_crit_toolkit), MIT), in Python. Used for the AR-order sweeps. |

Settings throughout the paper: 40 ms bins, AR order 20, 30 s windows stepped 2 s. In MATLAB, put the folder on the
path with `addpath(genpath('methods/d2'))`; the per-session d2 workers are in
[`Revision_Rv2/sleep/analysis_scripts/matlab/`](../Revision_Rv2/sleep/analysis_scripts/matlab).

## criticality: branching ratio and avalanches (Python package)

| file | what it is |
|---|---|
| `criticality/WiltingPriesemann.py` | The multistep-regression (MR) estimator of the branching ratio and the branching-process simulation of Wilting and Priesemann (2018), Nat. Commun. 9, 2325, from the authors' repository [github.com/jwilting/WiltingPriesemann2018](https://github.com/jwilting/WiltingPriesemann2018). [`get_wiltingpriesemann.py`](criticality/get_wiltingpriesemann.py) downloads it at commit `0c26b11` and rewrites its three Python 2 print statements for Python 3. |
| [`criticality/av_analysis.py`](criticality/av_analysis.py) | Neuronal avalanches: detection on binned population activity, power-law fits of sizes and durations (powerlaw package), and the deviation from criticality coefficient (DCC). |

Install once from the repository root:

```
pip install -e methods
python methods/criticality/get_wiltingpriesemann.py
```

then `from criticality import WiltingPriesemann`.
