"""
r35_make_ppp4day11_pkl.py -- writes PPP4_day11 into the replay/non-replay input
directory of the submitted Fig. 2e notebook, in place of PPP4_day10.

`Fig2_E_replay_non_replay.ipynb` reads every file in `data/ripple_barr/replay_non_replay/`.
PPP4_day11 is the session used for this animal in the rest of Figure 2. This script rebuilds PPP4_day11 in
`generate_replay_non_replay.ipynb`'s own schema: d2 timeline from the stored per-window
table, ripples intersected with NREM, replay intervals from one template per session over
epochs 1/3/5 with left- and right-significant events appended separately (so `replay_rate`
matches the pipeline's per-direction counting), and a ripple counted as replay-carrying if
it overlaps any significant interval. The new file is written only if the rebuilt ripple
and replay counts match the stored per-window counts, window for window; the PPP4_day10
file moves to `_superseded/`.

Inputs: data/replay/replay_probality/PPP4_day11.pkl (d2 timeline, ripple_count,
replay_count); W:\\data\\PPP\\PPP4\\day11 (ripples, sleep state); the R_Project replay
sequences (PPP4_day11_replay_sequences.mat); SESSION_TEMPLATE / NOTEBOOK_EPOCHS from
r35_swr_property_control.py.

    -> data/ripple_barr/replay_non_replay/PPP4_day11_CA1.pkl
    -> data/ripple_barr/replay_non_replay/_superseded/PPP4_day10_CA1.pkl
"""
from __future__ import annotations

import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r35_swr_property_control import (NOTEBOOK_EPOCHS, SESSION_TEMPLATE,  # noqa: E402
                                      assert_local_path)

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
RES = r"D:\Documents\GitHub\R_Project\PPP\codebook\Criticality_analysis\replay_results"
OUT_DIR = os.path.join(REPO, "data", "ripple_barr", "replay_non_replay")
PROB_DIR = os.path.join(REPO, "data", "replay", "replay_probality")
SESS = "PPP4_day11"
BASEPATH = r"W:\data\PPP\PPP4\day11"
RETIRE = "PPP4_day10_CA1.pkl"
WIN = 30.0

COLS = ["center_time", "d2_value", "d2_6tile", "d2_err", "replay_rate",
        "ripple_rate_all", "ripple_rate_in", "ripple_rate_out"]


def replay_intervals_notebook(sess):
    """Replay intervals from one template per session, epochs 1/3/5, with left- and
    right-significant events appended separately."""
    d = scipy.io.loadmat(os.path.join(RES, f"{sess}_replay_sequences.mat"))
    t = SESSION_TEMPLATE[sess]
    left, right = [], []
    for e in NOTEBOOK_EPOCHS:
        k = f"{e}_{t}"
        if f"{k}_eventtimes" not in d or k not in d:
            continue
        ev = np.squeeze(d[k + "_eventtimes"]).reshape(-1, 2)
        pv = np.squeeze(d[k + "_pvalue"]).reshape(len(ev), -1)[:, :2]
        left.append(ev[pv[:, 0] < 0.05])
        right.append(ev[pv[:, 1] < 0.05])
    L = np.vstack(left) if left else np.empty((0, 2))
    R = np.vstack(right) if right else np.empty((0, 2))
    print(f"  {sess}: template {t}, {len(L)} left-significant, {len(R)} right-significant"
          f"  -> combined {len(L) + len(R)} (duplicates intentional, as in cell 9)")
    return np.vstack([L, R]) if len(L) + len(R) else np.empty((0, 2))


def count_in_windows(iv, centers, win=WIN):
    if iv.size == 0:
        return np.zeros(len(centers), int)
    return np.array([int(np.sum((iv[:, 0] < c + win / 2) & (iv[:, 1] > c - win / 2)))
                     for c in centers])


def main():
    import nelpy as nel
    from neuro_py.io import loading

    with open(os.path.join(PROB_DIR, f"{SESS}.pkl"), "rb") as fh:
        per = pickle.load(fh)["per_center"]
    nrem = per[per["state"].astype(str).str.upper().eq("NREM")].sort_values("center_s")
    c = nrem["center_s"].to_numpy(float)
    print(f"  d2 timeline: {len(c)} NREM windows")

    rip = loading.load_ripples_events(BASEPATH)
    sd = loading.load_SleepState_states(BASEPATH)
    riv = (nel.EpochArray(sd["NREMstate"]) &
           nel.EpochArray(np.array(rip[["start", "stop"]]))).data
    print(f"  ripples: {len(rip)} total, {len(riv)} in NREM")

    S = replay_intervals_notebook(SESS)
    mask = np.fromiter((bool(np.any((r[0] < S[:, 1]) & (r[1] > S[:, 0]))) if S.size
                        else False for r in riv), bool, len(riv))
    print(f"  replay-carrying ripples: {int(mask.sum())} / {len(riv)} "
          f"({mask.mean():.1%})")

    n_all = count_in_windows(riv, c)
    n_in = count_in_windows(riv[mask], c)
    n_out = count_in_windows(riv[~mask], c)
    n_rep = count_in_windows(S, c)
    assert np.array_equal(n_all, n_in + n_out), "in+out must partition all"

    # --- counts must match the stored per-window table before writing -------
    exp_all = nrem["ripple_count"].to_numpy(float)
    exp_in = nrem["replay_count"].to_numpy(float)
    ok_all = np.mean(np.isclose(n_all, exp_all))
    ok_in = np.mean(np.isclose(n_in, exp_in))
    print(f"\n  VERIFY vs replay_probality/{SESS}.pkl:")
    print(f"    ripple_count matches replay_probality: {ok_all:.1%}")
    print(f"    replay_count matches replay_probality: {ok_in:.1%}")
    if ok_all < 1.0 or ok_in < 1.0:
        raise SystemExit(
            "  ABORT: rebuilt counts do not match the author's stored counts; "
            "refusing to write a file that would silently differ from the pipeline.")

    df = pd.DataFrame({
        "center_time": c,
        "d2_value": nrem["d2"].to_numpy(float),
        "d2_6tile": nrem["d2"].to_numpy(float),
        "d2_err": nrem["d2_err"].to_numpy(float),
        "replay_rate": n_rep / WIN,
        "ripple_rate_all": n_all / WIN,
        "ripple_rate_in": n_in / WIN,
        "ripple_rate_out": n_out / WIN,
    })[COLS]

    dst = assert_local_path(os.path.join(OUT_DIR, f"{SESS}_CA1.pkl"))
    df.to_pickle(dst)
    print(f"\n  wrote {dst}  {df.shape}")

    # --- move the replaced file to _superseded/ ------------------------------
    src = os.path.join(OUT_DIR, RETIRE)
    if os.path.exists(src):
        sup = assert_local_path(os.path.join(OUT_DIR, "_superseded"))
        os.makedirs(sup, exist_ok=True)
        shutil.move(src, os.path.join(sup, RETIRE))
        with open(assert_local_path(os.path.join(sup, "README.md")), "w") as fh:
            fh.write(
                "# superseded inputs\n\n"
                "`PPP4_day10_CA1.pkl` was in the Fig. 2e glob but is not the session the\n"
                "author used -- the record and Fig. 2d both use PPP4_day11. day10 has only\n"
                "one epoch decoded (`epoch5_template02`), giving 437 replay SWRs against\n"
                "1677-3322 in properly covered sessions, and it was one of the two anomalies\n"
                "in the equal-sample test. The file is valid, just the wrong session; kept\n"
                "here byte-identical and out of `glob('*.pkl')`.\n\n"
                "Replaced by `PPP4_day11_CA1.pkl` (see revision/r35_make_ppp4day11_pkl.py).\n")
        print(f"  moved {RETIRE} -> _superseded/ (preserved, out of the glob)")

    print("\n  Fig. 2e input directory now:")
    for f in sorted(x for x in os.listdir(OUT_DIR) if x.endswith(".pkl")):
        print(f"    {f}")


if __name__ == "__main__":
    main()
