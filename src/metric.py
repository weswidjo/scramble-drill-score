"""
Separation metrics for the Scramble Drill Score project.

receiver_separation computes, for every route runner:
  - sep_start  : distance to nearest Coverage defender at scramble_start_frame
  - sep_throw  : distance to nearest Coverage defender at throw_frame
  - sep_gained : sep_throw - sep_start
  - speed_during_scramble : mean s over scramble_start_frame..throw_frame

Traditional comparison plays get sep_throw only (no scramble window).

Nearest-defender distances use numpy broadcasting: within a frame we build a
[receivers x defenders] matrix with np.hypot and take the row-wise min. No
per-receiver Python loops.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scramble import build_windows

CACHE_DIR = Path(__file__).resolve().parent.parent / "output" / "cache"


def _nearest_defender_at_frames(tracking: pd.DataFrame,
                                frame_lookup: pd.DataFrame,
                                dist_col: str) -> pd.DataFrame:
    """Nearest Coverage-defender distance for each receiver at a target frame.

    Parameters
    ----------
    tracking : full merged tracking (needs pff_role, x, y, frameId, keys).
    frame_lookup : one row per (gameId, playId) with a column `target_frame`.
    dist_col : output column name for the distance.

    Returns
    -------
    DataFrame: gameId, playId, nflId, <dist_col> for every route runner on the
    target frame of its play.
    """
    keys = ["gameId", "playId"]

    # Restrict tracking to each play's target frame.
    tr = tracking.merge(frame_lookup, on=keys, how="inner")
    tr = tr[tr["frameId"] == tr["target_frame"]]

    receivers = tr[tr["pff_role"] == "Pass Route"]
    defenders = tr[tr["pff_role"] == "Coverage"]

    out = []
    # Loop is over plays (hundreds), not frames or players. Within each play the
    # distance is a single vectorized np.hypot over a [receivers x defenders]
    # grid, satisfying the "no per-row loops over frames" rule.
    rec_groups = dict(tuple(receivers.groupby(keys)))
    def_groups = dict(tuple(defenders.groupby(keys)))

    for key, rec in rec_groups.items():
        deff = def_groups.get(key)
        rec_xy = rec[["x", "y"]].to_numpy()
        if deff is None or len(deff) == 0:
            dists = np.full(len(rec), np.nan)
        else:
            def_xy = deff[["x", "y"]].to_numpy()
            # [receivers x defenders] distance matrix via broadcasting.
            dx = rec_xy[:, 0][:, None] - def_xy[:, 0][None, :]
            dy = rec_xy[:, 1][:, None] - def_xy[:, 1][None, :]
            dmat = np.hypot(dx, dy)
            dists = dmat.min(axis=1)
        res = rec[keys + ["nflId"]].copy()
        res[dist_col] = dists
        out.append(res)

    if not out:
        return pd.DataFrame(columns=keys + ["nflId", dist_col])
    return pd.concat(out, ignore_index=True)


def _mean_speed_in_window(tracking: pd.DataFrame,
                          windows: pd.DataFrame) -> pd.DataFrame:
    """Mean speed per route runner over scramble_start..throw (vectorized)."""
    keys = ["gameId", "playId"]
    scr = windows[windows["is_scramble"]][
        keys + ["scramble_start_frame", "throw_frame"]]

    tr = tracking[tracking["pff_role"] == "Pass Route"].merge(
        scr, on=keys, how="inner")
    in_window = ((tr["frameId"] >= tr["scramble_start_frame"]) &
                 (tr["frameId"] <= tr["throw_frame"]))
    tr = tr[in_window]
    speed = tr.groupby(keys + ["nflId"]).s.mean().rename(
        "speed_during_scramble").reset_index()
    return speed


def receiver_separation(tracking: pd.DataFrame,
                        windows: pd.DataFrame) -> pd.DataFrame:
    """One row per route runner per play with separation metrics."""
    keys = ["gameId", "playId"]
    scr_keys = windows.loc[windows["is_scramble"], keys]

    # --- sep at throw (all plays) ---
    throw_lookup = windows[keys + ["throw_frame"]].rename(
        columns={"throw_frame": "target_frame"})
    sep_throw = _nearest_defender_at_frames(tracking, throw_lookup, "sep_throw")

    # --- sep at scramble start (scramble plays only) ---
    start_lookup = windows.loc[windows["is_scramble"],
                               keys + ["scramble_start_frame"]].rename(
        columns={"scramble_start_frame": "target_frame"})
    sep_start = _nearest_defender_at_frames(tracking, start_lookup, "sep_start")

    # --- mean speed during the scramble window (scramble plays only) ---
    speed = _mean_speed_in_window(tracking, windows)

    # --- receiver identity + is_scramble ---
    route = tracking[tracking["pff_role"] == "Pass Route"][
        keys + ["nflId", "displayName", "officialPosition", "is_scramble"]
    ].drop_duplicates()

    df = route.merge(sep_throw, on=keys + ["nflId"], how="left")
    df = df.merge(sep_start, on=keys + ["nflId"], how="left")
    df = df.merge(speed, on=keys + ["nflId"], how="left")

    df["sep_gained"] = df["sep_throw"] - df["sep_start"]

    cols = keys + ["nflId", "displayName", "officialPosition", "is_scramble",
                   "sep_start", "sep_throw", "sep_gained",
                   "speed_during_scramble"]
    return df[cols]


def build_receivers():
    windows, tracking = build_windows()
    receivers = receiver_separation(tracking, windows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    receivers.to_csv(CACHE_DIR / "receivers.csv", index=False)
    return receivers


if __name__ == "__main__":
    receivers = build_receivers()
    scr = receivers[receivers["is_scramble"]]

    print(f"receiver-play rows total : {len(receivers)}")
    print(f"  scramble rows          : {len(scr)}")
    print(f"  traditional rows       : {(~receivers['is_scramble']).sum()}")

    print("\n=== mean separation by officialPosition (scramble plays) ===")
    summary = scr.groupby("officialPosition")[
        ["sep_start", "sep_throw", "sep_gained"]].mean().sort_values(
        "sep_gained", ascending=False)
    print(summary.to_string())

    print(f"\ncached -> {CACHE_DIR / 'receivers.csv'}")
