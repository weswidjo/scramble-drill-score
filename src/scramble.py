"""
Scramble play selection and timing windows for the Scramble Drill Score project.

Pipeline:
  1. select_plays  -- scramble throws + a TRADITIONAL comparison set
  2. attach_tracking -- load tracking and merge pff roles + player names
  3. scramble_windows -- snap / throw / scramble-start frames per play

All frame logic is vectorized (groupby/transform), no per-row loops over frames,
per the steering spec.
"""
from pathlib import Path

import pandas as pd

from load import (
    load_plays, load_pff, load_players, load_tracking,
)

CACHE_DIR = Path(__file__).resolve().parent.parent / "output" / "cache"

PASS_RESULTS = ["C", "I", "IN"]
SNAP_EVENTS = ["ball_snap", "autoevent_ballsnap"]
THROW_EVENTS = ["pass_forward", "autoevent_passforward"]
QB_LATERAL_THRESHOLD = 3.0  # yards of QB |y - y_at_snap| that defines a scramble


def select_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """Return scramble throws plus a TRADITIONAL comparison set.

    Scramble throws : dropBackType contains "SCRAMBLE" AND passResult in C/I/IN.
    Comparison set  : dropBackType == "TRADITIONAL" AND the same passResult filter.

    Adds a boolean column is_scramble.
    """
    valid_result = plays["passResult"].isin(PASS_RESULTS)

    scramble = plays[
        plays["dropBackType"].str.contains("SCRAMBLE", na=False) & valid_result
    ].copy()
    scramble["is_scramble"] = True

    traditional = plays[
        (plays["dropBackType"] == "TRADITIONAL") & valid_result
    ].copy()
    traditional["is_scramble"] = False

    selected = pd.concat([scramble, traditional], ignore_index=True)
    return selected


def attach_tracking(selected: pd.DataFrame,
                    pff: pd.DataFrame,
                    players: pd.DataFrame) -> pd.DataFrame:
    """Load tracking for the selected plays and merge role + player info."""
    tracking = load_tracking(selected[["gameId", "playId"]])

    roles = pff[["gameId", "playId", "nflId",
                 "pff_role", "pff_positionLinedUp"]]
    tracking = tracking.merge(roles, on=["gameId", "playId", "nflId"],
                              how="left")

    info = players[["nflId", "displayName", "officialPosition"]]
    tracking = tracking.merge(info, on="nflId", how="left")

    # Carry the is_scramble flag onto tracking for convenience downstream.
    tracking = tracking.merge(
        selected[["gameId", "playId", "is_scramble"]],
        on=["gameId", "playId"], how="left")
    return tracking


def _event_frame(tracking: pd.DataFrame, events) -> pd.Series:
    """First frameId per play that carries any of `events` (vectorized).

    Returns a Series indexed by (gameId, playId).
    """
    mask = tracking["event"].isin(events)
    hits = tracking.loc[mask, ["gameId", "playId", "frameId"]]
    return hits.groupby(["gameId", "playId"]).frameId.min()


def scramble_windows(tracking: pd.DataFrame) -> pd.DataFrame:
    """Per-play timing windows: snap, throw, and scramble-start frames.

    scramble_start_frame = first frame between snap and throw where the QB's
    lateral displacement from his snap position exceeds QB_LATERAL_THRESHOLD.

    Scramble plays that never cross the threshold are dropped. Traditional
    comparison plays keep snap/throw but have NaN scramble_start.

    Returns one row per (gameId, playId).
    """
    keys = ["gameId", "playId"]

    snap = _event_frame(tracking, SNAP_EVENTS).rename("snap_frame")
    throw = _event_frame(tracking, THROW_EVENTS).rename("throw_frame")

    windows = pd.concat([snap, throw], axis=1).reset_index()
    windows = windows.dropna(subset=["snap_frame", "throw_frame"])
    windows[["snap_frame", "throw_frame"]] = \
        windows[["snap_frame", "throw_frame"]].astype(int)

    # QB rows only (pff_role == "Pass"), limited to between snap and throw.
    qb = tracking[tracking["pff_role"] == "Pass"].merge(windows, on=keys,
                                                         how="inner")
    qb = qb[(qb["frameId"] >= qb["snap_frame"]) &
            (qb["frameId"] <= qb["throw_frame"])]

    # QB y at snap (vectorized): the y where frameId == snap_frame.
    at_snap = qb[qb["frameId"] == qb["snap_frame"]][keys + ["y"]].rename(
        columns={"y": "qb_y_at_snap"})
    qb = qb.merge(at_snap, on=keys, how="left")
    qb["lateral"] = (qb["y"] - qb["qb_y_at_snap"]).abs()

    # First frame where lateral exceeds the threshold, per play.
    crossed = qb[qb["lateral"] > QB_LATERAL_THRESHOLD]
    scramble_start = crossed.groupby(keys).frameId.min().rename(
        "scramble_start_frame")

    windows = windows.merge(scramble_start, on=keys, how="left")

    # Attach the is_scramble flag.
    flag = tracking[keys + ["is_scramble"]].drop_duplicates()
    windows = windows.merge(flag, on=keys, how="left")

    # Drop scramble plays that never met the lateral threshold.
    drop_mask = windows["is_scramble"] & windows["scramble_start_frame"].isna()
    windows = windows[~drop_mask].copy()

    windows["scramble_duration_sec"] = (
        windows["throw_frame"] - windows["scramble_start_frame"]) / 10.0
    return windows


def build_windows():
    """Full pipeline -> cached windows DataFrame."""
    plays = load_plays()
    pff = load_pff()
    players = load_players()

    selected = select_plays(plays)
    tracking = attach_tracking(selected, pff, players)
    windows = scramble_windows(tracking)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    windows.to_csv(CACHE_DIR / "windows.csv", index=False)
    return windows, tracking


if __name__ == "__main__":
    windows, _ = build_windows()

    n_scramble = int(windows["is_scramble"].sum())
    n_trad = int((~windows["is_scramble"]).sum())
    print(f"scramble plays (with valid window): {n_scramble}")
    print(f"traditional comparison plays        : {n_trad}")

    print("\n=== scramble_duration_sec (scramble plays) ===")
    print(windows.loc[windows["is_scramble"], "scramble_duration_sec"]
          .describe())

    print(f"\ncached -> {CACHE_DIR / 'windows.csv'}")
