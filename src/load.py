"""
Data loading for the Scramble Drill Score project.

Loads the static CSVs (plays, pff, players) and tracking data. Tracking is
large (one file per game, 10 fps), so load_tracking reads only the columns we
need and keeps only the plays requested via play_keys. Direction is normalized
per the steering spec so every play reads left-to-right.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRACKING_DIR = DATA_DIR / "tracking"

# Field constants (yards)
FIELD_X = 120.0
FIELD_Y = 160.0 / 3.0  # 53.333...

TRACKING_COLS = [
    "gameId", "playId", "nflId", "frameId", "team", "playDirection",
    "x", "y", "s", "a", "dir", "event",
]


def load_plays() -> pd.DataFrame:
    """Play-level info (one row per gameId, playId)."""
    return pd.read_csv(DATA_DIR / "plays.csv")


def load_pff() -> pd.DataFrame:
    """PFF scouting data (one row per gameId, playId, nflId)."""
    return pd.read_csv(DATA_DIR / "pffScoutingData.csv")


def load_players() -> pd.DataFrame:
    """Player attributes (one row per nflId)."""
    return pd.read_csv(DATA_DIR / "players.csv")


def _normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Flip left-direction plays so every play moves left-to-right.

    Per the steering spec: if playDirection == "left",
        x = 120 - x
        y = 160/3 - y
    Applied vectorized across the whole frame.
    """
    left = df["playDirection"].values == "left"
    df.loc[left, "x"] = FIELD_X - df.loc[left, "x"]
    df.loc[left, "y"] = FIELD_Y - df.loc[left, "y"]
    return df


def load_tracking(play_keys) -> pd.DataFrame:
    """Load tracking rows for the requested plays only.

    Parameters
    ----------
    play_keys : iterable of (gameId, playId) tuples, or a DataFrame with
        columns gameId and playId.

    Returns
    -------
    DataFrame with TRACKING_COLS, direction-normalized, for the requested
    plays across all tracking files.
    """
    # Accept a DataFrame or an iterable of pairs; build a fast lookup set.
    if isinstance(play_keys, pd.DataFrame):
        keys = set(map(tuple, play_keys[["gameId", "playId"]].to_numpy()))
    else:
        keys = set(map(tuple, play_keys))

    # Only touch tracking files for the games we actually need.
    wanted_games = {int(g) for g, _ in keys}

    parts = []
    for path in sorted(TRACKING_DIR.glob("tracking_*.csv")):
        gid = int(path.stem.split("_")[-1])
        if gid not in wanted_games:
            continue
        df = pd.read_csv(path, usecols=TRACKING_COLS)
        # Keep only rows whose (gameId, playId) are requested.
        play_ids = {p for g, p in keys if g == gid}
        df = df[df["playId"].isin(play_ids)]
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame(columns=TRACKING_COLS)

    tracking = pd.concat(parts, ignore_index=True)
    tracking = _normalize_direction(tracking)
    return tracking


if __name__ == "__main__":
    plays = load_plays()
    pff = load_pff()
    players = load_players()

    print("plays shape   :", plays.shape)
    print("pff shape     :", pff.shape)
    print("players shape :", players.shape)

    print("\n=== plays.passResult ===")
    print(plays["passResult"].value_counts(dropna=False))

    print("\n=== plays.dropBackType ===")
    print(plays["dropBackType"].value_counts(dropna=False))

    # Smoke-test tracking load on the first few scramble throws.
    scramble = plays[
        plays["dropBackType"].str.contains("SCRAMBLE", na=False)
        & plays["passResult"].isin(["C", "I", "IN"])
    ]
    sample_keys = scramble[["gameId", "playId"]].head(5)
    tr = load_tracking(sample_keys)
    print("\n=== load_tracking smoke test (5 scramble throws) ===")
    print("tracking shape:", tr.shape)
    print("columns       :", list(tr.columns))
    print("playDirection :", tr["playDirection"].value_counts().to_dict())
