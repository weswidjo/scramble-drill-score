"""
Targeted-receiver identification for the Scramble Drill Score project.

Parses the intended receiver out of plays.playDescription, matches it to a
route runner on the possessing team via a short name (first initial + last
name), and tags the receivers table with target/outcome columns.

Outputs output/cache/receivers_with_outcome.csv.
"""
import re
from pathlib import Path

import pandas as pd

from load import load_plays
from metric import build_receivers

CACHE_DIR = Path(__file__).resolve().parent.parent / "output" / "cache"

TARGET_RE = re.compile(
    r"pass (?:incomplete )?(?:short|deep) \w+ to "
    r"([A-Z][\w']*\.\s?[\w'\-]+)")


def _short_name(display_name: str) -> str:
    """First initial + '.' + last name, e.g. 'Stefon Diggs' -> 'S.Diggs'."""
    if not isinstance(display_name, str):
        return ""
    parts = display_name.split(" ", 1)
    if len(parts) != 2:
        return display_name
    first, last = parts
    return f"{first[0]}.{last}"


def _norm(s: str) -> str:
    """Normalize a short name for matching (drop internal whitespace)."""
    return re.sub(r"\s+", "", s) if isinstance(s, str) else ""


def parse_targets(plays: pd.DataFrame) -> pd.DataFrame:
    """Extract the targeted short name per play from playDescription."""
    target = plays["playDescription"].str.extract(TARGET_RE, expand=False)
    out = plays[["gameId", "playId", "possessionTeam",
                 "passResult", "playResult"]].copy()
    out["target_short"] = target.map(_norm)
    return out


def add_targets(receivers: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """Tag each receiver row with targeted / outcome columns.

    A receiver is `targeted` when their short name matches the parsed target
    AND they are a route runner on the possessing team for that play.
    """
    keys = ["gameId", "playId"]
    targets = parse_targets(plays)

    df = receivers.merge(targets, on=keys, how="left")
    df["short_name"] = df["displayName"].map(_short_name).map(_norm)

    # We only have possession-team filtering implicitly: route runners in the
    # receivers table are offensive players, so matching short name to the
    # play's parsed target identifies the target among that play's routes.
    df["targeted"] = (
        (df["short_name"] == df["target_short"]) &
        df["target_short"].astype(bool)
    )

    df["completed"] = df["passResult"] == "C"
    return df


def build():
    """Run receiver separation + target tagging; cache and return the table."""
    receivers = build_receivers()
    plays = load_plays()
    tagged = add_targets(receivers, plays)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tagged.to_csv(CACHE_DIR / "receivers_with_outcome.csv", index=False)
    return tagged


if __name__ == "__main__":
    tagged = build()
    keys = ["gameId", "playId"]

    # % of plays with exactly one matched target among route runners.
    per_play = tagged.groupby(keys)["targeted"].sum()
    pct_one = (per_play == 1).mean() * 100
    print(f"plays with exactly one matched target: {pct_one:.1f}% "
          f"({(per_play == 1).sum()} of {len(per_play)} plays)")
    print(f"  plays with zero matches : {(per_play == 0).sum()}")
    print(f"  plays with >1 match     : {(per_play > 1).sum()}")

    print("\n=== crosstab: is_scramble x completed (targeted rows) ===")
    tgt = tagged[tagged["targeted"]]
    print(pd.crosstab(tgt["is_scramble"], tgt["completed"],
                      margins=True))

    print("\n=== completion rate among targeted, by is_scramble ===")
    print(tgt.groupby("is_scramble")["completed"].mean())

    print(f"\ncached -> {CACHE_DIR / 'receivers_with_outcome.csv'}")
