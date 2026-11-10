"""
Scramble Drill Score -- single entry point.

Runs the full pipeline and produces every output:

  Pipeline (cached to output/cache/):
    load -> scramble windows -> receiver separation -> targets & outcomes

  Deliverables (output/):
    separation_payoff.png   coaches / fans: does getting open pay off?
    leaderboard.png/.csv    scouts: who gets open when the play breaks down?
    showcase_play.gif       broadcasters: the best scramble-drill rep
    side_by_side.gif        kept working vs stopped working

Usage:
    python main.py              # reuse caches if present, else build them
    python main.py --rebuild    # force the pipeline to recompute caches
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from scramble import build_windows            # noqa: E402
from metric import receiver_separation        # noqa: E402
from target import add_targets                # noqa: E402
from load import load_plays                   # noqa: E402
import charts                                 # noqa: E402
import animate                                # noqa: E402
import stats as sds_stats                     # noqa: E402

CACHE_DIR = ROOT / "output" / "cache"
OUTPUT_DIR = ROOT / "output"
RECEIVERS_OUTCOME = CACHE_DIR / "receivers_with_outcome.csv"


def build_pipeline():
    """Run the full data pipeline and write all cache files."""
    print("== building pipeline ==")
    windows, tracking = build_windows()
    print(f"  windows: {len(windows)} plays")

    receivers = receiver_separation(tracking, windows)
    receivers.to_csv(CACHE_DIR / "receivers.csv", index=False)
    print(f"  receivers: {len(receivers)} receiver-play rows")

    tagged = add_targets(receivers, load_plays())
    tagged.to_csv(RECEIVERS_OUTCOME, index=False)
    print(f"  tagged with outcomes -> {RECEIVERS_OUTCOME}")
    return tagged


def load_or_build(rebuild: bool) -> pd.DataFrame:
    if rebuild or not RECEIVERS_OUTCOME.exists():
        return build_pipeline()
    print(f"== using cached {RECEIVERS_OUTCOME.name} ==")
    return pd.read_csv(RECEIVERS_OUTCOME)


def main():
    ap = argparse.ArgumentParser(description="Scramble Drill Score pipeline")
    ap.add_argument("--rebuild", action="store_true",
                    help="force recompute of all cached intermediates")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_build(args.rebuild)

    print("\n== statistical validation ==")
    sds_stats.run(df)

    print("\n== charts ==")
    charts.separation_payoff(df)
    charts.leaderboard(df)

    print("\n== animations ==")
    good = animate.pick_showcase_play(df)
    bad = animate.pick_antishowcase_play(df)
    animate.animate_play(*good, OUTPUT_DIR / "showcase_play.gif")
    animate.side_by_side(good, bad, OUTPUT_DIR / "side_by_side.gif")

    print_headline_summary(df)

    print("\n== done ==")
    print("outputs in", OUTPUT_DIR)


def print_headline_summary(df: pd.DataFrame):
    """Print the headline numbers cited in the README (single source of truth)."""
    keys = ["gameId", "playId"]
    windows = pd.read_csv(CACHE_DIR / "windows.csv")

    n_scramble = int(windows["is_scramble"].sum())
    tgt_scr = df[df["is_scramble"] & df["targeted"]]
    m_targets = len(tgt_scr)

    tgt = df[df["targeted"]]
    comp = tgt.groupby("is_scramble")["completed"].mean() * 100

    # most-open rate on single-target scramble plays
    scr = df[df["is_scramble"]].dropna(subset=["sep_throw"]).copy()
    tcount = scr.groupby(keys)["targeted"].sum()
    good = tcount[tcount == 1].index
    scr = scr.set_index(keys).loc[good].reset_index()
    scr["is_most_open"] = (
        scr["sep_throw"] == scr.groupby(keys)["sep_throw"].transform("max"))
    most_open = scr[scr["targeted"]]["is_most_open"].mean() * 100

    print("\n== headline summary (README numbers) ==")
    print(f"scramble throws (N)                 : {n_scramble}")
    print(f"targeted scramble receivers (M)     : {m_targets}")
    print(f"completion%% traditional vs scramble : "
          f"{comp[False]:.1f}% vs {comp[True]:.1f}%")
    print(f"QB targeted the most-open receiver  : {most_open:.1f}% of scrambles")


if __name__ == "__main__":
    main()
