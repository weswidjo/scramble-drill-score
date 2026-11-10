"""
Charts for the Scramble Drill Score project. All charts use the shared theme
in src/style.py.

separation_payoff: does gaining separation during a scramble pay off? Buckets
targeted scramble-throw receivers by separation gained and shows completion %
and yards per attempt per bucket.

leaderboard: which players get open when the play breaks down? Ranks route
runners by mean separation gained on scramble plays (top 10 / bottom 5).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import style

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
CACHE_DIR = OUTPUT_DIR / "cache"

BUCKET_ORDER = ["Lost 1+ yd", "Held (\u00b11)", "Gained 1\u20133 yds",
                "Gained 3+ yds"]

MIN_SNAPS = 12  # minimum scramble snaps to qualify for the leaderboard

# Position colors (offense gold tones + highlight), with a legend.
POSITION_COLORS = {
    "WR": style.GOLD,
    "TE": style.ORANGE,
    "RB": style.SLATE,
}


def _bucket(sep_gained: pd.Series) -> pd.Categorical:
    """Bucket sep_gained into the four labeled categories."""
    # edges: (-inf,-1] , (-1,1] , (1,3] , (3, inf)
    bins = [-np.inf, -1.0, 1.0, 3.0, np.inf]
    labels = BUCKET_ORDER
    return pd.cut(sep_gained, bins=bins, labels=labels, right=True)


def separation_payoff(df: pd.DataFrame):
    """Build the separation-payoff two-panel chart from the receivers table."""
    # Targeted receivers on scramble plays only (one row per targeted attempt).
    scr = df[df["is_scramble"] & df["targeted"]].copy()
    scr = scr.dropna(subset=["sep_gained"])
    scr = scr.drop_duplicates(["gameId", "playId"])  # one per attempt

    scr["bucket"] = _bucket(scr["sep_gained"])

    grp = scr.groupby("bucket", observed=True)
    comp_pct = grp["completed"].mean() * 100
    ypa = grp["playResult"].mean()
    counts = grp.size()

    # keep bucket order and only buckets that have data
    present = [b for b in BUCKET_ORDER if b in comp_pct.index and counts[b] > 0]
    comp_pct = comp_pct.reindex(present)
    ypa = ypa.reindex(present)
    counts = counts.reindex(present)

    overall_comp = scr["completed"].mean() * 100
    overall_ypa = scr["playResult"].mean()

    fig, axes = style.new_figure(ncols=2, figsize=(16, 9))
    x = np.arange(len(present))

    # Panel 1: completion %
    ax = axes[0]
    bars = ax.bar(x, comp_pct.values, color=style.GOLD, width=0.65,
                  zorder=3)
    ax.axhline(overall_comp, ls="--", color=style.ORANGE, lw=2, zorder=2,
               label=f"overall {overall_comp:.0f}%")
    ax.set_title("Completion %", color=style.WHITE)
    ax.set_ylabel("completion %")
    ax.set_ylim(0, max(comp_pct.max(), overall_comp) * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=15, ha="right")
    style.label_bars(ax, bars, comp_pct.values, counts.values, fmt="{:.0f}%")
    ax.legend(loc="upper left", facecolor=style.NAVY, edgecolor=style.SLATE,
              labelcolor=style.WHITE)

    # Panel 2: yards per attempt
    ax = axes[1]
    bars = ax.bar(x, ypa.values, color=style.SLATE, width=0.65, zorder=3)
    ax.axhline(overall_ypa, ls="--", color=style.ORANGE, lw=2, zorder=2,
               label=f"overall {overall_ypa:.1f} yd")
    ax.set_title("Yards per attempt", color=style.WHITE)
    ax.set_ylabel("yards per attempt (playResult)")
    ax.set_ylim(min(0, ypa.min() * 1.25), max(ypa.max(), overall_ypa) * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=15, ha="right")
    style.label_bars(ax, bars, ypa.values, counts.values, fmt="{:.1f}")
    ax.legend(loc="upper left", facecolor=style.NAVY, edgecolor=style.SLATE,
              labelcolor=style.WHITE)

    style.title_block(
        fig,
        "Keep working: receivers who get open during a scramble get paid",
        "2021 Weeks 1\u20138, targeted WR/TE/RB on QB scramble throws. "
        "Separation gained = defender distance at throw minus at scramble "
        "start.")
    style.footer(fig)

    fig.tight_layout(rect=[0, 0.05, 1, 0.88])
    out = OUTPUT_DIR / "separation_payoff.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print("wrote", out)
    print("\nbucket summary:")
    print(pd.DataFrame({"completion_%": comp_pct.round(1),
                        "ypa": ypa.round(2),
                        "n": counts}))
    return out


def leaderboard(df: pd.DataFrame):
    """Rank route runners by mean separation gained on scramble plays.

    Shows the top 10 and bottom 5 qualifiers (min MIN_SNAPS scramble snaps),
    bars colored by position, with a visible gap between the two groups.
    """
    scr = df[df["is_scramble"]].copy()
    scr = scr.dropna(subset=["sep_gained"])

    # Completion on a target = the player was targeted AND the pass completed.
    scr["completed_target"] = scr["targeted"] & scr["completed"]

    agg = scr.groupby(["nflId", "displayName", "officialPosition"]).agg(
        snaps=("sep_gained", "size"),
        mean_sep_gained=("sep_gained", "mean"),
        targets=("targeted", "sum"),
        completions=("completed_target", "sum"),
    ).reset_index()

    agg = agg[agg["snaps"] >= MIN_SNAPS].sort_values(
        "mean_sep_gained", ascending=False).reset_index(drop=True)
    agg.to_csv(OUTPUT_DIR / "leaderboard.csv", index=False)

    # Limit to WR/TE/RB per the project scope.
    agg = agg[agg["officialPosition"].isin(POSITION_COLORS)]

    top = agg.head(10)
    bottom = agg.tail(5)

    # Build plotting order bottom-up: bottom 5 first, gap, then top 10.
    GAP = object()  # sentinel for the visual gap row
    rows = list(bottom.iloc[::-1].itertuples(index=False)) + [GAP] + \
        list(top.iloc[::-1].itertuples(index=False))

    labels, values, colors, snaps, positions = [], [], [], [], []
    for r in rows:
        if r is GAP:
            labels.append("")
            values.append(0.0)
            colors.append(style.NAVY)
            snaps.append(None)
            positions.append(None)
            continue
        labels.append(r.displayName)
        values.append(r.mean_sep_gained)
        colors.append(POSITION_COLORS[r.officialPosition])
        snaps.append(r.snaps)
        positions.append(r.officialPosition)

    fig, ax = style.new_figure(ncols=1, figsize=(16, 9))
    y = np.arange(len(rows))
    bars = ax.barh(y, values, color=colors, zorder=3, height=0.72)

    ax.axvline(0, color=style.WHITE, lw=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean separation gained during scramble (yards)")

    xmax = max(values) if values else 1
    xmin = min(values) if values else 0
    span = (xmax - xmin) or 1
    for yi, (v, n, pos) in enumerate(zip(values, snaps, positions)):
        if n is None:
            continue
        off = span * 0.015
        ha = "left" if v >= 0 else "right"
        xtext = v + (off if v >= 0 else -off)
        ax.text(xtext, yi, f"{v:+.2f}", va="center", ha=ha,
                color=style.WHITE, fontsize=12, fontweight="bold")
        ax.text(xtext, yi - 0.3, f"{int(n)} snaps", va="center", ha=ha,
                color=style.SLATE, fontsize=9)

    ax.set_xlim(xmin - span * 0.18, xmax + span * 0.18)

    # Legend for positions.
    handles = [Patch(facecolor=POSITION_COLORS[p], label=p)
               for p in ["WR", "TE", "RB"]]
    ax.legend(handles=handles, loc="lower right", facecolor=style.NAVY,
              edgecolor=style.SLATE, labelcolor=style.WHITE,
              title="position", title_fontsize=11)

    # Group labels on the gap.
    gap_idx = rows.index(GAP)
    ax.text(xmin - span * 0.16, gap_idx + 3, "TOP 10", color=style.GOLD,
            fontsize=12, fontweight="bold", va="center")
    ax.text(xmin - span * 0.16, gap_idx - 3, "BOTTOM 5", color=style.SLATE,
            fontsize=12, fontweight="bold", va="center")

    style.title_block(
        fig,
        "Who gets open when the play breaks down?",
        f"2021 Weeks 1\u20138, route runners with \u2265{MIN_SNAPS} scramble "
        "snaps. Separation gained = defender distance at throw minus at "
        "scramble start.")
    style.footer(fig)

    fig.tight_layout(rect=[0, 0.05, 1, 0.88])
    out = OUTPUT_DIR / "leaderboard.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print("wrote", out)
    print("wrote", OUTPUT_DIR / "leaderboard.csv")
    print("\ntop 10 by mean sep_gained:")
    print(top[["displayName", "officialPosition", "snaps",
               "mean_sep_gained", "targets", "completions"]].to_string(
        index=False))
    return out


if __name__ == "__main__":
    df = pd.read_csv(CACHE_DIR / "receivers_with_outcome.csv")
    separation_payoff(df)
    leaderboard(df)
