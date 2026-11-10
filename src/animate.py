"""
Broadcaster-facing play animation for the Scramble Drill Score project.

animate_play renders a single scramble play as a GIF: cropped field, players as
dots (jersey numbers only for the key players), a QB scramble trail, the
featured receiver highlighted with a live separation readout, story banners
(SCRAMBLE / before-after separation / throw callout), a generated title, a
legend, and a play-description caption below the field. The play plays from the
snap to the throw, then holds the final frame for 2 seconds before looping.
Uses PillowWriter (no ffmpeg needed).

pick_showcase_play chooses the most illustrative completed scramble throw.
"""
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import style
from load import (
    load_plays, load_pff, load_players, DATA_DIR, FIELD_X, FIELD_Y,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
CACHE_DIR = OUTPUT_DIR / "cache"

SNAP_EVENTS = ["ball_snap", "autoevent_ballsnap"]
THROW_EVENTS = ["pass_forward", "autoevent_passforward"]
QB_LATERAL_THRESHOLD = 3.0
FB_BROWN = "#8B5A2B"
FPS = 10
HOLD_FRAMES = 20  # 2s hold on the throw frame before looping


# --------------------------------------------------------------------------- #
# Play selection
# --------------------------------------------------------------------------- #
def pick_showcase_play(df: pd.DataFrame):
    """Pick a completed scramble throw with the biggest separation gained.

    Restricted to plays with scramble_duration_sec >= 1.5. Returns
    (gameId, playId, nflId).
    """
    windows = pd.read_csv(CACHE_DIR / "windows.csv")
    m = df.merge(windows[["gameId", "playId", "scramble_duration_sec"]],
                 on=["gameId", "playId"], how="left")
    cand = m[(m["is_scramble"]) & (m["targeted"]) & (m["completed"]) &
             (m["scramble_duration_sec"] >= 1.5)].copy()
    cand = cand.dropna(subset=["sep_gained"]).sort_values(
        "sep_gained", ascending=False)
    top = cand.iloc[0]
    return int(top["gameId"]), int(top["playId"]), int(top["nflId"])


# --------------------------------------------------------------------------- #
# Tracking with jersey numbers (animation needs per-frame detail)
# --------------------------------------------------------------------------- #
def _load_play_tracking(game_id, play_id):
    cols = ["gameId", "playId", "nflId", "frameId", "jerseyNumber", "team",
            "playDirection", "x", "y", "s", "event"]
    tr = pd.read_csv(DATA_DIR / "tracking" / f"tracking_{game_id}.csv",
                     usecols=cols)
    tr = tr[tr["playId"] == play_id].copy()
    # Direction normalization (same rule as load.py).
    left = tr["playDirection"].values == "left"
    tr.loc[left, "x"] = FIELD_X - tr.loc[left, "x"]
    tr.loc[left, "y"] = FIELD_Y - tr.loc[left, "y"]

    pff = load_pff()
    pff = pff[(pff["gameId"] == game_id) & (pff["playId"] == play_id)]
    tr = tr.merge(pff[["nflId", "pff_role"]], on="nflId", how="left")
    return tr


def _event_frame(tr, events):
    f = tr.loc[tr["event"].isin(events), "frameId"]
    return int(f.min()) if not f.empty else None


def _scramble_start_frame(tr, snap_frame, throw_frame):
    qb = tr[tr["pff_role"] == "Pass"]
    qb = qb[(qb["frameId"] >= snap_frame) & (qb["frameId"] <= throw_frame)]
    if qb.empty:
        return None
    y0 = qb.loc[qb["frameId"] == snap_frame, "y"]
    if y0.empty:
        return None
    y0 = float(y0.iloc[0])
    crossed = qb[(qb["y"] - y0).abs() > QB_LATERAL_THRESHOLD]
    return int(crossed["frameId"].min()) if not crossed.empty else None


# --------------------------------------------------------------------------- #
# Main animation
# --------------------------------------------------------------------------- #
def animate_play(game_id, play_id, receiver_nflId, out_path):
    tr = _load_play_tracking(game_id, play_id)
    plays = load_plays()
    players = load_players()
    meta = plays[(plays["gameId"] == game_id) &
                 (plays["playId"] == play_id)].iloc[0]
    rec_name = players.loc[players["nflId"] == receiver_nflId,
                           "displayName"].iloc[0]

    snap_frame = _event_frame(tr, SNAP_EVENTS) or int(tr["frameId"].min())
    throw_frame = _event_frame(tr, THROW_EVENTS) or int(tr["frameId"].max())
    scr_frame = _scramble_start_frame(tr, snap_frame, throw_frame)

    # Offense = possession team; football separate; defense = the rest.
    off_team = meta["possessionTeam"]
    tr["side"] = np.where(tr["team"] == "football", "ball",
                          np.where(tr["team"] == off_team, "off", "def"))

    qb_id = tr.loc[tr["pff_role"] == "Pass", "nflId"]
    qb_id = int(qb_id.iloc[0]) if not qb_id.empty else None

    # (1) Play runs snap -> throw only, then holds the final frame.
    frames = [f for f in sorted(tr["frameId"].unique())
              if snap_frame <= f <= throw_frame]

    # Field crop around the line of scrimmage; expand to fit the action and
    # guarantee a landscape frame.
    los = float(meta["absoluteYardlineNumber"])
    x_lo, x_hi = los - 10, los + 30
    anim_x = tr.loc[tr["frameId"].isin(frames), "x"]
    if not anim_x.empty:
        x_lo = min(x_lo, float(anim_x.min()) - 3)
        x_hi = max(x_hi, float(anim_x.max()) + 3)
    if (x_hi - x_lo) < FIELD_Y + 6:
        pad = (FIELD_Y + 6 - (x_hi - x_lo)) / 2
        x_lo, x_hi = x_lo - pad, x_hi + pad

    # Separation: featured receiver -> nearest defender, per frame.
    def nearest_def(frame_df, rx, ry):
        d = frame_df[frame_df["side"] == "def"]
        if d.empty:
            return np.nan, None, None, None
        dist = np.hypot(d["x"].values - rx, d["y"].values - ry)
        i = int(dist.argmin())
        return (float(dist.min()), float(d["x"].values[i]),
                float(d["y"].values[i]), int(d["nflId"].values[i]))

    sep_at = {}
    for f in frames:
        fr = tr[tr["frameId"] == f]
        r = fr[fr["nflId"] == receiver_nflId]
        if r.empty:
            sep_at[f] = (np.nan, None, None, None)
        else:
            sep_at[f] = nearest_def(fr, float(r["x"].iloc[0]),
                                    float(r["y"].iloc[0]))

    sep_start = sep_at.get(scr_frame, (np.nan,))[0] if scr_frame else np.nan
    sep_throw = sep_at.get(throw_frame, (np.nan,))[0]
    sep_gained = (sep_throw - sep_start
                  if not np.isnan(sep_start) and not np.isnan(sep_throw)
                  else np.nan)
    result_txt = "Complete" if meta["passResult"] == "C" else "Incomplete"
    qb_name = (players.loc[players["nflId"] == qb_id, "displayName"].iloc[0]
               if qb_id is not None else "QB")

    # (2) Generated title from data.
    verb = "keeps working" if (not np.isnan(sep_gained) and sep_gained >= 0) \
        else "stops working"
    title_txt = (f"{rec_name} {verb}: {sep_gained:+.1f} yds of separation "
                 f"during {qb_name}'s scramble")

    # (1) Build sequence: frames, then hold the throw frame.
    seq = list(frames) + [throw_frame] * HOLD_FRAMES

    # --------------------------------------------------------------------- #
    # Figure: 16:9, field panel ~90% width.
    # --------------------------------------------------------------------- #
    style.apply_theme()
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.05, 0.14, 0.90, 0.72])  # field takes ~90% width

    def draw_field():
        ax.clear()
        ax.add_patch(Rectangle((x_lo, 0), x_hi - x_lo, FIELD_Y,
                               facecolor="#13432A", zorder=0))
        for yd in range(int(x_lo) - int(x_lo) % 5, int(x_hi) + 5, 5):
            ax.axvline(yd, color="white", lw=0.8, alpha=0.35, zorder=1)
        # (5) Yard numbers every 10 yds (field tens digit).
        for yd in range(int(x_lo) - int(x_lo) % 10, int(x_hi) + 10, 10):
            if x_lo + 1 < yd < x_hi - 1:
                marker = int(round(yd)) % 100
                ax.text(yd, 3, f"{marker:02d}", color="white", alpha=0.45,
                        fontsize=13, fontweight="bold", ha="center",
                        va="bottom", zorder=1)
                ax.text(yd, FIELD_Y - 3, f"{marker:02d}", color="white",
                        alpha=0.45, fontsize=13, fontweight="bold",
                        ha="center", va="top", zorder=1)
        ax.axvline(los, color="#3FA7FF", lw=2.5, zorder=2)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, FIELD_Y)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    def label_box(xx, yy, text, color, fontsize=11):
        """Dark rounded label so text never overlaps a dot."""
        ax.text(xx, yy, text, color=color, fontsize=fontsize,
                fontweight="bold", ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#05101C",
                          edgecolor=color, linewidth=1.2, alpha=0.92))

    def update(i):
        f = seq[i]
        draw_field()

        past = tr[(tr["frameId"] >= snap_frame) & (tr["frameId"] <= f)]
        # (4) Brighter, thicker trails.
        qb_tr = past[past["pff_role"] == "Pass"].sort_values("frameId")
        if len(qb_tr) > 1:
            ax.plot(qb_tr["x"], qb_tr["y"], color=style.WHITE, lw=3,
                    alpha=0.6, zorder=3)
        rec_tr = past[past["nflId"] == receiver_nflId].sort_values("frameId")
        if len(rec_tr) > 1:
            ax.plot(rec_tr["x"], rec_tr["y"], color=style.ORANGE, lw=3,
                    alpha=0.9, zorder=3)

        fr = tr[tr["frameId"] == f]
        dist, dx, dy, nd_id = sep_at[f]

        # Draw players (dots). Jersey numbers only for featured rec, QB, and
        # the nearest defender (6).
        show_numbers = {receiver_nflId, qb_id, nd_id}
        qb_xy = None
        for _, p in fr.iterrows():
            if p["side"] == "ball":
                continue  # drawn after the QB so his number stays readable
            is_feat = p["nflId"] == receiver_nflId
            color = (style.ORANGE if is_feat else
                     style.GOLD if p["side"] == "off" else style.SLATE)
            size = 460 if is_feat else 340
            ax.scatter(p["x"], p["y"], s=size, color=color,
                       edgecolors="white", linewidths=1.4, zorder=5)
            if p["nflId"] == qb_id:
                qb_xy = (float(p["x"]), float(p["y"]))
            if p["nflId"] in show_numbers and pd.notna(p["jerseyNumber"]):
                ax.annotate(str(int(p["jerseyNumber"])), (p["x"], p["y"]),
                            color=style.NAVY, ha="center", va="center",
                            fontsize=9, fontweight="bold", zorder=6)

        # (6) Football slightly offset from, and on top of, the QB.
        ball = fr[fr["side"] == "ball"]
        if not ball.empty:
            bx, by = float(ball["x"].iloc[0]), float(ball["y"].iloc[0])
            if qb_xy is not None:
                bx, by = qb_xy[0] + 0.8, qb_xy[1] + 0.8
            ax.scatter(bx, by, s=70, color=FB_BROWN, edgecolors="white",
                       zorder=7)

        # Separation line + offset label (3).
        r = fr[fr["nflId"] == receiver_nflId]
        if not r.empty and dx is not None:
            rx, ry = float(r["x"].iloc[0]), float(r["y"].iloc[0])
            ax.plot([rx, dx], [ry, dy], color="yellow", ls="--", lw=2,
                    zorder=4)
            # place label 1.5 yds from the receiver along the line
            vx, vy = dx - rx, dy - ry
            norm = np.hypot(vx, vy) or 1.0
            lx = rx + vx / norm * 1.5
            ly = ry + vy / norm * 1.5 + 1.2
            label_box(lx, ly, f"{dist:.1f} yds", "yellow")

        # Clock top-left.
        t = (f - snap_frame) / 10.0
        ax.text(0.02, 0.96, f"{t:.1f}s after snap", transform=ax.transAxes,
                color=style.WHITE, fontsize=14, fontweight="bold",
                va="top", ha="left", zorder=8)

        # (7) SCRAMBLE banner from scramble start; persistent "at start" marker.
        if scr_frame is not None and f >= scr_frame:
            ax.text(0.5, 0.95, "SCRAMBLE", transform=ax.transAxes,
                    color=style.NAVY, fontsize=20, fontweight="bold",
                    ha="center", va="top", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=style.ORANGE,
                              edgecolor="white"))
            if not np.isnan(sep_start):
                ax.text(0.98, 0.96,
                        f"{sep_start:.1f} yds at scramble start",
                        transform=ax.transAxes, color=style.SLATE,
                        fontsize=12, fontweight="bold", va="top", ha="right",
                        zorder=8,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#05101C", edgecolor=style.SLATE,
                                  alpha=0.9))

        # Throw callout during the held frames.
        if f == throw_frame and i >= len(frames) - 1:
            callout = (f"{rec_name}: {sep_start:.1f} \u2192 {sep_throw:.1f} "
                       f"yds (+{sep_gained:.1f})   |   Result: {result_txt}, "
                       f"{int(meta['playResult'])} yds")
            ax.text(0.5, 0.05, callout, transform=ax.transAxes,
                    color=style.WHITE, fontsize=15, fontweight="bold",
                    ha="center", va="bottom", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor=style.NAVY,
                              edgecolor=style.GOLD, linewidth=2.5))
        return []

    anim = FuncAnimation(fig, update, frames=len(seq), blit=False)

    # (2) Title above the field.
    fig.text(0.5, 0.95, title_txt, color=style.WHITE, fontsize=20,
             fontweight="bold", ha="center", va="top")

    # (2) Legend.
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=style.GOLD,
               markeredgecolor="white", markersize=12, label="offense"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=style.SLATE,
               markeredgecolor="white", markersize=12, label="defense"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=style.ORANGE, markeredgecolor="white",
               markersize=12, label="featured receiver"),
        Line2D([0], [0], color="yellow", ls="--", lw=2,
               label="distance to nearest defender"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.905), ncol=4, frameon=False,
               labelcolor=style.WHITE, fontsize=11)

    # (5) Play description below the field, white, larger.
    desc = "  ".join(textwrap.wrap(str(meta["playDescription"]), 120))
    fig.text(0.5, 0.075, desc, color=style.WHITE, fontsize=12,
             ha="center", va="top")
    style.footer(fig)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print("wrote", out_path)
    print(f"  {rec_name}: sep {sep_start:.1f} -> {sep_throw:.1f} "
          f"(+{sep_gained:.1f}), {result_txt}, {int(meta['playResult'])} yds")
    return out_path


# --------------------------------------------------------------------------- #
# Side-by-side contrast ("Kept working" vs "Stopped working")
# --------------------------------------------------------------------------- #
def pick_antishowcase_play(df: pd.DataFrame):
    """Pick the scramble throw with the most NEGATIVE separation gained.

    Completed or incomplete, scramble_duration_sec >= 1.5. Returns
    (gameId, playId, nflId).
    """
    windows = pd.read_csv(CACHE_DIR / "windows.csv")
    m = df.merge(windows[["gameId", "playId", "scramble_duration_sec"]],
                 on=["gameId", "playId"], how="left")
    cand = m[(m["is_scramble"]) & (m["targeted"]) &
             (m["passResult"].isin(["C", "I"])) &
             (m["scramble_duration_sec"] >= 1.5)].copy()
    cand = cand.dropna(subset=["sep_gained"]).sort_values("sep_gained")
    top = cand.iloc[0]
    return int(top["gameId"]), int(top["playId"]), int(top["nflId"])


def _prepare_play(game_id, play_id, receiver_nflId):
    """Precompute everything needed to render a play onto an axis."""
    tr = _load_play_tracking(game_id, play_id)
    plays = load_plays()
    players = load_players()
    meta = plays[(plays["gameId"] == game_id) &
                 (plays["playId"] == play_id)].iloc[0]
    rec_name = players.loc[players["nflId"] == receiver_nflId,
                           "displayName"].iloc[0]

    snap_frame = _event_frame(tr, SNAP_EVENTS) or int(tr["frameId"].min())
    throw_frame = _event_frame(tr, THROW_EVENTS) or int(tr["frameId"].max())
    scr_frame = _scramble_start_frame(tr, snap_frame, throw_frame)

    off_team = meta["possessionTeam"]
    tr["side"] = np.where(tr["team"] == "football", "ball",
                          np.where(tr["team"] == off_team, "off", "def"))
    qb_id = tr.loc[tr["pff_role"] == "Pass", "nflId"]
    qb_id = int(qb_id.iloc[0]) if not qb_id.empty else None

    frames = [f for f in sorted(tr["frameId"].unique())
              if snap_frame <= f <= throw_frame]

    los = float(meta["absoluteYardlineNumber"])
    x_lo, x_hi = los - 10, los + 30
    anim_x = tr.loc[tr["frameId"].isin(frames), "x"]
    if not anim_x.empty:
        x_lo = min(x_lo, float(anim_x.min()) - 3)
        x_hi = max(x_hi, float(anim_x.max()) + 3)
    if (x_hi - x_lo) < FIELD_Y + 6:
        pad = (FIELD_Y + 6 - (x_hi - x_lo)) / 2
        x_lo, x_hi = x_lo - pad, x_hi + pad

    def nearest_def(frame_df, rx, ry):
        d = frame_df[frame_df["side"] == "def"]
        if d.empty:
            return np.nan, None, None, None
        dist = np.hypot(d["x"].values - rx, d["y"].values - ry)
        i = int(dist.argmin())
        return (float(dist.min()), float(d["x"].values[i]),
                float(d["y"].values[i]), int(d["nflId"].values[i]))

    sep_at = {}
    for f in frames:
        fr = tr[tr["frameId"] == f]
        r = fr[fr["nflId"] == receiver_nflId]
        sep_at[f] = (nearest_def(fr, float(r["x"].iloc[0]),
                                 float(r["y"].iloc[0]))
                     if not r.empty else (np.nan, None, None, None))

    sep_start = sep_at.get(scr_frame, (np.nan,))[0] if scr_frame else np.nan
    sep_throw = sep_at.get(throw_frame, (np.nan,))[0]
    sep_gained = (sep_throw - sep_start
                  if not np.isnan(sep_start) and not np.isnan(sep_throw)
                  else np.nan)

    return dict(
        tr=tr, meta=meta, rec_name=rec_name, receiver=receiver_nflId,
        qb_id=qb_id, snap_frame=snap_frame, throw_frame=throw_frame,
        scr_frame=scr_frame, frames=frames, x_lo=x_lo, x_hi=x_hi, los=los,
        sep_at=sep_at, sep_start=sep_start, sep_throw=sep_throw,
        sep_gained=sep_gained,
        result_txt="Complete" if meta["passResult"] == "C" else "Incomplete",
    )


def _render_panel(ax, P, frame_idx):
    """Draw play P onto ax at the given sequence index (holds last frame)."""
    frames = P["frames"]
    f = frames[frame_idx] if frame_idx < len(frames) else frames[-1]
    tr, rec, qb_id = P["tr"], P["receiver"], P["qb_id"]

    ax.clear()
    ax.add_patch(Rectangle((P["x_lo"], 0), P["x_hi"] - P["x_lo"], FIELD_Y,
                           facecolor="#13432A", zorder=0))
    for yd in range(int(P["x_lo"]) - int(P["x_lo"]) % 5, int(P["x_hi"]) + 5, 5):
        ax.axvline(yd, color="white", lw=0.7, alpha=0.3, zorder=1)
    ax.axvline(P["los"], color="#3FA7FF", lw=2.2, zorder=2)
    ax.set_xlim(P["x_lo"], P["x_hi"])
    ax.set_ylim(0, FIELD_Y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    past = tr[(tr["frameId"] >= P["snap_frame"]) & (tr["frameId"] <= f)]
    qb_tr = past[past["pff_role"] == "Pass"].sort_values("frameId")
    if len(qb_tr) > 1:
        ax.plot(qb_tr["x"], qb_tr["y"], color=style.WHITE, lw=3, alpha=0.6,
                zorder=3)
    rec_tr = past[past["nflId"] == rec].sort_values("frameId")
    if len(rec_tr) > 1:
        ax.plot(rec_tr["x"], rec_tr["y"], color=style.ORANGE, lw=3, alpha=0.9,
                zorder=3)

    fr = tr[tr["frameId"] == f]
    dist, dx, dy, nd_id = P["sep_at"][f]
    show_numbers = {rec, qb_id, nd_id}
    qb_xy = None
    for _, p in fr.iterrows():
        if p["side"] == "ball":
            continue
        is_feat = p["nflId"] == rec
        color = (style.ORANGE if is_feat else
                 style.GOLD if p["side"] == "off" else style.SLATE)
        ax.scatter(p["x"], p["y"], s=360 if is_feat else 260, color=color,
                   edgecolors="white", linewidths=1.2, zorder=5)
        if p["nflId"] == qb_id:
            qb_xy = (float(p["x"]), float(p["y"]))
        if p["nflId"] in show_numbers and pd.notna(p["jerseyNumber"]):
            ax.annotate(str(int(p["jerseyNumber"])), (p["x"], p["y"]),
                        color=style.NAVY, ha="center", va="center",
                        fontsize=8, fontweight="bold", zorder=6)
    ball = fr[fr["side"] == "ball"]
    if not ball.empty:
        bx, by = (qb_xy[0] + 0.8, qb_xy[1] + 0.8) if qb_xy else (
            float(ball["x"].iloc[0]), float(ball["y"].iloc[0]))
        ax.scatter(bx, by, s=55, color=FB_BROWN, edgecolors="white", zorder=7)

    r = fr[fr["nflId"] == rec]
    if not r.empty and dx is not None:
        rx, ry = float(r["x"].iloc[0]), float(r["y"].iloc[0])
        ax.plot([rx, dx], [ry, dy], color="yellow", ls="--", lw=2, zorder=4)
        vx, vy = dx - rx, dy - ry
        norm = np.hypot(vx, vy) or 1.0
        lx, ly = rx + vx / norm * 1.5, ry + vy / norm * 1.5 + 1.2
        ax.text(lx, ly, f"{dist:.1f} yds", color="yellow", fontsize=10,
                fontweight="bold", ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#05101C",
                          edgecolor="yellow", alpha=0.92))

    t = (f - P["snap_frame"]) / 10.0
    ax.text(0.03, 0.95, f"{t:.1f}s", transform=ax.transAxes, color=style.WHITE,
            fontsize=12, fontweight="bold", va="top", ha="left", zorder=8)
    if P["scr_frame"] is not None and f >= P["scr_frame"]:
        ax.text(0.5, 0.95, "SCRAMBLE", transform=ax.transAxes,
                color=style.NAVY, fontsize=14, fontweight="bold", ha="center",
                va="top", zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=style.ORANGE,
                          edgecolor="white"))

    if frame_idx >= len(frames) - 1:
        callout = (f"{P['rec_name']}: {P['sep_start']:.1f} \u2192 "
                   f"{P['sep_throw']:.1f} yds ({P['sep_gained']:+.1f})\n"
                   f"{P['result_txt']}, {int(P['meta']['playResult'])} yds")
        ax.text(0.5, 0.05, callout, transform=ax.transAxes, color=style.WHITE,
                fontsize=11, fontweight="bold", ha="center", va="bottom",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=style.NAVY,
                          edgecolor=style.GOLD, linewidth=2))


def side_by_side(good, bad, out_path):
    """Render two plays side by side, frame-synced, as a GIF.

    good / bad : (gameId, playId, nflId) tuples.
    The shorter play is padded by holding its last frame.
    """
    P_good = _prepare_play(*good)
    P_bad = _prepare_play(*bad)

    # Frame-sync: longest play length + the hold, both padded to match.
    n = max(len(P_good["frames"]), len(P_bad["frames"])) + HOLD_FRAMES

    style.apply_theme()
    fig = plt.figure(figsize=(16, 9))
    ax_l = fig.add_axes([0.03, 0.10, 0.45, 0.74])
    ax_r = fig.add_axes([0.52, 0.10, 0.45, 0.74])

    def update(i):
        _render_panel(ax_l, P_good, i)
        _render_panel(ax_r, P_bad, i)
        ax_l.set_title("Kept working", color=style.GOLD, fontsize=18,
                       fontweight="bold", pad=12)
        ax_r.set_title("Stopped working", color=style.SLATE, fontsize=18,
                       fontweight="bold", pad=12)
        return []

    anim = FuncAnimation(fig, update, frames=n, blit=False)
    fig.text(0.5, 0.95,
             "Keep working: two scramble throws, two outcomes",
             color=style.WHITE, fontsize=20, fontweight="bold", ha="center",
             va="top")
    style.footer(fig)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print("wrote", out_path)
    print(f"  LEFT  {P_good['rec_name']}: {P_good['sep_gained']:+.1f} yds")
    print(f"  RIGHT {P_bad['rec_name']}: {P_bad['sep_gained']:+.1f} yds")
    return out_path


if __name__ == "__main__":
    df = pd.read_csv(CACHE_DIR / "receivers_with_outcome.csv")
    gid, pid, nid = pick_showcase_play(df)
    print(f"showcase play: game {gid} play {pid} receiver {nid}")
    animate_play(gid, pid, nid, OUTPUT_DIR / "showcase_play.gif")

    good = pick_showcase_play(df)
    bad = pick_antishowcase_play(df)
    print(f"side-by-side: good {good}  bad {bad}")
    side_by_side(good, bad, OUTPUT_DIR / "side_by_side.gif")
