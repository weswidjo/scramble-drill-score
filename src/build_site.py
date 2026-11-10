"""
Build the static site data for the Scramble Drill Score project.

Writes:
  docs/data/site.json                      -- all summary numbers for the page
  docs/data/plays/<gameId>_<playId>.json   -- per-frame tracking for showcased plays

Every number is computed from the pipeline outputs (output/cache/*.csv) and the
tests in src/stats.py. Nothing is hard-coded.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stats as sds_stats
from load import load_plays, load_tracking, FIELD_Y

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "output" / "cache"
DOCS_DATA = ROOT / "docs" / "data"
PLAYS_DIR = DOCS_DATA / "plays"

MIN_SNAPS = 12
TOP_N = 40
BOTTOM_N = 10
SNAP_EVENTS = ["ball_snap", "autoevent_ballsnap"]


def _round(v, nd=1):
    return None if pd.isna(v) else round(float(v), nd)


# --------------------------------------------------------------------------- #
# Summary (site.json)
# --------------------------------------------------------------------------- #
def _headline(df):
    gain, held = sds_stats._groups(df)
    z = sds_stats.two_proportion_z(gain, held)
    ctrl = sds_stats.logistic_controlled(df)
    gx, gn = int(gain["completed"].sum()), len(gain)
    hx, hn = int(held["completed"].sum()), len(held)
    return {
        "gainers_caught_pct": _round(z["comp_gained"]),
        "gainers_detail": f"{gx} of {gn}",
        "nongainers_caught_pct": _round(z["comp_held"]),
        "nongainers_detail": f"{hx} of {hn}",
        "z_test_p": _round(z["p_value"], 3),
        "logistic_odds_ratio_per_yd": _round(ctrl["odds_ratio"], 2),
        "logistic_p_controlled": _round(ctrl["p_value"], 4),
        "logistic_controls": ctrl["controls"],
    }


def _key_numbers(df, windows):
    scr = windows[windows["is_scramble"]]
    targeted = df[df["is_scramble"] & df["targeted"]].drop_duplicates(
        ["gameId", "playId"])
    uni = sds_stats.logistic_separation(df)
    gain, held = sds_stats._groups(df)
    z = sds_stats.two_proportion_z(gain, held)
    return {
        "scramble_dropbacks": {
            "label": "Scramble dropbacks (tagged SCRAMBLE, threw C/I/IN)",
            "value": int(df[df["is_scramble"]]
                         .drop_duplicates(["gameId", "playId"]).shape[0])},
        "scramble_throws_with_start": {
            "label": "Scramble throws with a detected scramble start",
            "value": int(len(scr))},
        "median_scramble_duration_s": {
            "label": "Median scramble duration (s)",
            "value": _round(scr["scramble_duration_sec"].median(), 2)},
        "targeted_throws": {
            "label": "Targeted scramble throws analyzed",
            "value": int(len(targeted))},
        "z_test_p": {
            "label": "Completion lift p-value (gainers vs non-gainers)",
            "value": _round(z["p_value"], 3)},
        "odds_ratio_per_yd": {
            "label": "Odds of completion per +1 yd separation",
            "value": _round(uni["odds_ratio"], 2)},
    }


def _payoff_buckets(df):
    d = df[df["is_scramble"] & df["targeted"]].dropna(
        subset=["sep_gained"]).drop_duplicates(["gameId", "playId"]).copy()
    bins = [-np.inf, -1.0, 1.0, 3.0, np.inf]
    labels = ["Lost 1+", "Held \u00b11", "Gained 1\u20133", "Gained 3+"]
    d["bucket"] = pd.cut(d["sep_gained"], bins=bins, labels=labels, right=True)
    out = []
    for b in labels:
        g = d[d["bucket"] == b]
        if g.empty:
            continue
        out.append({
            "bucket": b,
            "n": int(len(g)),
            "completion_pct": _round(g["completed"].mean() * 100),
            "yards_per_attempt": _round(g["playResult"].mean()),
            "mean_sep_start": _round(g["sep_start"].mean()),
            "mean_sep_throw": _round(g["sep_throw"].mean()),
        })
    return out


def _leaderboard(df):
    scr = df[df["is_scramble"]].dropna(subset=["sep_gained"]).copy()
    scr["completed_target"] = scr["targeted"] & scr["completed"]
    agg = scr.groupby(["nflId", "displayName", "officialPosition"]).agg(
        snaps=("sep_gained", "size"),
        mean_sep_gained=("sep_gained", "mean"),
        targets=("targeted", "sum"),
        completions=("completed_target", "sum"),
    ).reset_index()
    agg = agg[agg["snaps"] >= MIN_SNAPS].sort_values(
        "mean_sep_gained", ascending=False)
    # team: most frequent possessionTeam for that player's scramble rows
    team = scr.groupby("nflId")["possessionTeam"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else "")
    out = []
    for _, r in agg.iterrows():
        out.append({
            "name": r["displayName"],
            "position": r["officialPosition"],
            "team": team.get(r["nflId"], ""),
            "snaps": int(r["snaps"]),
            "mean_sep_gained": _round(r["mean_sep_gained"], 2),
            "targets": int(r["targets"]),
            "completions": int(r["completions"]),
        })
    return out


def _by_position(df):
    d = df[df["is_scramble"]].dropna(subset=["sep_gained"]).drop_duplicates(
        ["gameId", "playId", "nflId"])
    out = []
    for pos in ["WR", "TE", "RB"]:
        g = d[d["officialPosition"] == pos]
        if g.empty:
            continue
        tgt = g[g["targeted"]]
        out.append({
            "position": pos,
            "n": int(len(g)),
            "mean_sep_start": _round(g["sep_start"].mean()),
            "mean_sep_throw": _round(g["sep_throw"].mean()),
            "mean_sep_gained": _round(g["sep_gained"].mean(), 2),
            "targeted_completion_pct": (
                _round(tgt["completed"].mean() * 100) if len(tgt) else None),
        })
    return out


def _qb_most_open(df):
    scr = df[df["is_scramble"]].dropna(subset=["sep_throw"]).copy()
    keys = ["gameId", "playId"]
    tcount = scr.groupby(keys)["targeted"].sum()
    good = tcount[tcount == 1].index
    scr = scr.set_index(keys).loc[good].reset_index()
    scr["is_most_open"] = (
        scr["sep_throw"] == scr.groupby(keys)["sep_throw"].transform("max"))
    tgt = scr[scr["targeted"]]
    most = tgt[tgt["is_most_open"]]
    notmost = tgt[~tgt["is_most_open"]]
    return {
        "pct_to_most_open": _round(tgt["is_most_open"].mean() * 100),
        "completion_pct_most_open": (
            _round(most["completed"].mean() * 100) if len(most) else None),
        "completion_pct_not_most_open": (
            _round(notmost["completed"].mean() * 100) if len(notmost) else None),
    }


def _selected_plays(df, plays, games):
    """Top TOP_N + bottom BOTTOM_N targeted scramble throws by sep_gained."""
    d = df[df["is_scramble"] & df["targeted"]].dropna(
        subset=["sep_gained"]).drop_duplicates(["gameId", "playId"]).copy()
    d = d.sort_values("sep_gained", ascending=False)
    picked = pd.concat([d.head(TOP_N), d.tail(BOTTOM_N)]).drop_duplicates(
        ["gameId", "playId"])

    meta = plays[["gameId", "playId", "playDescription", "quarter", "down",
                  "yardsToGo", "possessionTeam", "defensiveTeam"]]
    picked = picked.merge(meta, on=["gameId", "playId"], how="left",
                          suffixes=("", "_p"))
    wk = games[["gameId", "week"]]
    picked = picked.merge(wk, on="gameId", how="left")

    rows = []
    for _, r in picked.iterrows():
        rows.append({
            "gameId": int(r["gameId"]),
            "playId": int(r["playId"]),
            "receiver": r["displayName"],
            "position": r["officialPosition"],
            "team": r["possessionTeam"],
            "opponent": r.get("defensiveTeam", ""),
            "week": int(r["week"]) if pd.notna(r["week"]) else None,
            "quarter": int(r["quarter"]) if pd.notna(r["quarter"]) else None,
            "down_distance": (f"{int(r['down'])} & {int(r['yardsToGo'])}"
                              if pd.notna(r["down"]) else ""),
            "playDescription": r["playDescription"],
            "sep_gained": _round(r["sep_gained"], 2),
            "result": r["passResult"],
            "yards": int(r["playResult"]) if pd.notna(r["playResult"]) else None,
        })
    return rows, picked[["gameId", "playId", "nflId"]]


# --------------------------------------------------------------------------- #
# Per-play frame files
# --------------------------------------------------------------------------- #
def _write_play_files(selected_keys, windows, plays):
    keys = ["gameId", "playId"]
    tracking = load_tracking(selected_keys[keys].drop_duplicates())

    # jersey numbers + roles need re-attaching; load_tracking dropped jersey.
    # Re-read jersey from raw tracking via a light merge on the few games.
    from load import TRACKING_DIR
    jersey_parts = []
    for gid in selected_keys["gameId"].unique():
        jp = pd.read_csv(TRACKING_DIR / f"tracking_{gid}.csv",
                         usecols=["gameId", "playId", "nflId", "frameId",
                                  "jerseyNumber"])
        jersey_parts.append(jp)
    jersey = pd.concat(jersey_parts, ignore_index=True)
    tracking = tracking.merge(
        jersey, on=["gameId", "playId", "nflId", "frameId"], how="left")

    roles = pd.read_csv(ROOT / "data" / "pffScoutingData.csv")[
        ["gameId", "playId", "nflId", "pff_role"]]
    tracking = tracking.merge(roles, on=["gameId", "playId", "nflId"],
                              how="left")

    plays_meta = plays.set_index(["gameId", "playId"])
    win = windows.set_index(["gameId", "playId"])

    PLAYS_DIR.mkdir(parents=True, exist_ok=True)
    for (gid, pid), featured in selected_keys.set_index(
            keys)["nflId"].items():
        tp = tracking[(tracking["gameId"] == gid) &
                      (tracking["playId"] == pid)].copy()
        if tp.empty or (gid, pid) not in win.index:
            continue
        snap = int(win.loc[(gid, pid), "snap_frame"])
        throw = int(win.loc[(gid, pid), "throw_frame"])
        scr_start = win.loc[(gid, pid), "scramble_start_frame"]
        scr_start = int(scr_start) if pd.notna(scr_start) else None
        off_team = plays_meta.loc[(gid, pid), "possessionTeam"]
        los = float(plays_meta.loc[(gid, pid), "absoluteYardlineNumber"])

        # snap-1s (10 frames) through last frame
        start = snap - 10
        tp = tp[tp["frameId"] >= start].sort_values("frameId")

        def side(team):
            if team == "football":
                return "ball"
            return "off" if team == off_team else "def"

        frames = {}
        for _, p in tp.iterrows():
            frames.setdefault(int(p["frameId"]), []).append({
                "nflId": None if pd.isna(p["nflId"]) else int(p["nflId"]),
                "jersey": None if pd.isna(p["jerseyNumber"]) else int(
                    p["jerseyNumber"]),
                "team": side(p["team"]),
                "role": None if pd.isna(p["pff_role"]) else p["pff_role"],
                "x": round(float(p["x"]), 2),
                "y": round(float(p["y"]), 2),
            })
        frame_list = [{"frameId": f, "players": frames[f]}
                      for f in sorted(frames)]

        obj = {
            "gameId": int(gid), "playId": int(pid),
            "featured_nflId": int(featured),
            "snap_frame": snap,
            "scramble_start_frame": scr_start,
            "throw_frame": throw,
            "los_x": round(los, 2),
            "field_width": round(FIELD_Y, 2),
            "frames": frame_list,
        }
        out = PLAYS_DIR / f"{gid}_{pid}.json"
        out.write_text(json.dumps(obj, separators=(",", ":")))


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def build_site():
    df = pd.read_csv(CACHE_DIR / "receivers_with_outcome.csv")
    windows = pd.read_csv(CACHE_DIR / "windows.csv")
    plays = load_plays()
    games = pd.read_csv(ROOT / "data" / "games.csv")

    plays_list, selected_keys = _selected_plays(df, plays, games)

    site = {
        "headline": _headline(df),
        "key_numbers": _key_numbers(df, windows),
        "payoff_buckets": _payoff_buckets(df),
        "leaderboard": _leaderboard(df),
        "by_position": _by_position(df),
        "qb_most_open": _qb_most_open(df),
        "plays": plays_list,
    }

    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA / "site.json").write_text(json.dumps(site, indent=2))

    _write_play_files(selected_keys, windows, plays)

    total = sum(f.stat().st_size for f in DOCS_DATA.rglob("*") if f.is_file())
    mb = total / (1024 * 1024)
    print(f"wrote {DOCS_DATA} ({mb:.1f} MB, "
          f"{len(list(PLAYS_DIR.glob('*.json')))} play files)")
    if mb > 15:
        print(f"WARNING: docs/data is {mb:.1f} MB, exceeds 15 MB budget")
    return site


if __name__ == "__main__":
    build_site()
