# Project: Scramble Drill Score

Question: When the QB scrambles, does it pay for WR/TE/RB to keep working to get open?

Audience: NFL coaches (actionable), broadcasters (visual), scouts (leaderboard).

Data (data/): plays.csv, pffScoutingData.csv, players.csv, games.csv, tracking/tracking_<gameId>.csv (10 fps).

Key facts:

- Scramble throws = plays.dropBackType contains "SCRAMBLE" AND passResult in ["C","I","IN"].

- Snap event: "ball_snap" or "autoevent_ballsnap". Throw event: "pass_forward" or "autoevent_passforward".

- Tracking ends 0.5s (5 frames) after the throw on every play. Outcomes come from plays.csv (passResult, playResult).

- QB = pff_role "Pass"; receivers = pff_role "Pass Route"; defenders = pff_role "Coverage".

- The football is rows where team == "football" (nflId is NaN).

- Normalize direction: if playDirection == "left", x = 120 - x and y = 160/3 - y.

- Scramble start = first frame between snap and throw where the QB's |y - y_at_snap| > 3 yards.

- Separation = distance to nearest Coverage defender. sep_gained = sep_at_throw - sep_at_scramble_start.

Code rules: Python, pandas/numpy/matplotlib/seaborn only. Vectorize with numpy, no per-row Python loops over frames.

Modules live in src/, outputs in output/, a single entry point main.py. Cache intermediate results to output/cache/ as CSV.

Visual style: one shared theme in src/style.py (dark navy background #0B1E33, offense #E8B23A gold, defense #8FA3B8 slate,

highlight #E4572E orange, white text, large readable fonts, sample sizes labeled on every bar).
