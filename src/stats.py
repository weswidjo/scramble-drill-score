"""
Statistical validation for the Scramble Drill Score project.

Answers the question a skeptical analytics judge will ask: is the headline
finding real, or small-sample noise? Four tests, all on scramble throws:

  1. Logistic regression: does separation at the throw predict completion?
     (coefficient, p-value, odds ratio per +1 yd)
  2. Two-proportion z-test: is the completion gap between receivers who GAINED
     1+ yd vs HELD (within 1 yd) significant?
  3. Welch t-test: is yards per attempt higher for separation-gainers?
  4. Bootstrap 95% CI on the completion-rate lift (gained vs held).

Pure numpy + scipy (no statsmodels dependency). Writes a one-line verdict to
output/stats_verdict.txt and prints a full report.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
CACHE_DIR = OUTPUT_DIR / "cache"


# --------------------------------------------------------------------------- #
# 1. Logistic regression (Newton-Raphson, with Wald p-values)
# --------------------------------------------------------------------------- #
def _logit_fit(X, y, iters=50, tol=1e-8):
    """Fit logistic regression via Newton-Raphson. X includes intercept col.

    Returns (beta, cov) where cov is the coefficient covariance matrix.
    """
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        # Hessian = X' W X ; gradient = X'(y - p)
        XtW = X.T * W
        H = XtW @ X
        g = X.T @ (y - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    W = p * (1 - p)
    H = (X.T * W) @ X
    cov = np.linalg.inv(H)
    return beta, cov


def logistic_separation(df):
    """Completion ~ separation_at_throw on scramble throws (one row/target)."""
    d = df[df["is_scramble"] & df["targeted"]].dropna(
        subset=["sep_throw", "completed"]).drop_duplicates(["gameId", "playId"])
    y = d["completed"].astype(float).to_numpy()
    sep = d["sep_throw"].to_numpy()
    X = np.column_stack([np.ones_like(sep), sep])

    beta, cov = _logit_fit(X, y)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    odds_ratio = np.exp(beta[1])  # per +1 yd of separation
    return {
        "n": len(d),
        "coef": beta[1],
        "p_value": pvals[1],
        "odds_ratio": odds_ratio,
    }


def logistic_controlled(df):
    """Completion ~ sep_throw + sep_start + position dummies.

    Returns the odds ratio and p-value for sep_throw after controlling for how
    open the receiver started (sep_start) and his position.
    """
    d = df[df["is_scramble"] & df["targeted"]].dropna(
        subset=["sep_throw", "sep_start", "completed"]).drop_duplicates(
        ["gameId", "playId"]).copy()
    y = d["completed"].astype(float).to_numpy()

    cols = [np.ones(len(d)), d["sep_throw"].to_numpy(),
            d["sep_start"].to_numpy()]
    names = ["intercept", "sep_throw", "sep_start"]
    # Position dummies (drop first level as reference).
    pos = pd.get_dummies(d["officialPosition"], prefix="pos", drop_first=True)
    for c in pos.columns:
        cols.append(pos[c].to_numpy().astype(float))
        names.append(c)
    X = np.column_stack(cols)

    beta, cov = _logit_fit(X, y)
    se = np.sqrt(np.diag(cov))
    zscores = beta / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(zscores)))
    i = names.index("sep_throw")
    return {
        "n": len(d),
        "coef": float(beta[i]),
        "p_value": float(pvals[i]),
        "odds_ratio": float(np.exp(beta[i])),
        "controls": names[2:],
    }


# --------------------------------------------------------------------------- #
# 2 & 3 & 4. Group comparisons: gained 1+ vs held (within 1 yd)
# --------------------------------------------------------------------------- #
def _groups(df):
    d = df[df["is_scramble"] & df["targeted"]].dropna(
        subset=["sep_gained"]).drop_duplicates(["gameId", "playId"]).copy()
    gained = d[d["sep_gained"] >= 1.0]
    held = d[d["sep_gained"].abs() < 1.0]
    return gained, held


def two_proportion_z(gained, held):
    """Z-test for difference in completion rate (gained vs held)."""
    x1, n1 = int(gained["completed"].sum()), len(gained)
    x2, n2 = int(held["completed"].sum()), len(held)
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"comp_gained": p1 * 100, "comp_held": p2 * 100,
            "lift_pts": (p1 - p2) * 100, "z": z, "p_value": p,
            "n_gained": n1, "n_held": n2}


def ypa_ttest(gained, held):
    """Welch t-test for yards per attempt (gained vs held)."""
    a = gained["playResult"].to_numpy()
    b = held["playResult"].to_numpy()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"ypa_gained": a.mean(), "ypa_held": b.mean(), "t": t, "p_value": p}


def bootstrap_lift_ci(gained, held, n_boot=10000):
    """Bootstrap 95% CI on the completion-rate lift (gained - held).

    Uses a fresh seeded RNG so the CI is identical no matter when or how often
    this is called (single source of truth across main.py and stats.py).
    """
    rng = np.random.default_rng(42)
    g = gained["completed"].to_numpy().astype(float)
    h = held["completed"].to_numpy().astype(float)
    lifts = np.empty(n_boot)
    for i in range(n_boot):
        gb = rng.choice(g, size=len(g), replace=True).mean()
        hb = rng.choice(h, size=len(h), replace=True).mean()
        lifts[i] = (gb - hb) * 100
    lo, hi = np.percentile(lifts, [2.5, 97.5])
    return {"lift_pts": (g.mean() - h.mean()) * 100, "ci_lo": lo, "ci_hi": hi}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def run(df):
    log = logistic_separation(df)
    gained, held = _groups(df)
    z = two_proportion_z(gained, held)
    t = ypa_ttest(gained, held)
    boot = bootstrap_lift_ci(gained, held)

    print("== statistical validation (scramble throws) ==\n")
    print(f"1. Logistic regression  completion ~ separation at throw "
          f"(n={log['n']})")
    print(f"   coefficient = {log['coef']:+.3f}  "
          f"p = {log['p_value']:.4f}  "
          f"odds ratio per +1 yd = {log['odds_ratio']:.2f}x")
    print(f"   -> each extra yard of separation multiplies the odds of a "
          f"completion by {log['odds_ratio']:.2f}.\n")

    print(f"2. Completion rate: gained 1+ yd vs held (within 1 yd)")
    print(f"   gained {z['comp_gained']:.1f}% (n={z['n_gained']})  vs  "
          f"held {z['comp_held']:.1f}% (n={z['n_held']})")
    print(f"   lift = {z['lift_pts']:+.1f} pts  z = {z['z']:.2f}  "
          f"p = {z['p_value']:.4f}\n")

    print(f"3. Yards per attempt: gained vs held")
    print(f"   gained {t['ypa_gained']:.2f}  vs  held {t['ypa_held']:.2f}  "
          f"t = {t['t']:.2f}  p = {t['p_value']:.4f}\n")

    print(f"4. Bootstrap 95% CI on completion lift (10k resamples)")
    print(f"   lift = {boot['lift_pts']:+.1f} pts  "
          f"95% CI [{boot['ci_lo']:.1f}, {boot['ci_hi']:.1f}]\n")

    sig = log["p_value"] < 0.05 and z["p_value"] < 0.05
    verdict = (
        f"Separation at the throw significantly predicts completion on "
        f"scramble throws (logistic p={log['p_value']:.3f}, "
        f"odds ratio {log['odds_ratio']:.2f}x per yard). Receivers who gained "
        f"1+ yd completed {z['comp_gained']:.1f}% vs {z['comp_held']:.1f}% for "
        f"those who held even (+{z['lift_pts']:.1f} pts, p={z['p_value']:.3f}, "
        f"95% CI [{boot['ci_lo']:.1f}, {boot['ci_hi']:.1f}] pts). "
        + ("The effect is statistically significant."
           if sig else "Effect not significant at 0.05."))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "stats_verdict.txt").write_text(verdict + "\n")
    print("VERDICT:", verdict)
    print("\nwrote", OUTPUT_DIR / "stats_verdict.txt")
    return {"logistic": log, "ztest": z, "ttest": t, "bootstrap": boot,
            "verdict": verdict}


if __name__ == "__main__":
    df = pd.read_csv(CACHE_DIR / "receivers_with_outcome.csv")
    run(df)
