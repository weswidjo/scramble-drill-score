"""
Shared visual theme for the Scramble Drill Score project.

One place for colors, fonts, and helpers so every chart looks the same:
dark navy background, gold offense, slate defense, orange highlight, white
text, large readable fonts. Sample sizes get labeled on every bar (see
label_bars).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Palette (from the steering spec)
NAVY = "#0B1E33"        # background
GOLD = "#E8B23A"        # offense
SLATE = "#8FA3B8"       # defense
ORANGE = "#E4572E"      # highlight
WHITE = "#FFFFFF"       # text
GRID = "#1E3A52"        # subtle gridlines on navy

DATA_SOURCE = ("Data: NFL Big Data Bowl 2023 (NFL Next Gen Stats player "
               "tracking + PFF scouting), 2021 season Weeks 1\u20138.")


def apply_theme():
    """Apply the project theme to matplotlib rcParams."""
    mpl.rcParams.update({
        "figure.facecolor": NAVY,
        "axes.facecolor": NAVY,
        "savefig.facecolor": NAVY,
        "axes.edgecolor": SLATE,
        "axes.labelcolor": WHITE,
        "axes.titlecolor": WHITE,
        "text.color": WHITE,
        "xtick.color": WHITE,
        "ytick.color": WHITE,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.dpi": 100,
    })


def new_figure(ncols=1, figsize=(16, 9)):
    """Create a themed figure + axes sized 16:9 by default."""
    apply_theme()
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    return fig, axes


def title_block(fig, title, subtitle=None):
    """Standard title + optional subtitle at the top of the figure."""
    fig.suptitle(title, color=WHITE, fontsize=22, fontweight="bold",
                 x=0.5, y=0.97, ha="center")
    if subtitle:
        fig.text(0.5, 0.905, subtitle, color=SLATE, fontsize=13,
                 ha="center", va="top", wrap=True)


def footer(fig, extra=None):
    """Data-source footer along the bottom."""
    text = DATA_SOURCE if extra is None else f"{DATA_SOURCE}  {extra}"
    fig.text(0.5, 0.02, text, color=SLATE, fontsize=10, ha="center",
             va="bottom")


def label_bars(ax, bars, values, counts, fmt="{:.0f}", y_pad_frac=0.02):
    """Annotate each bar with its value and sample size (n=...).

    bars    : the BarContainer / list of Rectangle patches
    values  : the displayed value per bar
    counts  : the sample size per bar
    """
    ymax = ax.get_ylim()[1]
    pad = ymax * y_pad_frac
    for bar, val, n in zip(bars, values, counts):
        x = bar.get_x() + bar.get_width() / 2
        h = bar.get_height()
        # Value sits just above the bar; sample size stacks above the value so
        # neither label overlaps the bar edge.
        ax.text(x, h + pad, fmt.format(val), ha="center", va="bottom",
                color=WHITE, fontsize=13, fontweight="bold")
        ax.text(x, h + pad * 2.6, f"n={int(n)}", ha="center", va="bottom",
                color=SLATE, fontsize=10)
