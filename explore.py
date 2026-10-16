"""Exploratory analysis of the raw bina.az file.

Read-only. This script never writes to data/ — it only looks, draws and
reports. Every decision in src/data_prep.py should be traceable to a
measurement printed here.

    python explore.py --data data/bina_az_sale.csv --out reports/eda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no display needed; write straight to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Colours. One fixed set, used the same way in every figure, so that a colour
# always means the same thing. Taken from a colour-blind-safe palette.
# --------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT = "#0b0b0b", "#52514e"
GRID = "#e3e1dc"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "font.size": 9,
    "figure.dpi": 130,
})

TARGET = "price"
ADVERT_ID = "estate_rel_url_x"
APARTMENTS = ["Yeni tikili", "Köhnə tikili"]
RESIDENTIAL = APARTMENTS + ["Həyət evi/Bağ evi"]

# Columns we suspect are the target in disguise.
SUSPECTED_LEAKAGE = ["total_price", "unit_price"]

# Columns the scraper only ever writes when the answer is "yes".
PRESENCE_FLAGS = ["vip", "featured", "repair", "bill_of_sale", "mortgage"]

# Azerbaijan's real bounding box, used to spot impossible coordinates.
AZ_LAT, AZ_LNG = (38.3, 41.95), (44.7, 50.6)



class Report:
    """Collects every line printed, so the run can be saved as a text file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def header(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def table(self, frame: pd.DataFrame | pd.Series | str) -> None:
        self(frame if isinstance(frame, str) else frame.to_string())

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def save(fig: plt.Figure, out_dir: Path, name: str, rep: Report) -> None:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    rep(f"    [figure] {path}")


def style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """One place for the look of every axis: recessive grid, no box."""
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=INK_SOFT)
    ax.set_ylabel(ylabel, color=INK_SOFT)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def money(value: float) -> str:
    return f"{value:,.0f}"


def compact(value: float) -> str:
    """1234567 -> '1.2M'. Keeps crowded axes readable."""
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= size:
            text = f"{value / size:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return f"{value:,.0f}"

# ==========================================================================
# Light parsers. The raw file stores numbers inside text, e.g. "145 m²" or
# "7 / 9". We unpack them here ONLY so the plots can be drawn. The real,
# tested versions belong in src/data_prep.py.
# ==========================================================================
def parse_number_from_text(series: pd.Series) -> pd.Series:
    """'145 m²' -> 145.0 ; '1.3 sot' -> 1.3 ; '3 450 AZN/m²' -> 3450.0

    We delete spaces used as thousand separators, keep digits and one dot,
    and let anything unparseable become NaN.
    """
    cleaned = (
        series.astype("string")
        .str.replace(r"\s", "", regex=True)       # "3 450" -> "3450"
        .str.replace(",", ".", regex=False)       # decimal comma -> dot
        .str.extract(r"(\d+\.?\d*)", expand=False)  # first number in the text
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_floor(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'7 / 9' -> (floor 7, building height 9)."""
    parts = series.astype("string").str.extract(r"(\d+)\s*/\s*(\d+)")
    return (
        pd.to_numeric(parts[0], errors="coerce"),
        pd.to_numeric(parts[1], errors="coerce"),
    )

# ==========================================================================
# 2. Missing values
# ==========================================================================
def section_missing(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("2. MISSING VALUES — and whether 'missing' really means 'unknown'")
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    rep("Percentage missing per column (highest first, only columns > 0):")
    rep.table(missing[missing > 0].round(1))

    rep("")
    rep("THE TRAP. Some columns are never written as 'no'. The scraper only")
    rep("records them when the answer is 'yes', so NaN means 'no', not 'unknown'.")
    rep("Check: every one of these columns has exactly ONE distinct value.")
    rep("")
    for col in PRESENCE_FLAGS + ["İpoteka"]:
        values = df[col].dropna().unique().tolist()
        rep(f"  {col:<14} distinct non-missing values = {values}"
            f"   present in {df[col].notna().mean() * 100:5.1f}% of rows")
    rep("")
    rep("So these must be filled with 0/1, NOT dropped and NOT imputed with a")
    rep("mean. Dropping rows where 'vip' is missing would delete 91% of the data")
    rep("for no reason.")

    top = missing[missing > 0].head(18).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    flaglike = set(PRESENCE_FLAGS) | {"İpoteka"}
    colors = [ORANGE if c in flaglike else BLUE for c in top.index]
    ax.barh(top.index, top.to_numpy(), color=colors, height=0.72)
    for name, value in top.items():
        ax.text(value + 0.8, name, f"{value:.0f}%", va="center", color=INK_SOFT, fontsize=8)
    ax.set_xlim(0, 108)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style(ax, "Missing values per column", "% of rows missing")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="flag column — missing means 'no'"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="genuinely unknown"),
    ], frameon=False, loc="lower right", fontsize=8)
    save(fig, out_dir, "fig01_missing_values.png", rep)

# ==========================================================================
# 5. The target
# ==========================================================================
def section_target(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("5. THE TARGET `price` — what are we predicting?")
    price = df[TARGET]
    rep("Summary in AZN:")
    quantiles = price.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    rep.table(quantiles.apply(money))
    rep("")
    rep(f"The mean ({money(price.mean())}) is far above the median "
        f"({money(price.median())}).")
    rep("That gap is the signature of a long right tail: a few enormous values")
    rep("drag the average up. Look at both ends:")
    rep("")
    rep(f"  cheapest 10 prices: {sorted(price.nsmallest(10).tolist())}")
    rep(f"  priciest 5 prices:  {[money(v) for v in price.nlargest(5)]}")
    rep("")
    rep(f"  rows priced under 1,000 AZN:      {(price < 1_000).sum():,}")
    rep(f"  rows priced over 10,000,000 AZN:  {(price > 10_000_000).sum():,}")
    rep("  These are data-entry errors or whole buildings, not flats. A flat")
    rep("  does not cost 11 AZN. data_prep should cut an explicit price window")
    rep("  and the report must name the cut.")
    rep("")
    rep("WHY log(price). Prices span six orders of magnitude. In AZN, one")
    rep("mistake on a 10,000,000 flat outweighs thousands of ordinary flats, so")
    rep("squared error chases the outliers. log(price) turns 'twice as")
    rep("expensive' into a constant step, which is how people compare houses,")
    rep("and makes the distribution close to symmetric — see the figure.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    window = price[(price > 0) & (price < 1_000_000)]
    axes[0].hist(window, bins=80, color=BLUE)
    axes[0].axvline(price.median(), color=ORANGE, lw=2)
    axes[0].text(price.median() * 1.08, axes[0].get_ylim()[1] * 0.85,
                 f"median\n{money(price.median())} AZN", color=ORANGE, fontsize=8)
    axes[0].xaxis.set_major_formatter(lambda v, _: f"{v / 1000:,.0f}k")
    style(axes[0], "Raw price — squeezed against the left wall",
          "price, AZN (shown up to 1,000,000)", "number of adverts")

    logged = np.log10(price[price > 0])
    axes[1].hist(logged, bins=80, color=AQUA)
    axes[1].axvline(np.log10(price.median()), color=ORANGE, lw=2)
    # One tick per power of ten, labelled compactly, so nothing overlaps.
    decades = range(int(np.floor(logged.min())), int(np.ceil(logged.max())) + 1)
    axes[1].set_xticks(list(decades), [compact(10 ** d) for d in decades])
    style(axes[1], "log10(price) — roughly symmetric, a usable target",
          "price, AZN on a log scale", "number of adverts")
    save(fig, out_dir, "fig03_price_distribution.png", rep)
