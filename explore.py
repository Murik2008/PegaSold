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

    # ==========================================================================
    # 3. Duplicates
    # ==========================================================================
    def section_duplicates(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
        rep.header("3. DUPLICATES — is every row a different flat?")
        rep(f"Identical rows (all 51 columns equal): {df.duplicated().sum():,}")
        rep("")
        rep("But each advert has a stable page address in `estate_rel_url_x`")
        rep("(e.g. /items/4521724). If the same address appears twice, the same")
        rep("flat was scraped twice.")
        listings = df["estate_rel_url_x"].value_counts()
        rep(f"  rows in file:            {len(df):,}")
        rep(f"  distinct adverts:        {listings.size:,}")
        rep(f"  extra (repeat) rows:     {len(df) - listings.size:,}"
            f"  = {(len(df) - listings.size) / len(df) * 100:.1f}% of the file")
        rep("")
        rep("How many times each advert appears:")
        rep.table(listings.value_counts().sort_index().rename("adverts").to_frame())
        rep("")
        rep("Does the price change between the copies of one advert?")
        spread = df.groupby("estate_rel_url_x")[TARGET].nunique()
        rep(f"  adverts with a single price across their copies: {(spread == 1).sum():,}")
        rep(f"  adverts whose price changed:                     {(spread > 1).sum():,}")
        rep("")
        rep("WHY THIS MATTERS. If we split the data randomly without removing these,")
        rep("the same flat can land in both train and test. The model then recalls a")
        rep("price it has already seen and the test score is a lie.")

        counts = listings.value_counts().sort_index()
        counts = counts[counts.index <= 6]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(counts.index.astype(str), counts.to_numpy(), color=BLUE, width=0.6)
        for x, y in zip(counts.index.astype(str), counts.to_numpy()):
            ax.text(x, y, f"{y:,}", ha="center", va="bottom", color=INK_SOFT, fontsize=8)
        ax.xaxis.grid(False)
        style(ax, "Most adverts were scraped once, many twice",
              "times the same advert appears in the file", "number of adverts")
        save(fig, out_dir, "fig02_duplicate_adverts.png", rep)

    # ==========================================================================
    # 4. Leakage
    # ==========================================================================
    def section_leakage(df: pd.DataFrame, rep: Report) -> None:
        rep.header("4. LEAKAGE — which columns already contain the answer?")
        rep("A feature leaks if it could only be known once the price is known.")
        rep("Training on one makes the score perfect and the model worthless.")
        rep("")

        same = (df["total_price"] == df[TARGET]).mean() * 100
        rep(f"`total_price` equals `price` in {same:.2f}% of rows.")
        rep("  -> it is the target under a second name. DROP.")
        rep("")

        both = df[["unit_price_azn_m2", "area_m2"]].notna().all(axis=1)
        reconstructed = df.loc[both, "unit_price_azn_m2"] * df.loc[both, "area_m2"]
        error = (reconstructed - df.loc[both, TARGET]).abs() / df.loc[both, TARGET]
        rep(f"`unit_price` is price per m². On the {both.sum():,} rows where both it")
        rep("and the area are present, unit_price x area reconstructs the price:")
        rep(f"  within  1% of the true price: {(error < 0.01).mean() * 100:.1f}% of rows")
        rep(f"  within  5% of the true price: {(error < 0.05).mean() * 100:.1f}% of rows")
        rep("  -> it is the target divided by a feature we keep. DROP.")
        rep("")
        rep("`description` is free text written by the seller and very often states")
        rep("the price in words. Treat as leakage unless the price is stripped out:")
        sample = df["description"].dropna().head(4000).astype(str)
        mentions = sample.str.contains(r"\b(?:qiymət|azn|man\.?)\b", case=False, regex=True).mean()
        rep(f"  {mentions * 100:.0f}% of a 4,000-row sample mention a price word.")
        rep("")
        rep("VERDICT — drop before training: total_price, unit_price, description.")

    # ==========================================================================
    # 10. Task B preview
    # ==========================================================================
    def section_tiers(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
        rep.header("10. TASK B PREVIEW — the premium / standard split")
        threshold = df[TARGET].median()
        premium = df[TARGET] > threshold
        rep("Task B asks for a price tier made by splitting at the median.")
        rep(f"On the whole file the median is {money(threshold)} AZN, which gives:")
        rep(f"  standard (<= median): {(~premium).sum():,}  ({(~premium).mean() * 100:.1f}%)")
        rep(f"  premium  (>  median): {premium.sum():,}  ({premium.mean() * 100:.1f}%)")
        rep("")
        rep("The split is balanced by construction, so accuracy is a fair metric")
        rep("here and a coin flip scores 50%. IMPORTANT: the brief says the median")
        rep("must be computed on the TRAINING SET ONLY. Using the median of the")
        rep("whole file would let test-set prices decide the class boundary, which")
        rep("is leakage.")
        rep("")
        rep("Share of premium adverts inside each property type:")
        share = (df.assign(premium=premium)
                 .groupby("Kateqoriya")["premium"].agg(["mean", "size"])
                 .sort_values("mean", ascending=False))
        share.columns = ["premium_share", "rows"]
        rep.table(share.assign(premium_share=lambda t: (t["premium_share"] * 100).round(1)))

        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        order = share.index
        vals = share["premium_share"].to_numpy() * 100
        ax.barh(order[::-1], vals[::-1], color=BLUE, height=0.7)
        ax.axvline(50, color=ORANGE, lw=1.5, ls="--")
        ax.text(51, -0.45, "50% = balanced", color=ORANGE, fontsize=8)
        for name, value in zip(order[::-1], vals[::-1]):
            ax.text(value + 1.5, name, f"{value:.0f}%", va="center", color=INK_SOFT, fontsize=8)
        ax.set_xlim(0, 108)
        ax.yaxis.grid(False)
        style(ax, "Share of 'premium' adverts within each property type",
              "% of the type priced above the overall median")
        save(fig, out_dir, "fig11_price_tiers.png", rep)

        # ==========================================================================
        # 14. Conclusions
        # ==========================================================================
        def section_conclusions(df: pd.DataFrame, rep: Report) -> None:
            rep.header("14. CONCLUSIONS — what data_prep.py has to do")
            rep("DROP — no information:")
            rep(f"  identifiers and scrape bookkeeping ({len(ID_AND_BOOKKEEPING)} columns): "
                f"{', '.join(ID_AND_BOOKKEEPING)}")
            rep("  constant columns: currency_x, currency_y, shop_title, (Binanın növü: 99.8% empty)")
            rep("")
            rep("DROP — leakage:")
            rep("  total_price (identical to price), unit_price (price per m²), description")
            rep("")
            rep("DROP — not knowable before the sale:")
            rep("  views")
            rep("")
            rep("DEDUPLICATE:")
            rep("  one row per estate_rel_url_x, BEFORE splitting into train/test")
            rep("")
            rep("DECIDE A SCOPE:")
            rep("  city is 99.7% Baku, and 'Torpaq' (bare land) / 'Obyekt' / 'Qaraj' are")
            rep("  priced on different logic than flats. Narrowing the scope makes the")
            rep("  task coherent; whatever we choose must be stated in the report.")
            rep("")
            rep("PARSE OUT OF TEXT:")
            rep("  Sahə -> area_m2,  Mərtəbə -> floor + building_floors,")
            rep("  Torpaq sahəsi -> land_sot")
            rep("")
            rep("FILL FLAGS WITH 0/1, NOT MEANS:")
            rep(f"  {', '.join(PRESENCE_FLAGS)}")
            rep("")
            rep("CUT IMPOSSIBLE VALUES (and count every row removed):")
            rep("  price outside a sane window, area <= 5 or > 1000 m²,")
            rep("  floor above building height, coordinates outside Azerbaijan")
            rep("")
            rep("ENCODE:")
            rep("  Kateqoriya and the frequent `location` values as one-hot, with a")
            rep("  shared 'other' bucket; yes/no columns as 0/1")
            rep("")
            rep("TARGET:")
            rep("  Task A on log(price);  Task B on a median split of the TRAINING set")
            rep("")
            rep("SCALING:")
            rep("  the SVM needs standardised features (area is in hundreds, flags are")
            rep("  0/1). Fit the scaler on train only, then apply it to valid/test.")
            rep("  The decision tree does not care about scale.")

    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(description="Read-only exploration of the bina.az dataset.")
        p.add_argument("--data", default="data/bina_az_sale.csv", help="path to the raw CSV")
        p.add_argument("--out", default="reports/eda", help="where the report and figures go")
        return p.parse_args()

    def main() -> int:
        args = parse_args()
        out_dir = Path(args.out)
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        rep = Report()
        rep("bina.az sale — exploratory data analysis")
        rep(f"source file: {args.data}")
        rep("this script reads the data only; it writes nothing into data/")

        raw = pd.read_csv(args.data)
        df = build_exploration_frame(raw)

        section_overview(raw, rep)
        section_missing(raw, fig_dir, rep)
        section_duplicates(raw, fig_dir, rep)
        section_leakage(df, rep)
        section_target(df, fig_dir, rep)
        section_numeric(df, fig_dir, rep)
        section_categorical(df, fig_dir, rep)
        section_geography(df, fig_dir, rep)
        section_correlation(df, fig_dir, rep)
        section_tiers(df, fig_dir, rep)
        section_scope(df, fig_dir, rep)
        section_dedup(df, fig_dir, rep)
        section_window(df, fig_dir, rep)
        section_conclusions(df, rep)

        rep.save(out_dir / "eda_report.txt")
        print(f"\nReport written to {out_dir / 'eda_report.txt'}")
        print(f"Figures written to {fig_dir}")
        return 0

    if __name__ == "__main__":
        raise SystemExit(main())


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