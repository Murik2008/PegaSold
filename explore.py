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
