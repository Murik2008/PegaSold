"""Loading, cleaning, splitting and labelling the bina.az sale data.

Every statistic used to transform the data is measured on the TRAINING
SPLIT ONLY, so nothing leaks from validation or test into the model.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42 #for reproducibility
DATA_PATH = "data/bina_az_sale.csv"
OUTPUT_DIR = "data/processed"

LEAKAGE_COLUMNS = ["total_price", "unit_price","description"] #leakage columns

#The column holding the advert's page address, e.g. "/items/4521724".
ADVERT_URL = "estate_rel_url_x"

#Azerbaijan's bounding box. A coordinate outside this is broken data.
LAT_RANGE, LNG_RANGE = (38.3, 41.95), (44.7, 50.6)

#For def apply_scope
KEEP_CITY = "bakı"
KEEP_CATEGORIES = ("Yeni tikili", "Köhnə tikili", "Həyət evi/Bağ evi")

#Fixed, declared thresholds.
MIN_PRICE, MAX_PRICE = 10_000, 5_000_000            # AZN
MIN_PRICE_PER_M2, MAX_PRICE_PER_M2 = 200, 20_000    # AZN per m²
MIN_AREA, MAX_AREA = 10, 2_000                      # m²


def load(path: str|Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw scv"""
    df = pd.read_csv(path, low_memory=False)
    print(f"  [1] loaded {len(df):,} rows x {df.shape[1]} columns from {path}")
    return df

def number_from_text(series: pd.Series) -> pd.Series:
    """Convert raw to real numbers, anything unreadable becomes NaN"""
    cleaned = (
        series.astype("string")
        .str.replace(r"\s", "", regex=True)
        .str.replace(",", ".", regex=False)
        .str.extract(r"(\d+\.?\d*)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")

def parse(df: pd.DataFrame) -> pd.DataFrame:
    """Build tidy, correctly typed columns from the raw text ones."""
    
    out = pd.DataFrame(index=df.index)
    # "/items/4521724" -> "4521724". One id per advert, used for deduplication
    # and for the split fingerprints.
    out["item_id"] = df[ADVERT_URL].astype("string").str.extract(r"(\d+)", expand=False)
    out["scraped"] = pd.to_datetime(df["day_x"], format="%d.%m.%Y", errors="coerce")

    out["price"] = pd.to_numeric(df["price"], errors="coerce")

    area = number_from_text(df["Sahə"])
    is_land = df["Kateqoriya"].eq("Torpaq")
    out["area_m2"] = area.where(~is_land)   # land's area is in the wrong unit -> NaN

    # Convert to m² so every area in the frame is in one unit. Flats have no plot, so NaN
    # here means "no land"
    out["land_m2"] = number_from_text(df["Torpaq sahəsi"]) * 100

    out["rooms"] = pd.to_numeric(df["Otaq sayı"], errors="coerce")

    # "7 / 9" means the 7th floor of a 9-storey building.
    floors = df["Mərtəbə"].astype("string").str.extract(r"(\d+)\s*/\s*(\d+)")
    out["floor"] = pd.to_numeric(floors[0], errors="coerce")
    out["n_floors"] = pd.to_numeric(floors[1], errors="coerce")

    out["city"] = df["city"]
    out["category"] = df["Kateqoriya"]
    out["location"] = df["location"]

    # A coordinate outside Azerbaijan is broken. We blank the coordinate rather
    # than delete the row: the rest of that advert is still perfectly usable.
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lng = pd.to_numeric(df["lng"], errors="coerce")
    inside = lat.between(*LAT_RANGE) & lng.between(*LNG_RANGE)
    out["lat"] = lat.where(inside)
    out["lng"] = lng.where(inside)

    # These five are only ever written when the answer is yes, convert it to column with 1/0.
    out["has_repair"] = df["Təmir"].eq("var").astype(int)
    out["has_bill_of_sale"] = df["Çıxarış"].eq("var").astype(int)
    out["has_mortgage"] = df["İpoteka"].notna().astype(int)
    out["is_vip"] = df["vip"].notna().astype(int)
    out["is_featured"] = df["featured"].notna().astype(int)

    # Who is selling. Agents list 90% of adverts and at a higher median price.
    out["sold_by_owner"] = df["owner_title"].eq("mülkiyyətçi").astype(int)

    # `repair`, `bill_of_sale` and `mortgage` are not read at all. They
    # agree with Təmir / Çıxarış / İpoteka on 100.0% of rows (reports/eda
    # section 7) — the same column twice. We keep one of each pair.
    # `views` is not read either: it counts people who opened the advert AFTER
    # it went live, so it cannot be known about a flat you are about to price.

    print(f"  [2] parsed {out.shape[1]} working columns")
    return out

def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    #100,775 → 63,241, removing 37% of rows
    """Reduce the file to one row per property.
    Two passes, because there are two different kinds of duplicate, and both
    must happen before the split — otherwise the same flat lands in train and
    test and the test score becomes a memory test.
    """
    start = len(df)

    # Pass 1: the same advert scraped on several days. Sorting by scrape date
    # and keeping the last row keeps the most recent asking price.
    df = (df.sort_values(["item_id", "scraped"])
            .drop_duplicates(subset=["item_id"], keep="last"))
    after_advert = len(df)

    # Pass 2: one property posted twice under two advert addresses. If two
    # adverts agree on position, size, rooms AND price, it is the same flat.
    twin_key = ["lat", "lng", "area_m2", "rooms", "price"]
    complete = df[twin_key].notna().all(axis=1)
    is_twin = complete & df.duplicated(subset=twin_key, keep="first")
    df = df[~is_twin]
    after_twins = len(df)

    stats = {
        "rows_before": start,
        "rows_removed_by_repeat_scrapes": start - after_advert,
        "rows_removed_by_reposted_twins": after_advert - after_twins,
        "rows_after": after_twins,
    }
    print(f"  [3] deduplicated {start:,} -> {after_twins:,} rows "
          f"(-{start - after_advert:,} repeat scrapes, "
          f"-{after_advert - after_twins:,} reposts)")
    return df.reset_index(drop=True), stats

def apply_scope(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    #63,241 → 57,069 rows
    """Keep only Baku residential property.
    Two reasons, both measured in reports/eda section 11:
      - 99.6% of adverts are in Baku. The 260 elsewhere are a different market
        sampled 260 times: too few to learn from, enough to add noise.
      - land / commercial units / offices / garages record no rooms and no
        floor, and land's area is in sot rather than m². They are not the same
        prediction problem.
    `category` stays as a feature, so the model can still tell a house from a
    new-build flat.
    """
    start = len(df)
    keep = df["city"].eq(KEEP_CITY) & df["category"].isin(KEEP_CATEGORIES)
    df = df[keep]

    stats = {
        "rows_before": start,
        "rows_removed_by_scope": start - len(df),
        "rows_after": len(df),
        "kept_city": KEEP_CITY,
        "kept_categories": list(KEEP_CATEGORIES),
    }
    print(f"  [4] scope: {start:,} -> {len(df):,} rows "
          f"(Baku + {len(KEEP_CATEGORIES)} residential categories)")
    return df.reset_index(drop=True), stats

def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop records that cannot be true, and count every one we drop.

    The thresholds are the fixed numbers declared at the top of this file.
    They are NOT percentiles of this data: a rule like "drop the top 1%"
    computed over the whole file would let the test set decide what the
    training set contains.
    """
    start = len(df)
    df = df.copy()

    # A flat has no plot of land, so a missing land size means 0, not unknown.
    df["land_m2"] = df["land_m2"].fillna(0.0)

    # Each rule, kept separately so the report can say what each one cost.
    price_per_m2 = df["price"] / df["area_m2"]
    rules = {
        "price_outside_window": ~df["price"].between(MIN_PRICE, MAX_PRICE),
        "area_missing_or_outside_window": ~df["area_m2"].between(MIN_AREA, MAX_AREA),
        "price_per_m2_outside_window": ~price_per_m2.between(MIN_PRICE_PER_M2, MAX_PRICE_PER_M2),
        "floor_above_building_height": df["floor"] > df["n_floors"],
    }
    removed = {name: int(mask.sum()) for name, mask in rules.items()}

    drop = pd.concat(rules.values(), axis=1).any(axis=1)
    df = df[~drop]

    stats = {
        "rows_before": start,
        "removed_by_rule": removed,          # rules overlap, so these overlap too
        "total_removed_by_domain_filters": int(drop.sum()),
        "rows_after": len(df),
        "price_window_AZN": [MIN_PRICE, MAX_PRICE],
        "price_per_m2_window_AZN": [MIN_PRICE_PER_M2, MAX_PRICE_PER_M2],
        "area_window_m2": [MIN_AREA, MAX_AREA],
    }
    print(f"  [5] filters: {start:,} -> {len(df):,} rows (-{int(drop.sum()):,})")
    for name, n in removed.items():
        print(f"        {name}: {n:,}")
    return df.reset_index(drop=True), stats


