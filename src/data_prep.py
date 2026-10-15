"""Loading, cleaning, splitting and labelling the bina.az sale data.

Every statistic used to transform the data is measured on the TRAINING
SPLIT ONLY, so nothing leaks from validation or test into the model.
"""
from __future__ import annotations

from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import hashlib #for fingerprint
import json #for report

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


def fingerprint(ids: pd.Series) -> str:
    """A short, stable hash of a set of item ids."""
    joined = "\n".join(sorted(ids.astype(str)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def run(input_path: str | Path, output_dir: str | Path) -> dict:
    """Run every step in order and return the report dictionary."""
    print(f"\ndata_prep: {input_path} -> {output_dir}")

    raw = load(input_path)
    raw_rows = len(raw)

    df = parse(raw)
    df, dedup_stats = deduplicate(df)
    df, scope_stats = apply_scope(df)
    df, filter_stats = apply_filters(df)

    parts = split(df)
    rules = fit_preprocessor(parts["train"])

    # Build the training matrix once here purely to count the columns for the
    # report; save() rebuilds it per split.
    n_features = build_features(parts["train"], rules).shape[1]
    print(f"  [8] built {n_features} features per row")

    report = {
        "random_seed": RANDOM_SEED,
        "raw_rows": raw_rows,

        "rows_after_deduplication": dedup_stats["rows_after"],
        "rows_removed_by_item_deduplication":
            dedup_stats["rows_removed_by_repeat_scrapes"]
            + dedup_stats["rows_removed_by_reposted_twins"],
        "deduplication": dedup_stats,

        "scope": {
            "rows_after_scope": scope_stats["rows_after"],
            "rows_removed_by_scope": scope_stats["rows_removed_by_scope"],
            "kept_city": scope_stats["kept_city"],
            "kept_categories": scope_stats["kept_categories"],
        },

        "rows_after_cleaning": filter_stats["rows_after"],
        "domain_filters": filter_stats,

        # The ranges every surviving row is guaranteed to lie inside. We do not
        # winsorize (clip an extreme value back to a fence): a row outside a
        # window is a broken record, so step 5 removes it outright. The bounds
        # are recorded here so src/validate.py can confirm they hold.
        "value_ranges": {
            "method": "rows outside the window are removed, not clipped",
            "fences": {
                "price": {"low": MIN_PRICE, "high": MAX_PRICE},
                "area_m2": {"low": MIN_AREA, "high": MAX_AREA},
            },
        },

        "final_rows": len(df),
        "split_rows": {k: len(v) for k, v in parts.items()},
        "split_item_id_sha256": {k: fingerprint(v["item_id"]) for k, v in parts.items()},

        "leakage_columns_removed": LEAKAGE_COLUMNS,
        "tier_threshold_AZN": rules["tier_threshold_AZN"],

        "n_processed_features": n_features,
        "numeric_features": CONTINUOUS_FEATURES,
        "binary_features": BINARY_FEATURES,
        "n_location_columns": len(rules["locations"]) + 1,   # + "other"
        "feature_names": rules["feature_names"],
        "scaler_mean": rules["scaler_mean"],
        "scaler_std": rules["scaler_std"],
    }

    save(parts, rules, report, output_dir)
    print(f"done: {len(df):,} rows, {n_features} features, "
          f"tier threshold {rules['tier_threshold_AZN']:,.0f} AZN\n")
    return report

    # --- 1. SPLIT ---
TRAIN_SHARE = 0.70
VALID_SHARE = 0.15

def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(df))

    n_train = int(len(df) * TRAIN_SHARE)
    n_valid = int(len(df) * VALID_SHARE)

    parts = {
        "train": df.iloc[order[:n_train]],
        "validation": df.iloc[order[n_train:n_train + n_valid]],
        "test": df.iloc[order[n_train + n_valid:]],
    }
    parts = {k: v.reset_index(drop=True) for k, v in parts.items()}
    print("  [6] split " + ", ".join(f"{k}={len(v):,}" for k, v in parts.items()))
    return parts


# --- 2. FIT PREPROCESSOR ---
BAKU_CENTRE_LAT, BAKU_CENTRE_LNG = 40.3667, 49.8352
KM_PER_DEGREE_LAT = 111.0
KM_PER_DEGREE_LNG = 111.0 * float(np.cos(np.radians(BAKU_CENTRE_LAT)))
MIN_ROWS_PER_LOCATION = 100

CONTINUOUS_FEATURES = [
    "log_area_m2", "rooms", "floor", "n_floors", "log_land_m2", "lat", "lng",
    "distance_to_centre_km", "floor_ratio", "area_per_room", "log_land_ratio",
]

BINARY_FEATURES = [
    "has_floor_info", "has_repair", "has_bill_of_sale", "has_mortgage",
    "is_vip", "is_featured", "sold_by_owner",
    "is_ground_floor", "is_top_floor",
]

def fit_preprocessor(train: pd.DataFrame) -> dict:
    counts = train["location"].value_counts()
    locations = sorted(counts[counts >= MIN_ROWS_PER_LOCATION].index.tolist())

    fill_values = {col: float(train[col].median())
                   for col in ["rooms", "lat", "lng"]}

    tier_threshold = float(train["price"].median())

    rules: dict = {
        "locations": locations,
        "categories": list(KEEP_CATEGORIES),
        "fill_values": fill_values,
        "tier_threshold_AZN": tier_threshold,
    }

    unscaled = build_features(train, rules, scale=False)
    rules["scaler_mean"] = {c: float(unscaled[c].mean()) for c in CONTINUOUS_FEATURES}
    rules["scaler_std"] = {c: float(unscaled[c].std(ddof=0)) or 1.0
                           for c in CONTINUOUS_FEATURES}
    rules["feature_names"] = unscaled.columns.tolist()

    print(f"  [7] fitted on train: {len(locations)} named locations, "
          f"tier threshold {tier_threshold:,.0f} AZN")
    return rules


# --- 3. BUILD FEATURES ---
def build_features(df: pd.DataFrame, rules: dict, scale: bool = True) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["log_area_m2"] = np.log(df["area_m2"])
    out["log_land_m2"] = np.log1p(df["land_m2"])
    out["rooms"] = df["rooms"].fillna(rules["fill_values"]["rooms"])
    out["lat"] = df["lat"].fillna(rules["fill_values"]["lat"])
    out["lng"] = df["lng"].fillna(rules["fill_values"]["lng"])

    has_floor = df["floor"].notna() & df["n_floors"].notna()
    out["floor"] = df["floor"].where(has_floor, 0.0)
    out["n_floors"] = df["n_floors"].where(has_floor, 0.0)

    km_north = (out["lat"] - BAKU_CENTRE_LAT) * KM_PER_DEGREE_LAT
    km_east = (out["lng"] - BAKU_CENTRE_LNG) * KM_PER_DEGREE_LNG
    out["distance_to_centre_km"] = np.sqrt(km_north ** 2 + km_east ** 2)

    out["floor_ratio"] = np.where(
        has_floor & (df["n_floors"] > 0),
        df["floor"] / df["n_floors"].replace(0, np.nan),
        0.0,
    )
    out["is_ground_floor"] = (has_floor & df["floor"].eq(1)).astype(int)
    out["is_top_floor"] = (has_floor & df["floor"].eq(df["n_floors"])).astype(int)

    out["area_per_room"] = df["area_m2"] / out["rooms"].clip(lower=1)
    out["log_land_ratio"] = np.log1p(df["land_m2"] / df["area_m2"])

    out["has_floor_info"] = has_floor.astype(int)
    for col in BINARY_FEATURES:
        if col not in ("has_floor_info", "is_ground_floor", "is_top_floor"):
            out[col] = df[col].astype(int)

    for name in rules["categories"]:
        out[f"category={name}"] = df["category"].eq(name).astype(int)

    known = set(rules["locations"])
    for name in rules["locations"]:
        out[f"location={name}"] = df["location"].eq(name).astype(int)
    out["location=other"] = (~df["location"].isin(known)).astype(int)

    if scale:
        for col in CONTINUOUS_FEATURES:
            out[col] = (out[col] - rules["scaler_mean"][col]) / rules["scaler_std"][col]

    return out.astype(float)


# --- 4. SAVE ---
READABLE_COLUMNS = [
    "item_id", "price", "log_price", "area_m2", "rooms", "floor",
    "n_floors", "land_m2", "lat", "lng", "category", "location", "scraped",
    "has_repair", "has_bill_of_sale", "has_mortgage",
    "is_vip", "is_featured", "sold_by_owner",
]

def save(parts: dict[str, pd.DataFrame], rules: dict, report: dict,
         output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    suffix = {"train": "train", "validation": "valid", "test": "test"}

    for name, frame in parts.items():
        X = build_features(frame, rules).to_numpy()
        y_log, tier = build_targets(frame, rules)
        s = suffix[name]
        np.save(out / f"X_{s}.npy", X)
        np.save(out / f"y_{s}_log.npy", y_log)
        np.save(out / f"tier_{s}.npy", tier)
        readable = frame.assign(log_price=np.log(frame["price"]))
        readable[READABLE_COLUMNS].to_csv(out / f"{s}_clean.csv", index=False)

    pd.DataFrame({"feature": rules["feature_names"]}).to_csv(
        out / "feature_names.csv", index=False)
    (out / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  [9] wrote arrays, tables and preprocessing_report.json to {out}/")

# Build Targets:
def build_targets(df: pd.DataFrame, rules: dict) -> tuple[np.ndarray, np.ndarray]:
    y_log = np.log(df["price"].to_numpy(dtype=float)) # Task A
    tier = (df["price"] >= rules["tier_threshold_AZN"]).to_numpy().astype(int) # Task B
    return y_log, tier

# Argparse, run and main:
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare the bina.az data for modelling.")
    p.add_argument("--input", default="data/bina_az_sale.csv")
    p.add_argument("--output-dir", default="data/processed")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run(args.input, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())