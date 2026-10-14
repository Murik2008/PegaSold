"""Loading, cleaning, splitting and labelling the bina.az sale data.

Every statistic used to transform the data is measured on the TRAINING
SPLIT ONLY, so nothing leaks from validation or test into the model.
"""
from __future__ import annotations

from pathlib import Path
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


def run_cleaning(input_path: str | Path = DATA_PATH, output_dir: str | Path = OUTPUT_DIR) -> dict:
    """Run the cleaning stages and record every number they produced.

    Temporary harness for the cleaning half of the pipeline. Once split,
    fit_preprocessor and build_features exist, this folds into run().
    """
    print(f"\ndata_prep (cleaning): {input_path}")

    raw = load(input_path)
    raw_rows = len(raw)

    df = parse(raw)
    df, dedup_stats = deduplicate(df)
    df, scope_stats = apply_scope(df)
    df, filter_stats = apply_filters(df)

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
        # winsorize: a row outside a window is a broken record, so apply_filters removes it outright.
        # The bounds are recorded here so src/validate.py can confirm they hold.
        "value_ranges": {
            "method": "rows outside the window are removed, not clipped",
            "fences": {
                "price": {"low": MIN_PRICE, "high": MAX_PRICE},
                "area_m2": {"low": MIN_AREA, "high": MAX_AREA},
            },
        },

        "final_rows": len(df),
        "cleaned_item_id_sha256": fingerprint(df["item_id"]),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote -> {out / 'preprocessing_report.json'}")
    return report




#  The rule for everything below:
#      the preprocessor is FITTED on the training split and only APPLIED to
#      validation and test. No number that shapes a feature is ever measured
#      on rows the model is later judged on.

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15          # test gets the remainder (0.15)
N_STRATA = 10                              # price deciles, used only to place rows

# Approximate centre of Baku (Fountain Square / Old City area), used for
# `distance_to_centre_km`. Fixed constant, not learned from the data.
BAKU_CENTRE = (40.3725, 49.8352)           # (lat, lng)

# The locations kept by name: the TOP_K_LOCATIONS most frequent in the TRAIN
# split. Every other district (and a missing one) shares one "other" column.
TOP_K_LOCATIONS = 58
OTHER_LOCATION = "other"

# Continuous engineered features are standardised; the 0/1 flags and one-hot
# columns are left alone (scaling a rare 0/1 column by its tiny std would
# blow it up to huge values and hurt the SVM).
CONTINUOUS_FEATURES = [
    "log_area_m2", "log_land_m2", "rooms", "floor", "n_floors",
    "floor_ratio", "area_per_room", "log_land_ratio",
    "lat", "lng", "distance_to_centre_km",
]
BINARY_FEATURES = [
    "is_ground_floor", "is_top_floor", "has_repair", "has_bill_of_sale",
    "has_mortgage", "is_vip", "is_featured", "sold_by_owner", "coords_missing",
]
NUMERIC_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES      # 20 columns

# Columns whose gaps are filled with a TRAIN median.
MEDIAN_FILL_COLUMNS = ["rooms", "floor", "n_floors", "floor_ratio", "lat", "lng"]


def split(df: pd.DataFrame, seed: int = RANDOM_SEED
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic 70 / 15 / 15 train / validation / test split.

    * Stratified on price decile, so every split sees the whole price range
      and the premium/standard balance (Task B) is the same in all three.
      The deciles only decide WHERE a row goes; no model statistic is taken
      from them. The tier threshold itself is measured later, on train only.
    * Depends only on the SET of item ids and the seed, not on the row order
      of the input, so a rerun gives the identical split (see `fingerprint`).
    * Sizes are exact: n_train = int(0.70 n), n_val = int(0.15 n), and the
      test split takes whatever is left (57,003 rows -> 39,902 / 8,550 / 8,551).
    """
    df = df.sort_values("item_id", kind="stable").reset_index(drop=True)
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    rng = np.random.default_rng(seed)
    strata = pd.qcut(df["price"], q=N_STRATA, labels=False, duplicates="drop").to_numpy()

    # Give every row a position in [0, 1) that is spread evenly inside its
    # stratum. Cutting the sorted positions at 70% and 85% then hands each
    # split an equal share of every stratum, with exact overall sizes.
    position = np.empty(n)
    for s in np.unique(strata):
        members = rng.permutation(np.flatnonzero(strata == s))
        position[members] = (np.arange(len(members)) + rng.random()) / len(members)
    order = np.argsort(position, kind="stable")

    parts = (order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:])
    train, val, test = (df.iloc[np.sort(idx)].reset_index(drop=True) for idx in parts)

    print(f"  [6] split: train {len(train):,} / validation {len(val):,} / test {len(test):,}")
    return train, val, test


def _floor_ratio(df: pd.DataFrame) -> pd.Series:
    """floor / n_floors, NaN when either is unknown (or the building has 0 floors)."""
    n_floors = df["n_floors"].where(df["n_floors"] > 0)
    return (df["floor"] / n_floors).clip(0, 1)


def _haversine_km(lat: pd.Series, lng: pd.Series) -> pd.Series:
    """Great-circle distance from each point to BAKU_CENTRE, in km."""
    lat0, lng0 = np.radians(BAKU_CENTRE[0]), np.radians(BAKU_CENTRE[1])
    lat1, lng1 = np.radians(lat), np.radians(lng)
    a = (np.sin((lat1 - lat0) / 2) ** 2
         + np.cos(lat0) * np.cos(lat1) * np.sin((lng1 - lng0) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def _engineer(df: pd.DataFrame, fill: dict) -> pd.DataFrame:
    """The 20 numeric features, UNSCALED. `fill` holds the train medians."""
    f = pd.DataFrame(index=df.index)

    area = df["area_m2"].astype(float)
    land = df["land_m2"].fillna(0.0).astype(float)      # no plot -> 0
    ratio = _floor_ratio(df)

    # Flags are read from the RAW floor numbers, before any filling, so an
    # advert with no floor info is "not ground, not top" instead of getting
    # the median floor's flag by accident.
    f["is_ground_floor"] = (df["floor"] == 1).astype(int)
    f["is_top_floor"] = ((df["floor"] == df["n_floors"]) & (df["n_floors"] > 1)).astype(int)
    f["coords_missing"] = (df["lat"].isna() | df["lng"].isna()).astype(int)

    rooms = df["rooms"].fillna(fill["rooms"])
    lat = df["lat"].fillna(fill["lat"])
    lng = df["lng"].fillna(fill["lng"])

    f["log_area_m2"] = np.log(area)
    f["log_land_m2"] = np.log1p(land)
    f["rooms"] = rooms
    f["floor"] = df["floor"].fillna(fill["floor"])
    f["n_floors"] = df["n_floors"].fillna(fill["n_floors"])
    f["floor_ratio"] = ratio.fillna(fill["floor_ratio"])
    f["area_per_room"] = area / rooms.clip(lower=1)
    f["log_land_ratio"] = np.log1p(land / area)
    f["lat"] = lat
    f["lng"] = lng
    f["distance_to_centre_km"] = _haversine_km(lat, lng)

    for col in ("has_repair", "has_bill_of_sale", "has_mortgage",
                "is_vip", "is_featured", "sold_by_owner"):
        f[col] = df[col].astype(int)

    return f[NUMERIC_FEATURES]



def fit_preprocessor(train: pd.DataFrame) -> dict:
    """Measure every statistic the features need -- on the TRAIN split only.

    Returns a plain, JSON-serialisable dict:
      fill            median used for each gap in MEDIAN_FILL_COLUMNS
      locations       the 58 most frequent train locations (by name)
      categories      the residential categories, in a fixed order
      scale           mean and std of each continuous feature
      tier_threshold  median train price: premium = price >= threshold
    """
    fill = {
        "rooms": float(train["rooms"].median()),
        "floor": float(train["floor"].median()),
        "n_floors": float(train["n_floors"].median()),
        "floor_ratio": float(_floor_ratio(train).median()),
        "lat": float(train["lat"].median()),
        "lng": float(train["lng"].median()),
    }

    # Most frequent first; ties broken alphabetically so the list is the
    # same on every machine.
    counts = train["location"].dropna().value_counts()
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    locations = [name for name, _ in ranked[:TOP_K_LOCATIONS]]

    engineered = _engineer(train, fill)
    scale = {}
    for col in CONTINUOUS_FEATURES:
        std = float(engineered[col].std(ddof=0))
        scale[col] = {"mean": float(engineered[col].mean()),
                      "std": std if std > 0 else 1.0}      # constant column: do not divide by 0

    prep = {
        "fit_on": "train split only",
        "n_train_rows": int(len(train)),
        "fill": fill,
        "locations": locations,
        "categories": list(KEEP_CATEGORIES),
        "scale": scale,
        "tier_threshold": float(train["price"].median()),
        "baku_centre": list(BAKU_CENTRE),
    }
    print(f"  [7] fit_preprocessor on {len(train):,} train rows: "
          f"{len(locations)} named locations, "
          f"tier threshold {prep['tier_threshold']:,.0f} AZN")
    return prep



def feature_names(prep: dict) -> list[str]:
    """Column names of X, in order."""
    return (list(NUMERIC_FEATURES)
            + [f"loc_{name}" for name in prep["locations"]] + [f"loc_{OTHER_LOCATION}"]
            + [f"cat_{name}" for name in prep["categories"]])


def build_features(df: pd.DataFrame, prep: dict) -> np.ndarray:
    """Clean rows -> the model matrix X (rows x 82), using a FITTED preprocessor.

    Same function for train, validation and test: only `prep` (measured on
    train) is ever used to fill gaps and scale, never the rows in `df`.
    """
    eng = _engineer(df, prep["fill"])
    for col in CONTINUOUS_FEATURES:
        s = prep["scale"][col]
        eng[col] = (eng[col] - s["mean"]) / s["std"]

    # One-hot location: the 58 named ones + "other" (rare or missing district).
    named = set(prep["locations"])
    loc = df["location"].where(df["location"].isin(named), OTHER_LOCATION)
    loc_cols = [f"loc_{name}" for name in prep["locations"]] + [f"loc_{OTHER_LOCATION}"]
    loc_levels = prep["locations"] + [OTHER_LOCATION]
    loc_onehot = np.column_stack([(loc == level).to_numpy() for level in loc_levels])

    cat_onehot = np.column_stack([(df["category"] == c).to_numpy() for c in prep["categories"]])

    X = np.column_stack([eng.to_numpy(dtype=float),
                         loc_onehot.astype(float),
                         cat_onehot.astype(float)])

    assert X.shape[1] == len(NUMERIC_FEATURES) + len(loc_cols) + len(prep["categories"])
    if not np.isfinite(X).all():
        bad = np.flatnonzero(~np.isfinite(X).all(axis=0))
        raise ValueError(f"non-finite values in feature columns {bad.tolist()}")
    return X


def build_targets(df: pd.DataFrame, prep: dict) -> tuple[np.ndarray, np.ndarray]:
    """Task A target: log(price).  Task B target: 1 = premium, 0 = standard,
    cut at the TRAIN median stored in `prep`."""
    y_log = np.log(df["price"].to_numpy(dtype=float))
    tier = (df["price"].to_numpy(dtype=float) >= prep["tier_threshold"]).astype(np.int8)
    return y_log, tier



def save(splits: dict[str, pd.DataFrame], X: dict[str, np.ndarray],
         y_log: dict[str, np.ndarray], tier: dict[str, np.ndarray],
         prep: dict, output_dir: str | Path = OUTPUT_DIR) -> dict:
    """Write everything the models and validate.py read back.

    Per split name in ("train", "val", "test"):
        X_<name>.npy        model matrix
        y_<name>_log.npy    log(price)            (Task A)
        tier_<name>.npy     0/1 price tier        (Task B)
        clean_<name>.csv    the cleaned rows (with item_id) behind that split
    Shared:
        feature_names.json  the 82 column names, in order
        preprocessor.json   every fitted statistic (medians, scaling, threshold)
        split_report.json   sizes, tier balance, id fingerprints
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = feature_names(prep)
    report = {"random_seed": RANDOM_SEED, "n_features": len(names),
              "tier_threshold_AZN": prep["tier_threshold"], "splits": {}}

    for name, frame in splits.items():
        np.save(out / f"X_{name}.npy", X[name])
        np.save(out / f"y_{name}_log.npy", y_log[name])
        np.save(out / f"tier_{name}.npy", tier[name])
        frame.to_csv(out / f"clean_{name}.csv", index=False, encoding="utf-8")
        report["splits"][name] = {
            "rows": int(len(frame)),
            "premium_share": round(float(tier[name].mean()), 4),
            "item_id_sha256": fingerprint(frame["item_id"]),
        }

    (out / "feature_names.json").write_text(
        json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "preprocessor.json").write_text(
        json.dumps(prep, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  [8] saved X ({len(names)} features), targets, clean CSVs -> {out}")
    return report


def run_features(input_path: str | Path = DATA_PATH,
                 output_dir: str | Path = OUTPUT_DIR) -> dict:
    print(f"\ndata_prep (cleaning + features): {input_path}")
    raw = load(input_path)
    df = parse(raw)
    df, _ = deduplicate(df)
    df, _ = apply_scope(df)
    df, _ = apply_filters(df)

    train, val, test = split(df)
    prep = fit_preprocessor(train)           # <- train only

    splits = {"train": train, "val": val, "test": test}
    X = {k: build_features(v, prep) for k, v in splits.items()}
    targets = {k: build_targets(v, prep) for k, v in splits.items()}
    y_log = {k: t[0] for k, t in targets.items()}
    tier = {k: t[1] for k, t in targets.items()}
    return save(splits, X, y_log, tier, prep, output_dir)


if __name__ == "__main__":
    run_features()