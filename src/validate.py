"""validate.py -- read-only verification of the cleaned data produced by data_prep.py.

Loads model-ready arrays, cleaned split tables, feature names and the
preprocessing report, checks every invariant, prints PASS/FAIL, writes
cleanliness plots (val_*.png) + validation_report.md next to the EDA report."""

import hashlib
import json

from pathlib import Path

import pandas as pd
import numpy as np

SPLIT_TOLERANCE = 0.02

def leakage_audit(v, report, feature_names, tables):
    """
    Check that nothing in the model inputs gives away the price.

    Three checks: no leakage column survives into the feature names, the
    report records which columns were removed, and no numeric feature is
    almost a copy of the price (|correlation| above 0.95 on the train table).

    :param v: Validator that records the PASS/FAIL results.
    :param report: Preprocessing report (leakage columns, numeric feature list).
    :param feature_names: Names of the processed features.
    :param tables: Dict of cleaned split tables; the "train" table is used.
    """
    v.group("LEAKAGE")

    # 1. No processed feature may be derived from a column that contains the answer
    leaked = [
        name
        for name in feature_names
        if any(tag in name for tag in ("unit_price", "total_price", "description"))
    ]
    v.check("leakage columns absent from feature_names", not leaked, f"{leaked}")

    # 2. The preprocessing step must have recorded exactly these removals
    v.check(
        "leakage columns recorded in report",
        set(report["leakage_columns_removed"]) == {"unit_price", "total_price", "description"},
        ",".join(sorted(report["leakage_columns_removed"])),
    )

    # 3. No numeric feature may be (almost) a copy of the price
    numeric_features = report.get("numeric_features", [])
    train = tables["train"]
    price = pd.to_numeric(train["price"], errors="coerce")

    # Strongest correlation seen so far, as (column, |r|)
    worst = ("", 0.0)
    for column in numeric_features:
        # Skip features that are not columns of the cleaned table
        if column not in train.columns:
            continue
        r = abs(train[column].corr(price))
        # corr() gives NaN for constant columns, so those are ignored
        if pd.notna(r) and r > worst[1]:
            worst = (column, float(r))
    v.check(
        "no numeric feature |corr| > 0.95 with price",
        worst[1] <= 0.95,
        f"max {worst[0]}={worst[1]:.3f}",
    )

class Validator:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.current_group = 0
        self.rows: list[tuple[str, str, bool, str]] = []

    def group(self, title: str) -> None:
        self.current_group = title
        print()
        print(title)
        print("-" * len(title))

    def check(self, name: str, ok: bool, detail: str) -> None:
        mark = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        self.rows.append((self.current_group or "", name, ok, detail))
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{mark}] {name}{suffix}")

def sha256_ids(series: pd.Series) -> str:
    joined = "\n".join(sorted(series.astype(str)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

def load(data_dir: Path):
    report = json.loads((data_dir / "preprocessing_report.json").read_text(encoding="utf-8"))
    feature_names = pd.read_csv(data_dir / "feature_names.csv")["feature"].tolist()
    arrays = {
        "X_train": np.load(data_dir / "X_train.npy"),
        "X_valid": np.load(data_dir / "X_valid.npy"),
        "X_test": np.load(data_dir / "X_test.npy"),
        "y_train_log": np.load(data_dir / "y_train_log.npy"),
        "y_valid_log": np.load(data_dir / "y_valid_log.npy"),
        "y_test_log": np.load(data_dir / "y_test_log.npy"),
    }
    tiers = {
        "train": np.load(data_dir / "tier_train.npy"),
        "validation": np.load(data_dir / "tier_valid.npy"),
        "test": np.load(data_dir / "tier_test.npy"),
    }
    tables = {
        "train": pd.read_csv(data_dir / "train_clean.csv", low_memory=False),
        "validation": pd.read_csv(data_dir / "valid_clean.csv", low_memory=False),
        "test": pd.read_csv(data_dir / "test_clean.csv", low_memory=False),
    }
    return report, feature_names, arrays, tiers, tables

def validate(report, feature_names, arrays, tiesr, tables) -> Validator:
    v = Validator()

    # Split Integrity:
    v.group("SPLIT INTEGRITY")
    sizes = {k: len(df) for k, df in tables.items()}
    reported = report["split_rows"]
    v.check(
        "split sizes match report",
        all(sizes[k] == reported[k] for k in ("train", "validation", "test")),
        f"train={sizes['train']} valid={sizes['validation']} test={sizes['test']}",
    )
    v.check(
        "splits sum to final_rows",
        sum(sizes.values()) == report["final_rows"],
        f"{sum(sizes.values())} == {report['final_rows']}",
    )
    total = sum(sizes.values())
    splits =("train", "validation", "test")
    expected_ratio = {"train": 0.70, "validation": 0.15, "test": 0.15}
    for k in splits:
        actual = sizes[k] / total
        v.check(
            f"'{k}' ratio ~ {expected_ratio[k]:.2f}",
            abs(actual - expected_ratio[k]) <= SPLIT_TOLERANCE,
            f"{actual:.3f}",
        )

    ids = {k: tables[k]["term_id"] for k in splits}
    for k in splits:
        v.check(
            f"no duplicate item_id in '{k}'",
            int(ids[k].duplicated().sum()) == 0,
            f"{int(ids[k].duplicated().sum())} dupes",
        )
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = len(set(ids[a]) & set(ids[b]))
        v.check(
            f"no item_id overlap {a} <-> {b}",
            overlap == 0,
            f"{overlap} shared"
        )

    if "split_item_id_sha256" in report:
        for k in splits:
            expected = report["split_item_id_sha256"][k]
            v.check(
                f"'{k}' id fingerprint matches report",
                actual == expected,
                actual,
            )
    else:
        v.check("report contains split id fingerprints", False, "missing key")

    # ---
    leakage_audit(v, report, feature_names, tables)
    # ---
