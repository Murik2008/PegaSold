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