"""validate.py -- read-only verification of the cleaned data produced by data_prep.py.

Loads model-ready arrays, cleaned split tables, feature names and the
preprocessing report, checks every invariant, prints PASS/FAIL, writes
cleanliness plots (val_*.png) + validation_report.md next to the EDA report."""

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