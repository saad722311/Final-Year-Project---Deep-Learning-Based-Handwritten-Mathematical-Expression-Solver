# src/utils/clean_latex_labels.py
from __future__ import annotations

import csv
from pathlib import Path

from src.utils.latex_norm import normalize_latex_label


def clean_csv(csv_path: Path) -> None:
    print(f"\nProcessing: {csv_path}")

    backup_path = csv_path.with_suffix(".csv.bak")
    if not backup_path.exists():
        csv_path.replace(backup_path)
        print(f"  Backup created: {backup_path.name}")
    else:
        print(f"  Backup already exists: {backup_path.name}")

    rows = []
    with backup_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        assert fieldnames is not None
        assert "label" in fieldnames

        for r in reader:
            r["label"] = normalize_latex_label(r["label"])
            rows.append(r)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Cleaned and rewritten: {csv_path.name}")


def main():
    base = Path("03-development/datasets/TC11_CROHME23/processed")
    for split in ("train", "valid", "test"):
        clean_csv(base / f"{split}_labels.csv")


if __name__ == "__main__":
    main()