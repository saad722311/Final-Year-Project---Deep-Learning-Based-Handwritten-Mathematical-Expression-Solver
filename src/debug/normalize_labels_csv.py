from __future__ import annotations

import csv
from pathlib import Path

from src.utils.latex_norm import normalize_latex_label


def normalize_csv(in_path: Path, out_path: Path) -> None:
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0
    empty_after = 0

    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {in_path}")

        # Expect at least these
        if "filename" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"Expected columns filename,label. Found: {reader.fieldnames}"
            )

        rows = []
        for row in reader:
            total += 1
            fn = (row.get("filename") or "").strip()
            raw = (row.get("label") or "").strip()

            norm = normalize_latex_label(raw)

            if norm != raw:
                changed += 1

            if not norm:
                empty_after += 1

            rows.append({"filename": fn, "label": norm})

    # Write with proper CSV quoting so commas etc. stay valid
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"], quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote: {out_path}")
    print(f"     rows: {total}")
    print(f"     changed labels: {changed}")
    print(f"     empty after norm: {empty_after}")


if __name__ == "__main__":
    processed_dir = Path("03-development/datasets/TC11_CROHME23/processed")

    in_csv = processed_dir / "train_labels.csv"
    out_csv = processed_dir / "train_labels.normalized.csv"

    normalize_csv(in_csv, out_csv)