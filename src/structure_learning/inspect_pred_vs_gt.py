from __future__ import annotations

import argparse
import csv
import difflib
from pathlib import Path


def pretty_diff(a: str, b: str) -> str:
    """
    Character-level diff between strings.
    """
    return "".join(
        difflib.ndiff(a, b)
    )


def main():
    ap = argparse.ArgumentParser(description="Inspect predicted vs GT expressions")
    ap.add_argument("--csv", required=True, help="pred_latex.csv file")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--only_errors", action="store_true", help="Show only EM=0 rows")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    shown = 0
    for r in rows:
        if args.only_errors and r["em"] == "1":
            continue

        ui = r["ui"]
        pred = r["pred"]
        gt = r["gt"]
        em = r["em"]
        nem = r["nem"]
        n_nodes = r.get("n_nodes", "?")
        n_edges = r.get("n_pred_edges", "?")

        print("=" * 100)
        print(f"UI        : {ui}")
        print(f"Nodes     : {n_nodes} | Pred edges: {n_edges}")
        print(f"EM / nEM  : {em} / {nem}")
        print("-" * 100)
        print("GT:")
        print(gt)
        print("-" * 100)
        print("PRED:")
        print(pred)
        print("-" * 100)
        print("DIFF (GT → PRED):")
        print(pretty_diff(gt, pred))
        print()

        shown += 1
        if args.limit and shown >= args.limit:
            break

    print(f"\nShown {shown} items.")


if __name__ == "__main__":
    main()