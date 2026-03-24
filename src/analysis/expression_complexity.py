from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path


def classify_complexity(latex: str) -> str:
    flags = {
        "sup": "^" in latex,
        "frac": "\\frac" in latex,
        "root": "\\sqrt" in latex,
    }
    n = sum(flags.values())

    if n == 0:
        return "linear"
    if n == 1:
        if flags["sup"]:
            return "superscript"
        if flags["frac"]:
            return "fraction"
        if flags["root"]:
            return "root"
    return "complex"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear_csv", required=True,
                    help="linear_reconstruct_v3.csv")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.linear_csv)

    if "gt_linear" not in df.columns:
        raise SystemExit("Expected gt_linear column")

    df["complexity"] = df["gt_linear"].astype(str).apply(classify_complexity)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("Saved:", out)
    print(df["complexity"].value_counts())


if __name__ == "__main__":
    main()