from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="linear_with_complexity.csv")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if "complexity" not in df.columns or "em" not in df.columns:
        raise SystemExit("Expected columns: complexity, em")

    summary = (
        df.groupby("complexity")
          .agg(
              n=("em", "count"),
              em_rate=("em", "mean")
          )
          .reset_index()
          .sort_values("em_rate", ascending=False)
    )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    print("EM by complexity:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()