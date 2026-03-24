from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True, help="Training labels CSV")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--topk", type=int, default=100, help="Top-K classes to show")
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv)

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column")

    counts = df["label"].value_counts().head(args.topk)

    plt.figure(figsize=(max(12, len(counts) * 0.25), 6))
    counts.plot(kind="bar")

    plt.title("Training Set Symbol Distribution")
    plt.xlabel("Symbol")
    plt.ylabel("Count")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved training distribution plot to {out}")
    print(f"Total classes (shown): {len(counts)}")
    print(f"Total samples: {len(df)}")


if __name__ == "__main__":
    main()