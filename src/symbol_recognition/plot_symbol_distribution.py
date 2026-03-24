from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rotate", type=int, default=90)
    ap.add_argument("--fig_w", type=int, default=20)
    ap.add_argument("--fig_h", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv)

    # Count symbols
    counts = df["label"].value_counts().sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(args.fig_w, args.fig_h))
    counts.plot(kind="bar")

    plt.title("Training Set Symbol Class Distribution")
    plt.xlabel("Symbol Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=args.rotate)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved symbol distribution plot to {out}")
    print(f"Total classes: {len(counts)}")
    print(f"Total samples: {counts.sum()}")


if __name__ == "__main__":
    main()