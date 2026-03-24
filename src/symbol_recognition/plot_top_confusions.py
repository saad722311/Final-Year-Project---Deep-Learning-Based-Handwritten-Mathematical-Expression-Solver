from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusions_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.confusions_csv)

    # Take top-k confusions
    df = df.head(args.topk)

    labels = [f"{gt} → {pred}" for gt, pred in zip(df["gt"], df["pred"])]
    counts = df["count"].tolist()

    plt.figure(figsize=(10, 4))
    plt.bar(labels, counts)
    plt.ylabel("Count")
    plt.title("Top Symbol Confusions (Validation Set)")
    plt.xticks(rotation=45, ha="right")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved confusion plot to {out}")


if __name__ == "__main__":
    main()