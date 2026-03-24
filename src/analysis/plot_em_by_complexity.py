from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="em_by_complexity.csv")
    ap.add_argument("--out", required=True,
                    help="output png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    plt.figure(figsize=(5, 4))
    plt.bar(df["complexity"], df["em_rate"])
    plt.ylabel("Exact Match (EM)")
    plt.xlabel("Expression Complexity")
    plt.ylim(0, 1.0)
    plt.title("Linear Expression EM by Complexity (Validation Set)")

    for i, v in enumerate(df["em_rate"]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()