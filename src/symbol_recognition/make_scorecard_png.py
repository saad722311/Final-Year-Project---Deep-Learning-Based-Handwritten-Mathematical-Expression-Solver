from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--train_acc", type=float, required=True)
    ap.add_argument("--valid_acc", type=float, required=True)
    ap.add_argument("--test_acc", type=float, required=True)
    ap.add_argument("--valid_linear_em", type=float, required=True)
    ap.add_argument("--n_valid_expr", type=int, required=True)
    args = ap.parse_args()

    rows = [
        ["Symbol Acc (Train)", f"{args.train_acc*100:.2f}%"],
        ["Symbol Acc (Valid)", f"{args.valid_acc*100:.2f}%"],
        ["Symbol Acc (Test)", f"{args.test_acc*100:.2f}%"],
        ["Linear Expr EM (Valid)", f"{args.valid_linear_em*100:.2f}%  (n={args.n_valid_expr})"],
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])

    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.4)

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved scorecard to: {out}")


if __name__ == "__main__":
    main()