from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def read_em_txt(path: Path) -> tuple[float, float, int]:
    txt = path.read_text(encoding="utf-8")
    items = int(re.search(r"items=(\d+)", txt).group(1))
    em = float(re.search(r"EM=([0-9.]+)", txt).group(1))
    nem = float(re.search(r"nEM=([0-9.]+)", txt).group(1))
    return em, nem, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1_em_txt", required=True)
    ap.add_argument("--v2_em_txt", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--title", default="Learned Structure (Edge MLP) — v1 vs v2 (Validation)")
    args = ap.parse_args()

    v1_em, v1_nem, n1 = read_em_txt(Path(args.v1_em_txt))
    v2_em, v2_nem, n2 = read_em_txt(Path(args.v2_em_txt))

    rows = [
        {"model": "edge_mlp_v1", "items": n1, "EM": v1_em, "nEM": v1_nem},
        {"model": "edge_mlp_v2", "items": n2, "EM": v2_em, "nEM": v2_nem},
    ]
    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Plot (no manual colors)
    labels = ["v1 EM", "v1 nEM", "v2 EM", "v2 nEM"]
    vals = [v1_em, v1_nem, v2_em, v2_nem]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, vals)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v*100:.2f}%", ha="center", va="bottom", fontsize=10)
    plt.ylim(0, max(vals) + 0.12)
    plt.ylabel("Score")
    plt.title(args.title)
    plt.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print("saved png:", out_png)
    print("saved csv:", out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()