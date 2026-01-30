from __future__ import annotations

import argparse
from pathlib import Path
import random

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples_per_class", type=int, default=5)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    df = pd.read_csv(args.labels_csv)

    grouped = df.groupby("label")
    labels = sorted(grouped.groups.keys())

    n_classes = len(labels)
    k = args.samples_per_class

    fig, axes = plt.subplots(
        n_classes, k, figsize=(k * 2, n_classes * 1.5)
    )

    if n_classes == 1:
        axes = [axes]

    for row, label in enumerate(labels):
        rows = grouped.get_group(label)
        samples = rows.sample(min(k, len(rows)), random_state=0)

        for col, (_, r) in enumerate(samples.iterrows()):
            img = Image.open(images_dir / r["filename"]).convert("L")
            ax = axes[row][col]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_title(label, fontsize=10, loc="left")

        for col in range(len(samples), k):
            axes[row][col].axis("off")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved class overview to {out}")
    print(f"Total classes: {n_classes}")


if __name__ == "__main__":
    main()