from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples_per_class", type=int, default=5)
    ap.add_argument("--topk", type=int, default=25)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    df = pd.read_csv(args.labels_csv)

    # Count frequency per class
    counts = df["label"].value_counts().head(args.topk)
    labels = counts.index.tolist()

    k = args.samples_per_class
    n_classes = len(labels)

    fig, axes = plt.subplots(
        n_classes, k,
        figsize=(k * 1.6, n_classes * 1.3)
    )

    if n_classes == 1:
        axes = [axes]

    for row, label in enumerate(labels):
        rows = df[df["label"] == label]
        samples = rows.sample(min(k, len(rows)), random_state=0)

        for col, (_, r) in enumerate(samples.iterrows()):
            img = Image.open(images_dir / r["filename"]).convert("L")
            ax = axes[row][col]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(label, rotation=0, labelpad=25, fontsize=9)

        for col in range(len(samples), k):
            axes[row][col].axis("off")

    plt.suptitle(
        f"Top {args.topk} Most Frequent Symbol Classes (Training Set)",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved visualization to {out}")
    print(f"Displayed classes: {args.topk}")


if __name__ == "__main__":
    main()