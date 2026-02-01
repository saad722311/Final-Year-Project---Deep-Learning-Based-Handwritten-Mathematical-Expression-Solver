# src/symbol_recognition/visualize_eval.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_csv", type=str, required=True)
    ap.add_argument("--confusions_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--topk_classes", type=int, default=40, help="Show top-K classes by support in the plot")
    ap.add_argument("--topk_confusions", type=int, default=30, help="Show top-K confusion pairs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)

    # Expect columns: y_true, y_pred, correct (0/1)
    if "correct" not in df.columns:
        df["correct"] = (df["y_true"].astype(str) == df["y_pred"].astype(str)).astype(int)

    df["y_true"] = df["y_true"].astype(str)
    df["y_pred"] = df["y_pred"].astype(str)

    overall = df["correct"].mean()

    # -------------------------
    # 1) Per-class accuracy
    # -------------------------
    g = (
        df.groupby("y_true", as_index=False)
        .agg(support=("correct", "size"), correct=("correct", "sum"))
    )
    g["acc"] = g["correct"] / g["support"]
    g = g.sort_values(["support", "acc"], ascending=[False, False])

    g.to_csv(out_dir / "per_class_accuracy.csv", index=False)

    plot_df = g.head(args.topk_classes)

    # Dynamic width so labels fit
    fig_w = max(14, 0.35 * len(plot_df))
    plt.figure(figsize=(fig_w, 6))
    plt.bar(plot_df["y_true"].tolist(), plot_df["acc"].tolist())
    plt.xticks(rotation=80, ha="right")
    plt.ylim(0, 1.0)
    plt.title(f"Per-class Accuracy (Top {len(plot_df)} by support) | overall={overall:.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_accuracy.png", dpi=200)
    plt.close()

    # -------------------------
    # 2) Confusion pairs chart
    # -------------------------
    conf = pd.read_csv(args.confusions_csv)

    # Expected columns: y_true, y_pred, count OR size
    # normalize column names defensively
    cols = [c.lower() for c in conf.columns]
    conf.columns = cols

    # try to find likely names
    true_col = "y_true" if "y_true" in cols else ("true" if "true" in cols else cols[0])
    pred_col = "y_pred" if "y_pred" in cols else ("pred" if "pred" in cols else cols[1])

    count_col = None
    for c in ["count", "size", "n", "freq"]:
        if c in cols:
            count_col = c
            break
    if count_col is None:
        # last column fallback
        count_col = cols[-1]

    conf = conf.sort_values(count_col, ascending=False).head(args.topk_confusions).copy()
    conf["pair"] = conf[true_col].astype(str) + " → " + conf[pred_col].astype(str)

    plt.figure(figsize=(14, 6))
    plt.bar(conf["pair"].tolist(), conf[count_col].tolist())
    plt.xticks(rotation=80, ha="right")
    plt.title(f"Top {len(conf)} Confusion Pairs")
    plt.tight_layout()
    plt.savefig(out_dir / "top_confusion_pairs.png", dpi=200)
    plt.close()

    # -------------------------
    # 3) Save a quick summary txt
    # -------------------------
    (out_dir / "summary.txt").write_text(
        f"rows={len(df)}\n"
        f"overall_acc={overall:.6f}\n"
        f"classes={g.shape[0]}\n"
        f"topk_classes_plotted={len(plot_df)}\n"
        f"topk_confusions_plotted={len(conf)}\n",
        encoding="utf-8",
    )

    print(f"[viz] overall_acc={overall:.4f}")
    print(f"[viz] saved: {out_dir / 'per_class_accuracy.png'}")
    print(f"[viz] saved: {out_dir / 'top_confusion_pairs.png'}")
    print(f"[viz] saved: {out_dir / 'per_class_accuracy.csv'}")
    print(f"[viz] saved: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()