from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _load_full_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Make it robust to different column naming
    # Expect something like: gt/label and pred/pred_label
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("gt", "gold", "truth", "label", "y_true"):
            col_map[c] = "gt"
        if lc in ("pred", "y_pred", "pred_label"):
            col_map[c] = "pred"
    df = df.rename(columns=col_map)

    if "gt" not in df.columns or "pred" not in df.columns:
        raise ValueError(f"{path} must contain gt/label and pred columns. Found: {list(df.columns)}")

    df["gt"] = df["gt"].astype(str)
    df["pred"] = df["pred"].astype(str)
    df["correct"] = (df["gt"] == df["pred"]).astype(int)
    return df


def _top_confusions(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    wrong = df[df["correct"] == 0].copy()
    if len(wrong) == 0:
        return pd.DataFrame(columns=["gt", "pred", "count"])
    out = (
        wrong.groupby(["gt", "pred"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(k)
    )
    return out


def _read_unseen_list(path: Path) -> str:
    if not path.exists():
        return "-"
    txt = path.read_text(encoding="utf-8").strip()
    return txt if txt else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_full", required=True, help="Full predictions CSV for train (per-sample).")
    ap.add_argument("--valid_full", required=True, help="Full predictions CSV for valid (per-sample).")
    ap.add_argument("--test_full", required=True, help="Full predictions CSV for test (per-sample).")
    ap.add_argument("--test_unseen", default="", help="Optional unseen_labels.txt for test.")
    ap.add_argument("--out_png", required=True, help="Output PNG path.")
    ap.add_argument("--topk", type=int, default=10, help="Top-K confusions per split.")
    args = ap.parse_args()

    train_df = _load_full_preds(Path(args.train_full))
    valid_df = _load_full_preds(Path(args.valid_full))
    test_df  = _load_full_preds(Path(args.test_full))

    def summary_row(name: str, df: pd.DataFrame):
        return {
            "split": name,
            "n": len(df),
            "acc": float(df["correct"].mean()),
        }

    summary = pd.DataFrame([
        summary_row("train", train_df),
        summary_row("valid", valid_df),
        summary_row("test(seen)", test_df),
    ])

    unseen_txt = _read_unseen_list(Path(args.test_unseen)) if args.test_unseen else "-"

    train_conf = _top_confusions(train_df, args.topk)
    valid_conf = _top_confusions(valid_df, args.topk)
    test_conf  = _top_confusions(test_df,  args.topk)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # --- Plot ---
    fig = plt.figure(figsize=(16, 10))

    # Summary table
    ax0 = fig.add_subplot(2, 2, 1)
    ax0.axis("off")
    t0 = summary.copy()
    t0["acc"] = t0["acc"].map(lambda x: f"{x:.4f}")
    table0 = ax0.table(
        cellText=t0.values,
        colLabels=t0.columns,
        loc="center",
        cellLoc="center",
    )
    table0.auto_set_font_size(False)
    table0.set_fontsize(11)
    table0.scale(1, 1.6)
    ax0.set_title("Symbol Classifier — Accuracy Summary", fontsize=14)

    ax0.text(
        0.0, -0.15,
        f"Test unseen labels (dropped in eval): {unseen_txt}",
        transform=ax0.transAxes,
        fontsize=10,
        va="top"
    )

    def add_conf_table(ax, title: str, conf: pd.DataFrame):
        ax.axis("off")
        if len(conf) == 0:
            ax.set_title(title + " (no errors)", fontsize=13)
            return
        tt = conf.copy()
        table = ax.table(
            cellText=tt.values,
            colLabels=tt.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)
        ax.set_title(title, fontsize=13)

    ax1 = fig.add_subplot(2, 2, 2)
    add_conf_table(ax1, f"Top {args.topk} Confusions — Train", train_conf)

    ax2 = fig.add_subplot(2, 2, 3)
    add_conf_table(ax2, f"Top {args.topk} Confusions — Valid", valid_conf)

    ax3 = fig.add_subplot(2, 2, 4)
    add_conf_table(ax3, f"Top {args.topk} Confusions — Test(seen)", test_conf)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Saved report PNG: {out_png}")


if __name__ == "__main__":
    main()