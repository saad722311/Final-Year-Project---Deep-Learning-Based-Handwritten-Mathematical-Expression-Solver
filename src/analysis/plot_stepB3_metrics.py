# src/analysis/plot_stepB3_metrics.py
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def read_rule_em_file(pred_em_txt: str) -> dict:
    """
    Reads:
      items=548
      EM=0.1277
      nEM=0.2172
    """
    txt = Path(pred_em_txt).read_text(encoding="utf-8", errors="ignore")
    out = {"items": None, "EM": None, "nEM": None}

    m = re.search(r"items\s*=\s*(\d+)", txt)
    if m:
        out["items"] = int(m.group(1))

    m = re.search(r"EM\s*=\s*([0-9.]+)", txt)
    if m:
        out["EM"] = float(m.group(1))

    m = re.search(r"nEM\s*=\s*([0-9.]+)", txt)
    if m:
        out["nEM"] = float(m.group(1))

    if out["EM"] is None:
        raise ValueError(f"Could not parse EM from: {pred_em_txt}")
    if out["nEM"] is None:
        raise ValueError(f"Could not parse nEM from: {pred_em_txt}")

    return out


def linear_em_from_csv(linear_csv: str) -> dict:
    """
    linear_reconstruct_v3.csv columns:
      ui,n_symbols,pred_linear,gt_linear,em
    EM here is linear token EM, across full set.
    """
    df = pd.read_csv(linear_csv)
    if "em" not in df.columns:
        raise ValueError(f"'em' column not found in {linear_csv}. Found: {list(df.columns)}")
    # em may be int/float/str. Coerce safely.
    em = pd.to_numeric(df["em"], errors="coerce").fillna(0).mean()
    return {"items": len(df), "EM": float(em)}


def save_rule_only_png(rule_metrics: dict, out_path: Path, title: str):
    em = rule_metrics["EM"]
    nem = rule_metrics["nEM"]

    labels = ["EM", "nEM"]
    values = [em, nem]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.ylabel("Score")

    # annotate
    for i, v in enumerate(values):
        plt.text(i, min(0.98, v + 0.02), f"{v*100:.2f}%", ha="center", va="bottom", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_comparison_png(linear_metrics: dict, rule_metrics: dict, out_path: Path, title: str):
    lin_em = linear_metrics["EM"]
    rb_em = rule_metrics["EM"]
    rb_nem = rule_metrics["nEM"]

    labels = ["Linear EM", "Rule EM", "Rule nEM"]
    values = [lin_em, rb_em, rb_nem]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.ylabel("Score")

    for i, v in enumerate(values):
        plt.text(i, min(0.98, v + 0.02), f"{v*100:.2f}%", ha="center", va="bottom", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear_csv", required=True, help="Full linear CSV (e.g., linear_reconstruct_v3.csv)")
    ap.add_argument("--rule_pred_em", required=True, help="Rule-based pred_em.txt from pred_edges_to_latex_eval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="val", help="Used in filenames/titles, e.g. val/test")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lin = linear_em_from_csv(args.linear_csv)
    rb = read_rule_em_file(args.rule_pred_em)

    # Print summary (copy/paste-able into report)
    print("=== Step B3 Metrics Summary ===")
    print(f"Linear (tokens, linear parse) : items={lin['items']}  EM={lin['EM']:.4f} ({lin['EM']*100:.2f}%)")
    print(f"Rule-based (structure parse)  : items={rb['items']}   EM={rb['EM']:.4f} ({rb['EM']*100:.2f}%)  "
          f"nEM={rb['nEM']:.4f} ({rb['nEM']*100:.2f}%)")

    # PNG 1: rule-only
    png_rule = out_dir / f"stepB3_rule_only_{args.tag}.png"
    save_rule_only_png(
        rb,
        png_rule,
        title=f"Rule-based Parsing ({args.tag})"
    )
    print(f"Saved: {png_rule}")

    # PNG 2: comparison
    png_cmp = out_dir / f"stepB3_comparison_{args.tag}.png"
    save_comparison_png(
        lin,
        rb,
        png_cmp,
        title=f"Baseline Comparison ({args.tag})"
    )
    print(f"Saved: {png_cmp}")


if __name__ == "__main__":
    main()