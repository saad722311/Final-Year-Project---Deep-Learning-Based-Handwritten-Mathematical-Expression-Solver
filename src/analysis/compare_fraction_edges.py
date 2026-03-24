# src/analysis/compare_fraction_edges.py
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def load_edges(jsonl_path: str, pred: bool):
    """
    Returns Counter of edge relations across all graphs.
    pred=True  -> expects 'pred_edges'
    pred=False -> expects 'edges'
    """
    c = Counter()
    n_graphs = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            edges = obj.get("pred_edges" if pred else "edges", [])
            for e in edges:
                c[e[2]] += 1
            n_graphs += 1

    return c, n_graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_jsonl", required=True)
    ap.add_argument("--v1_jsonl", required=True)
    ap.add_argument("--v2_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    gt_cnt, n_gt = load_edges(args.gt_jsonl, pred=False)
    v1_cnt, n_v1 = load_edges(args.v1_jsonl, pred=True)
    v2_cnt, n_v2 = load_edges(args.v2_jsonl, pred=True)

    rows = []
    for rel in ["FRAC_NUM", "FRAC_DEN"]:
        rows.append({
            "relation": rel,
            "GT": gt_cnt.get(rel, 0),
            "EdgeMLP_v1": v1_cnt.get(rel, 0),
            "EdgeMLP_v2": v2_cnt.get(rel, 0),
        })

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nFraction edge comparison:")
    print(df.to_string(index=False))
    print("\nGraphs:")
    print(f"  GT graphs : {n_gt}")
    print(f"  v1 graphs : {n_v1}")
    print(f"  v2 graphs : {n_v2}")
    print(f"\nSaved CSV -> {out_path}")


if __name__ == "__main__":
    main()