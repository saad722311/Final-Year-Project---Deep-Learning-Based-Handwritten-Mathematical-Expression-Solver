from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd


def load_map(path: str) -> dict:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ui = o.get("ui")
            if ui:
                m[ui] = o
    return m


def edge_set(edges):
    """
    Normalize to a set of (s,t,rel). Ignores score.
    """
    out = set()
    for e in edges:
        if len(e) < 3:
            continue
        s, t, rel = e[0], e[1], e[2]
        out.add((int(s), int(t), str(rel)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--gt_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = load_map(args.pred_jsonl)
    gt = load_map(args.gt_jsonl)
    common = sorted(set(pred) & set(gt))

    rows = []
    rel_fp = Counter()
    rel_fn = Counter()

    for ui in common:
        po = pred[ui]
        go = gt[ui]

        pe = po.get("pred_edges", [])
        ge = go.get("edges", go.get("gt_edges", []))

        P = edge_set(pe)
        G = edge_set(ge)

        fp = P - G
        fn = G - P

        # counts by relation
        for (_, _, r) in fp:
            rel_fp[r] += 1
        for (_, _, r) in fn:
            rel_fn[r] += 1

        rows.append(
            {
                "ui": ui,
                "n_gt": len(G),
                "n_pred": len(P),
                "fp": len(fp),
                "fn": len(fn),
                "mismatch": len(fp) + len(fn),
            }
        )

    df = pd.DataFrame(rows).sort_values("mismatch", ascending=False)

    # save table
    csv_path = out_dir / "edge_mismatch_rank.csv"
    df.to_csv(csv_path, index=False)

    # save picked list
    picked = df.head(args.topk)["ui"].tolist()
    (out_dir / "picked_ui.txt").write_text("\n".join(picked), encoding="utf-8")

    print("saved:", csv_path)
    print("saved:", out_dir / "picked_ui.txt")
    print("\nTop mismatch cases:")
    print(df.head(args.topk).to_string(index=False))

    print("\n[Aggregate FP by relation]:", rel_fp)
    print("[Aggregate FN by relation]:", rel_fn)


if __name__ == "__main__":
    main()