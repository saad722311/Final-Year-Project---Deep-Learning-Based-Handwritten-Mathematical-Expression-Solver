# src/analysis/fraction_edge_audit.py
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple, Any


def load_jsonl_map(path: str) -> Dict[str, Any]:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            ui = o.get("ui")
            if ui is not None:
                m[str(ui)] = o
    return m


def frac_edges_from(obj: dict, key_edges: str) -> Dict[int, Dict[str, Tuple[int, float]]]:
    """
    Return: frac_src -> {"FRAC_NUM": (dst, score), "FRAC_DEN": (dst, score)}
    Works for:
      GT edges: edges = [src, dst, rel]  (no score)
      Pred edges: pred_edges = [src, dst, rel, score]
    """
    out = defaultdict(dict)

    edges = obj.get(key_edges, [])
    for e in edges:
        if key_edges == "edges":
            s, t, r = e
            if r in ("FRAC_NUM", "FRAC_DEN"):
                out[int(s)][r] = (int(t), 1.0)
        else:
            s, t, r, sc = e
            if r in ("FRAC_NUM", "FRAC_DEN"):
                out[int(s)][r] = (int(t), float(sc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_jsonl", required=True)
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_jsonl_map(args.gt_jsonl)
    pr = load_jsonl_map(args.pred_jsonl)

    common = sorted(set(gt) & set(pr))
    print("GT graphs:", len(gt))
    print("Pred graphs:", len(pr))
    print("Intersection:", len(common))

    # graph-level stats
    stats = Counter()
    rows = []

    for ui in common:
        gto = gt[ui]
        pro = pr[ui]

        gt_frac = frac_edges_from(gto, "edges")
        pr_frac = frac_edges_from(pro, "pred_edges")

        # only consider graphs that truly have fractions in GT
        if not gt_frac:
            continue

        # For each frac_src in GT, compare
        for frac_src, gt_map in gt_frac.items():
            gt_num = gt_map.get("FRAC_NUM", (None, 0.0))[0]
            gt_den = gt_map.get("FRAC_DEN", (None, 0.0))[0]

            pr_map = pr_frac.get(frac_src, {})
            pr_num = pr_map.get("FRAC_NUM", (None, 0.0))[0]
            pr_den = pr_map.get("FRAC_DEN", (None, 0.0))[0]

            case = None
            if pr_num is None and pr_den is None:
                case = "missing_both"
            elif pr_num is None and pr_den is not None:
                case = "missing_num_only"
            elif pr_num is not None and pr_den is None:
                case = "missing_den_only"
            else:
                # both present
                if pr_num == gt_num and pr_den == gt_den:
                    case = "both_correct"
                elif pr_num == gt_den and pr_den == gt_num:
                    case = "swapped"
                elif pr_num == gt_num and pr_den != gt_den:
                    case = "num_correct_den_wrong"
                elif pr_num != gt_num and pr_den == gt_den:
                    case = "num_wrong_den_correct"
                else:
                    case = "both_wrong"

            stats[case] += 1
            rows.append(
                {
                    "ui": ui,
                    "frac_src": frac_src,
                    "gt_num": gt_num,
                    "gt_den": gt_den,
                    "pr_num": pr_num,
                    "pr_den": pr_den,
                    "case": case,
                }
            )

    # Save CSV
    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = out_dir / "fraction_edge_audit.csv"
    df.to_csv(csv_path, index=False)
    print("\nSaved:", csv_path)

    # Print summary
    print("\n[Fraction edge audit summary]")
    total = sum(stats.values())
    for k, v in stats.most_common():
        print(f"{k:20s}: {v:5d}  ({v/max(1,total):.3f})")

    # Simple bar plot (no fixed colors)
    import matplotlib.pyplot as plt

    keys = [k for k, _ in stats.most_common()]
    vals = [stats[k] for k in keys]

    plt.figure(figsize=(10, 4))
    plt.bar(keys, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Fraction edge cases (per FRAC virtual node)")
    plt.tight_layout()
    png_path = out_dir / "fraction_edge_audit.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print("Saved:", png_path)


if __name__ == "__main__":
    main()