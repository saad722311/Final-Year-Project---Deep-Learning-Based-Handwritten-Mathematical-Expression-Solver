from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="Path to structure dataset jsonl")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to save outputs")
    ap.add_argument("--topk", type=int, default=40, help="Top-K node labels for plot")
    args = ap.parse_args()

    p = Path(args.jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_counts = Counter()
    node_label_counts = Counter()

    n_items = 0
    n_nodes = 0
    n_edges = 0

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            n_items += 1

            nodes = item.get("nodes", [])
            edges = item.get("edges", [])

            n_nodes += len(nodes)
            n_edges += len(edges)

            for nd in nodes:
                lab = (nd.get("label") or "").strip()
                if lab:
                    node_label_counts[lab] += 1

            for e in edges:
                # e = [src, dst, rel]
                if len(e) >= 3:
                    edge_counts[str(e[2])] += 1

    # summary txt
    summary_txt = out_dir / (p.stem + "_summary.txt")
    with summary_txt.open("w", encoding="utf-8") as w:
        w.write(f"file: {p}\n")
        w.write(f"items: {n_items}\n")
        w.write(f"total_nodes: {n_nodes}\n")
        w.write(f"total_edges: {n_edges}\n")
        w.write(f"unique_node_labels: {len(node_label_counts)}\n")
        w.write(f"unique_edge_types: {len(edge_counts)}\n\n")

        w.write("EDGE TYPES:\n")
        for k, v in edge_counts.most_common():
            w.write(f"{k}\t{v}\n")

        w.write("\nTOP NODE LABELS:\n")
        for k, v in node_label_counts.most_common(100):
            w.write(f"{k}\t{v}\n")

    # edge plot
    if edge_counts:
        labs = [k for k, _ in edge_counts.most_common()]
        vals = [v for _, v in edge_counts.most_common()]

        plt.figure(figsize=(10, 4))
        plt.bar(labs, vals)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / (p.stem + "_edge_types.png"), dpi=200)
        plt.close()

    # top node plot
    if node_label_counts:
        items = node_label_counts.most_common(args.topk)
        labs = [k for k, _ in items]
        vals = [v for _, v in items]

        plt.figure(figsize=(14, 5))
        plt.bar(labs, vals)
        plt.xticks(rotation=70, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / (p.stem + "_top_nodes.png"), dpi=200)
        plt.close()

    print(f"[stats] items={n_items} nodes={n_nodes} edges={n_edges}")
    print(f"[stats] unique_edge_types={len(edge_counts)} unique_node_labels={len(node_label_counts)}")
    print(f"[stats] saved: {summary_txt}")


if __name__ == "__main__":
    main()