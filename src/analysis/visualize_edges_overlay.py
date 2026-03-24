from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt


def load_map(path: str) -> dict:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ui = o.get("ui")
            if ui:
                m[ui] = o
    return m


def node_xy(node: dict):
    """
    Try to get a 2D position for a node.
    Priority: bbox center -> (cx,cy)
    Returns (x,y) or None.
    """
    bbox = node.get("bbox")
    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    cx, cy = node.get("cx"), node.get("cy")
    if cx is not None and cy is not None:
        return (float(cx), float(cy))

    return None


def build_node_pos(graph: dict) -> dict[int, tuple[float, float]]:
    """
    Returns dict: node_index -> (x,y)
    Skips nodes without coordinates (often virtual nodes).
    """
    pos = {}
    nodes = graph.get("nodes", graph.get("symbols", []))
    for i, n in enumerate(nodes):
        p = node_xy(n)
        if p is not None:
            pos[i] = p
    return pos


def draw_edges(ax, edges, pos, color: str, alpha: float, label: str, lw: float = 1.5):
    """
    edges: list of [s, t, rel, score?]
    pos: dict node_index -> (x,y)
    """
    drawn = 0
    rel_counter = Counter()

    for e in edges:
        if len(e) < 3:
            continue
        s, t, rel = e[0], e[1], e[2]
        if s not in pos or t not in pos:
            continue

        x1, y1 = pos[s]
        x2, y2 = pos[t]

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=alpha),
        )
        # small label near midpoint
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax.text(mx, my, rel, color=color, fontsize=7, alpha=min(1.0, alpha + 0.15))

        drawn += 1
        rel_counter[rel] += 1

    ax.plot([], [], color=color, lw=lw, alpha=alpha, label=f"{label} (drawn={drawn})")
    return drawn, rel_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--gt_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--pick_mode", choices=["first", "random"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = load_map(args.pred_jsonl)
    gt = load_map(args.gt_jsonl)

    common = sorted(set(pred) & set(gt))
    print("pred graphs:", len(pred))
    print("gt graphs  :", len(gt))
    print("intersection:", len(common))

    if not common:
        raise SystemExit("No common UIs found between pred and gt JSONL")

    if args.pick_mode == "random":
        sample = common[:]
        random.shuffle(sample)
        sample = sample[: args.limit]
    else:
        sample = common[: args.limit]

    # also save the chosen list
    (out_dir / "picked_ui.txt").write_text("\n".join(sample), encoding="utf-8")
    print("picked:", len(sample), "->", str(out_dir / "picked_ui.txt"))

    total_drawn_pred = 0
    total_drawn_gt = 0
    rel_pred_all = Counter()
    rel_gt_all = Counter()

    for ui in sample:
        po = pred[ui]
        go = gt[ui]

        # positions (use GT nodes because they have consistent layout)
        pos = build_node_pos(go)

        nodes = go.get("nodes", go.get("symbols", []))

        # edges
        pred_edges = po.get("pred_edges", [])
        gt_edges = go.get("edges", go.get("gt_edges", []))

        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # draw nodes
        for i, n in enumerate(nodes):
            p = pos.get(i)
            if p is None:
                continue
            x, y = p
            lab = n.get("label", n.get("sym", n.get("text", str(i))))
            ax.scatter([x], [y], s=35)
            ax.text(x, y, str(lab), fontsize=9)

        # draw edges overlay
        drawn_gt, rel_gt = draw_edges(ax, gt_edges, pos, color="green", alpha=0.55, label="GT", lw=2.0)
        drawn_pr, rel_pr = draw_edges(ax, pred_edges, pos, color="red", alpha=0.55, label="Pred", lw=1.5)

        total_drawn_gt += drawn_gt
        total_drawn_pred += drawn_pr
        rel_gt_all.update(rel_gt)
        rel_pred_all.update(rel_pr)

        ax.set_title(f"Edges Overlay — {ui}  (green=GT, red=Pred)")
        ax.invert_yaxis()  # usually needed for image coordinate systems
        ax.axis("off")
        ax.legend(loc="upper right")

        out_path = out_dir / f"{ui}_overlay.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # summary
    print("\n[summary]")
    print("total drawn GT  :", total_drawn_gt)
    print("total drawn Pred:", total_drawn_pred)
    print("GT rel dist:", rel_gt_all)
    print("Pred rel dist:", rel_pred_all)
    print("saved overlays ->", str(out_dir))


if __name__ == "__main__":
    main()