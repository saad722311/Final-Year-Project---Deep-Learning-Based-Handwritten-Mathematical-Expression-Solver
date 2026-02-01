from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.structure_learning.edge_dataset import (
    EDGE_TYPES,
    build_label_vocab,
    candidate_pairs,
    pair_features,
)
from src.structure_learning.edge_model import EdgeMLP


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_label(s: str) -> str:
    return (s or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt")
    ap.add_argument("--meta", default="", help="meta.json (default: next to ckpt)")
    ap.add_argument("--train_jsonl", required=True, help="needed to rebuild label_vocab reliably")
    ap.add_argument("--in_jsonl", required=True, help="val/test jsonl to run inference on")
    ap.add_argument("--out_jsonl", required=True, help="where to write predictions")

    ap.add_argument("--k_right", type=int, default=6)
    ap.add_argument("--k_any", type=int, default=6)
    ap.add_argument("--thr", type=float, default=0.50, help="keep non-NONE edges with prob >= thr")

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    ckpt_path = Path(args.ckpt)
    meta_path = Path(args.meta) if args.meta else (ckpt_path.parent / "meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # edge types
    edge_type_to_id = meta.get("edge_type_to_id", {t: i for i, t in enumerate(EDGE_TYPES)})
    id_to_edge = {v: k for k, v in edge_type_to_id.items()}
    n_edge_types = len(edge_type_to_id)

    # label vocab (rebuild; your meta didn't store it)
    label_vocab = build_label_vocab(args.train_jsonl, max_items=0)
    n_labels = len(label_vocab)

    feat_dim = int(meta.get("feat_dim", 13))

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = EdgeMLP(n_labels=n_labels, n_edge_types=n_edge_types, feat_dim=feat_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept_graphs = 0
    total_edges_kept = 0

    with Path(args.in_jsonl).open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if args.limit and idx >= args.limit:
                break
            if not line.strip():
                continue

            obj = json.loads(line)
            nodes = obj.get("nodes", [])
            if not nodes:
                continue

            cand = candidate_pairs(nodes, k_right=args.k_right, k_any=args.k_any)

            # build tensors
            xi_ids = []
            xj_ids = []
            feats = []
            for (i, j) in cand:
                li = _safe_label(nodes[i].get("label", ""))
                lj = _safe_label(nodes[j].get("label", ""))
                xi_ids.append(label_vocab.get(li, label_vocab["<UNK>"]))
                xj_ids.append(label_vocab.get(lj, label_vocab["<UNK>"]))
                feats.append(pair_features(nodes[i], nodes[j]))

            xi = torch.tensor(xi_ids, dtype=torch.long, device=device)
            xj = torch.tensor(xj_ids, dtype=torch.long, device=device)
            feat = torch.tensor(feats, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits = model(xi, xj, feat)
                probs = F.softmax(logits, dim=-1)  # (M, C)
                pred = torch.argmax(probs, dim=-1)  # (M,)

            pred_edges: List[List] = []
            for (i, j), cid, pvec in zip(cand, pred.tolist(), probs.tolist()):
                et = id_to_edge[int(cid)]
                prob = float(pvec[int(cid)])
                if et != "NONE" and prob >= args.thr:
                    pred_edges.append([int(i), int(j), et, prob])

            out_obj = {
                "ui": obj.get("ui"),
                "inkml_path": obj.get("inkml_path"),
                "nodes": nodes,
                "pred_edges": pred_edges,      # [src, dst, type, prob]
                "thr": args.thr,
                "k_right": args.k_right,
                "k_any": args.k_any,
            }

            fout.write(json.dumps(out_obj) + "\n")
            kept_graphs += 1
            total_edges_kept += len(pred_edges)

    print(f"[predict] graphs={kept_graphs} total_pred_edges_kept={total_edges_kept}")
    print(f"[predict] saved: {out_path}")


if __name__ == "__main__":
    main()