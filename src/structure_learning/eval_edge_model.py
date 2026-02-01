# src/structure_learning/eval_edge_model.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.structure_learning.edge_dataset import (
    EDGE_TYPES,
    build_label_vocab,
    EdgeDataset,
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


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def compute_metrics(y_true, y_pred, n_classes):
    conf = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1

    support = [sum(conf[c]) for c in range(n_classes)]
    pred_count = [sum(conf[r][c] for r in range(n_classes)) for c in range(n_classes)]
    tp = [conf[c][c] for c in range(n_classes)]

    precision, recall, f1 = [], [], []
    for c in range(n_classes):
        p = safe_div(tp[c], pred_count[c])
        r = safe_div(tp[c], support[c])
        f = safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)

    total = sum(support)
    acc = safe_div(sum(tp), total)
    macro_f1 = sum(f1) / n_classes
    weighted_f1 = safe_div(sum(f1[c] * support[c] for c in range(n_classes)), total)

    return {
        "conf": conf,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total": total,
    }


def save_confusion_png(conf, labels, out_png: Path):
    plt.figure(figsize=(10, 8))
    plt.imshow(conf)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)

    # IMPORTANT: needed if meta.json doesn't store label_vocab
    ap.add_argument("--train_jsonl", default="", help="Train JSONL used to rebuild label_vocab if missing in meta.json")

    ap.add_argument("--neg_per_pos", type=int, default=3)
    ap.add_argument("--k_right", type=int, default=6)
    ap.add_argument("--k_any", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    ckpt_path = Path(args.ckpt)
    meta_path = ckpt_path.parent / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"[eval] meta.json not found next to ckpt: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # -------------------------
    # edge types
    # -------------------------
    if "edge_type_to_id" in meta:
        edge_type_to_id = meta["edge_type_to_id"]
    else:
        edge_type_to_id = {t: i for i, t in enumerate(EDGE_TYPES)}

    id_to_edge = {v: k for k, v in edge_type_to_id.items()}
    n_edge_types = len(edge_type_to_id)

    # -------------------------
    # label vocab (robust)
    # -------------------------
    if "label_vocab" in meta and isinstance(meta["label_vocab"], dict) and len(meta["label_vocab"]) > 0:
        label_vocab = meta["label_vocab"]
        print(f"[eval] label_vocab loaded from meta.json (size={len(label_vocab)})")
    else:
        if not args.train_jsonl:
            raise SystemExit(
                "[eval] meta.json has no label_vocab, so you MUST pass --train_jsonl "
                "(the train structure dataset jsonl) to rebuild it."
            )
        print("[eval] meta.json missing label_vocab -> rebuilding from train_jsonl ...")
        label_vocab = build_label_vocab(args.train_jsonl, max_items=0)
        print(f"[eval] rebuilt label_vocab size={len(label_vocab)}")

    n_labels = len(label_vocab)

    # -------------------------
    # feat_dim (robust)
    # -------------------------
    feat_dim = int(meta.get("feat_dim", 13))  # default=13 for your pair_features
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    val_ds = EdgeDataset(
        jsonl_path=args.val_jsonl,
        label_vocab=label_vocab,
        edge_type_to_id=edge_type_to_id,
        neg_per_pos=args.neg_per_pos,
        k_right=args.k_right,
        k_any=args.k_any,
        seed=123,
    )
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"val_samples={len(val_ds)}")

    # Model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = EdgeMLP(n_labels=n_labels, n_edge_types=n_edge_types, feat_dim=feat_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ce = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    y_true, y_pred = [], []
    conf_pairs = defaultdict(int)

    with torch.no_grad():
        for xi, xj, feat, lab in loader:
            xi, xj, feat, lab = xi.to(device), xj.to(device), feat.to(device), lab.to(device)
            logits = model(xi, xj, feat)
            pred = logits.argmax(1)

            total += lab.size(0)
            correct += (pred == lab).sum().item()

            lt = lab.cpu().tolist()
            lp = pred.cpu().tolist()
            y_true.extend(lt)
            y_pred.extend(lp)

            for t, p in zip(lt, lp):
                if t != p:
                    conf_pairs[(t, p)] += 1

    m = compute_metrics(y_true, y_pred, n_edge_types)

    print(f"Eval acc={m['acc']:.4f}")
    print(f"Macro-F1={m['macro_f1']:.4f}  Weighted-F1={m['weighted_f1']:.4f}")

    # Per-class report
    with (out_dir / "per_class_report.csv").open("w", encoding="utf-8") as f:
        f.write("edge,precision,recall,f1,support\n")
        for i in range(n_edge_types):
            f.write(
                f"{id_to_edge[i]},{m['precision'][i]:.6f},{m['recall'][i]:.6f},{m['f1'][i]:.6f},{m['support'][i]}\n"
            )

    save_confusion_png(m["conf"], [id_to_edge[i] for i in range(n_edge_types)],
                       out_dir / "confusion_matrix.png")

    with (out_dir / "top_confusions.csv").open("w", encoding="utf-8") as f:
        f.write("gt,pred,count\n")
        for (t, p), c in sorted(conf_pairs.items(), key=lambda x: -x[1])[:80]:
            f.write(f"{id_to_edge[t]},{id_to_edge[p]},{c}\n")

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "accuracy": m["acc"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "total": m["total"],
                "n_labels": n_labels,
                "n_edge_types": n_edge_types,
                "feat_dim": feat_dim,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()