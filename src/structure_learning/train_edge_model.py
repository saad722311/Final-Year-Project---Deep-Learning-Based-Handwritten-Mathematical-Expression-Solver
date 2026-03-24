# src/structure_learning/train_edge_model.py
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.structure_learning.edge_dataset import EdgeDataset, EDGE_TYPES, build_label_vocab
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


@torch.no_grad()
def evaluate(model, loader, device, n_edge_types: int):
    model.eval()
    total = 0
    correct = 0
    conf = torch.zeros((n_edge_types, n_edge_types), dtype=torch.long)

    for li, lj, feat, y in loader:
        li, lj, feat, y = li.to(device), lj.to(device), feat.to(device), y.to(device)
        logits = model(li, lj, feat)
        pred = logits.argmax(dim=1)
        total += int(y.numel())
        correct += int((pred == y).sum().item())
        for g, p in zip(y.tolist(), pred.tolist()):
            conf[g, p] += 1

    acc = correct / max(1, total)
    return acc, conf


def save_confusion_png(conf: torch.Tensor, labels: list[str], out_png: Path, top_only: bool = False):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Plot as an image (no manual colors)
    plt.figure(figsize=(8, 7))
    plt.imshow(conf.cpu().numpy())
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)

    ap.add_argument("--neg_per_pos", type=int, default=3)
    ap.add_argument("--k_right", type=int, default=6)
    ap.add_argument("--k_any", type=int, default=6)

    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Device:", device)

    # vocab from train
    label_vocab = build_label_vocab(args.train_jsonl, max_items=0)
    edge_type_to_id = {t: i for i, t in enumerate(EDGE_TYPES)}

    # datasets
    train_ds = EdgeDataset(
        args.train_jsonl,
        label_vocab=label_vocab,
        edge_type_to_id=edge_type_to_id,
        max_items=0,
        neg_per_pos=args.neg_per_pos,
        k_right=args.k_right,
        k_any=args.k_any,
        seed=42,
    )
    val_ds = EdgeDataset(
        args.val_jsonl,
        label_vocab=label_vocab,
        edge_type_to_id=edge_type_to_id,
        max_items=0,
        neg_per_pos=args.neg_per_pos,
        k_right=args.k_right,
        k_any=args.k_any,
        seed=123,
    )

    print(f"train_samples={len(train_ds)} val_samples={len(val_ds)}")
    # class balance info
    train_counts = Counter()
    for _, _, _, y in DataLoader(train_ds, batch_size=4096):
        for v in y.tolist():
            train_counts[int(v)] += 1
    print("Train class counts:")
    for k, v in sorted(train_counts.items()):
        print(f"  {EDGE_TYPES[k]}: {v}")

    # class weights (helps rare relations)
    weights = torch.ones(len(EDGE_TYPES), dtype=torch.float32)
    total = sum(train_counts.values())
    for k, v in train_counts.items():
        # inverse frequency (smoothed)
        weights[k] = total / max(1.0, float(v))
    # normalize weights to mean ~1
    weights = weights / weights.mean()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = EdgeMLP(
        n_labels=len(label_vocab),
        n_edge_types=len(EDGE_TYPES),
        feat_dim=16,  # from pair_features()
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(weight=weights.to(device))

    best_acc = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for li, lj, feat, y in train_loader:
            li, lj, feat, y = li.to(device), lj.to(device), feat.to(device), y.to(device)
            logits = model(li, lj, feat)
            loss = ce(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * int(y.size(0))
            total_n += int(y.size(0))

        avg_loss = total_loss / max(1, total_n)
        val_acc, conf = evaluate(model, val_loader, device, n_edge_types=len(EDGE_TYPES))

        print(f"Epoch {epoch:02d}: train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "label_vocab": label_vocab,
                    "edge_types": EDGE_TYPES,
                    "config": vars(args),
                },
                best_path,
            )
            save_confusion_png(conf, EDGE_TYPES, out_dir / "val_confusion.png")
            print(f"  saved best -> {best_path}")

    print("Best val acc:", best_acc)

    # save small metadata
    meta = out_dir / "meta.json"
    meta.write_text(json.dumps({"best_val_acc": best_acc}, indent=2), encoding="utf-8")
    print("Saved:", meta)


if __name__ == "__main__":
    main()