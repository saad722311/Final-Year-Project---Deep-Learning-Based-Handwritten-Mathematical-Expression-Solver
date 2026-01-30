from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.symbol_recognition.dataset import SymbolDataset, SymbolDatasetConfig
from src.symbol_recognition.model import TinySymbolCNN
from src.utils.seed import seed_everything


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
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = ce(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0

    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = ce(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * x.size(0)
        total += int(x.size(0))

    return total_loss / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    seed_everything(int(cfg.get("seed", 42)))

    run_name = cfg["run"]["name"]
    out_dir = Path(cfg["run"]["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    device = get_device(cfg["train"].get("device", "auto"))
    print(f"Device: {device}")

    ds_cfg = SymbolDatasetConfig(
        images_dir=str(cfg["data"]["train_images_dir"]),
        labels_csv=str(cfg["data"]["train_labels_csv"]),
        image_size=int(cfg["data"].get("image_size", 64)),
        invert=bool(cfg["data"].get("invert", False)),
    )
    train_ds = SymbolDataset(ds_cfg)
    print(f"Train symbols: {len(train_ds)} | classes: {train_ds.num_classes}")

    # Validation (optional but recommended)
    val_images_dir = cfg["data"].get("valid_images_dir", None)
    val_labels_csv = cfg["data"].get("valid_labels_csv", None)
    if val_images_dir and val_labels_csv:
        val_cfg = SymbolDatasetConfig(
            images_dir=str(val_images_dir),
            labels_csv=str(val_labels_csv),
            image_size=int(cfg["data"].get("image_size", 64)),
            invert=bool(cfg["data"].get("invert", False)),
        )
        valid_ds = SymbolDataset(val_cfg)
        print(f"Valid symbols: {len(valid_ds)} | classes: {valid_ds.num_classes}")
    else:
        valid_ds = None
        print("Valid dataset: (not provided)")

    # Sanity: class sets must match
    if valid_ds is not None and valid_ds.stoi.keys() != train_ds.stoi.keys():
        print("⚠️ Warning: Train/valid class vocab differs. Accuracy may be misleading.")
        # Still continue, but better to use same split generation method.

    bs = int(cfg["data"].get("batch_size", 128))
    nw = int(cfg["data"].get("num_workers", 0))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=nw) if valid_ds else None

    model = TinySymbolCNN(num_classes=train_ds.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))

    epochs = int(cfg["train"]["epochs"])

    log_csv = out_dir / "log.csv"
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "valid_loss", "valid_acc", "time_sec"])

    best_acc = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, opt, device)

        if valid_loader is not None:
            va_loss, va_acc = eval_epoch(model, valid_loader, device)
        else:
            va_loss, va_acc = float("nan"), float("nan")

        dt = time.time() - t0
        print(f"Epoch {ep}/{epochs} | train_loss={tr_loss:.4f} | valid_loss={va_loss:.4f} | valid_acc={va_acc:.4f} | {dt:.1f}s")

        # save last
        torch.save(
            {
                "model": model.state_dict(),
                "stoi": train_ds.stoi,
                "itos": train_ds.itos,
                "image_size": ds_cfg.image_size,
            },
            last_path,
        )

        # save best by val acc (if val exists)
        if valid_loader is not None and va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "stoi": train_ds.stoi,
                    "itos": train_ds.itos,
                    "image_size": ds_cfg.image_size,
                    "best_acc": best_acc,
                },
                best_path,
            )

        with log_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ep, f"{tr_loss:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}", f"{dt:.2f}"])

    print(f"\nDone. Logs: {log_csv}")
    if valid_loader is not None:
        print(f"Best valid acc: {best_acc:.4f} | checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()