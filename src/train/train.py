# src/train/train.py
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from xml.parsers.expat import model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.utils.seed import seed_everything
from src.utils.checkpoints import save_checkpoint

from src.data.tokenizer import CharTokenizer
from src.data.datasets import CROHMEProcessedConfig, CROHMEProcessedDataset
from src.data.transforms import ImageTransformConfig
from src.data.collate import HMERBatchCollator  # picklable collator (no lambda)

from src.models.hmer_model import HMERModel


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pad_id: int,
    grad_clip: float | None = None,
    max_batches: int | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["images"].to(device)
        image_widths = batch["image_widths"].to(device)  # ✅ NEW
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        # Forward -> logits (pass widths so encoder can build memory_mask)
        logits = model(images=images, input_ids=input_ids, image_widths=image_widths)  # ✅ UPDATED

        # Cross entropy: ignore padding
        loss = nn.functional.cross_entropy(
            logits.transpose(1, 2),  # (B, V, L)
            target_ids,              # (B, L)
            ignore_index=pad_id,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    seed_everything(int(cfg.get("seed", 42)))

    run_name = cfg["run"]["name"]
    output_dir = Path(cfg["run"]["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of config for reproducibility
    (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    device = get_device(cfg["train"].get("device", "auto"))
    print(f"Device: {device}")

    # -------------------
    # Tokenizer
    # -------------------
    processed_dir = Path(cfg["data"]["processed_dir"])
    train_labels_csv = processed_dir / "train_labels.csv"

    tokenizer = CharTokenizer.build_from_labels_csv(str(train_labels_csv))
    tokenizer.save(output_dir / "tokenizer.json")
    print(f"Vocab size: {tokenizer.vocab_size}")

    pad_id = tokenizer.pad_id

    # picklable collator instance (no lambda)
    collator = HMERBatchCollator(pad_id=pad_id)

    # -------------------
    # Dataset + Loader
    # -------------------
    tf_cfg = ImageTransformConfig(
        target_height=int(cfg["data"]["target_height"]),
        max_width=int(cfg["data"]["max_width"]),
        invert=bool(cfg["data"]["invert"]),
    )

    train_ds = CROHMEProcessedDataset(
        cfg=CROHMEProcessedConfig(
            processed_dir=str(processed_dir),
            split=cfg["data"]["train_split"],
        ),
        tokenizer=tokenizer,
        img_tf_cfg=tf_cfg,
    )

    valid_ds = CROHMEProcessedDataset(
        cfg=CROHMEProcessedConfig(
            processed_dir=str(processed_dir),
            split=cfg["data"]["valid_split"],
        ),
        tokenizer=tokenizer,
        img_tf_cfg=tf_cfg,
    )

    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])

    # On macOS, pin_memory does not help for MPS; keep it False.
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        collate_fn=collator,
        persistent_workers=(nw > 0),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        collate_fn=collator,
        persistent_workers=(nw > 0),
    )

    model = HMERModel(
    vocab_size=tokenizer.vocab_size,
    pad_id=tokenizer.pad_id,
    sos_id=tokenizer.sos_id,
    eos_id=tokenizer.eos_id,
    encoder_d_model=int(cfg["model"]["d_model"]),
    decoder_hidden=int(cfg["model"]["hidden_size"]),
).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    # -------------------
    # Logging
    # -------------------
    log_path = output_dir / "log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "valid_loss", "time_sec"])

    best_val = float("inf")
    epochs = int(cfg["train"]["epochs"])
    grad_clip = float(cfg["train"]["grad_clip"])
    max_batches = cfg["train"].get("max_batches_per_epoch", None)
    if max_batches is not None:
        max_batches = int(max_batches)

    global_step = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_id=pad_id,
            grad_clip=grad_clip,
            max_batches=max_batches,
        )

        with torch.no_grad():
            valid_loss = run_epoch(
                model=model,
                loader=valid_loader,
                optimizer=None,
                device=device,
                pad_id=pad_id,
                grad_clip=None,
                max_batches=max_batches,
            )

        dt = time.time() - t0
        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | {dt:.1f}s")

        # save last
        save_checkpoint(
            path=output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
        )

        # save best
        if valid_loss < best_val:
            best_val = valid_loss
            save_checkpoint(
                path=output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                extra={"best_val": best_val},
            )

        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, valid_loss, f"{dt:.2f}"])

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Run saved to: {output_dir}")


if __name__ == "__main__":
    main()