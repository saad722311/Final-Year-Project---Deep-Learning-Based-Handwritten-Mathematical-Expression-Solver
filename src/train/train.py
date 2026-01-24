# src/train/train.py
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
    label_smoothing: float = 0.0,  # ✅ passed in from main()
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["images"].to(device)

        image_widths = batch.get("image_widths", None)
        if image_widths is not None:
            image_widths = image_widths.to(device)

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        # Forward -> logits
        logits = model(images=images, input_ids=input_ids, image_widths=image_widths)  # (B,L,V)

        # Cross entropy (ignore PAD)
        # logits: (B,L,V) -> (B,V,L) for CE
        loss = nn.functional.cross_entropy(
            logits.transpose(1, 2),
            target_ids,
            ignore_index=pad_id,
            label_smoothing=float(label_smoothing),
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

    # ✅ label smoothing (train only). Keep val at 0.0 for "true" CE.
    train_label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))

    # -------------------
    # Tokenizer
    # -------------------
    processed_dir = Path(cfg["data"]["processed_dir"])
    train_labels_csv = processed_dir / "train_labels.csv"

    tokenizer = CharTokenizer.build_from_labels_csv(str(train_labels_csv))
    tokenizer.save(output_dir / "tokenizer.json")
    print(f"Vocab size: {tokenizer.vocab_size}")

    pad_id = tokenizer.pad_id
    collator = HMERBatchCollator(pad_id=pad_id)

    # -------------------
    # Dataset + Loader
    # -------------------
    tf_cfg = ImageTransformConfig(
        target_height=int(cfg["data"]["target_height"]),
        max_width=int(cfg["data"]["max_width"]),
        invert=bool(cfg["data"].get("invert", False)),
    )

    # Optional debug toggles (only affect printing, not training)
    debug_print = bool(cfg.get("debug", {}).get("dataset_print", False))
    debug_limit = int(cfg.get("debug", {}).get("dataset_print_limit", 5))
    warn_unk = bool(cfg.get("debug", {}).get("warn_unk", True))

    train_ds = CROHMEProcessedDataset(
        cfg=CROHMEProcessedConfig(
            processed_dir=str(processed_dir),
            split=cfg["data"]["train_split"],
        ),
        tokenizer=tokenizer,
        img_tf_cfg=tf_cfg,
        debug_print=debug_print,
        debug_limit=debug_limit,
        warn_unk=warn_unk,
    )

    valid_ds = CROHMEProcessedDataset(
        cfg=CROHMEProcessedConfig(
            processed_dir=str(processed_dir),
            split=cfg["data"]["valid_split"],
        ),
        tokenizer=tokenizer,
        img_tf_cfg=tf_cfg,
        debug_print=False,  # keep valid clean
        debug_limit=0,
        warn_unk=warn_unk,
    )

    # ✅ TRUE LIMITING: restrict dataset size for real overfit sanity check
    limit_train = cfg["data"].get("limit_train", None)
    limit_valid = cfg["data"].get("limit_valid", None)

    if limit_train is not None:
        train_ds = Subset(train_ds, list(range(int(limit_train))))
    if limit_valid is not None:
        valid_ds = Subset(valid_ds, list(range(int(limit_valid))))

    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])

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

    # -------------------
    # Model
    # -------------------
    model = HMERModel(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
        unk_id=tokenizer.unk_id,
        encoder_d_model=int(cfg["model"]["d_model"]),
        decoder_hidden=int(cfg["model"].get("hidden_size", 256)),
        # supports transformer too
        decoder_type=str(cfg["model"].get("decoder_type", "lstm")),
        n_heads=int(cfg["model"].get("n_heads", 4)),
        n_layers=int(cfg["model"].get("n_layers", 4)),
        ff_dim=int(cfg["model"].get("ff_dim", 1024)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        max_len=int(cfg["data"].get("max_decode_len", 256)),
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
            label_smoothing=train_label_smoothing,  # ✅ train smoothing
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
                label_smoothing=0.0,  # ✅ keep validation CE "true"
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