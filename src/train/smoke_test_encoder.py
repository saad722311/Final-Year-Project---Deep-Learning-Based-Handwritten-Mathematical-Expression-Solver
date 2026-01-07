# src/train/smoke_test_encoder.py
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.tokenizer import CharTokenizer
from src.data.datasets import CROHMEProcessedDataset, CROHMEProcessedConfig
from src.data.collate import collate_batch
from src.models.cnn_encoder import CNNEncoder, CNNEncoderConfig


def main():
    processed_dir = Path("03-development/datasets/TC11_CROHME23/processed")
    train_csv = processed_dir / "train_labels.csv"

    # Build tokenizer from train labels (uses your CSV: filename,label)
    texts = []
    with train_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            texts.append(r["label"])
    tok = CharTokenizer.build_from_texts(texts)
    print("Vocab size:", tok.vocab_size)

    # Dataset + DataLoader
    ds = CROHMEProcessedDataset(
        cfg=CROHMEProcessedConfig(processed_dir=str(processed_dir), split="train"),
        tokenizer=tok,
    )

    dl = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_batch(b, pad_id=tok.pad_id),
    )

    batch = next(iter(dl))
    images = batch["images"]  # (B,1,128,W)

    print("Input images:", images.shape)

    # Encoder
    enc = CNNEncoder(CNNEncoderConfig(in_channels=1, d_model=256))
    enc.eval()

    with torch.no_grad():
        memory = enc(images)

    print("Encoder memory:", memory.shape)
    print("Meaning: (B, T, D) = (batch, time_steps_across_width, feature_dim)")
    print("T (time steps) =", memory.shape[1])
    print("D (feature dim) =", memory.shape[2])


if __name__ == "__main__":
    main()