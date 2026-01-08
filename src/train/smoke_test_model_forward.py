# src/train/smoke_test_model_forward.py
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.tokenizer import CharTokenizer
from src.data.datasets import CROHMEProcessedDataset, CROHMEProcessedConfig
from src.data.collate import collate_batch
from src.models.hmer_model import HMERModel


def main():
    processed_dir = Path("03-development/datasets/TC11_CROHME23/processed")
    train_csv = processed_dir / "train_labels.csv"

    # Build tokenizer from train labels
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
    images = batch["images"]          # (B,1,128,W)
    input_ids = batch["input_ids"]    # (B,L)
    target_ids = batch["target_ids"]  # (B,L)

    print("images:", images.shape)
    print("input_ids:", input_ids.shape)
    print("target_ids:", target_ids.shape)

    # Model
    model = HMERModel(vocab_size=tok.vocab_size, pad_id=tok.pad_id)
    model.train()

    logits = model(images, input_ids)   # (B,L,V)
    print("logits:", logits.shape)

    # Loss: CrossEntropy over vocab, ignore PAD
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)

    # reshape to (B*L, V) and (B*L)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
    print("loss:", float(loss.item()))


if __name__ == "__main__":
    main()