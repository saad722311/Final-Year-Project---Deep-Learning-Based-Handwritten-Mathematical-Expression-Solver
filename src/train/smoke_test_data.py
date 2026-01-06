# src/train/smoke_test_data.py
from __future__ import annotations

from pathlib import Path
import csv

from torch.utils.data import DataLoader

from src.data.tokenizer import CharTokenizer
from src.data.datasets import CROHMEProcessedDataset, CROHMEProcessedConfig
from src.data.collate import collate_batch


def main():
    processed_dir = Path("03-development/datasets/TC11_CROHME23/processed")
    train_csv = processed_dir / "train_labels.csv"

    # Build tokenizer from training labels
    texts = []
    with train_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            texts.append(r["label"])
    tok = CharTokenizer.build_from_texts(texts)
    print("Vocab size:", tok.vocab_size)

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
    print("images:", batch["images"].shape)       # (B,1,H,W)
    print("input_ids:", batch["input_ids"].shape)
    print("target_ids:", batch["target_ids"].shape)
    print("sample label:", batch["labels"][0])
    print("decoded check:", tok.decode(batch["input_ids"][0].tolist()))

if __name__ == "__main__":
    main()