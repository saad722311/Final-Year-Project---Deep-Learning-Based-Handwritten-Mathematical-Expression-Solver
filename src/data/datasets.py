# src/data/datasets.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.tokenizer import CharTokenizer
from src.data.transforms import DefaultImageTransform, ImageTransformConfig


@dataclass
class CROHMEProcessedConfig:
    processed_dir: str = "03-development/datasets/TC11_CROHME23/processed"
    split: str = "train"  # train | valid | test
    digits: int = 6       # filenames like 000000.png


class CROHMEProcessedDataset(Dataset):
    """
    Loads MathWriting-style processed CROHME:
      processed/{split}_images/*.png
      processed/{split}_labels.csv with columns: filename,label
    """
    def __init__(
        self,
        cfg: CROHMEProcessedConfig,
        tokenizer: CharTokenizer,
        img_tf_cfg: ImageTransformConfig | None = None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.processed_dir = Path(cfg.processed_dir)

        if cfg.split not in ("train", "valid", "test"):
            raise ValueError("split must be one of: train, valid, test")

        self.images_dir = self.processed_dir / f"{cfg.split}_images"
        self.csv_path = self.processed_dir / f"{cfg.split}_labels.csv"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing labels csv: {self.csv_path}")

        self.rows: List[Tuple[str, str]] = []
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fn = r.get("filename") or r.get("image")  # tolerate older schema
                lb = r.get("label") or r.get("latex")
                if not fn or lb is None:
                    continue
                self.rows.append((fn, lb))

        self.transform = DefaultImageTransform(img_tf_cfg)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        filename, label = self.rows[idx]
        img_path = self.images_dir / filename
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image referenced by CSV: {img_path}")

        img = Image.open(img_path)
        x: torch.Tensor = self.transform(img)  # (1, H, W)

        token_ids = self.tokenizer.encode(label)

        # Teacher forcing setup:
        # input_ids:  <SOS> a b c
        # target_ids: a b c <EOS>
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)

        return {
            "image": x,
            "label_str": label,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "filename": filename,
        }