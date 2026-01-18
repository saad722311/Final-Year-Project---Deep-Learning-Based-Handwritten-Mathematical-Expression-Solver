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
        debug_print: bool = False,
        debug_limit: int = 5,
        warn_unk: bool = True,
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

        # Debug printing controls
        self.debug_print = debug_print
        self.debug_limit = debug_limit
        self._debug_count = 0

        self.warn_unk = warn_unk

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        filename, label = self.rows[idx]
        img_path = self.images_dir / filename
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image referenced by CSV: {img_path}")

        img = Image.open(img_path)
        x: torch.Tensor = self.transform(img)  # (1, H, W)

        # IMPORTANT: encode() is expected to include special tokens: [SOS, ..., EOS]
        token_ids = self.tokenizer.encode(label)

        # Optional UNK warnings (helps diagnose “can’t overfit”)
        unk_id = getattr(self.tokenizer, "unk_id", None)
        if self.warn_unk and unk_id is not None:
            unk_count = sum(1 for t in token_ids if t == unk_id)
            if unk_count > 0:
                print(f"[WARN] UNK in label | file={filename} | unk_count={unk_count}")
                print("LABEL:", label)
                print("ENC  :", self.tokenizer.decode(token_ids, remove_special=False))
                print("----")

        # Teacher forcing setup:
        # input_ids:  <SOS> a b c
        # target_ids: a b c <EOS>
        if len(token_ids) < 2:
            # Extremely defensive: should not happen if encode adds SOS/EOS
            input_ids = torch.tensor([self.tokenizer.sos_id], dtype=torch.long)
            target_ids = torch.tensor([self.tokenizer.eos_id], dtype=torch.long)
        else:
            input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
            target_ids = torch.tensor(token_ids[1:], dtype=torch.long)

        # Debug print (limited)
        if self.debug_print and self._debug_count < self.debug_limit:
            self._debug_count += 1
            print("FILE :", filename)
            print("LABEL:", label)
            print("ENC  :", self.tokenizer.decode(token_ids, remove_special=False))
            print("IN   :", self.tokenizer.decode(input_ids.tolist(), remove_special=False))
            print("TGT  :", self.tokenizer.decode(target_ids.tolist(), remove_special=False))
            print("----")

        return {
            "image": x,
            "label_str": label,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "filename": filename,
        }