from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SymbolDatasetConfig:
    images_dir: str
    labels_csv: str
    image_size: int = 64  # resize to square for the CNN
    invert: bool = False  # keep False if your PNGs are black strokes on white background


def _build_vocab(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniq = sorted(set(labels))
    stoi = {s: i for i, s in enumerate(uniq)}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


class SymbolDataset(Dataset):
    """
    CSV format expected (you already have this):
      filename,label,ui,inkml,trace_ids

    Uses:
      images_dir/<filename>  -> PNG
      label -> class
    """

    def __init__(self, cfg: SymbolDatasetConfig):
        self.cfg = cfg
        self.images_dir = Path(cfg.images_dir)
        self.labels_csv = Path(cfg.labels_csv)

        rows = []
        labels = []

        with self.labels_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fn = (r.get("filename") or "").strip()
                lab = (r.get("label") or "").strip()
                if not fn or not lab:
                    continue
                rows.append((fn, lab))
                labels.append(lab)

        if not rows:
            raise RuntimeError(f"No usable rows found in: {self.labels_csv}")

        self.rows = rows
        self.stoi, self.itos = _build_vocab(labels)
        self.num_classes = len(self.stoi)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, filename: str) -> torch.Tensor:
        p = self.images_dir / filename
        if not p.exists():
            raise FileNotFoundError(f"Missing image: {p}")

        img = Image.open(p).convert("L")  # grayscale

        # Resize (no fancy aspect logic; your renderer already outputs square-ish crops)
        img = img.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)

        x = torch.tensor(list(img.getdata()), dtype=torch.float32).view(
            self.cfg.image_size, self.cfg.image_size
        )
        x = x / 255.0  # [0,1]

        # if invert, flip white<->black
        if self.cfg.invert:
            x = 1.0 - x

        # add channel: (1,H,W)
        x = x.unsqueeze(0)
        return x

    def __getitem__(self, idx: int):
        filename, label = self.rows[idx]
        x = self._load_image(filename)
        y = self.stoi[label]
        return x, y, filename, label