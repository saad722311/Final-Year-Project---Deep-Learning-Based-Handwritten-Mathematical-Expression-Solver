# src/data/collate.py
from __future__ import annotations

from typing import Dict, List

import torch


def pad_1d(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def collate_batch(batch: List[Dict], pad_id: int) -> Dict:
    """
    Pads:
      - images along width to max in batch
      - token sequences along length to max in batch

    Also returns:
      - image_widths: true widths BEFORE padding (needed for encoder memory mask)
    """
    # Images: (1,H,W) but W may vary; pad width to max in batch
    images = [b["image"] for b in batch]
    image_widths = torch.tensor([im.size(-1) for im in images], dtype=torch.long)

    max_w = int(image_widths.max().item())
    h = images[0].size(-2)

    padded_images = torch.zeros((len(images), 1, h, max_w), dtype=images[0].dtype)
    for i, im in enumerate(images):
        w = im.size(-1)
        padded_images[i, :, :, :w] = im

    input_ids = pad_1d([b["input_ids"] for b in batch], pad_value=pad_id)
    target_ids = pad_1d([b["target_ids"] for b in batch], pad_value=pad_id)

    # lengths help later (masking/metrics)
    input_lens = torch.tensor([b["input_ids"].size(0) for b in batch], dtype=torch.long)
    target_lens = torch.tensor([b["target_ids"].size(0) for b in batch], dtype=torch.long)

    return {
        "images": padded_images,          # (B,1,H,Wmax)
        "image_widths": image_widths,     # (B,) true widths before padding

        "input_ids": input_ids,           # (B,L)
        "target_ids": target_ids,         # (B,L)
        "input_lens": input_lens,
        "target_lens": target_lens,

        "labels": [b["label_str"] for b in batch],
        "filenames": [b["filename"] for b in batch],
    }


class HMERBatchCollator:
    """
    Picklable collator wrapper so DataLoader can use num_workers > 0 on macOS.
    Avoids lambda / nested functions (not picklable under spawn).
    """
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        return collate_batch(batch, pad_id=self.pad_id)