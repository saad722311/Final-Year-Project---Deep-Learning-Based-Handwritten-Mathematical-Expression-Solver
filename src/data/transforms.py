# src/data/transforms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image
import torch


@dataclass
class ImageTransformConfig:
    target_height: int = 128
    max_width: int = 1024  # safety cap to avoid extreme widths
    invert: bool = False   # set True if your images are white ink on black bg


def _resize_keep_aspect(img: Image.Image, target_h: int, max_w: int) -> Image.Image:
    w, h = img.size
    if h == 0:
        raise ValueError("Image has zero height")

    scale = target_h / float(h)
    new_w = int(round(w * scale))
    new_w = min(new_w, max_w)
    img = img.resize((new_w, target_h), resample=Image.BILINEAR)
    return img


def pil_to_tensor(img: Image.Image, invert: bool = False) -> torch.Tensor:
    """
    Returns float tensor shape (1, H, W), values in [0,1], normalized to mean=0.5 std=0.5.
    """
    img = img.convert("L")  # grayscale
    arr = np.array(img, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr

    # (H, W) -> (1, H, W)
    t = torch.from_numpy(arr).unsqueeze(0)
    # Normalize roughly to [-1, 1]
    t = (t - 0.5) / 0.5
    return t


class DefaultImageTransform:
    def __init__(self, cfg: ImageTransformConfig | None = None):
        self.cfg = cfg or ImageTransformConfig()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = _resize_keep_aspect(img, self.cfg.target_height, self.cfg.max_width)
        return pil_to_tensor(img, invert=self.cfg.invert)