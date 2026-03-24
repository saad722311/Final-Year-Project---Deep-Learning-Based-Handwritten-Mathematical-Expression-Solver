from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image

from src.end2end.render_inkml_to_image import render_expression_image, read_truth_latex
from src.end2end.latex_tokenizer import tokenize_latex, Vocab


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    # img is L and already inverted (ink=high). Normalize to [0,1]
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, ...]  # (1,H,W)
    return t


class CrohmeE2EDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        vocab: Vocab,
        image_hw: Tuple[int, int] = (256, 256),
        max_items: int = 0,
    ):
        self.path = Path(jsonl_path)
        self.vocab = vocab
        self.image_hw = image_hw
        self.items: List[Dict[str, Any]] = []

        with self.path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                o = json.loads(line)
                self.items.append(o)
                if max_items and (i + 1) >= max_items:
                    break

    def __len__(self) -> int:
        return len(self.items)

    def _get_inkml_path(self, o):
        if "inkml_path" in o:
            return o["inkml_path"]
        if "inkml" in o:
            return o["inkml"]
        if "path" in o:
            return o["path"]
        raise KeyError(f"Cannot find inkml path key in item keys={list(o.keys())}")

    def __getitem__(self, idx: int):
        o = self.items[idx]
        inkml = self._get_inkml_path(o)

        img = render_expression_image(inkml, out_hw=self.image_hw)
        x = pil_to_tensor(img)

        latex = read_truth_latex(inkml)
        toks = tokenize_latex(latex)
        y = torch.tensor(self.vocab.encode(toks, add_bos_eos=True), dtype=torch.long)

        return x, y, inkml