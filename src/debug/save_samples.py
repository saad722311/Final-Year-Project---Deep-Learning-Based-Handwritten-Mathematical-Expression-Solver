# src/debug/save_samples.py
from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
import yaml

from src.data.tokenizer import CharTokenizer
from src.data.datasets import CROHMEProcessedConfig, CROHMEProcessedDataset
from src.data.transforms import ImageTransformConfig
from src.data.collate import HMERBatchCollator


def save_tensor_image(img: torch.Tensor, out_path: Path) -> None:
    """
    img: (1, H, W) or (H, W) float tensor in [0,1] (or close).
    Saves as PNG using PIL.
    """
    from PIL import Image
    if img.dim() == 3:
        img = img[0]
    x = img.detach().cpu().clamp(0, 1).mul(255).byte().numpy()
    Image.fromarray(x, mode="L").save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train", "valid", "test"], required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out_dir", type=str, default="results/debug_samples")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    run_name = cfg["run"]["name"]
    output_dir = Path(cfg["run"]["output_dir"]) / run_name
    tok_path = output_dir / "tokenizer.json"
    tokenizer = CharTokenizer.load(tok_path)

    processed_dir = Path(cfg["data"]["processed_dir"])

    tf_cfg = ImageTransformConfig(
        target_height=int(cfg["data"]["target_height"]),
        max_width=int(cfg["data"]["max_width"]),
        invert=bool(cfg["data"].get("invert", False)),
    )

    ds = CROHMEProcessedDataset(
        cfg=CROHMEProcessedConfig(processed_dir=str(processed_dir), split=args.split),
        tokenizer=tokenizer,
        img_tf_cfg=tf_cfg,
        debug_print=False,
    )

    collator = HMERBatchCollator(pad_id=tokenizer.pad_id)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collator)

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # sample first n items from shuffled loader
    for i, batch in enumerate(loader):
        if i >= args.n:
            break
        img = batch["images"][0]  # (1,H,W)
        label = batch["labels"][0]
        fname = batch["filenames"][0]

        save_tensor_image(img, out_dir / f"{i:02d}_{fname}")
        (out_dir / f"{i:02d}_{fname}.txt").write_text(label, encoding="utf-8")

    print(f"Saved {args.n} samples to: {out_dir}")


if __name__ == "__main__":
    main()