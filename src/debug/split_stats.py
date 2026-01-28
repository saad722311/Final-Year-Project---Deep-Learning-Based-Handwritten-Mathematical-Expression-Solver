# src/debug/split_stats.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import yaml

from src.data.tokenizer import CharTokenizer
from src.data.datasets import CROHMEProcessedConfig, CROHMEProcessedDataset
from src.data.transforms import ImageTransformConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train", "valid", "test"], required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_name = cfg["run"]["name"]
    output_dir = Path(cfg["run"]["output_dir"]) / run_name
    tokenizer = CharTokenizer.load(output_dir / "tokenizer.json")

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

    lengths = []
    for i in range(len(ds)):
        item = ds[i]
        # count non-pad later, but here target_ids is per-sample (no pad), includes EOS
        lengths.append(len(item["target_ids"]))

    arr = np.array(lengths, dtype=np.int32)

    def pct(x): 
        return float(np.percentile(arr, x))

    print(f"Split={args.split} n={len(arr)}")
    print(f"len mean={arr.mean():.2f} std={arr.std():.2f}")
    print(f"len min={arr.min()} p50={pct(50):.0f} p90={pct(90):.0f} p95={pct(95):.0f} max={arr.max()}")


if __name__ == "__main__":
    main()