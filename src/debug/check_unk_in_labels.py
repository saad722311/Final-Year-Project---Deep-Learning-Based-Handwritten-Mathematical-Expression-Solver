# src/debug/check_unk_in_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from src.data.tokenizer import CharTokenizer
from src.data.datasets import CROHMEProcessedConfig, CROHMEProcessedDataset
from src.data.transforms import ImageTransformConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train", "valid", "test"], required=True)
    ap.add_argument("--max_show", type=int, default=20)
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

    total = 0
    unk_count = 0
    examples = []

    for i in range(len(ds)):
        item = ds[i]
        # item["target_ids"] contains tokenized sequence
        ids = item["target_ids"]
        total += 1
        if tokenizer.unk_id in ids:
            unk_count += 1
            if len(examples) < args.max_show:
                examples.append((item["filename"], item["label"]))

    rate = 100.0 * unk_count / max(1, total)
    print(f"Split={args.split} | samples={total} | UNK_in_GT={unk_count} ({rate:.2f}%)")

    if examples:
        print("\nExamples with UNK in GT:")
        for fn, lab in examples:
            print(f"- {fn}: {lab}")


if __name__ == "__main__":
    main()