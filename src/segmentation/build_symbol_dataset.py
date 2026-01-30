# src/segmentation/build_symbol_dataset.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import xml.etree.ElementTree as ET

from src.segmentation.inkml_parser import parse_inkml_symbols, SymbolSeg
from src.segmentation.render import RenderConfig, render_symbol


Point = Tuple[float, float]


def _inkml_ns(tag: str) -> str:
    # CROHME InkML uses default namespace: http://www.w3.org/2003/InkML
    return f"{{http://www.w3.org/2003/InkML}}{tag}"


def load_traces(inkml_path: str | Path) -> Dict[int, List[Point]]:
    """
    Return mapping: trace_id -> [(x,y), ...]
    """
    inkml_path = Path(inkml_path)
    tree = ET.parse(str(inkml_path))
    root = tree.getroot()

    traces: Dict[int, List[Point]] = {}

    for trace_el in root.findall(_inkml_ns("trace")):
        tid_str = trace_el.get("id")
        if tid_str is None:
            continue
        tid = int(tid_str)

        text = (trace_el.text or "").strip()
        if not text:
            traces[tid] = []
            continue

        # Format: "x y, x y, ..."
        pts: List[Point] = []
        chunks = [c.strip() for c in text.split(",") if c.strip()]
        for ch in chunks:
            parts = ch.split()
            if len(parts) < 2:
                continue
            x = float(parts[0])
            y = float(parts[1])
            pts.append((x, y))

        traces[tid] = pts

    return traces


def build_split(
    inkml_dir: Path,
    out_img_dir: Path,
    out_csv_path: Path,
    limit: int | None = None,
    cfg: RenderConfig | None = None,
) -> None:
    cfg = cfg or RenderConfig()
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    inkml_files = sorted(inkml_dir.glob("*.inkml"))
    if limit is not None:
        inkml_files = inkml_files[: int(limit)]

    rows = []
    written = 0
    skipped = 0

    for inkml_path in inkml_files:
        ui, syms = parse_inkml_symbols(str(inkml_path))
        if not syms:
            skipped += 1
            continue

        traces = load_traces(inkml_path)

        for k, seg in enumerate(syms):
            # gather points for trace ids
            pts_per_trace: List[List[Point]] = []
            ok = True
            for tid in seg.trace_ids:
                if tid not in traces:
                    ok = False
                    break
                pts_per_trace.append(traces[tid])

            if not ok:
                skipped += 1
                continue

            img = render_symbol(pts_per_trace, cfg)

            # filename is unique and traceable
            out_name = f"{inkml_path.stem}__sym{k:03d}.png"
            img.save(out_img_dir / out_name)

            rows.append(
                {
                    "filename": out_name,
                    "label": seg.symbol,
                    "ui": ui,
                    "inkml": inkml_path.name,
                    "trace_ids": " ".join(str(t) for t in seg.trace_ids),
                }
            )
            written += 1

    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "label", "ui", "inkml", "trace_ids"])
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] {inkml_dir}")
    print(f"  inkml_files: {len(inkml_files)}")
    print(f"  symbols_written: {written}")
    print(f"  skipped_items: {skipped}")
    print(f"  images_dir: {out_img_dir}")
    print(f"  labels_csv: {out_csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inkml_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--canvas", type=int, default=128)
    ap.add_argument("--stroke", type=int, default=3)
    args = ap.parse_args()

    inkml_dir = Path(args.inkml_dir)
    out_dir = Path(args.out_dir)

    out_img_dir = out_dir / f"symbols_{args.split}_images"
    out_csv = out_dir / f"symbols_{args.split}_labels.csv"

    cfg = RenderConfig(canvas_size=args.canvas, stroke_width=args.stroke)

    build_split(
        inkml_dir=inkml_dir,
        out_img_dir=out_img_dir,
        out_csv_path=out_csv,
        limit=args.limit,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()