# src/structure_from_segments/debug_dump.py
from __future__ import annotations

import argparse

from dataclasses import asdict, is_dataclass
from pathlib import Path

from src.segmentation.inkml_parser import parse_inkml_symbols
from src.structure_from_segments.inkml_geom import build_symbol_geoms


def _pick(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inkml", type=str, required=True)
    args = ap.parse_args()
    p = Path(args.inkml)

    ui, segs = parse_inkml_symbols(str(p))
    geoms = build_symbol_geoms(p, segs)

    print("UI:", ui)
    print("n_symbols:", len(geoms))
    print("-----")

    for i, g in enumerate(geoms):
        # Convert dataclass -> dict, fallback to __dict__
        if is_dataclass(g):
            d = asdict(g)
        else:
            d = dict(getattr(g, "__dict__", {}))

        # Try common names used across codebases
        label = _pick(d, ["label", "token", "sym", "symbol", "truth", "pred", "name"], "?")

        # bbox / center fields
        cx = _pick(d, ["cx", "x_center", "xc", "center_x"], None)
        cy = _pick(d, ["cy", "y_center", "yc", "center_y"], None)
        w = _pick(d, ["w", "width"], None)
        h = _pick(d, ["h", "height"], None)

        # sometimes bbox is stored as (x0,y0,x1,y1)
        x0 = _pick(d, ["x0", "xmin", "left"], None)
        y0 = _pick(d, ["y0", "ymin", "top"], None)
        x1 = _pick(d, ["x1", "xmax", "right"], None)
        y1 = _pick(d, ["y1", "ymax", "bottom"], None)

        if cx is None and x0 is not None and x1 is not None:
            cx = (float(x0) + float(x1)) / 2
        if cy is None and y0 is not None and y1 is not None:
            cy = (float(y0) + float(y1)) / 2
        if w is None and x0 is not None and x1 is not None:
            w = float(x1) - float(x0)
        if h is None and y0 is not None and y1 is not None:
            h = float(y1) - float(y0)

        # Print a compact line + then full dict keys once (first item) if needed
        print(f"[{i:03d}] label={label} cx={cx} cy={cy} w={w} h={h}")

        # If you want to see EVERYTHING for the first few:
        if i < 3:
            print("    keys:", sorted(d.keys()))
            # print("    full:", d)  # uncomment if needed


if __name__ == "__main__":
    main()