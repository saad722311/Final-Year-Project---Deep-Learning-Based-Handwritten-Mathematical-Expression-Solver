from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image, ImageDraw


def _inkml_ns(tag: str) -> str:
    return f"{{http://www.w3.org/2003/InkML}}{tag}"


def read_truth_latex(inkml_path: str) -> str:
    tree = ET.parse(inkml_path)
    root = tree.getroot()
    for ann in root.findall(_inkml_ns("annotation")):
        if ann.attrib.get("type", "").lower() == "truth":
            return (ann.text or "").strip()
    return ""


def read_traces_xy(inkml_path: str) -> Dict[int, np.ndarray]:
    tree = ET.parse(inkml_path)
    root = tree.getroot()
    traces: Dict[int, np.ndarray] = {}
    for tr in root.findall(_inkml_ns("trace")):
        tid = int(tr.attrib["id"])
        txt = (tr.text or "").strip()
        pts = []
        for chunk in txt.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split()
            if len(parts) >= 2:
                pts.append((float(parts[0]), float(parts[1])))
        if pts:
            traces[tid] = np.array(pts, dtype=np.float32)
    return traces


def _bbox_all_traces(traces: Dict[int, np.ndarray]) -> Tuple[float, float, float, float]:
    all_pts = np.concatenate(list(traces.values()), axis=0)
    minx = float(all_pts[:, 0].min())
    miny = float(all_pts[:, 1].min())
    maxx = float(all_pts[:, 0].max())
    maxy = float(all_pts[:, 1].max())
    return minx, miny, maxx, maxy


def render_expression_image(
    inkml_path: str,
    out_hw: Tuple[int, int] = (256, 256),
    pad: int = 10,
    stroke_width: int = 3,
    invert: bool = True,
) -> Image.Image:
    """
    Render ALL traces in the InkML into a fixed-size grayscale image.
    Returns PIL Image (L).
    """
    traces = read_traces_xy(inkml_path)
    if not traces:
        return Image.new("L", out_hw, color=255)

    minx, miny, maxx, maxy = _bbox_all_traces(traces)
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)

    # Make a padded coordinate space then scale into out_hw
    W, H = out_hw
    sx = (W - 2 * pad) / w
    sy = (H - 2 * pad) / h
    s = min(sx, sy)

    def tx(x): return pad + (x - minx) * s
    def ty(y): return pad + (y - miny) * s

    img = Image.new("L", out_hw, color=255)
    draw = ImageDraw.Draw(img)

    for pts in traces.values():
        xy = [(tx(x), ty(y)) for x, y in pts]
        if len(xy) >= 2:
            draw.line(xy, fill=0, width=stroke_width, joint="curve")
        else:
            x0, y0 = xy[0]
            draw.ellipse((x0 - 1, y0 - 1, x0 + 1, y0 + 1), fill=0)

    if invert:
        arr = 255 - np.array(img, dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")

    return img