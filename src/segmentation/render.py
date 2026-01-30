# src/segmentation/render.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
from PIL import Image, ImageDraw


Point = Tuple[float, float]


@dataclass
class RenderConfig:
    canvas_size: int = 128        # output image size (square)
    stroke_width: int = 3
    padding: float = 2.0          # padding in "ink space" before scaling
    margin_px: int = 8            # pixel margin inside canvas
    bg: int = 255                 # white
    fg: int = 0                   # black


def _bbox(points: List[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def render_symbol(points_per_trace: List[List[Point]], cfg: RenderConfig) -> Image.Image:
    """
    Render one symbol (multiple traces) into a square PNG.
    """
    # Flatten all points for bbox
    flat: List[Point] = [p for tr in points_per_trace for p in tr]
    if len(flat) < 2:
        return Image.new("L", (cfg.canvas_size, cfg.canvas_size), color=cfg.bg)

    x0, y0, x1, y1 = _bbox(flat)
    # Expand bbox slightly (ink-space padding)
    x0 -= cfg.padding
    y0 -= cfg.padding
    x1 += cfg.padding
    y1 += cfg.padding

    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)

    # Fit into canvas with margin
    usable = cfg.canvas_size - 2 * cfg.margin_px
    scale = usable / max(w, h)

    def tx(p: Point) -> Tuple[float, float]:
        # translate to bbox origin then scale then add margin
        return (
            (p[0] - x0) * scale + cfg.margin_px,
            (p[1] - y0) * scale + cfg.margin_px,
        )

    img = Image.new("L", (cfg.canvas_size, cfg.canvas_size), color=cfg.bg)
    draw = ImageDraw.Draw(img)

    for tr in points_per_trace:
        if len(tr) < 2:
            continue
        pts = [tx(p) for p in tr]
        draw.line(pts, fill=cfg.fg, width=cfg.stroke_width, joint="curve")

    return img