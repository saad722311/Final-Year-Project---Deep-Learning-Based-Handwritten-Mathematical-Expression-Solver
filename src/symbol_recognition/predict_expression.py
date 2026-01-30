from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.segmentation.inkml_parser import parse_inkml_symbols, SymbolSeg
from src.symbol_recognition.model import TinySymbolCNN


# -------------------------
# InkML utilities
# -------------------------
def _inkml_ns(tag: str) -> str:
    # CROHME InkML uses default namespace
    return f"{{http://www.w3.org/2003/InkML}}{tag}"


def read_truth_latex(inkml_path: str) -> str:
    """Read <annotation type="truth"> ... </annotation>."""
    tree = ET.parse(inkml_path)
    root = tree.getroot()
    for ann in root.findall(_inkml_ns("annotation")):
        if ann.attrib.get("type", "").lower() == "truth":
            if ann.text:
                return ann.text.strip()
    return ""


def read_traces_xy(inkml_path: str) -> dict[int, np.ndarray]:
    """Return {trace_id: array([[x,y],...])}"""
    tree = ET.parse(inkml_path)
    root = tree.getroot()
    traces = {}
    for tr in root.findall(_inkml_ns("trace")):
        tid = int(tr.attrib["id"])
        txt = (tr.text or "").strip()
        pts = []
        # points format: "x y, x y, ..."
        for chunk in txt.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split()
            if len(parts) >= 2:
                x = float(parts[0])
                y = float(parts[1])
                pts.append((x, y))
        if pts:
            traces[tid] = np.array(pts, dtype=np.float32)
    return traces


@dataclass
class SymBox:
    symbol_gt: str
    trace_ids: List[int]
    bbox: Tuple[float, float, float, float]  # minx, miny, maxx, maxy


def bbox_from_traces(traces_xy: dict[int, np.ndarray], trace_ids: List[int]) -> Tuple[float, float, float, float]:
    pts = []
    for tid in trace_ids:
        if tid in traces_xy:
            pts.append(traces_xy[tid])
    if not pts:
        return (0, 0, 0, 0)
    all_pts = np.concatenate(pts, axis=0)
    minx = float(all_pts[:, 0].min())
    miny = float(all_pts[:, 1].min())
    maxx = float(all_pts[:, 0].max())
    maxy = float(all_pts[:, 1].max())
    return (minx, miny, maxx, maxy)


def render_symbol_crop(
    traces_xy: dict[int, np.ndarray],
    trace_ids: List[int],
    out_size: int = 64,
    pad: int = 2,
    stroke_width: int = 2,
) -> Image.Image:
    """Render selected traces into a 64x64 grayscale image."""
    # get bbox in trace coordinate space
    minx, miny, maxx, maxy = bbox_from_traces(traces_xy, trace_ids)
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)

    # create a high-res canvas first
    scale = 8  # oversample
    canvas_w = int((w + 2 * pad) * scale)
    canvas_h = int((h + 2 * pad) * scale)
    canvas = Image.new("L", (max(canvas_w, 16), max(canvas_h, 16)), color=255)
    draw = ImageDraw.Draw(canvas)

    def tx(x): return (x - minx + pad) * scale
    def ty(y): return (y - miny + pad) * scale

    for tid in trace_ids:
        if tid not in traces_xy:
            continue
        pts = traces_xy[tid]
        xy = [(tx(x), ty(y)) for x, y in pts]
        if len(xy) >= 2:
            draw.line(xy, fill=0, width=stroke_width * scale, joint="curve")
        else:
            x0, y0 = xy[0]
            draw.ellipse((x0 - 2, y0 - 2, x0 + 2, y0 + 2), fill=0)

    # downsample to out_size
    img = canvas.resize((out_size, out_size), resample=Image.BILINEAR)
    return img


# -------------------------
# Model utilities
# -------------------------
def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0  # [0..1]
    arr = 1.0 - arr  # invert so ink = 1
    t = torch.from_numpy(arr)[None, ...]  # (1,H,W)
    return t


# -------------------------
# Simple structure heuristics
# -------------------------
def normalize_truth(s: str) -> str:
    s = s.strip()
    # remove surrounding $...$ if present
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def reconstruct_latex(symbols: List[Tuple[str, Tuple[float, float, float, float]]]) -> str:
    """
    Heuristic:
    - If a '-' exists and it is "wide", treat it as fraction bar.
    - Split symbols into numerator (above bar) and denominator (below bar) based on center_y.
    - Handle superscripts: if a symbol is above previous baseline and close in x, make ^{sym}.
    """
    if not symbols:
        return ""

    # sort by x center
    def cx(b): return 0.5 * (b[0] + b[2])
    def cy(b): return 0.5 * (b[1] + b[3])

    # detect fraction bar candidate: symbol '-' with large width
    bar_idx = None
    best_width = 0.0
    for i, (sym, b) in enumerate(symbols):
        if sym == "-":
            width = b[2] - b[0]
            if width > best_width:
                best_width = width
                bar_idx = i

    if bar_idx is not None and best_width > 10:  # threshold in trace space (works well for CROHME scale)
        bar_b = symbols[bar_idx][1]
        bar_y = cy(bar_b)

        num = []
        den = []
        others = []
        for i, (sym, b) in enumerate(symbols):
            if i == bar_idx:
                continue
            if cy(b) < bar_y:
                num.append((sym, b))
            else:
                den.append((sym, b))

        num = sorted(num, key=lambda x: cx(x[1]))
        den = sorted(den, key=lambda x: cx(x[1]))

        num_str = linear_with_superscripts(num)
        den_str = linear_with_superscripts(den)

        return f"\\frac{{{num_str}}}{{{den_str}}}"

    # otherwise just linear
    return linear_with_superscripts(sorted(symbols, key=lambda x: cx(x[1])))


def linear_with_superscripts(seq: List[Tuple[str, Tuple[float, float, float, float]]]) -> str:
    """
    Very simple superscript rule:
    if current symbol center_y is significantly above previous symbol center_y,
    treat as superscript of previous token.
    """
    if not seq:
        return ""

    def cx(b): return 0.5 * (b[0] + b[2])
    def cy(b): return 0.5 * (b[1] + b[3])
    def h(b): return (b[3] - b[1])

    out = []
    prev_b = None
    for sym, b in seq:
        if prev_b is not None:
            # superscript if above and x is close-ish
            y_up = cy(b) < (cy(prev_b) - 0.25 * h(prev_b))
            x_close = abs(cx(b) - cx(prev_b)) < (0.9 * h(prev_b) + 5.0)
            if y_up and x_close and len(out) > 0:
                out[-1] = f"{out[-1]}^{{{sym}}}"
                prev_b = b
                continue

        out.append(sym)
        prev_b = b

    return " ".join(out)


# -------------------------
# Main
# -------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inkml", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_size", type=int, default=64)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    image_size = int(ckpt.get("image_size", args.out_size))

    model = TinySymbolCNN(num_classes=len(stoi))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # InkML -> segmentation
    ui, segs = parse_inkml_symbols(args.inkml)
    truth = normalize_truth(read_truth_latex(args.inkml))
    traces_xy = read_traces_xy(args.inkml)

    # build bboxes + predict each segment
    pred_symbols = []
    for s in segs:
        b = bbox_from_traces(traces_xy, s.trace_ids)
        img = render_symbol_crop(traces_xy, s.trace_ids, out_size=image_size)

        x = pil_to_tensor(img).unsqueeze(0).to(device)  # (1,1,H,W)
        logits = model(x)
        pred_id = int(logits.argmax(dim=1).item())
        sym_pred = itos[pred_id]

        pred_symbols.append((sym_pred, b))

    pred_latex = reconstruct_latex(pred_symbols)
    pred_latex_norm = normalize_truth(pred_latex)

    em = 1 if pred_latex_norm == truth else 0

    print(f"UI:   {ui}")
    print(f"GT:   {truth}")
    print(f"PRED: {pred_latex_norm}")
    print(f"EM:   {em}")


if __name__ == "__main__":
    main()