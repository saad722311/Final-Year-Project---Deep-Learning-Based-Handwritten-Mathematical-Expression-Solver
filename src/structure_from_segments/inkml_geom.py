from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Trace:
    tid: int
    points: List[Tuple[float, float]]  # (x, y)


@dataclass
class SymbolGeom:
    """
    Geometry for a single symbol segment.

    Fields are explicit (cx/cy/w/h) so downstream rule-based reconstruction
    and debug tools can work without guessing keys.
    """
    label: str
    trace_ids: List[int]

    # raw bbox
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)

    # derived geometry
    x0: float
    y0: float
    x1: float
    y1: float
    cx: float
    cy: float
    w: float
    h: float


# ----------------------------
# InkML parsing helpers
# ----------------------------

def _ns(tag: str, ns: str) -> str:
    return f"{{{ns}}}{tag}"


def load_traces(inkml_path: Path) -> Dict[int, Trace]:
    """Parse <trace id=".."> blocks into point lists."""
    tree = ET.parse(str(inkml_path))
    root = tree.getroot()
    ns = root.tag.split("}")[0].strip("{")

    traces: Dict[int, Trace] = {}
    for t in root.findall(_ns("trace", ns)):
        tid = int(t.attrib["id"])
        raw = (t.text or "").strip()

        pts: List[Tuple[float, float]] = []
        if raw:
            # Format: "x y, x y, ..."
            for chunk in raw.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                parts = chunk.split()
                if len(parts) < 2:
                    continue
                x_str, y_str = parts[0], parts[1]
                pts.append((float(x_str), float(y_str)))

        traces[tid] = Trace(tid=tid, points=pts)

    return traces


def bbox_of_trace_ids(
    traces: Dict[int, Trace],
    trace_ids: List[int],
) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []

    for tid in trace_ids:
        tr = traces.get(int(tid))
        if tr is None:
            continue
        for x, y in tr.points:
            xs.append(x)
            ys.append(y)

    if not xs:
        return (0.0, 0.0, 0.0, 0.0)

    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_to_geom(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Convert bbox -> (x0,y0,x1,y1,cx,cy,w,h)
    """
    x0, y0, x1, y1 = b
    w = float(x1 - x0)
    h = float(y1 - y0)
    cx = float((x0 + x1) / 2.0)
    cy = float((y0 + y1) / 2.0)
    return float(x0), float(y0), float(x1), float(y1), cx, cy, w, h


# ----------------------------
# Main builder
# ----------------------------

def build_symbol_geoms(inkml_path: Path, symbol_segs) -> List[SymbolGeom]:
    """
    symbol_segs: list[SymbolSeg] from src.segmentation.inkml_parser
    Must have .symbol and .trace_ids
    """
    traces = load_traces(inkml_path)

    out: List[SymbolGeom] = []
    for s in symbol_segs:
        b = bbox_of_trace_ids(traces, list(s.trace_ids))
        x0, y0, x1, y1, cx, cy, w, h = _bbox_to_geom(b)

        out.append(
            SymbolGeom(
                label=str(s.symbol),
                trace_ids=list(s.trace_ids),
                bbox=b,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
            )
        )

    return out