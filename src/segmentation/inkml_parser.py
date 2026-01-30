# src/segmentation/inkml_parser.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET


@dataclass
class SymbolSeg:
    symbol: str
    trace_ids: List[int]
    group_id: str | None = None


def _ns(tag: str, ns: str) -> str:
    """Helper to build namespaced tags."""
    return f"{{{ns}}}{tag}"


def _get_inkml_namespace(root: ET.Element) -> str:
    """
    InkML uses a default namespace like:
      <ink xmlns="http://www.w3.org/2003/InkML">
    In ElementTree, tags become '{namespace}tag'.
    """
    if root.tag.startswith("{") and "}" in root.tag:
        return root.tag.split("}")[0].strip("{")
    return ""


def parse_inkml_symbols(inkml_path: str | Path) -> Tuple[str, List[SymbolSeg]]:
    """
    Returns:
      ui: string (from <annotation type="UI">)
      symbols: list of SymbolSeg(symbol, trace_ids)

    Works for CROHME InkML where symbol segmentation is represented as a traceGroup
    containing nested traceGroups, each with an annotation (truth) = symbol and traceView refs.
    """
    inkml_path = Path(inkml_path)
    tree = ET.parse(str(inkml_path))
    root = tree.getroot()

    ns = _get_inkml_namespace(root)
    # If namespace exists, use namespaced tags, else plain.
    def T(name: str) -> str:
        return _ns(name, ns) if ns else name

    # 1) UI
    ui = ""
    for ann in root.findall(T("annotation")):
        if ann.attrib.get("type") == "UI":
            ui = (ann.text or "").strip()
            break

    # 2) Find a "segmentation root" traceGroup.
    # CROHME sometimes labels it as "Segmentation" or "Closest Strk" etc.
    segmentation_keywords = {
        "segmentation",
        "closest strk",
        "closest stroke",
        "symbol segmentation",
        "segmentation result",
    }

    seg_root: Optional[ET.Element] = None

    # scan all traceGroups, find one whose *direct* annotation truth is a known segmentation label
    for tg in root.findall(f".//{T('traceGroup')}"):
        ann = tg.find(T("annotation"))
        if ann is None:
            continue
        if ann.attrib.get("type") != "truth":
            continue
        label = (ann.text or "").strip().lower()
        if label in segmentation_keywords:
            seg_root = tg
            break

    # If we didn't find via keyword, fallback: choose the deepest traceGroup that contains nested traceGroups
    # and has child traceGroups with symbol annotations.
    if seg_root is None:
        for tg in root.findall(f".//{T('traceGroup')}"):
            child_tgs = tg.findall(T("traceGroup"))
            if not child_tgs:
                continue
            # if any child traceGroup has an annotation truth that is 1–3 chars (likely symbol)
            for child in child_tgs:
                ann = child.find(T("annotation"))
                if ann is None or ann.attrib.get("type") != "truth":
                    continue
                sym = (ann.text or "").strip()
                if sym and len(sym) <= 5:  # heuristic: symbols are short
                    seg_root = tg
                    break
            if seg_root is not None:
                break

    if seg_root is None:
        return ui, []

    # 3) Extract symbols from children traceGroups of seg_root
    symbols: List[SymbolSeg] = []
    for sym_group in seg_root.findall(T("traceGroup")):
        ann = sym_group.find(T("annotation"))
        if ann is None or ann.attrib.get("type") != "truth":
            continue
        sym = (ann.text or "").strip()
        if not sym:
            continue

        trace_ids: List[int] = []
        for tv in sym_group.findall(T("traceView")):
            ref = tv.attrib.get("traceDataRef")
            if ref is None:
                continue
            try:
                trace_ids.append(int(ref))
            except ValueError:
                pass

        if trace_ids:
            symbols.append(
                SymbolSeg(
                    symbol=sym,
                    trace_ids=trace_ids,
                    group_id=sym_group.attrib.get("xml:id") or sym_group.attrib.get("id"),
                )
            )

    return ui, symbols