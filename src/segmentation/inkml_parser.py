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
    href: str | None = None   # NEW: link to MathML token id (e.g., "x_1", "cos_2")


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


def _get_xml_id(elem: ET.Element) -> Optional[str]:
    """
    Read xml:id correctly (ElementTree stores it under the XML namespace).
    """
    # xml namespace for xml:id
    xml_ns = "http://www.w3.org/XML/1998/namespace"
    if f"{{{xml_ns}}}id" in elem.attrib:
        return elem.attrib.get(f"{{{xml_ns}}}id")
    # sometimes datasets also store as plain "xml:id" or "id"
    return elem.attrib.get("xml:id") or elem.attrib.get("id")


def parse_inkml_symbols(inkml_path: str | Path) -> Tuple[str, List[SymbolSeg]]:
    """
    Returns:
      ui: string (from <annotation type="UI">)
      symbols: list of SymbolSeg(symbol, trace_ids, group_id, href)

    CROHME InkML:
      - symbols live in nested traceGroups
      - each symbol traceGroup has:
          <annotation type="truth">SYMBOL</annotation>
          <traceView traceDataRef="..."/>
          <annotationXML href="TOKEN_ID"/>   <-- links to MathML token (important!)
    """
    inkml_path = Path(inkml_path)
    tree = ET.parse(str(inkml_path))
    root = tree.getroot()

    ns = _get_inkml_namespace(root)

    def T(name: str) -> str:
        return _ns(name, ns) if ns else name

    # 1) UI
    ui = ""
    for ann in root.findall(T("annotation")):
        if ann.attrib.get("type") == "UI":
            ui = (ann.text or "").strip()
            break

    # 2) Find segmentation root traceGroup
    segmentation_keywords = {
        "segmentation",
        "closest strk",
        "closest stroke",
        "symbol segmentation",
        "segmentation result",
    }

    seg_root: Optional[ET.Element] = None

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

    # fallback: choose a traceGroup that contains child traceGroups with short truth labels
    if seg_root is None:
        for tg in root.findall(f".//{T('traceGroup')}"):
            child_tgs = tg.findall(T("traceGroup"))
            if not child_tgs:
                continue
            for child in child_tgs:
                ann = child.find(T("annotation"))
                if ann is None or ann.attrib.get("type") != "truth":
                    continue
                sym = (ann.text or "").strip()
                if sym and len(sym) <= 8:
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

        # NEW: grab MathML link from annotationXML href
        href = None
        ax = sym_group.find(T("annotationXML"))
        if ax is not None:
            href = ax.attrib.get("href")
            if href:
                href = href.strip()
                # some files may prefix href with '#'
                if href.startswith("#"):
                    href = href[1:]

        if trace_ids:
            symbols.append(
                SymbolSeg(
                    symbol=sym,
                    trace_ids=trace_ids,
                    group_id=_get_xml_id(sym_group),
                    href=href,
                )
            )

    return ui, symbols