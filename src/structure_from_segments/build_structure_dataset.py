from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

from src.segmentation.inkml_parser import parse_inkml_symbols, SymbolSeg
from src.structure_from_segments.inkml_geom import build_symbol_geoms
from src.structure_from_segments.mathml_edges import (
    find_content_mathml_root,
    extract_edges_from_mathml,
)


def _iter_inkml_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.inkml"))


def _safe_get(obj: Any, key: str, default=None):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _geom_to_dict(g: Any) -> Dict[str, Any]:
    """
    SymbolGeom is likely a dataclass. Convert safely.
    """
    if is_dataclass(g):
        d = asdict(g)
    else:
        d = {
            "label": _safe_get(g, "label"),
            "cx": _safe_get(g, "cx"),
            "cy": _safe_get(g, "cy"),
            "bbox": _safe_get(g, "bbox"),
            "trace_ids": _safe_get(g, "trace_ids"),
            "href": _safe_get(g, "href"),
            "group_id": _safe_get(g, "group_id"),
        }
    return d


def _build_href_map(segs: List[SymbolSeg]) -> Dict[str, int]:
    """
    Map MathML token id (href like cos_1) -> node index in our oracle nodes list.
    """
    href_map: Dict[str, int] = {}
    for i, s in enumerate(segs):
        href = getattr(s, "href", None)
        if href:
            href_map[href] = i
    return href_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inkml_dir", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    inkml_dir = Path(args.inkml_dir)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    files = _iter_inkml_files(inkml_dir)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    n_total = 0
    n_kept = 0
    n_skipped_no_mathml = 0
    n_skipped_no_links = 0

    # coverage stats
    edge_total = 0
    edge_kept = 0
    edge_dropped = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in files:
            n_total += 1

            # ---- oracle symbols (with href, after your parser update) ----
            try:
                ui, segs = parse_inkml_symbols(str(p))
            except Exception:
                continue

            # ---- geometry for oracle symbols ----
            # build_symbol_geoms expects: (inkml_path, segs)
            try:
                geoms = build_symbol_geoms(p, segs)
            except Exception:
                geoms = []

            # ---- mathml edges + virtual nodes ----
            try:
                tree = ET.parse(str(p))
                ink_root = tree.getroot()
                math = find_content_mathml_root(ink_root)
            except Exception:
                math = None

            if math is None:
                n_skipped_no_mathml += 1
                continue

            token_ids, edges, vnodes = extract_edges_from_mathml(math)

            # If we can’t link mathml token ids to symbol nodes, skip (for now)
            href_map = _build_href_map(segs)
            if not href_map:
                n_skipped_no_links += 1
                continue

            # Build node list: first oracle symbols, then virtual nodes
            nodes: List[Dict[str, Any]] = []
            for g in geoms:
                nodes.append(_geom_to_dict(g))

            base_n = len(nodes)

            for vn in vnodes:
                nodes.append({
                    "label": vn.kind,
                    "cx": None, "cy": None, "bbox": None,
                    "href": vn.node_id,
                    "is_virtual": True,
                })

            # virtual node id -> index
            vmap = {vn.node_id: base_n + i for i, vn in enumerate(vnodes)}

            # Convert edges to (src_idx, dst_idx, rel) and drop un-linkable edges
            edge_rows = []
            for e in edges:
                edge_total += 1

                src_idx = None
                dst_idx = None

                # src may be a virtual node or a mathml token id
                if e.src in vmap:
                    src_idx = vmap[e.src]
                elif e.src in href_map:
                    src_idx = href_map[e.src]

                # dst is always a mathml token id (for our current relations)
                if e.dst in href_map:
                    dst_idx = href_map[e.dst]

                if src_idx is None or dst_idx is None:
                    edge_dropped += 1
                    continue

                edge_kept += 1
                edge_rows.append([src_idx, dst_idx, e.rel])

            # if no edges survive, skip
            if not edge_rows:
                continue

            item = {
                "ui": ui,
                "inkml_path": str(p),
                "n_nodes": len(nodes),
                "n_edges": len(edge_rows),
                "nodes": nodes,
                "edges": edge_rows,
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"[build-struct] files_total={n_total} kept={n_kept}")
    print(f"[build-struct] skipped_no_mathml={n_skipped_no_mathml} skipped_no_links={n_skipped_no_links}")
    print(f"[build-struct] edges_total={edge_total} kept={edge_kept} dropped={edge_dropped}")
    if edge_total > 0:
        print(f"[build-struct] edge_coverage={100.0 * edge_kept / edge_total:.2f}%")
    print(f"[build-struct] saved: {out_jsonl}")


if __name__ == "__main__":
    main()