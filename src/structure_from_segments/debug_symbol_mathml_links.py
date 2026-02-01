# src/structure_from_segments/debug_symbol_mathml_links.py
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from src.segmentation.inkml_parser import parse_inkml_symbols


INKML_NS = {"ink": "http://www.w3.org/2003/InkML"}


def _find_group_hrefs(inkml_path: Path) -> dict[str, str]:
    """
    Returns mapping: traceGroup xml:id -> href target (e.g. 'x_1', '2_1', 'sum_1')
    Only for symbol-level traceGroups that contain an <annotationXML href="..."/>.
    """
    tree = ET.parse(str(inkml_path))
    root = tree.getroot()

    group_to_href: dict[str, str] = {}

    # all traceGroups
    for tg in root.findall(".//ink:traceGroup", INKML_NS):
        gid = tg.get("{http://www.w3.org/XML/1998/namespace}id") or tg.get("xml:id") or tg.get("id")
        if not gid:
            continue

        annxml = tg.find("./ink:annotationXML", INKML_NS)
        if annxml is None:
            continue

        href = annxml.get("href")
        if href:
            # href values look like "x_1" (no leading '#') in CROHME files
            group_to_href[gid] = href.strip()

    return group_to_href


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inkml_dir", type=str, required=True)
    ap.add_argument("--limit", type=int, default=3, help="how many files to print")
    args = ap.parse_args()

    inkml_dir = Path(args.inkml_dir)
    files = sorted(inkml_dir.rglob("*.inkml"))
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        raise SystemExit(f"No .inkml found under {inkml_dir}")

    for p in files:
        ui, segs = parse_inkml_symbols(str(p))
        g2h = _find_group_hrefs(p)

        print("=" * 80)
        print("FILE:", p)
        print("UI  :", ui)
        print("segs:", len(segs), "| groups_with_href:", len(g2h))

        # show first ~30 segments (or all if small)
        show_n = min(30, len(segs))
        linked = 0

        for i, s in enumerate(segs[:show_n]):
            # Your SymbolSeg fields (you printed earlier): symbol, trace_ids, group_id
            sym = getattr(s, "symbol", None)
            tids = getattr(s, "trace_ids", None)
            gid = getattr(s, "group_id", None)

            href = None
            if gid is not None and str(gid) in g2h:
                href = g2h[str(gid)]
            if href:
                linked += 1

            print(f"[{i:02d}] group_id={gid} symbol={sym} trace_ids={tids} href={href}")

        print(f"[summary] shown={show_n} linked_in_shown={linked}")

        # check overall linkage rate
        total_linked = 0
        for s in segs:
            gid = getattr(s, "group_id", None)
            if gid is not None and str(gid) in g2h:
                total_linked += 1
        print(f"[summary] total_segs={len(segs)} linked_total={total_linked}")

    print("\nDone.")


if __name__ == "__main__":
    main()