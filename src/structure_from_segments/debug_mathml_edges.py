from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from src.structure_from_segments.mathml_edges import (
    find_content_mathml_root,
    extract_edges_from_mathml,
)
from src.segmentation.inkml_parser import parse_inkml_symbols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inkml_dir", type=str, required=True)
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.inkml_dir)
    files = sorted(root.rglob("*.inkml"))[: args.limit]

    for p in files:
        print("=" * 80)
        print("FILE:", p)
        ui, segs = parse_inkml_symbols(str(p))
        print("UI:", ui, "n_segs:", len(segs))

        tree = ET.parse(str(p))
        ink_root = tree.getroot()

        math = find_content_mathml_root(ink_root)
        if math is None:
            print("[warn] no Content-MathML found")
            continue

        token_ids, edges, vnodes = extract_edges_from_mathml(math)
        print("mathml_token_ids:", len(token_ids))
        print("virtual_nodes:", len(vnodes))
        print("edges:", len(edges))

        if vnodes:
            print("Virtual nodes:")
            for vn in vnodes[:20]:
                print(f"  {vn.node_id} ({vn.kind})")

        for e in edges[:80]:
            print(f"  {e.src} --{e.rel}--> {e.dst}")

    print("\nDone.")


if __name__ == "__main__":
    main()