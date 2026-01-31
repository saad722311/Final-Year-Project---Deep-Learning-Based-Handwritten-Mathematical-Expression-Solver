from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.segmentation.inkml_parser import parse_inkml_symbols
from src.structure.inkml_truth_mathml import extract_truth_latex_string
from src.structure_from_segments.inkml_geom import build_symbol_geoms
from src.structure_from_segments.reconstruct_rules import reconstruct_expression, node_to_latex
from src.structure_from_segments.latex_norm_plus import normalize_latex_plus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inkml_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--max_files", type=int, default=0)
    args = ap.parse_args()

    inkml_dir = Path(args.inkml_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(inkml_dir.glob("*.inkml"))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    used = 0
    em = 0
    nem = 0
    missing_truth = 0
    empty_segments = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["inkml", "gt", "pred", "EM", "nEM"])

        for p in files:
            gt = extract_truth_latex_string(p)
            if gt is None:
                missing_truth += 1
                continue

            ui, segs = parse_inkml_symbols(str(p))
            if not segs:
                empty_segments += 1
                continue

            geoms = build_symbol_geoms(p, segs)
            ast = reconstruct_expression(geoms)
            pred = node_to_latex(ast)

            used += 1
            is_em = int(pred == gt)
            is_nem = int(normalize_latex_plus(pred) == normalize_latex_plus(gt))
            em += is_em
            nem += is_nem

            w.writerow([p.name, gt, pred, is_em, is_nem])

    em_rate = (em / used) if used else 0.0
    nem_rate = (nem / used) if used else 0.0

    print(f"[oracle-seg] files_total={len(files)} used={used} missing_truth={missing_truth} empty_segments={empty_segments}")
    print(f"[oracle-seg] EM={em_rate*100:.2f}% ({em}/{used}) | nEM={nem_rate*100:.2f}% ({nem}/{used})")
    print(f"[oracle-seg] results saved to: {out_csv}")


if __name__ == "__main__":
    main()