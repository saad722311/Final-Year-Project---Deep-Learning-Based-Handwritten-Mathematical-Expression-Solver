# src/structure/eval_oracle.py
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional, List, Tuple

from src.structure.inkml_truth_mathml import extract_truth_mathml_root, extract_truth_latex_string
from src.structure.mathml_to_ast import mathml_to_ast
from src.structure.ast_to_latex import ast_to_latex


def _fallback_normalize(s: str) -> str:
    """
    Normalization for oracle reconstruction evaluation:
    - remove whitespace
    - ^{2} -> ^2 and _{i} -> _i for single-char tokens
    - strip \\left/\\right (formatting-only)
    """
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s)

    # superscripts/subscripts single token: ^{2} -> ^2, _{i} -> _i
    s = re.sub(r"\^\{([A-Za-z0-9])\}", r"^\1", s)
    s = re.sub(r"_\{([A-Za-z0-9])\}", r"_\1", s)

    # remove \left and \right wrappers
    s = s.replace(r"\left", "").replace(r"\right", "")
    return s


def _normalize(s: str) -> str:
    """
    Prefer your project normalizer if present; fall back otherwise.
    """
    try:
        from src.utils.latex_norm import normalize_latex  # type: ignore

        # normalize_latex may already remove whitespace etc.
        out = normalize_latex(s)
        # still apply brace simplification to avoid ^{2} vs ^2 discrepancies
        out = (out or "").strip()
        out = re.sub(r"\s+", "", out)
        out = re.sub(r"\^\{([A-Za-z0-9])\}", r"^\1", out)
        out = re.sub(r"_\{([A-Za-z0-9])\}", r"_\1", out)
        out = out.replace(r"\left", "").replace(r"\right", "")
        return out
    except Exception:
        return _fallback_normalize(s)


def oracle_predict_latex(inkml_path: Path) -> Optional[str]:
    """
    Oracle reconstruction: MathML truth -> AST -> LaTeX.
    Returns None if MathML truth is missing.
    """
    math_root = extract_truth_mathml_root(inkml_path)
    if math_root is None:
        return None
    ast = mathml_to_ast(math_root)
    return (ast_to_latex(ast) or "").strip()


def _iter_inkml_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(root.rglob("*.inkml"))
    return sorted(root.glob("*.inkml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inkml_dir", type=str, required=True, help="Directory containing InkMLs")
    ap.add_argument("--out_csv", type=str, required=True, help="Where to save per-file results")
    ap.add_argument("--max_files", type=int, default=0, help="0 = no limit")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders for .inkml")
    ap.add_argument("--print_examples", type=int, default=10, help="How many mismatches to print")
    args = ap.parse_args()

    inkml_dir = Path(args.inkml_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = _iter_inkml_files(inkml_dir, recursive=bool(args.recursive))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    used = 0
    em = 0
    nem = 0
    missing_mathml = 0
    missing_truth = 0

    # store mismatches to print
    examples_bad: List[Tuple[str, str, str, str, str]] = []

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["inkml", "gt_truth", "oracle_pred", "EM", "nEM", "gt_norm", "pred_norm"])

        for p in files:
            gt = extract_truth_latex_string(p)
            if gt is None:
                missing_truth += 1
                continue

            pred = oracle_predict_latex(p)
            if pred is None:
                missing_mathml += 1
                continue

            used += 1

            is_em = int(pred == gt)
            gt_n = _normalize(gt)
            pred_n = _normalize(pred)
            is_nem = int(pred_n == gt_n)

            em += is_em
            nem += is_nem

            if is_nem == 0 and len(examples_bad) < int(args.print_examples):
                examples_bad.append((p.name, gt, pred, gt_n, pred_n))

            w.writerow([p.name, gt, pred, is_em, is_nem, gt_n, pred_n])

    em_rate = (em / used) if used else 0.0
    nem_rate = (nem / used) if used else 0.0

    print(f"[oracle] files_total={len(files)} used={used} missing_truth={missing_truth} missing_mathml={missing_mathml}")
    print(f"[oracle] EM={em_rate*100:.2f}% ({em}/{used}) | nEM={nem_rate*100:.2f}% ({nem}/{used})")
    print(f"[oracle] per-file results saved to: {out_csv}")

    if examples_bad:
        print(f"\n[oracle] sample nEM mismatches (first {len(examples_bad)}):")
        for name, gt, pred, gt_n, pred_n in examples_bad:
            print(f"- {name}\n  GT   : {gt}\n  PRED : {pred}\n  nGT  : {gt_n}\n  nPRED: {pred_n}\n")


if __name__ == "__main__":
    main()