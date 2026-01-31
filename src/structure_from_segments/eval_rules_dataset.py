from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

from src.segmentation.inkml_parser import parse_inkml_symbols
from src.structure_from_segments.inkml_geom import build_symbol_geoms
from src.structure_from_segments.reconstruct_rules import reconstruct_expression, node_to_latex
from src.structure.inkml_truth_mathml import extract_truth_latex_string


# -------------------------
# Normalization (for nEM)
# -------------------------
_WS = re.compile(r"\s+")
def normalize_latex(s: str) -> str:
    """
    Light-weight normalization for comparing LaTeX strings.
    - remove whitespace
    - collapse redundant outer brackets for common CROHME formatting quirks
    - keep semantics mostly intact
    """
    if s is None:
        return ""
    s = str(s)

    # remove whitespace
    s = _WS.sub("", s)

    # common CROHME wrappers: [ ... ]
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        s = s[1:-1]

    # normalize \frac {a}{b} -> \frac{a}{b}
    s = s.replace(r"\frac{", r"\frac{")  # no-op but keeps clarity
    s = s.replace(r"\frac{", r"\frac{")
    s = s.replace(r"\frac{", r"\frac{")
    s = s.replace(r"\frac{", r"\frac{")
    s = s.replace(r"\frac{", r"\frac{")
    s = s.replace(r"\frac{", r"\frac{")
    s = s.replace(r"\frac{", r"\frac{")
    s = s.replace(r"\frac{", r"\frac{")
    s = s.replace(r"\frac{", r"\frac{")

    # but actually handle spaced form:
    s = s.replace(r"\frac{", r"\frac{")
    s = s.replace(r"\frac{", r"\frac{")
    s = s.replace(r"\frac{", r"\frac{")

    # robust: remove spaces already removed, now fix "\frac{...}" variants like "\frac{"
    s = s.replace(r"\frac{", r"\frac{")
    s = s.replace(r"\frac{", r"\frac{")

    # normalize \frac{a}{ b} etc. already handled by whitespace removal

    return s


def classify_failure(gt: str, pred: str) -> str:
    """Simple failure buckets for analysis."""
    gt_s = str(gt)
    pred_s = str(pred)

    if r"\frac" in gt_s:
        return "contains_frac"
    if r"\int" in gt_s or r"\sum" in gt_s:
        return "contains_bigop"
    if "^" in gt_s or "_" in gt_s:
        return "contains_scripts"
    if r"\sqrt" in gt_s:
        return "contains_sqrt"
    return "other"


def safe_truth(path: Path) -> str:
    try:
        return extract_truth_latex_string(path)
    except Exception as e:
        return f"__TRUTH_ERROR__:{type(e).__name__}"


def safe_pred(path: Path) -> Tuple[str, str]:
    """
    Returns (ui, pred_latex) or ("__PARSE_ERROR__", "...").
    """
    try:
        ui, segs = parse_inkml_symbols(str(path))
        geoms = build_symbol_geoms(path, segs)
        ast = reconstruct_expression(geoms)
        pred = node_to_latex(ast)
        return ui, pred
    except Exception as e:
        return "__PRED_ERROR__", f"__PRED_ERROR__:{type(e).__name__}"


def eval_dir(inkml_dir: Path, limit: int = 0) -> pd.DataFrame:
    inkml_paths = sorted(inkml_dir.glob("*.inkml"))
    if limit and limit > 0:
        inkml_paths = inkml_paths[:limit]

    rows: List[Dict] = []
    for p in inkml_paths:
        gt = safe_truth(p)
        ui, pred = safe_pred(p)

        gt_n = normalize_latex(gt)
        pred_n = normalize_latex(pred)

        em = int(pred == gt)
        nem = int(pred_n == gt_n)

        fail_type = ""
        if nem == 0:
            fail_type = classify_failure(gt, pred)

        rows.append(
            {
                "inkml": p.name,
                "ui": ui,
                "gt": gt,
                "pred": pred,
                "gt_norm": gt_n,
                "pred_norm": pred_n,
                "em": em,
                "nem": nem,
                "fail_type": fail_type,
            }
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inkml_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    inkml_dir = Path(args.inkml_dir)
    out_csv = Path(args.out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = eval_dir(inkml_dir, limit=args.limit)

    em = df["em"].mean() if len(df) else 0.0
    nem = df["nem"].mean() if len(df) else 0.0

    print(f"[rules-eval] files_total={len(df)}")
    print(f"[rules-eval] EM={em*100:.2f}% | nEM={nem*100:.2f}%")
    print(f"[rules-eval] saved: {out_csv}")

    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()