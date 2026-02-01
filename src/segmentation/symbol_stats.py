# src/segmentation/symbol_stats.py
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Any

import matplotlib.pyplot as plt

from src.segmentation.inkml_parser import parse_inkml_symbols


def _iter_inkml_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.inkml")


def _get_label(seg: Any) -> str:
    """
    Your SymbolSeg class doesn't have .label.
    Try common field names safely.
    """
    # Most likely candidates first
    for attr in ("label", "truth", "symbol", "sym", "text", "class_name", "cls", "y", "target"):
        if hasattr(seg, attr):
            v = getattr(seg, attr)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s

    # If dataclass, inspect its fields
    if is_dataclass(seg):
        d = asdict(seg)
        for k in ("label", "truth", "symbol", "sym", "text", "class_name", "cls", "y", "target"):
            if k in d and d[k] is not None:
                s = str(d[k]).strip()
                if s:
                    return s
        # otherwise take the first string-like field
        for k, v in d.items():
            if isinstance(v, str) and v.strip():
                return v.strip()

    # Fallback: if it prints nicely (last resort)
    s = str(seg).strip()
    return s if s else "<EMPTY>"


def main():
    ap = argparse.ArgumentParser(description="Count oracle symbol classes from CROHME InkML traceGroups.")
    ap.add_argument("--inkml_dir", type=str, required=True, help="Root directory containing .inkml files (recursive).")
    ap.add_argument("--out_png", type=str, required=True, help="Path to save bar chart PNG.")
    ap.add_argument("--out_txt", type=str, default=None, help="Path to save full counts txt (default: same stem as png).")
    ap.add_argument("--topk", type=int, default=40, help="Show top-K classes in plot.")
    ap.add_argument("--include_other", action="store_true", help="Include 'OTHER' bucket for classes outside topk.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = no limit).")
    args = ap.parse_args()

    inkml_dir = Path(args.inkml_dir)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    out_txt = Path(args.out_txt) if args.out_txt else out_png.with_suffix(".txt")

    files = list(_iter_inkml_files(inkml_dir))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    counts = Counter()
    n_files = 0
    n_syms = 0
    n_empty = 0
    n_parse_fail = 0

    for p in files:
        try:
            ui, segs = parse_inkml_symbols(str(p))
        except Exception:
            n_parse_fail += 1
            continue

        n_files += 1
        if not segs:
            n_empty += 1
            continue

        for seg in segs:
            lab = _get_label(seg)
            if not lab:
                lab = "<EMPTY>"
            counts[lab] += 1
            n_syms += 1

    # Save full counts
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"inkml_dir: {inkml_dir}\n")
        f.write(f"files_seen: {len(files)}\n")
        f.write(f"files_parsed_ok: {n_files}\n")
        f.write(f"parse_fail: {n_parse_fail}\n")
        f.write(f"files_with_no_symbols: {n_empty}\n")
        f.write(f"total_symbols: {n_syms}\n")
        f.write(f"unique_classes: {len(counts)}\n\n")
        for k, v in counts.most_common():
            f.write(f"{k}\t{v}\n")

    # Plot
    items = counts.most_common(args.topk)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    if args.include_other:
        top_set = set(labels)
        other = sum(v for k, v in counts.items() if k not in top_set)
        if other > 0:
            labels.append("OTHER")
            values.append(other)

    plt.figure(figsize=(14, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[symbol-stats] files={len(files)} parsed_ok={n_files} parse_fail={n_parse_fail} empty={n_empty}")
    print(f"[symbol-stats] total_symbols={n_syms} unique_classes={len(counts)}")
    print(f"[symbol-stats] saved plot: {out_png}")
    print(f"[symbol-stats] saved counts: {out_txt}")


if __name__ == "__main__":
    main()