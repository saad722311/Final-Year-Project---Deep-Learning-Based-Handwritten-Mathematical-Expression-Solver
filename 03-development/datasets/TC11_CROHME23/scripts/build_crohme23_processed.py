#!/usr/bin/env python3
"""
Create a MathWriting-style processed dataset for CROHME:

Output:
  processed/
    train_images/000000.png ...
    valid_images/000000.png ...
    test_images/000000.png ...
    train_labels.csv
    valid_labels.csv
    test_labels.csv

Filters out non-expression / metadata labels (e.g., 'Closest Strk').
Also removes rows with empty/invalid LaTeX.

CSV format:
  filename,label
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET


LATEX_CANDIDATE_TYPES = {
    "latex", "truth", "groundtruth", "ground_truth", "gt"
}

# Common junk / metadata labels seen in some CROHME subsets
BANNED_EXACT = {
    "closest strk",
    "closest stroke",
    "closeststrk",
    "segmentation",
    "symbol",
    "symbols",
    "relation",
    "relations",
}

BANNED_SUBSTRINGS = [
    "closest strk",
    "closest stroke",
]

# Minimal "math-like" tokens to accept as a real expression
MATH_TOKENS = set("\\{}^_[]=()+-/*<>|,.;:")  # includes backslash + structural chars


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def extract_latex_from_inkml(inkml_path: Path) -> str | None:
    """Extract LaTeX/GT text from InkML annotations."""
    try:
        tree = ET.parse(inkml_path)
    except ET.ParseError:
        return None

    root = tree.getroot()
    candidates: list[str] = []

    for elem in root.iter():
        if _strip_ns(elem.tag) != "annotation":
            continue
        ann_type = (elem.attrib.get("type") or "").strip().lower()
        text = (elem.text or "").strip()
        if not text:
            continue

        if ann_type in LATEX_CANDIDATE_TYPES:
            candidates.append(text)

    if candidates:
        return max(candidates, key=len)

    # fallback heuristic: looks like LaTeX/math
    for elem in root.iter():
        if _strip_ns(elem.tag) != "annotation":
            continue
        text = (elem.text or "").strip()
        if not text:
            continue
        if "\\" in text or "{" in text or "}" in text:
            return text

    return None


def normalize_latex(latex: str) -> str:
    latex = latex.replace("\u00a0", " ")
    latex = re.sub(r"\s+", " ", latex).strip()
    return latex


def is_valid_expression(label: str) -> bool:
    """
    Filter out non-expression labels like 'Closest Strk' and keep only
    math-like expressions.

    Rules:
      - remove exact/substring banned labels (case-insensitive)
      - require at least ONE "math token" OR a digit
      - reject very short plain words
    """
    if not label:
        return False

    s = label.strip()
    low = s.lower()

    if low in BANNED_EXACT:
        return False

    for sub in BANNED_SUBSTRINGS:
        if sub in low:
            return False

    # If label is purely alphabetic words/spaces, likely metadata
    if re.fullmatch(r"[A-Za-z ]+", s) is not None:
        return False

    # Require at least one math-ish token or a digit
    if not any(ch in MATH_TOKENS for ch in s) and not any(ch.isdigit() for ch in s):
        return False

    # Extra guard: extremely short strings that are not math
    if len(s) < 2:
        return False

    return True


def build_png_map(img_root: Path) -> dict[str, Path]:
    """Map: stem -> png path (first occurrence wins)."""
    m: dict[str, Path] = {}
    for p in img_root.rglob("*.png"):
        if p.stem not in m:
            m[p.stem] = p
    return m


def collect_pairs(img_root: Path, inkml_dirs: list[Path]) -> tuple[list[tuple[Path, str]], dict]:
    """
    Returns list of (png_path, label) pairs AFTER filtering invalid labels.
    """
    png_map = build_png_map(img_root)

    total_inkml = 0
    missing_png = 0
    missing_label = 0
    filtered_bad_label = 0
    pairs: list[tuple[Path, str]] = []

    for d in inkml_dirs:
        for inkml_path in sorted(d.rglob("*.inkml")):
            total_inkml += 1
            stem = inkml_path.stem

            label = extract_latex_from_inkml(inkml_path)
            if label is None:
                missing_label += 1
                continue

            label = normalize_latex(label)

            if not is_valid_expression(label):
                filtered_bad_label += 1
                continue

            png_path = png_map.get(stem)
            if png_path is None:
                missing_png += 1
                continue

            pairs.append((png_path, label))

    # Deterministic order helps reproducibility
    pairs.sort(key=lambda x: x[0].name)

    stats = {
        "total_inkml_scanned": total_inkml,
        "missing_png": missing_png,
        "missing_label": missing_label,
        "filtered_bad_label": filtered_bad_label,
        "pairs": len(pairs),
    }
    return pairs, stats


def write_processed_split(
    split_name: str,
    pairs: list[tuple[Path, str]],
    out_images_dir: Path,
    out_csv: Path,
    digits: int = 6,
    overwrite: bool = False,
) -> None:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        if any(out_images_dir.glob("*.png")) or out_csv.exists():
            raise FileExistsError(
                f"{split_name}: output already exists. Delete outputs or run with --overwrite."
            )

    if overwrite:
        if out_images_dir.exists():
            for p in out_images_dir.glob("*.png"):
                p.unlink()
        if out_csv.exists():
            out_csv.unlink()

    rows = []
    for i, (src_png, label) in enumerate(pairs):
        new_name = f"{i:0{digits}d}.png"
        dst_png = out_images_dir / new_name
        shutil.copy2(src_png, dst_png)
        rows.append((new_name, label))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(rows)


def latex_length_stats(pairs: list[tuple[Path, str]]) -> tuple[int, int, float]:
    if not pairs:
        return (0, 0, 0.0)
    lengths = [len(lbl) for _, lbl in pairs]
    return (len(pairs), max(lengths), sum(lengths) / len(lengths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="03-development/datasets/TC11_CROHME23")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite processed outputs if they exist")
    ap.add_argument("--digits", type=int, default=6, help="Zero-pad digits for filenames")
    ap.add_argument("--stats", action="store_true", help="Print split stats")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    img_root = dataset_root / "IMG"
    inkml_root = dataset_root / "INKML"
    processed = dataset_root / "processed"

    # TRAIN: use whatever exists in IMG/train (2013/2019/OffHME images).
    # InkML dirs: include those likely to match; any without PNG match are skipped.
    train_img = img_root / "train"
    train_inkml_dirs = [
        inkml_root / "train" / "CROHME2019",
        inkml_root / "train" / "CROHME2013_train",
        inkml_root / "train" / "OffHME",
        inkml_root / "train" / "CROHME2023_train",  # included but will only keep if PNG exists
        inkml_root / "train" / "Artificial_data",   # usually no PNGs; safe to include
    ]
    train_inkml_dirs = [d for d in train_inkml_dirs if d.exists()]

    # VALID: CROHME2023_val
    valid_img = img_root / "val" / "CROHME2023_val"
    valid_inkml_dirs = [inkml_root / "val" / "CROHME2023_val"]

    # TEST: CROHME2023_test
    test_img = img_root / "test" / "CROHME2023_test"
    test_inkml_dirs = [inkml_root / "test" / "CROHME2023_test"]

    jobs = [
        ("train", train_img, train_inkml_dirs, processed / "train_images", processed / "train_labels.csv"),
        ("valid", valid_img, valid_inkml_dirs, processed / "valid_images", processed / "valid_labels.csv"),
        ("test",  test_img,  test_inkml_dirs,  processed / "test_images",  processed / "test_labels.csv"),
    ]

    print(f"Dataset root : {dataset_root}")
    print(f"Processed    : {processed}")

    for name, img_dir, inkml_dirs, out_imgs, out_csv in jobs:
        if not img_dir.exists():
            raise FileNotFoundError(f"{name}: IMG dir not found: {img_dir}")
        for d in inkml_dirs:
            if not d.exists():
                raise FileNotFoundError(f"{name}: INKML dir not found: {d}")

        pairs, st = collect_pairs(img_dir, inkml_dirs)

        if args.stats:
            n, mx, avg = latex_length_stats(pairs)
            print(f"\n[{name}] pairs={st['pairs']} total_inkml_scanned={st['total_inkml_scanned']}")
            print(f"      missing_png={st['missing_png']} missing_label={st['missing_label']} filtered_bad_label={st['filtered_bad_label']}")
            print(f"      LaTeX length: max={mx} avg={avg:.2f}")

        write_processed_split(
            split_name=name,
            pairs=pairs,
            out_images_dir=out_imgs,
            out_csv=out_csv,
            digits=args.digits,
            overwrite=args.overwrite,
        )
        print(f"[{name}] wrote {out_imgs} and {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()