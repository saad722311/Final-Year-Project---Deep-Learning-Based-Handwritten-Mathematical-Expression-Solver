# src/utils/latex_norm.py
from __future__ import annotations

import re


def normalize_latex_label(s: str) -> str:
    """
    Normalize LaTeX labels consistently across train/valid/test.

    Goals:
    - Remove surrounding math mode ($...$, $$...$$)
    - Remove ALL remaining dollar signs ($) inside the string
    - Normalize whitespace
    - Keep internal LaTeX intact
    """

    if s is None:
        return ""

    s = s.strip()

    # Remove surrounding $$...$$
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()

    # Remove surrounding $...$
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    # 🔴 CRITICAL FIX: remove ALL remaining dollar signs
    # This handles dirty labels like:
    #   "$10,000 + $1,000 = $11,000"
    #   "\left| $\frac{a}{b} \right|"
    s = s.replace("$", "")

    # Normalize whitespace (collapse multiple spaces)
    s = re.sub(r"\s+", " ", s).strip()

    return s