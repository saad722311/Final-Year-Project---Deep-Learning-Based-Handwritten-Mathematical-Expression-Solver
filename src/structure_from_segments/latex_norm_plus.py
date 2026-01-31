from __future__ import annotations
import re

TOKEN_MAP = {
    "times": r"\times",
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "pi": r"\pi",
    "theta": r"\theta",
    "phi": r"\phi",
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "int": r"\int",
    "sum": r"\sum",
    "pm": r"\pm",
    "lt": r"\lt",
    "gt": r"\gt",
}

_SINGLE = r"(\\[A-Za-z]+|[A-Za-z0-9])"  # \alpha or a or 2

def normalize_latex_plus(s: str) -> str:
    s = s.strip()

    # collapse whitespace
    s = re.sub(r"\s+", " ", s)

    # map plain tokens to LaTeX (use lambda replacement to avoid backslash issues)
    for k, v in TOKEN_MAP.items():
        s = re.sub(rf"\b{k}\b", (lambda m, vv=v: vv), s)

    # remove spaces for comparison
    s = s.replace(" ", "")

    # normalize \sqrt2 -> \sqrt{2}
    s = re.sub(r"\\sqrt([A-Za-z0-9])", r"\\sqrt{\1}", s)

    # ✅ canonicalize braces for single-token scripts:
    # z^{2} -> z^2 ; x_{i} -> x_i ; \alpha^{2} -> \alpha^2
    s = re.sub(rf"\^\{{{_SINGLE}\}}", r"^\1", s)
    s = re.sub(rf"_\{{{_SINGLE}\}}", r"_\1", s)

    # optional: canonicalize double braces like {{10}^{-2}} -> {10}^{-2}
    s = re.sub(r"\{\{([^{}]+)\}\}", r"{\1}", s)

    return s