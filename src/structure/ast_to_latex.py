# src/structure/ast_to_latex.py
from __future__ import annotations
from typing import List

from src.structure.ast import (
    Node, Row, Mi, Mn, Mo, MText,
    Frac, Sqrt, Sup, Sub, SubSup, Fenced
)

# Minimal operator mapping (extend as you see mismatches)
OP_MAP = {
    "−": "-",   # unicode minus
    "×": r"\times",
    "·": r"\cdot",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "∞": r"\infty",
}

def _needs_braces(node: Node) -> bool:
    # when used as base/exp/sub, wrap if complex
    return isinstance(node, (Row, Frac, Sqrt, Sup, Sub, SubSup, Fenced))

def _wrap(node: Node) -> str:
    s = ast_to_latex(node)
    return "{" + s + "}" if _needs_braces(node) else s

def _join_row(parts: List[str]) -> str:
    # Join with spaces where needed to avoid merging tokens
    # (Simple rule: join with space always; safe for LaTeX tokens)
    return " ".join([p for p in parts if p != ""]).strip()

def ast_to_latex(node: Node) -> str:
    if isinstance(node, Row):
        return _join_row([ast_to_latex(x) for x in node.items])

    if isinstance(node, Mi):
        # In CROHME MathML, identifiers may already be like "a" or "z"
        # Greek typically appears as '\alpha' in the plain truth, but in MathML it might be 'α' or 'alpha'.
        # Keep as-is; you can add mapping later if needed.
        return node.text

    if isinstance(node, Mn):
        return node.text

    if isinstance(node, Mo):
        t = node.text
        t = OP_MAP.get(t, t)
        # If it's a single char operator, keep it
        return t

    if isinstance(node, MText):
        return node.text

    if isinstance(node, Frac):
        return r"\frac{" + ast_to_latex(node.num) + "}{" + ast_to_latex(node.den) + "}"

    if isinstance(node, Sqrt):
        return r"\sqrt{" + ast_to_latex(node.body) + "}"

    if isinstance(node, Sup):
        return _wrap(node.base) + "^" + "{" + ast_to_latex(node.exp) + "}"

    if isinstance(node, Sub):
        return _wrap(node.base) + "_" + "{" + ast_to_latex(node.sub) + "}"

    if isinstance(node, SubSup):
        return _wrap(node.base) + "_" + "{" + ast_to_latex(node.sub) + "}" + "^" + "{" + ast_to_latex(node.sup) + "}"

    if isinstance(node, Fenced):
        return node.open + " " + ast_to_latex(node.body) + " " + node.close

    raise TypeError(f"Unknown node: {type(node)}")