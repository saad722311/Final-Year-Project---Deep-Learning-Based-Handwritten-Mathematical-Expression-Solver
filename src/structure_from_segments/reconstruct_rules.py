# src/structure_from_segments/reconstruct_rules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from src.structure_from_segments.inkml_geom import SymbolGeom


# =========================
# AST nodes + constructors
# =========================
@dataclass
class Node:
    kind: str  # "seq", "frac", "sup", "sub", "sqrt", "tok"
    value: str | None = None
    children: List["Node"] | None = None


def tok(v: str) -> Node:
    return Node(kind="tok", value=v, children=[])


def seq(nodes: List[Node]) -> Node:
    return Node(kind="seq", children=nodes)


def frac(num: Node, den: Node) -> Node:
    return Node(kind="frac", children=[num, den])


def sup(base: Node, exp: Node) -> Node:
    return Node(kind="sup", children=[base, exp])


def sub(base: Node, subn: Node) -> Node:
    return Node(kind="sub", children=[base, subn])


def sqrt(inner: Node) -> Node:
    return Node(kind="sqrt", children=[inner])


# =========================
# Geometry helpers
# =========================
def _bbox_w_h(sym: SymbolGeom) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = sym.bbox
    return (max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6))


def _median_height(syms: List[SymbolGeom]) -> float:
    hs: List[float] = []
    for s in syms:
        _, h = _bbox_w_h(s)
        hs.append(h)
    hs.sort()
    return hs[len(hs) // 2] if hs else 1.0


def _is_bar_like(sym: SymbolGeom, H: float, n_total: int) -> bool:
    w, h = _bbox_w_h(sym)
    ratio = w / (h + 1e-6)

    if n_total <= 10:
        ratio_min = 2.6
        thin_max = 1.25 * H
        w_min = 3.5
    else:
        ratio_min = 6.5
        thin_max = 0.75 * H
        w_min = 7.0

    if sym.label.strip() == "-":
        ratio_min *= 0.85

    return (w >= w_min) and (h <= thin_max) and (ratio >= ratio_min)


def _is_tiny_minus(sym: SymbolGeom, H: float) -> bool:
    """
    Tiny/thin '-' often used as unary minus inside an exponent (e^{-\alpha k^2}).
    """
    if sym.label.strip() != "-":
        return False
    w, h = _bbox_w_h(sym)
    return (h <= 0.20 * H) and (w <= 1.50 * H)


def _guess_y_up(syms: List[SymbolGeom]) -> bool:
    """
    Infer whether smaller y is visually "higher".
    Uses \int with 0 and \infty if available. Defaults to True (CROHME typical).
    """
    y_up = True
    ints = [s for s in syms if s.label == r"\int"]
    if not ints:
        return y_up

    it = ints[0]
    it_w, _ = _bbox_w_h(it)
    x0 = it.cx - 0.4 * it_w
    x1 = it.cx + 1.8 * it_w

    zeros = [s for s in syms if s.label == "0" and x0 <= s.cx <= x1]
    infs = [s for s in syms if s.label == r"\infty" and x0 <= s.cx <= x1]
    if zeros and infs:
        z = min(zeros, key=lambda s: abs(s.cx - it.cx))
        f = min(infs, key=lambda s: abs(s.cx - it.cx))
        y_up = (f.cy < z.cy)
    return y_up


def _y_is_above(y_up: bool, y_candidate: float, y_base: float, thresh: float) -> bool:
    return (y_candidate < y_base - thresh) if y_up else (y_candidate > y_base + thresh)


def _y_is_below(y_up: bool, y_candidate: float, y_base: float, thresh: float) -> bool:
    return (y_candidate > y_base + thresh) if y_up else (y_candidate < y_base - thresh)


# =========================
# Fraction helpers
# =========================
def _split_by_bar_local(
    syms: List[SymbolGeom], bar: SymbolGeom
) -> Tuple[List[SymbolGeom], List[SymbolGeom], List[SymbolGeom]]:
    H = _median_height(syms)
    bx0, _, bx1, _ = bar.bbox
    by = bar.cy

    pad = 0.25 * (bx1 - bx0)
    x0 = bx0 - pad
    x1 = bx1 + pad
    band = 0.12 * H

    in_window = [
        s for s in syms
        if s is not bar and not (s.bbox[2] < x0 or s.bbox[0] > x1)
    ]

    num = [s for s in in_window if s.cy < (by - band)]
    den = [s for s in in_window if s.cy > (by + band)]
    mid = [s for s in in_window if abs(s.cy - by) <= band]
    return num, den, mid


def _pick_fraction_bar(syms: List[SymbolGeom]) -> Optional[SymbolGeom]:
    if not syms:
        return None

    H = _median_height(syms)
    n_total = len(syms)

    cands = [s for s in syms if _is_bar_like(s, H, n_total)]

    if not cands:
        for s in syms:
            if s.label.strip() != "-":
                continue
            w, h = _bbox_w_h(s)
            if w < 2.0 * H:
                continue
            if (w / (h + 1e-6)) < 2.3:
                continue
            num, den, _ = _split_by_bar_local(syms, s)
            if num and den:
                cands.append(s)

    if not cands:
        return None

    best = None
    best_score = -1.0

    for b in cands:
        w, h = _bbox_w_h(b)
        num, den, _ = _split_by_bar_local(syms, b)
        if not num or not den:
            continue
        ratio = w / (h + 1e-6)
        label_bonus = 1.25 if b.label.strip() == "-" else 1.0
        score = label_bonus * (ratio * w) * (1.0 + 0.15 * (len(num) + len(den)))
        if score > best_score:
            best_score = score
            best = b

    return best


# =========================
# Grouping helpers
# =========================
def _flatten_seq(n: Node) -> List[Node]:
    if n.kind == "seq":
        return list(n.children or [])
    return [n]


def _collect_region(
    syms: List[SymbolGeom],
    used: List[bool],
    x0: float, x1: float,
    y0: float, y1: float
) -> List[int]:
    idxs: List[int] = []
    for i, s in enumerate(syms):
        if used[i]:
            continue
        if x0 <= s.cx <= x1 and y0 <= s.cy <= y1:
            idxs.append(i)
    idxs.sort(key=lambda i: syms[i].cx)
    return idxs


def _parse_linear_with_group_scripts(syms: List[SymbolGeom]) -> Node:
    r"""
    Left-to-right parsing with grouped scripts + operator limits.

    Key robustness changes:
    - Scripts may NOT look past the next "baseline" symbol.
    - Script candidates must be smaller than the base.
    - Tiny unary minus '-' is allowed INSIDE a superscript group (for e^{-\alpha...})
      but '-' is NEVER attached as a script to ordinary tokens.
    """
    if not syms:
        return seq([])

    syms = sorted(syms, key=lambda s: s.cx)
    H = _median_height(syms)
    n = len(syms)
    used = [False] * n
    y_up = _guess_y_up(syms)

    def sym_w(i: int) -> float:
        w, _ = _bbox_w_h(syms[i])
        return w

    def sym_h(i: int) -> float:
        _, h = _bbox_w_h(syms[i])
        return h

    def is_small_script(j: int, base_h: float) -> bool:
        hj = sym_h(j)
        return hj <= 0.85 * max(base_h, H)

    def next_baseline_cx(i: int, base_y: float) -> Optional[float]:
        """
        Find the next unused symbol to the right that is on (roughly) the same baseline.
        Scripts must not extend past it.
        """
        band = 0.45 * H
        for k in range(i + 1, n):
            if used[k]:
                continue
            if abs(syms[k].cy - base_y) <= band:
                return syms[k].cx
        return None

    nodes: List[Node] = []
    i = 0
    while i < n:
        if used[i]:
            i += 1
            continue

        s = syms[i]
        used[i] = True
        base = tok(s.label)

        bw = sym_w(i)
        bh = sym_h(i)

        # ---- Operator limits: \int, \sum ----
        if s.label in (r"\int", r"\sum"):
            x0 = s.cx - 0.35 * bw
            x1 = s.cx + 1.35 * bw

            upper_y0 = min(s.cy - 3.0 * H, s.cy - 0.10 * H)
            upper_y1 = max(s.cy - 3.0 * H, s.cy - 0.10 * H)
            lower_y0 = min(s.cy + 0.10 * H, s.cy + 3.0 * H)
            lower_y1 = max(s.cy + 0.10 * H, s.cy + 3.0 * H)

            upper = _collect_region(syms, used, x0, x1, upper_y0, upper_y1)
            lower = _collect_region(syms, used, x0, x1, lower_y0, lower_y1)

            upper = [j for j in upper if is_small_script(j, bh)]
            lower = [j for j in lower if is_small_script(j, bh)]

            if lower:
                for j in lower:
                    used[j] = True
                base = sub(base, _parse_linear_with_group_scripts([syms[j] for j in lower]))

            if upper:
                for j in upper:
                    used[j] = True
                base = sup(base, _parse_linear_with_group_scripts([syms[j] for j in upper]))

            nodes.append(base)
            i += 1
            continue

        # ---- Normal grouped script detection ----
        # If base is a fraction bar-like minus, don't try scripts
        if (s.label.strip() == "-") and _is_bar_like(s, H, n):
            nodes.append(base)
            i += 1
            continue

        # Determine how far scripts may extend: stop at next baseline symbol
        stop_cx = next_baseline_cx(i, s.cy)
        x_right = max(2.0 * bw, 4.5 * H)
        x_limit = (stop_cx - 0.10 * H) if stop_cx is not None else (s.cx + x_right)
        x_limit = max(x_limit, s.cx + 0.35 * bw)  # ensure sane

        # Candidate region box
        x0 = s.cx + 0.15 * bw
        x1 = x_limit

        wide_y0 = min(s.cy - 2.6 * H, s.cy + 2.6 * H)
        wide_y1 = max(s.cy - 2.6 * H, s.cy + 2.6 * H)

        cand = _collect_region(syms, used, x0, x1, wide_y0, wide_y1)

        y_thresh = 0.55 * H
        sup_idxs: List[int] = []
        sub_idxs: List[int] = []

        for j in cand:
            # size gate
            if not is_small_script(j, bh):
                continue

            # allow tiny unary minus ONLY as part of superscript groups
            if syms[j].label.strip() == "-":
                if _is_tiny_minus(syms[j], H) and _y_is_above(y_up, syms[j].cy, s.cy, y_thresh):
                    sup_idxs.append(j)
                continue

            if _y_is_above(y_up, syms[j].cy, s.cy, y_thresh):
                sup_idxs.append(j)
            elif _y_is_below(y_up, syms[j].cy, s.cy, y_thresh):
                sub_idxs.append(j)

        sup_idxs.sort(key=lambda j: syms[j].cx)
        sub_idxs.sort(key=lambda j: syms[j].cx)

        if sub_idxs:
            for j in sub_idxs:
                used[j] = True
            base = sub(base, _parse_linear_with_group_scripts([syms[j] for j in sub_idxs]))

        if sup_idxs:
            for j in sup_idxs:
                used[j] = True
            base = sup(base, _parse_linear_with_group_scripts([syms[j] for j in sup_idxs]))

        nodes.append(base)
        i += 1

    return seq(nodes)


# =========================
# Structure reconstruction
# =========================
def reconstruct_expression(symbols: List[SymbolGeom]) -> Node:
    """
    v8:
    - Detect ONE fraction bar and embed it into the line.
    - Only consume numerator/denominator symbols + bar.
    - Parse everything else with grouped scripts + operator limits.
    """
    if not symbols:
        return seq([])

    syms = sorted(symbols, key=lambda s: s.cx)

    bar = _pick_fraction_bar(syms)
    if bar is None:
        return _parse_linear_with_group_scripts(syms)

    num_syms, den_syms, _mid = _split_by_bar_local(syms, bar)
    frac_node = frac(
        _parse_linear_with_group_scripts(num_syms),
        _parse_linear_with_group_scripts(den_syms),
    )

    consumed_ids = {id(s) for s in (num_syms + den_syms)}
    consumed_ids.add(id(bar))

    outside = [s for s in syms if id(s) not in consumed_ids]

    left_syms = [s for s in outside if s.cx < bar.cx]
    right_syms = [s for s in outside if s.cx >= bar.cx]

    left_node = _parse_linear_with_group_scripts(left_syms) if left_syms else seq([])
    right_node = _parse_linear_with_group_scripts(right_syms) if right_syms else seq([])

    combined: List[Node] = []
    combined += _flatten_seq(left_node)
    combined.append(frac_node)
    combined += _flatten_seq(right_node)

    return seq(combined)


# =========================
# AST -> LaTeX
# =========================
def node_to_latex(n: Node) -> str:
    if n.kind == "tok":
        return n.value or ""

    if n.kind == "seq":
        return "".join(node_to_latex(c) for c in (n.children or []))

    if n.kind == "frac":
        num, den = n.children or []
        return r"\frac{" + node_to_latex(num) + "}{" + node_to_latex(den) + "}"

    if n.kind == "sqrt":
        inner = (n.children or [])[0]
        return r"\sqrt{" + node_to_latex(inner) + "}"

    if n.kind == "sup":
        base, exp = n.children or []
        return node_to_latex(base) + "^{" + node_to_latex(exp) + "}"

    if n.kind == "sub":
        base, subn = n.children or []
        return node_to_latex(base) + "_{" + node_to_latex(subn) + "}"

    return ""