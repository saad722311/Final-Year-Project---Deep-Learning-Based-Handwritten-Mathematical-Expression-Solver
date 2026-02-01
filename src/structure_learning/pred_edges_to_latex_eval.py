# src/structure_learning/pred_edges_to_latex_eval.py
from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -------------------------
# GT helpers (InkML truth)
# -------------------------
def _get_inkml_namespace(root: ET.Element) -> str:
    if root.tag.startswith("{") and "}" in root.tag:
        return root.tag.split("}")[0].strip("{")
    return ""


def read_inkml_truth_latex(inkml_path: str) -> str:
    try:
        tree = ET.parse(inkml_path)
        root = tree.getroot()
        ns = _get_inkml_namespace(root)

        def T(name: str) -> str:
            return f"{{{ns}}}{name}" if ns else name

        for ann in root.findall(T("annotation")):
            if ann.attrib.get("type") == "truth":
                return (ann.text or "").strip()
    except Exception:
        return ""
    return ""


# -------------------------
# Normalization (nEM)
# -------------------------
_ws_re = re.compile(r"\s+")


def normalize_latex(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = _ws_re.sub("", s)
    s = re.sub(r"\^\{([A-Za-z0-9])\}", r"^\1", s)
    s = re.sub(r"_\{([A-Za-z0-9])\}", r"_\1", s)
    return s


# -------------------------
# Graph -> LaTeX
# -------------------------
def build_best_edges(pred_edges: List[list]) -> Dict[Tuple[int, str], Tuple[int, float]]:
    best: Dict[Tuple[int, str], Tuple[int, float]] = {}
    for e in pred_edges:
        if len(e) < 3:
            continue
        src = int(e[0])
        dst = int(e[1])
        rel = str(e[2])
        prob = float(e[3]) if len(e) >= 4 and e[3] is not None else 1.0
        key = (src, rel)
        cur = best.get(key)
        if cur is None or prob > cur[1]:
            best[key] = (dst, prob)
    return best


def incoming_next_counts(best_edges: Dict[Tuple[int, str], Tuple[int, float]], n_nodes: int) -> List[int]:
    inc = [0] * n_nodes
    for (src, rel), (dst, _p) in best_edges.items():
        if rel == "NEXT" and 0 <= dst < n_nodes:
            inc[dst] += 1
    return inc


def node_sort_key(nodes: List[dict], i: int) -> Tuple[float, float]:
    n = nodes[i]
    x = n.get("x0")
    y = n.get("y0")
    x = float(x) if isinstance(x, (int, float)) else 1e9
    y = float(y) if isinstance(y, (int, float)) else 1e9
    return (x, y)


def is_virtual_frac(node: dict) -> bool:
    return bool(node.get("is_virtual")) and str(node.get("label", "")).strip() == "FRAC"


# big operators where UNDER/OVER should become _{...}^{...}
BIGOPS = {
    r"\sum", r"\prod", r"\coprod", r"\int", r"\iint", r"\iiint", r"\oint", r"\bigcup", r"\bigcap", r"\bigoplus",
    r"\bigotimes", r"\bigsqcup", r"\bigvee", r"\bigwedge", r"\lim",
}


def render_graph_to_latex(nodes: List[dict], pred_edges: List[list]) -> str:
    n_nodes = len(nodes)
    best = build_best_edges(pred_edges)
    visited = [False] * n_nodes

    def best_dst(src: int, rel: str) -> Optional[int]:
        v = best.get((src, rel))
        if v is None:
            return None
        dst = int(v[0])
        if 0 <= dst < n_nodes:
            return dst
        return None

    def render_chain(start: Optional[int]) -> str:
        if start is None:
            return ""
        out_parts: List[str] = []
        cur = start
        guard = 0
        while cur is not None and 0 <= cur < n_nodes and not visited[cur] and guard < 800:
            visited[cur] = True
            out_parts.append(render_node(cur))
            cur = best_dst(cur, "NEXT")
            guard += 1
        return "".join(out_parts)

    def render_node(i: int) -> str:
        node = nodes[i]

        # --- Virtual FRAC ---
        if is_virtual_frac(node):
            num = best_dst(i, "FRAC_NUM")
            den = best_dst(i, "FRAC_DEN")
            num_s = render_chain(num)
            den_s = render_chain(den)
            return f"\\frac{{{num_s}}}{{{den_s}}}"

        base = str(node.get("label", "")).strip()
        if not base:
            base = ""

        # --- UNDER/OVER (only meaningful for big ops) ---
        under = best_dst(i, "UNDER")
        over = best_dst(i, "OVER")

        # Use chain rendering for under/over content
        under_s = render_chain(under) if under is not None else ""
        over_s = render_chain(over) if over is not None else ""

        # --- SUB/SUP ---
        sub = best_dst(i, "SUB")
        sup = best_dst(i, "SUP")
        sub_s = render_chain(sub) if sub is not None else ""
        sup_s = render_chain(sup) if sup is not None else ""

        # Decide how to attach scripts
        # Priority:
        # 1) BIGOPS: use UNDER/OVER if present, else SUB/SUP.
        # 2) Others: use SUB/SUP.
        if base in BIGOPS:
            if under is not None and over is not None:
                return f"{base}_{{{under_s}}}^{{{over_s}}}"
            if under is not None:
                # e.g., \lim_{x->0}
                return f"{base}_{{{under_s}}}"
            if over is not None:
                return f"{base}^{{{over_s}}}"

        # Fall back to SUB/SUP for normal nodes (or bigops without under/over)
        if sub is not None and sup is not None:
            return f"{base}_{{{sub_s}}}^{{{sup_s}}}"
        if sub is not None:
            return f"{base}_{{{sub_s}}}"
        if sup is not None:
            return f"{base}^{{{sup_s}}}"

        return base

    # roots = nodes with no incoming NEXT
    inc_next = incoming_next_counts(best, n_nodes)
    roots = [i for i in range(n_nodes) if inc_next[i] == 0]
    roots.sort(key=lambda i: node_sort_key(nodes, i))

    out: List[str] = []
    for r in roots:
        if visited[r]:
            continue
        out.append(render_chain(r))

    # any leftovers (broken chains/cycles)
    leftover = [i for i in range(n_nodes) if not visited[i]]
    leftover.sort(key=lambda i: node_sort_key(nodes, i))
    for i in leftover:
        if not visited[i]:
            out.append(render_chain(i))

    return "".join(out)


# -------------------------
# CLI: evaluate + save
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use_inkml_truth", action="store_true")
    args = ap.parse_args()

    pred_jsonl = Path(args.pred_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "pred_latex.csv"
    out_em = out_dir / "pred_em.txt"
    out_bad = out_dir / "bad_cases.txt"

    rows: List[dict] = []
    total = 0
    em_cnt = 0
    nem_cnt = 0

    with pred_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if args.limit and total >= args.limit:
                break

            obj = json.loads(line)
            ui = obj.get("ui", "")
            nodes = obj.get("nodes", [])
            pred_edges = obj.get("pred_edges", obj.get("edges", []))
            inkml_path = obj.get("inkml_path", "")

            pred = render_graph_to_latex(nodes, pred_edges)

            if args.use_inkml_truth and inkml_path:
                gt = read_inkml_truth_latex(inkml_path)
            else:
                gt = str(obj.get("gt_latex", obj.get("gt", "")) or "")

            em = 1 if pred.strip() == gt.strip() and gt.strip() != "" else 0
            nem = 1 if normalize_latex(pred) == normalize_latex(gt) and normalize_latex(gt) != "" else 0

            total += 1
            em_cnt += em
            nem_cnt += nem

            rows.append(
                {
                    "ui": ui,
                    "pred": pred,
                    "gt": gt,
                    "em": em,
                    "nem": nem,
                    "n_nodes": len(nodes),
                    "n_pred_edges": len(pred_edges),
                    "inkml_path": inkml_path,
                }
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["ui", "pred", "gt", "em", "nem", "n_nodes", "n_pred_edges", "inkml_path"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    em_rate = em_cnt / max(1, total)
    nem_rate = nem_cnt / max(1, total)

    out_em.write_text(f"items={total}\nEM={em_rate:.4f}\nnEM={nem_rate:.4f}\n", encoding="utf-8")

    bad_lines: List[str] = []
    for r in rows:
        if r["em"] == 0:
            bad_lines.append(f"UI: {r['ui']}\nPRED: {r['pred']}\nGT  : {r['gt']}\n---\n")
        if len(bad_lines) >= 200:
            break
    out_bad.write_text("".join(bad_lines), encoding="utf-8")

    print(f"[latex-eval] saved: {out_csv}")
    print(f"[latex-eval] saved: {out_em}")
    print(f"[latex-eval] saved: {out_bad}")


if __name__ == "__main__":
    main()