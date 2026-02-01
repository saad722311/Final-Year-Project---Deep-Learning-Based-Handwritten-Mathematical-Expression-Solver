# src/structure_learning/constrain_pred_edges_jsonl.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


REL_UNIQUE_OUT = {"NEXT", "SUP", "SUB", "UNDER", "OVER"}
FRAC_OUT = {"FRAC_NUM", "FRAC_DEN"}


def _get_edges(obj: dict) -> List[list]:
    # expected: pred_edges = [src, dst, rel, score]
    if "pred_edges" in obj:
        return obj["pred_edges"]
    if "edges_pred" in obj:
        return obj["edges_pred"]
    return obj.get("edges", [])


def _set_edges(obj: dict, edges: List[list]) -> None:
    obj["pred_edges"] = edges


def _score(e: list) -> float:
    # if score missing, treat as 1.0
    if len(e) >= 4 and e[3] is not None:
        try:
            return float(e[3])
        except Exception:
            return 1.0
    return 1.0


def _is_virtual_frac(node: dict) -> bool:
    if node.get("is_virtual") and str(node.get("label", "")).upper() == "FRAC":
        return True
    # also allow "FRAC@k" href-style
    href = str(node.get("href", ""))
    if node.get("is_virtual") and href.startswith("FRAC@"):
        return True
    return False


def _break_next_cycles(next_edges: Dict[int, Tuple[int, float]]) -> Dict[int, Tuple[int, float]]:
    """
    next_edges: src -> (dst, score)
    Remove edges that create cycles, keep higher-score ones.
    Simple approach: detect cycles by walking pointers.
    """
    # build reverse lookup for possible cycle resolution
    # We'll remove the lowest-score edge in each detected cycle.
    while True:
        visited_global = set()
        cycle_found = False

        for start in list(next_edges.keys()):
            if start in visited_global:
                continue
            path = []
            seen_local = {}  # node -> index in path
            cur = start

            while cur in next_edges:
                visited_global.add(cur)
                path.append(cur)
                seen_local[cur] = len(path) - 1
                nxt = next_edges[cur][0]

                if nxt in seen_local:
                    # cycle detected: nodes = path[idx:]
                    idx = seen_local[nxt]
                    cycle_nodes = path[idx:]  # these are src nodes in the cycle
                    # choose edge with lowest score to remove
                    worst_src = min(cycle_nodes, key=lambda s: next_edges[s][1])
                    del next_edges[worst_src]
                    cycle_found = True
                    break

                cur = nxt

            if cycle_found:
                break

        if not cycle_found:
            break

    return next_edges


def constrain_edges(obj: dict) -> List[list]:
    nodes = obj.get("nodes", [])
    edges = _get_edges(obj)

    # Filter: valid format, no self-loops
    cleaned = []
    for e in edges:
        if not isinstance(e, list) or len(e) < 3:
            continue
        s, t, rel = int(e[0]), int(e[1]), str(e[2])
        if s == t:
            continue
        sc = _score(e)
        cleaned.append([s, t, rel, sc])

    # Sort by score desc so we keep best edges first
    cleaned.sort(key=lambda e: e[3], reverse=True)

    # Determine which nodes are FRAC virtual nodes
    is_frac = set()
    for i, n in enumerate(nodes):
        if _is_virtual_frac(n):
            is_frac.add(i)

    # Keep only one outgoing for certain relations
    kept = []
    best_out: Dict[Tuple[int, str], Tuple[int, float]] = {}  # (src, rel) -> (dst, score)

    # For FRAC virtual nodes, track outgoing NUM/DEN separately
    best_frac: Dict[Tuple[int, str], Tuple[int, float]] = {}  # (frac_src, FRAC_NUM/DEN) -> (dst, score)

    # First pass: enforce uniqueness constraints
    for s, t, rel, sc in cleaned:
        if rel in FRAC_OUT and s in is_frac:
            key = (s, rel)
            if key not in best_frac:
                best_frac[key] = (t, sc)
            continue

        if rel in REL_UNIQUE_OUT:
            key = (s, rel)
            if key not in best_out:
                best_out[key] = (t, sc)
            continue

        # For other relations, just keep them (rare, but safe)
        kept.append([s, t, rel, sc])

    # Add unique-out edges
    for (s, rel), (t, sc) in best_out.items():
        kept.append([s, t, rel, sc])

    # Add FRAC NUM/DEN
    for (s, rel), (t, sc) in best_frac.items():
        kept.append([s, t, rel, sc])

    # Now specifically break NEXT cycles
    next_map: Dict[int, Tuple[int, float]] = {}
    other = []
    for s, t, rel, sc in kept:
        if rel == "NEXT":
            # keep best NEXT per source already ensured
            next_map[s] = (t, sc)
        else:
            other.append([s, t, rel, sc])

    next_map = _break_next_cycles(next_map)

    final_edges = other[:]
    for s, (t, sc) in next_map.items():
        final_edges.append([s, t, "NEXT", sc])

    # Sort final edges for readability (optional)
    final_edges.sort(key=lambda e: (e[2], e[0], e[1]))

    return final_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.in_jsonl)
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    before = 0
    after = 0

    with inp.open("r", encoding="utf-8") as f_in, outp.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            obj = json.loads(line)
            edges = _get_edges(obj)
            before += len(edges)

            new_edges = constrain_edges(obj)
            after += len(new_edges)

            _set_edges(obj, new_edges)
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"[constrain] graphs={n}")
    print(f"[constrain] edges_before={before} edges_after={after}")
    print(f"[constrain] saved: {outp}")


if __name__ == "__main__":
    main()