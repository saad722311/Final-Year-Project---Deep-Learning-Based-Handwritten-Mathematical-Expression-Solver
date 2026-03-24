# src/structure_learning/edge_dataset.py
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


EDGE_TYPES = ["NONE", "NEXT", "SUP", "SUB", "FRAC_NUM", "FRAC_DEN", "UNDER", "OVER"]


def _safe(x: float) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        if math.isfinite(x):
            return float(x)
    return 0.0


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0.0, hi - lo)


def _iou_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = _overlap_1d(a0, a1, b0, b1)
    union = max(1e-6, (a1 - a0) + (b1 - b0) - inter)
    return inter / union


def _is_frac_virtual(n: dict) -> bool:
    """Heuristic: virtual node that represents a fraction bar/container."""
    if not n.get("is_virtual"):
        return False
    lab = str(n.get("label", "")).upper()
    # be permissive: your pipeline uses FRAC_* relations so virtual label usually contains FRAC
    return "FRAC" in lab


def pair_features(ni: dict, nj: dict) -> List[float]:
    """
    Robust features for edge (i -> j).
    FIX: virtual nodes can have w/h ~ 0, which makes dx/dy explode.
         We normalize using a stable scale based on both nodes.

    C3.3.4 ADD: fraction-aware features:
      - when source node is a FRAC virtual node, add explicit flags indicating
        whether target is above or below that fraction bar. This helps FRAC_NUM vs FRAC_DEN.
    """
    xi, yi = _safe(ni.get("cx")), _safe(ni.get("cy"))
    xj, yj = _safe(nj.get("cx")), _safe(nj.get("cy"))

    wi, hi = _safe(ni.get("w")), _safe(ni.get("h"))
    wj, hj = _safe(nj.get("w")), _safe(nj.get("h"))

    # stable normalization (prevents virtual-node blowups)
    scale_w = max(wi, wj, 1.0)
    scale_h = max(hi, hj, 1.0)

    dx = (xj - xi) / scale_w
    dy = (yj - yi) / scale_h

    # clamp (extra safety)
    dx = max(-10.0, min(10.0, dx))
    dy = max(-10.0, min(10.0, dy))

    dist = math.sqrt(dx * dx + dy * dy)

    # bbox overlaps
    x0i, x1i = _safe(ni.get("x0")), _safe(ni.get("x1"))
    y0i, y1i = _safe(ni.get("y0")), _safe(ni.get("y1"))
    x0j, x1j = _safe(nj.get("x0")), _safe(nj.get("x1"))
    y0j, y1j = _safe(nj.get("y0")), _safe(nj.get("y1"))

    x_iou = _iou_1d(x0i, x1i, x0j, x1j)
    y_iou = _iou_1d(y0i, y1i, y0j, y1j)

    # relative size (safe)
    wi2 = max(1e-6, wi)
    hi2 = max(1e-6, hi)
    wr = max(1e-6, wj) / wi2
    hr = max(1e-6, hj) / hi2

    # direction flags
    right = 1.0 if xj > xi else 0.0
    left = 1.0 - right
    above = 1.0 if yj < yi else 0.0
    below = 1.0 - above

    vi = 1.0 if ni.get("is_virtual") else 0.0
    vj = 1.0 if nj.get("is_virtual") else 0.0

    # --- NEW: fraction-aware cues (only meaningful when i is a FRAC virtual node) ---
    frac_src = 1.0 if _is_frac_virtual(ni) else 0.0
    frac_tgt = 1.0 if _is_frac_virtual(nj) else 0.0  # usually 0, but keep for completeness
    frac_above = 0.0
    frac_below = 0.0
    if frac_src and not frac_tgt:
        if yj < yi:
            frac_above = 1.0
        elif yj > yi:
            frac_below = 1.0

    return [
        dx, dy, dist,
        x_iou, y_iou,
        wr, hr,
        right, left, above, below,
        vi, vj,
        frac_src, frac_above, frac_below
    ]


def build_label_vocab(jsonl_path: str, max_items: int = 0) -> Dict[str, int]:
    path = Path(jsonl_path)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            for n in obj.get("nodes", []):
                lab = str(n.get("label", "")).strip()
                if lab and lab not in vocab:
                    vocab[lab] = len(vocab)
            if max_items and (i + 1) >= max_items:
                break
    return vocab


def candidate_pairs(nodes: List[dict], k_right: int = 6, k_any: int = 6) -> List[Tuple[int, int]]:
    """
    Build a small candidate set per graph so we don't classify all N^2 pairs.

    C3.3.1 FIX (important for fractions):
      - keeps k_right (good for NEXT)
      - keeps k_any closest (general)
      - adds above/below candidates (critical for FRAC_NUM / FRAC_DEN)
      - keeps virtual-node full-connect safety net
    """
    pairs = set()
    N = len(nodes)

    # Safe geometry cache: (cx, cy, w, h, is_virtual)
    cxy = []
    for n in nodes:
        cx = _safe(n.get("cx"))
        cy = _safe(n.get("cy"))
        w = max(1e-6, _safe(n.get("w")))
        h = max(1e-6, _safe(n.get("h")))
        is_virtual = bool(n.get("is_virtual", False))
        cxy.append((cx, cy, w, h, is_virtual))

    # 1) Always fully-connect virtual nodes (both directions)
    virtual_idx = [i for i in range(N) if cxy[i][4]]
    if virtual_idx:
        for i in virtual_idx:
            for j in range(N):
                if i == j:
                    continue
                pairs.add((i, j))
                pairs.add((j, i))

    # vertical candidates budget
    k_vert = max(1, k_any // 2)  # above and below each get this many

    # 2) Normal top-k for non-virtual nodes (+ vertical mining)
    for i in range(N):
        xi, yi, wi, hi, vi = cxy[i]
        if vi:
            continue  # already fully-connected above

        right_cands = []
        any_cands = []
        above_cands = []
        below_cands = []

        for j in range(N):
            if i == j:
                continue
            xj, yj, wj, hj, vj = cxy[j]

            dx = (xj - xi) / wi
            dy = (yj - yi) / hi
            dist = math.sqrt(dx * dx + dy * dy)

            any_cands.append((dist, j))
            if xj > xi:
                right_cands.append((dx, j))

            # vertical mining (fractions/scripts live here)
            if yj < yi:
                above_cands.append((abs(dy), j))
            elif yj > yi:
                below_cands.append((abs(dy), j))

        right_cands.sort(key=lambda t: t[0])
        any_cands.sort(key=lambda t: t[0])
        above_cands.sort(key=lambda t: t[0])
        below_cands.sort(key=lambda t: t[0])

        for _, j in right_cands[:k_right]:
            pairs.add((i, j))
        for _, j in any_cands[:k_any]:
            pairs.add((i, j))

        # ensure above/below candidates exist
        for _, j in above_cands[:k_vert]:
            pairs.add((i, j))
        for _, j in below_cands[:k_vert]:
            pairs.add((i, j))

    return sorted(pairs)


@dataclass
class EdgeSample:
    xi: int
    xj: int
    feat: List[float]
    lab: int


class EdgeDataset(Dataset):
    """
    Builds edge classification samples across all graphs, with negative sampling.
    Each sample = (label_i, label_j, pair_features, edge_type)
    """
    def __init__(
        self,
        jsonl_path: str,
        label_vocab: Dict[str, int],
        edge_type_to_id: Dict[str, int],
        max_items: int = 0,
        neg_per_pos: int = 3,
        k_right: int = 6,
        k_any: int = 6,
        seed: int = 42,
    ):
        self.path = Path(jsonl_path)
        self.label_vocab = label_vocab
        self.edge_type_to_id = edge_type_to_id
        self.neg_per_pos = neg_per_pos
        self.k_right = k_right
        self.k_any = k_any
        self.rng = random.Random(seed)

        self.samples: List[EdgeSample] = []
        self._build(max_items=max_items)

    def _lab_id(self, s: str) -> int:
        s = (s or "").strip()
        return self.label_vocab.get(s, self.label_vocab["<UNK>"])

    def _build(self, max_items: int = 0):
        n_graphs = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                nodes = obj["nodes"]
                edges = obj["edges"]  # [src, dst, type]

                # gold map: (i,j)->etype
                gold = {(int(s), int(t)): str(r) for (s, t, r) in edges}

                # candidate pool
                cand = candidate_pairs(nodes, k_right=self.k_right, k_any=self.k_any)

                # ensure we don't lose gold edges
                gold_pairs = set(gold.keys())
                cand_set = set(cand) | gold_pairs

                # build positive samples
                pos = []
                for (i, j) in cand_set:
                    et = gold.get((i, j), "NONE")
                    if et != "NONE":
                        pos.append((i, j, et))

                # build negatives by sampling NONE among candidate pairs
                none_pairs = [(i, j) for (i, j) in cand_set if gold.get((i, j), "NONE") == "NONE"]

                # shuffle negatives once per graph
                self.rng.shuffle(none_pairs)

                # add positives
                for (i, j, et) in pos:
                    ni, nj = nodes[i], nodes[j]
                    xi = self._lab_id(str(ni.get("label", "")))
                    xj = self._lab_id(str(nj.get("label", "")))
                    feat = pair_features(ni, nj)
                    lab = self.edge_type_to_id[et]
                    self.samples.append(EdgeSample(xi, xj, feat, lab))

                # negative sampling
                if pos:
                    need = self.neg_per_pos * len(pos)
                    take = none_pairs[:need]
                else:
                    take = none_pairs[:10]

                for (i, j) in take:
                    ni, nj = nodes[i], nodes[j]
                    xi = self._lab_id(str(ni.get("label", "")))
                    xj = self._lab_id(str(nj.get("label", "")))
                    feat = pair_features(ni, nj)
                    lab = self.edge_type_to_id["NONE"]
                    self.samples.append(EdgeSample(xi, xj, feat, lab))

                n_graphs += 1
                if max_items and n_graphs >= max_items:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.tensor(s.xi, dtype=torch.long),
            torch.tensor(s.xj, dtype=torch.long),
            torch.tensor(s.feat, dtype=torch.float32),
            torch.tensor(s.lab, dtype=torch.long),
        )