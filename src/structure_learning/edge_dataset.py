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


def pair_features(ni: dict, nj: dict) -> List[float]:
    """
    Handcrafted features for edge (i -> j).
    All are cheap + strong baselines for CROHME relations.
    """
    xi, yi = _safe(ni.get("cx")), _safe(ni.get("cy"))
    xj, yj = _safe(nj.get("cx")), _safe(nj.get("cy"))
    wi, hi = max(1e-6, _safe(ni.get("w"))), max(1e-6, _safe(ni.get("h")))
    wj, hj = max(1e-6, _safe(nj.get("w"))), max(1e-6, _safe(nj.get("h")))

    dx = (xj - xi) / wi
    dy = (yj - yi) / hi
    dist = math.sqrt(dx * dx + dy * dy)

    # bbox overlaps (helpful for NEXT vs stacked relations)
    x0i, x1i = _safe(ni.get("x0")), _safe(ni.get("x1"))
    y0i, y1i = _safe(ni.get("y0")), _safe(ni.get("y1"))
    x0j, x1j = _safe(nj.get("x0")), _safe(nj.get("x1"))
    y0j, y1j = _safe(nj.get("y0")), _safe(nj.get("y1"))

    x_iou = _iou_1d(x0i, x1i, x0j, x1j)
    y_iou = _iou_1d(y0i, y1i, y0j, y1j)

    # relative size
    wr = wj / wi
    hr = hj / hi

    # directional indicators
    right = 1.0 if xj > xi else 0.0
    left = 1.0 - right
    above = 1.0 if yj < yi else 0.0
    below = 1.0 - above

    # virtual flags
    vi = 1.0 if ni.get("is_virtual") else 0.0
    vj = 1.0 if nj.get("is_virtual") else 0.0

    return [
        dx, dy, dist,
        x_iou, y_iou,
        wr, hr,
        right, left, above, below,
        vi, vj
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

    IMPORTANT FIX:
    Some nodes (especially virtual nodes like FRAC@k) can have cx/cy/w/h missing or None.
    We use _safe() so float(None) never crashes.

    Strategy:
      - for each i: take k_right closest nodes to the right (xj>xi) by dx
      - plus k_any closest overall by euclidean distance in normalized space
    """
    pairs = set()
    N = len(nodes)

    # Safe geometry cache: (cx, cy, w, h)
    cxy = []
    for n in nodes:
        cx = _safe(n.get("cx"))
        cy = _safe(n.get("cy"))
        w = max(1e-6, _safe(n.get("w")))
        h = max(1e-6, _safe(n.get("h")))
        cxy.append((cx, cy, w, h))

    for i in range(N):
        xi, yi, wi, hi = cxy[i]

        right_cands = []
        any_cands = []

        for j in range(N):
            if i == j:
                continue
            xj, yj, wj, hj = cxy[j]

            dx = (xj - xi) / wi
            dy = (yj - yi) / hi
            dist = math.sqrt(dx * dx + dy * dy)

            any_cands.append((dist, j))
            if xj > xi:
                right_cands.append((dx, j))

        right_cands.sort(key=lambda t: t[0])
        any_cands.sort(key=lambda t: t[0])

        for _, j in right_cands[:k_right]:
            pairs.add((i, j))
        for _, j in any_cands[:k_any]:
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

                # candidate pool (safe)
                cand = candidate_pairs(nodes, k_right=self.k_right, k_any=self.k_any)

                # positives are gold edges that exist in candidate pool; but we also keep gold edges even if not in cand
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
                    # graphs with no edges: take small amount
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