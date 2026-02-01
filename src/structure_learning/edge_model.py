# src/structure_learning/edge_model.py
from __future__ import annotations

import torch
import torch.nn as nn


class EdgeMLP(nn.Module):
    """
    Baseline edge classifier:
      Emb(label_i) + Emb(label_j) + pair_features -> MLP -> edge_type
    """
    def __init__(self, n_labels: int, n_edge_types: int, feat_dim: int, emb_dim: int = 64, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(n_labels, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2 + feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_edge_types),
        )

    def forward(self, li: torch.Tensor, lj: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        ei = self.emb(li)
        ej = self.emb(lj)
        x = torch.cat([ei, ej, feat], dim=-1)
        return self.net(x)