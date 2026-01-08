# src/models/decoder_lstm.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Bahdanau (additive) attention:
      score(h_t, m_i) = v^T tanh(W_h h_t + W_m m_i)

    h_t: (B, H)
    memory: (B, T, D)
    returns:
      context: (B, D)
      attn_weights: (B, T)
    """
    def __init__(self, hidden_size: int, memory_dim: int, attn_dim: int = 256):
        super().__init__()
        self.w_h = nn.Linear(hidden_size, attn_dim, bias=False)
        self.w_m = nn.Linear(memory_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h_t: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None):
        # h_t: (B, H) -> (B, 1, A)
        # memory: (B, T, D) -> (B, T, A)
        q = self.w_h(h_t).unsqueeze(1)          # (B, 1, A)
        k = self.w_m(memory)                    # (B, T, A)

        e = self.v(torch.tanh(q + k)).squeeze(-1)   # (B, T)

        if memory_mask is not None:
            # mask: True for valid, False for pad. We'll set pad positions to -inf.
            e = e.masked_fill(~memory_mask, float("-inf"))

        attn = F.softmax(e, dim=-1)             # (B, T)
        context = torch.bmm(attn.unsqueeze(1), memory).squeeze(1)  # (B, D)
        return context, attn


class LSTMDecoderConfig:
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,       # encoder memory dim
        emb_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        attn_dim: int = 256,
    ):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attn_dim = attn_dim


class LSTMDecoder(nn.Module):
    """
    Teacher-forcing decoder.

    Inputs:
      input_ids: (B, L)  -- decoder inputs (shifted with <SOS>)
      memory:    (B, T, D) -- encoder output

    Output:
      logits: (B, L, vocab_size)
    """
    def __init__(self, cfg: LSTMDecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_id)
        self.dropout = nn.Dropout(cfg.dropout)

        # LSTM reads embedded tokens
        self.lstm = nn.LSTM(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Attention over encoder memory
        self.attn = AdditiveAttention(
            hidden_size=cfg.hidden_size,
            memory_dim=cfg.d_model,
            attn_dim=cfg.attn_dim,
        )

        # Combine decoder hidden + attention context -> vocab logits
        self.out = nn.Linear(cfg.hidden_size + cfg.d_model, cfg.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids: (B, L)
        memory: (B, T, D)
        memory_mask: (B, T) bool, True = valid positions
        """
        B, L = input_ids.shape

        emb = self.embed(input_ids)     # (B, L, E)
        emb = self.dropout(emb)

        # Run LSTM over the whole sequence (teacher forcing)
        outputs, _ = self.lstm(emb)     # outputs: (B, L, H)

        # Compute attention per timestep
        logits_list = []
        for t in range(L):
            h_t = outputs[:, t, :]                      # (B, H)
            ctx, _ = self.attn(h_t, memory, memory_mask)  # ctx: (B, D)
            combined = torch.cat([h_t, ctx], dim=-1)     # (B, H+D)
            logits = self.out(combined)                 # (B, V)
            logits_list.append(logits)

        logits = torch.stack(logits_list, dim=1)         # (B, L, V)
        return logits