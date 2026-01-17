# src/models/decoder_lstm.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int, memory_dim: int, attn_dim: int = 256):
        super().__init__()
        self.w_h = nn.Linear(hidden_size, attn_dim, bias=False)
        self.w_m = nn.Linear(memory_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        h_t: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ):
        q = self.w_h(h_t).unsqueeze(1)      # (B,1,A)
        k = self.w_m(memory)                # (B,T,A)
        e = self.v(torch.tanh(q + k)).squeeze(-1)  # (B,T)

        if memory_mask is not None:
            e = e.masked_fill(~memory_mask, float("-inf"))

        attn = F.softmax(e, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), memory).squeeze(1)  # (B,D)
        return context, attn


class LSTMDecoderConfig:
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        sos_id: int,
        eos_id: int,
        d_model: int = 256,
        emb_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        attn_dim: int = 256,
    ):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attn_dim = attn_dim


class LSTMDecoder(nn.Module):
    """
    Teacher forcing forward (training) + greedy decode (inference).
    """
    def __init__(self, cfg: LSTMDecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_id)
        self.dropout = nn.Dropout(cfg.dropout)

        self.lstm = nn.LSTM(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        self.attn = AdditiveAttention(
            hidden_size=cfg.hidden_size,
            memory_dim=cfg.d_model,
            attn_dim=cfg.attn_dim,
        )

        self.out = nn.Linear(cfg.hidden_size + cfg.d_model, cfg.vocab_size)

        # NEW: initialize decoder state from encoder memory
        self.init_h = nn.Linear(cfg.d_model, cfg.hidden_size)
        self.init_c = nn.Linear(cfg.d_model, cfg.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Vectorized teacher-forcing forward (no Python loop over L).
        """
        B, L = input_ids.shape
        T = memory.size(1)
        D = memory.size(2)

        emb = self.dropout(self.embed(input_ids))      # (B, L, E)
        outputs, _ = self.lstm(emb)                    # (B, L, H)

        q = self.attn.w_h(outputs)                     # (B, L, A)
        k = self.attn.w_m(memory)                      # (B, T, A)

        e = self.attn.v(torch.tanh(q.unsqueeze(2) + k.unsqueeze(1))).squeeze(-1)  # (B, L, T)

        if memory_mask is not None:
            e = e.masked_fill(~memory_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(e, dim=-1)                # (B, L, T)

        attn_2d = attn.reshape(B * L, 1, T)            # (B*L, 1, T)
        mem_2d = memory.unsqueeze(1).expand(B, L, T, D).reshape(B * L, T, D)  # (B*L, T, D)
        ctx = torch.bmm(attn_2d, mem_2d).squeeze(1).reshape(B, L, D)          # (B, L, D)

        logits = self.out(torch.cat([outputs, ctx], dim=-1))  # (B, L, V)
        return logits

    def _pool_memory(self, memory: torch.Tensor, memory_mask: torch.Tensor | None) -> torch.Tensor:
        """
        Masked mean pool over time: (B,T,D) -> (B,D)
        """
        if memory_mask is None:
            return memory.mean(dim=1)

        m = memory_mask.unsqueeze(-1).float()          # (B,T,1)
        denom = m.sum(dim=1).clamp(min=1.0)            # (B,1)
        return (memory * m).sum(dim=1) / denom         # (B,D)

    @torch.no_grad()
    def greedy_decode(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        max_len: int = 160,
        min_len: int = 1,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        pad_to_max_len: bool = True,
    ) -> torch.Tensor:
        self.eval()
        device = memory.device
        B = memory.size(0)

        cur = torch.full((B,), self.cfg.sos_id, dtype=torch.long, device=device)

        # NEW: initialize h/c from pooled encoder memory
        pooled = self._pool_memory(memory, memory_mask)                # (B,D)
        h0 = torch.tanh(self.init_h(pooled)).unsqueeze(0)              # (1,B,H)
        c0 = torch.tanh(self.init_c(pooled)).unsqueeze(0)              # (1,B,H)
        h = h0.repeat(self.cfg.num_layers, 1, 1).contiguous()
        c = c0.repeat(self.cfg.num_layers, 1, 1).contiguous()

        finished = torch.zeros((B,), dtype=torch.bool, device=device)
        outputs: list[torch.Tensor] = []
        generated: list[list[int]] = [[] for _ in range(B)]

        def get_banned_tokens(prefix: list[int], n: int) -> set[int]:
            if n <= 1 or len(prefix) < n - 1:
                return set()
            mapping: dict[tuple[int, ...], set[int]] = {}
            for i in range(len(prefix) - n + 1):
                key = tuple(prefix[i : i + n - 1])
                nxt = prefix[i + n - 1]
                mapping.setdefault(key, set()).add(nxt)
            key_now = tuple(prefix[-(n - 1):])
            return mapping.get(key_now, set())

        for step in range(max_len):
            emb = self.embed(cur).unsqueeze(1)          # (B,1,E)
            out, (h, c) = self.lstm(emb, (h, c))        # (B,1,H)
            h_t = out[:, 0, :]                          # (B,H)

            ctx, _ = self.attn(h_t, memory, memory_mask)
            logits = self.out(torch.cat([h_t, ctx], dim=-1))  # (B,V)

            # NEVER generate PAD
            logits[:, self.cfg.pad_id] = float("-inf")
            if step < min_len:
                logits[:, self.cfg.eos_id] = float("-inf")

            for b in range(B):
                if finished[b]:
                    continue

                if generated[b] and repetition_penalty is not None and repetition_penalty > 1.0:
                    used = set(generated[b])
                    for tok in used:
                        if tok == self.cfg.pad_id:
                            continue
                        if logits[b, tok] > 0:
                            logits[b, tok] /= repetition_penalty
                        else:
                            logits[b, tok] *= repetition_penalty

                banned = get_banned_tokens(generated[b], no_repeat_ngram_size)
                for tok in banned:
                    logits[b, tok] = float("-inf")

            next_id = torch.argmax(logits, dim=-1)

            next_id = torch.where(
                finished,
                torch.tensor(self.cfg.eos_id, device=device),
                next_id,
            )

            outputs.append(next_id)

            next_list = next_id.detach().tolist()
            for b in range(B):
                if not finished[b]:
                    generated[b].append(int(next_list[b]))

            finished |= (next_id == self.cfg.eos_id)
            cur = next_id

            if bool(finished.all()):
                break

        ids = torch.stack(outputs, dim=1) if outputs else torch.empty((B, 0), dtype=torch.long, device=device)

        if pad_to_max_len and ids.size(1) < max_len:
            pad_len = max_len - ids.size(1)
            pad = torch.full((B, pad_len), self.cfg.pad_id, dtype=torch.long, device=device)
            ids = torch.cat([ids, pad], dim=1)

        return ids