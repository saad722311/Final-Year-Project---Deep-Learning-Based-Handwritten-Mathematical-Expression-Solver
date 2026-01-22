# src/models/decoder_transformer.py
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerDecoderConfig:
    vocab_size: int
    pad_id: int
    sos_id: int
    eos_id: int
    unk_id: int | None = None

    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    max_len: int = 256  # for positional embeddings


class PositionalEncoding(nn.Module):
    """
    Learnable positional embeddings (simple and effective for this baseline).
    """
    def __init__(self, max_len: int, d_model: int, dropout: float):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        idx = torch.arange(L, device=x.device)
        pe = self.pos(idx).unsqueeze(0).expand(B, L, D)
        return self.dropout(x + pe)


class TransformerDecoder(nn.Module):
    """
    Teacher forcing forward + greedy decode + beam decode.
    Matches the API of your LSTMDecoder.
    """
    def __init__(self, cfg: TransformerDecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos = PositionalEncoding(cfg.max_len, cfg.d_model, cfg.dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,   # crucial
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.n_layers)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)

    # -------------------------
    # Masks
    # -------------------------
    @staticmethod
    def _causal_mask(L: int, device: torch.device) -> torch.Tensor:
        # True = blocked (PyTorch transformer expects bool mask with True meaning "masked")
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def _key_padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        # True = pad positions
        return (ids == pad_id)

    # -------------------------
    # Training forward
    # -------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids: (B,L)  (teacher forcing input)
        memory: (B,T,D)
        memory_mask: (B,T) bool where True=valid, False=pad
        returns logits: (B,L,V)
        """
        B, L = input_ids.shape

        x = self.embed(input_ids)                  # (B,L,D)
        x = self.pos(x)                            # (B,L,D)

        tgt_mask = self._causal_mask(L, x.device)  # (L,L) bool
        tgt_key_padding = self._key_padding_mask(input_ids, self.cfg.pad_id)  # (B,L) bool

        # memory_key_padding_mask: True=pad
        mem_key_padding = None
        if memory_mask is not None:
            mem_key_padding = ~memory_mask  # (B,T)

        y = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=mem_key_padding,
        )  # (B,L,D)

        logits = self.out(y)  # (B,L,V)
        return logits

    # -------------------------
    # no-repeat ngram helper
    # -------------------------
    @staticmethod
    def _get_banned_tokens(prefix: list[int], n: int) -> set[int]:
        if n <= 1 or len(prefix) < n - 1:
            return set()
        mapping: dict[tuple[int, ...], set[int]] = {}
        for i in range(len(prefix) - n + 1):
            key = tuple(prefix[i : i + n - 1])
            nxt = prefix[i + n - 1]
            mapping.setdefault(key, set()).add(nxt)
        key_now = tuple(prefix[-(n - 1):])
        return mapping.get(key_now, set())

    # -------------------------
    # Greedy decode
    # -------------------------
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
        forbid_unk: bool = True,
    ) -> torch.Tensor:
        self.eval()
        device = memory.device
        B = memory.size(0)

        # decoded tokens so far (include SOS at pos 0)
        ys = torch.full((B, 1), self.cfg.sos_id, dtype=torch.long, device=device)
        finished = torch.zeros((B,), dtype=torch.bool, device=device)

        generated: list[list[int]] = [[] for _ in range(B)]  # excludes SOS (match your LSTM behavior)

        # memory padding mask
        mem_key_padding = None
        if memory_mask is not None:
            mem_key_padding = ~memory_mask  # True=pad

        for step in range(max_len):
            L = ys.size(1)
            x = self.embed(ys)
            x = self.pos(x)

            tgt_mask = self._causal_mask(L, device)
            tgt_key_padding = self._key_padding_mask(ys, self.cfg.pad_id)

            out = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding,
                memory_key_padding_mask=mem_key_padding,
            )  # (B,L,D)

            logits = self.out(out[:, -1, :])  # (B,V)

            # hard bans
            logits[:, self.cfg.pad_id] = float("-inf")
            if forbid_unk and self.cfg.unk_id is not None:
                logits[:, self.cfg.unk_id] = float("-inf")
            if step < min_len:
                logits[:, self.cfg.eos_id] = float("-inf")

            # penalties per sample (same style as your LSTM)
            for b in range(B):
                if finished[b]:
                    continue

                # repetition penalty
                if generated[b] and repetition_penalty is not None and repetition_penalty > 1.0:
                    used = set(generated[b])
                    for tok in used:
                        if tok == self.cfg.pad_id:
                            continue
                        if self.cfg.unk_id is not None and tok == self.cfg.unk_id:
                            continue
                        if logits[b, tok] > 0:
                            logits[b, tok] /= repetition_penalty
                        else:
                            logits[b, tok] *= repetition_penalty

                # no repeat ngram
                banned = self._get_banned_tokens(generated[b], no_repeat_ngram_size)
                for tok in banned:
                    logits[b, tok] = float("-inf")

            next_id = torch.argmax(logits, dim=-1)  # (B,)

            # if finished, force EOS
            next_id = torch.where(
                finished,
                torch.tensor(self.cfg.eos_id, device=device),
                next_id,
            )

            # append
            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)

            next_list = next_id.detach().tolist()
            for b in range(B):
                if not finished[b]:
                    generated[b].append(int(next_list[b]))

            finished |= (next_id == self.cfg.eos_id)
            if bool(finished.all()):
                break

        # drop SOS
        ids = ys[:, 1:]  # (B, <=max_len)

        if pad_to_max_len and ids.size(1) < max_len:
            pad_len = max_len - ids.size(1)
            pad = torch.full((B, pad_len), self.cfg.pad_id, dtype=torch.long, device=device)
            ids = torch.cat([ids, pad], dim=1)

        # truncate if longer (rare)
        ids = ids[:, :max_len]
        return ids

    # -------------------------
    # Beam decode
    # -------------------------
    @torch.no_grad()
    def beam_decode(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        max_len: int = 160,
        beam_size: int = 5,
        alpha: float = 0.6,
        min_len: int = 1,
        repetition_penalty: float = 1.10,
        no_repeat_ngram_size: int = 3,
        forbid_pad: bool = True,
        forbid_unk: bool = True,
        pad_to_max_len: bool = True,
    ) -> torch.Tensor:
        """
        Per-sample beam search (simple and correct, same style as your LSTM beam).
        Returns (B, max_len) padded with PAD.
        """
        self.eval()
        device = memory.device
        B = memory.size(0)

        pad_id = self.cfg.pad_id
        sos_id = self.cfg.sos_id
        eos_id = self.cfg.eos_id
        unk_id = self.cfg.unk_id

        mem_key_padding_all = None
        if memory_mask is not None:
            mem_key_padding_all = ~memory_mask  # (B,T) True=pad

        def length_norm(score: float, length: int) -> float:
            return score / ((max(1, length)) ** alpha)

        final = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)

        for b in range(B):
            mem_b = memory[b : b + 1]  # (1,T,D)
            mem_kpad_b = None
            if mem_key_padding_all is not None:
                mem_kpad_b = mem_key_padding_all[b : b + 1]  # (1,T)

            beams: list[tuple[list[int], float, bool]] = [
                ([sos_id], 0.0, False)
            ]

            for step in range(max_len):
                candidates: list[tuple[list[int], float, bool]] = []

                for seq, score, done in beams:
                    if done:
                        candidates.append((seq, score, True))
                        continue

                    ys = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1,L)
                    L = ys.size(1)

                    x = self.embed(ys)
                    x = self.pos(x)

                    tgt_mask = self._causal_mask(L, device)
                    tgt_kpad = self._key_padding_mask(ys, pad_id)

                    out = self.decoder(
                        tgt=x,
                        memory=mem_b,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_kpad,
                        memory_key_padding_mask=mem_kpad_b,
                    )  # (1,L,D)

                    logits = self.out(out[:, -1, :])  # (1,V)

                    if forbid_pad:
                        logits[:, pad_id] = float("-inf")
                    if forbid_unk and unk_id is not None:
                        logits[:, unk_id] = float("-inf")
                    if step < min_len:
                        logits[:, eos_id] = float("-inf")

                    # repetition penalty (beam-local)
                    if repetition_penalty is not None and repetition_penalty > 1.0 and len(seq) > 1:
                        used = set(seq[1:])  # exclude SOS
                        for tok in used:
                            if tok == pad_id:
                                continue
                            if unk_id is not None and tok == unk_id:
                                continue
                            if logits[0, tok] > 0:
                                logits[0, tok] /= repetition_penalty
                            else:
                                logits[0, tok] *= repetition_penalty

                    # no-repeat ngram (beam-local)
                    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 1 and len(seq) > 1:
                        banned = self._get_banned_tokens(seq[1:], int(no_repeat_ngram_size))
                        for tok in banned:
                            logits[0, tok] = float("-inf")

                    logp = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)
                    topk = min(beam_size, logp.numel())
                    topk_logp, topk_ids = torch.topk(logp, k=topk)

                    for lp, tok in zip(topk_logp.tolist(), topk_ids.tolist()):
                        new_seq = seq + [int(tok)]
                        new_score = score + float(lp)
                        new_done = (tok == eos_id)
                        candidates.append((new_seq, new_score, new_done))

                candidates.sort(
                    key=lambda x: length_norm(x[1], len(x[0]) - 1),
                    reverse=True,
                )
                beams = candidates[:beam_size]

                if all(done for (_, _, done) in beams):
                    break

            finished = [(s, sc) for (s, sc, done) in beams if done]
            if finished:
                best_seq, _ = max(finished, key=lambda x: length_norm(x[1], len(x[0]) - 1))
            else:
                best_seq = beams[0][0]

            best_seq = best_seq[1:]  # drop SOS
            if eos_id in best_seq:
                eos_pos = best_seq.index(eos_id)
                best_seq = best_seq[: eos_pos + 1]

            out_len = min(max_len, len(best_seq))
            final[b, :out_len] = torch.tensor(best_seq[:out_len], dtype=torch.long, device=device)

        if not pad_to_max_len:
            pass

        return final