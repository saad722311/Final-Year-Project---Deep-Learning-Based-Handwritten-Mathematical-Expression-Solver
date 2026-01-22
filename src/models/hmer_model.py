# src/models/hmer_model.py
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cnn_encoder import CNNEncoder, CNNEncoderConfig
from src.models.decoder_lstm import LSTMDecoder, LSTMDecoderConfig

# Optional transformer decoder (Day 7)
try:
    from src.models.decoder_transformer import TransformerDecoder, TransformerDecoderConfig
except Exception:
    TransformerDecoder = None
    TransformerDecoderConfig = None


class HMERModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        sos_id: int,
        eos_id: int,
        unk_id: int | None = None,
        encoder_d_model: int = 256,
        decoder_hidden: int = 256,
        # ✅ NEW: choose decoder by config
        decoder_type: str = "lstm",     # "lstm" | "transformer"
        # transformer params (safe defaults)
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.decoder_type = decoder_type

        self.encoder = CNNEncoder(CNNEncoderConfig(in_channels=1, d_model=encoder_d_model))

        if decoder_type == "transformer":
            if TransformerDecoder is None or TransformerDecoderConfig is None:
                raise RuntimeError(
                    "decoder_type='transformer' but src/models/decoder_transformer.py is missing or failed to import."
                )

            dec_cfg = TransformerDecoderConfig(
                vocab_size=vocab_size,
                pad_id=pad_id,
                sos_id=sos_id,
                eos_id=eos_id,
                unk_id=unk_id,
                d_model=encoder_d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                ff_dim=ff_dim,
                dropout=dropout,
                max_len=max_len,
            )
            self.decoder = TransformerDecoder(dec_cfg)
        else:
            # default: LSTM decoder (your current baseline)
            dec_cfg = LSTMDecoderConfig(
                vocab_size=vocab_size,
                pad_id=pad_id,
                sos_id=sos_id,
                eos_id=eos_id,
                unk_id=unk_id,
                d_model=encoder_d_model,
                emb_dim=encoder_d_model,
                hidden_size=decoder_hidden,
                dropout=dropout,
            )
            self.decoder = LSTMDecoder(dec_cfg)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        image_widths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Training forward (teacher forcing).
        """
        memory, memory_mask = self.encoder(images, image_widths=image_widths)
        logits = self.decoder(input_ids, memory, memory_mask=memory_mask)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        image_widths: torch.Tensor | None = None,
        max_len: int = 160,
        decode: str = "greedy",      # "greedy" | "beam"
        beam_size: int = 5,
        alpha: float = 0.6,
        min_len: int = 1,
        repetition_penalty: float = 1.10,
        no_repeat_ngram_size: int = 3,
        forbid_unk: bool = True,
    ) -> torch.Tensor:
        """
        Decode from images.
        """
        self.eval()
        memory, memory_mask = self.encoder(images, image_widths=image_widths)

        # LSTM decoder: supports greedy + beam
        if hasattr(self.decoder, "beam_decode") and decode == "beam":
            return self.decoder.beam_decode(
                memory=memory,
                memory_mask=memory_mask,
                max_len=max_len,
                beam_size=beam_size,
                alpha=alpha,
                min_len=min_len,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                forbid_unk=forbid_unk,
            )

        # fallback greedy (works for LSTM + Transformer)
        return self.decoder.greedy_decode(
            memory=memory,
            memory_mask=memory_mask,
            max_len=max_len,
            min_len=min_len,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            forbid_unk=forbid_unk,
        )