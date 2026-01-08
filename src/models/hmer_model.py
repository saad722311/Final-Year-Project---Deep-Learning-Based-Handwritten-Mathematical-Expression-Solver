# src/models/hmer_model.py
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cnn_encoder import CNNEncoder, CNNEncoderConfig
from src.models.decoder_lstm import LSTMDecoder, LSTMDecoderConfig


class HMERModel(nn.Module):
    """
    End-to-end recognition model:
      images -> CNN encoder -> memory
      input_ids + memory -> decoder -> logits
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        encoder_d_model: int = 256,
        decoder_hidden: int = 256,
    ):
        super().__init__()

        self.encoder = CNNEncoder(CNNEncoderConfig(in_channels=1, d_model=encoder_d_model))

        dec_cfg = LSTMDecoderConfig(
            vocab_size=vocab_size,
            pad_id=pad_id,
            d_model=encoder_d_model,
            emb_dim=encoder_d_model,
            hidden_size=decoder_hidden,
        )
        self.decoder = LSTMDecoder(dec_cfg)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        images: (B, 1, 128, W)
        input_ids: (B, L)
        return logits: (B, L, vocab_size)
        """
        memory = self.encoder(images)          # (B, T, D)
        logits = self.decoder(input_ids, memory)  # (B, L, V)
        return logits