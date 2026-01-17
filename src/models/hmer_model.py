# src/models/hmer_model.py
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cnn_encoder import CNNEncoder, CNNEncoderConfig
from src.models.decoder_lstm import LSTMDecoder, LSTMDecoderConfig


class HMERModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        sos_id: int,
        eos_id: int,
        encoder_d_model: int = 256,
        decoder_hidden: int = 256,
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            CNNEncoderConfig(in_channels=1, d_model=encoder_d_model)
        )

        dec_cfg = LSTMDecoderConfig(
            vocab_size=vocab_size,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            d_model=encoder_d_model,
            emb_dim=encoder_d_model,
            hidden_size=decoder_hidden,
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
        images: (B,1,128,Wmax)
        input_ids: (B,L)
        image_widths: (B,) true widths before padding (recommended)
        """
        memory, memory_mask = self.encoder(images, image_widths=image_widths)  # (B,T,D), (B,T) or None
        logits = self.decoder(input_ids, memory, memory_mask=memory_mask)      # (B,L,V)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        image_widths: torch.Tensor | None = None,
        max_len: int = 160,
    ) -> torch.Tensor:
        """
        Greedy decode from images.
        Returns token ids: (B, L_pred)
        """
        self.eval()
        memory, memory_mask = self.encoder(images, image_widths=image_widths)
        pred_ids = self.decoder.greedy_decode(
            memory=memory,
            memory_mask=memory_mask,
            max_len=max_len,
        )
        return pred_ids