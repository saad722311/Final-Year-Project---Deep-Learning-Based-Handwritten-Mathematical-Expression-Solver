from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNNEncoder(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 128
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 64
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32
            nn.Conv2d(128, d_model, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        # x: (B,1,H,W)
        f = self.net(x)               # (B,d_model,H',W')
        B, C, H, W = f.shape
        seq = f.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        return seq


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class E2ETransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.encoder_cnn = SmallCNNEncoder(d_model=d_model)
        self.enc_pos = PositionalEncoding(d_model)

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.dec_pos = PositionalEncoding(d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T: int, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, images, y_in):
        """
        images: (B,1,H,W)
        y_in:   (B,Ty) input tokens (teacher forcing), usually shifted right
        returns logits: (B,Ty,V)
        """
        enc = self.encoder_cnn(images)      # (B,Te,d)
        enc = self.enc_pos(enc)

        dec = self.tok_emb(y_in)            # (B,Ty,d)
        dec = self.dec_pos(dec)

        T = y_in.size(1)
        tgt_mask = self._causal_mask(T, y_in.device)

        # key padding mask for decoder input
        tgt_key_padding = (y_in == self.pad_id)

        h = self.decoder(
            tgt=dec,
            memory=enc,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding,
        )
        return self.out(h)