# src/models/cnn_encoder.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class CNNEncoderConfig:
    in_channels: int = 1
    d_model: int = 256          # output feature size per timestep
    dropout: float = 0.1


class ConvBlock(nn.Module):
    """
    Conv -> BN -> ReLU
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNEncoder(nn.Module):
    """
    Input:
      images: (B, 1, 128, W)

    Output:
      memory: (B, T, D)
        - T corresponds to width after downsampling (W')
        - D = d_model
    """
    def __init__(self, cfg: CNNEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Downsample both H and W gradually.
        # We aim for a small H' (e.g., 4) while keeping W' as time steps.
        self.cnn = nn.Sequential(
            ConvBlock(cfg.in_channels, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),     # /2

            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),     # /4

            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),     # /8

            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),     # /16

            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # /32 on height only
        )

        # After CNN: (B, 256, H', W')
        # We flatten (C * H') into features per timestep and project to d_model.
        self.proj = nn.Linear(256 * 4, cfg.d_model)  # assumes H'=4 when input H=128
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 1, 128, W)
        return memory: (B, T, D)
        """
        feat = self.cnn(images)  # (B, C=256, H', W')
        b, c, h, w = feat.shape

        # Safety: if H' is not 4, we can still support it by adjusting dynamically.
        # But proj layer expects fixed input features. We'll handle mismatch here.
        if h != 4:
            # Recreate proj lazily (rare if you keep H=128)
            self.proj = nn.Linear(c * h, self.cfg.d_model).to(feat.device)

        # Convert feature map to sequence along width:
        # (B, C, H', W') -> (B, W', C*H')
        feat = feat.permute(0, 3, 1, 2).contiguous()      # (B, W', C, H')
        feat = feat.view(b, w, c * h)                     # (B, W', C*H')

        memory = self.proj(feat)                          # (B, W', d_model)
        memory = self.dropout(memory)
        return memory