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


def _pool_out_len(L: torch.Tensor, kernel: int, stride: int, pad: int = 0, dilation: int = 1) -> torch.Tensor:
    """
    Output length formula used by PyTorch pooling/conv:
      out = floor((L + 2*pad - dilation*(kernel-1) - 1)/stride + 1)
    """
    return torch.floor((L + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1).to(torch.long)


class CNNEncoder(nn.Module):
    """
    Input:
      images: (B, 1, 128, W)
      image_widths: (B,) original widths BEFORE padding (important for mask)

    Output:
      memory: (B, T, D)
      memory_mask: (B, T) bool, True = valid timesteps, False = padding timesteps
    """
    def __init__(self, cfg: CNNEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.cnn = nn.Sequential(
            ConvBlock(cfg.in_channels, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),     # /2 (H,W)

            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),     # /4

            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),     # /8

            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),     # /16

            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # /32 on height only (W unchanged)
        )

        # After CNN: (B, 256, H', W')
        self.proj = nn.Linear(256 * 4, cfg.d_model)  # assumes H'=4 when input H=128
        self.dropout = nn.Dropout(cfg.dropout)

    def _compute_time_lengths(self, image_widths: torch.Tensor) -> torch.Tensor:
        """
        Compute pooled width after the 4 (2,2) maxpools.
        Uses the exact PyTorch formula for MaxPool2d with kernel=2, stride=2, pad=0.
        """
        w = image_widths.to(torch.long)

        # Four pools that downsample width by ~2 each time (exact floor formula)
        for _ in range(4):
            w = _pool_out_len(w, kernel=2, stride=2, pad=0, dilation=1)

        # Final pool is (2,1) so width unchanged.
        return torch.clamp(w, min=1)

    def forward(
        self,
        images: torch.Tensor,
        image_widths: torch.Tensor | None = None,
    ):
        """
        images: (B, 1, 128, Wmax)
        image_widths: (B,) true widths BEFORE padding (optional but recommended)

        Returns:
          memory: (B, T, D)
          memory_mask: (B, T) bool (or None if image_widths not provided)
        """
        feat = self.cnn(images)  # (B, C=256, H', W')
        b, c, h, w = feat.shape

        # If H' differs, adapt projection (rare if H=128)
        if h != 4:
            self.proj = nn.Linear(c * h, self.cfg.d_model).to(feat.device)

        # (B, C, H', W') -> (B, W', C*H')
        feat = feat.permute(0, 3, 1, 2).contiguous()   # (B, W', C, H')
        feat = feat.view(b, w, c * h)                  # (B, W', C*H')

        memory = self.proj(feat)                       # (B, W', d_model)
        memory = self.dropout(memory)

        memory_mask = None
        if image_widths is not None:
            t_lens = self._compute_time_lengths(image_widths).to(device=memory.device)  # (B,)
            T = memory.size(1)
            # mask True = valid, False = padded steps
            ar = torch.arange(T, device=memory.device).unsqueeze(0)  # (1, T)
            memory_mask = ar < t_lens.unsqueeze(1)                   # (B, T)

        return memory, memory_mask