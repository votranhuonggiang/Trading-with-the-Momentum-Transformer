"""Decoder-only Temporal Fusion Transformer model."""

from __future__ import annotations

import torch
from torch import nn

from models.decoder_transformer import DecoderTransformer


class GatedFeatureBlock(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.sigmoid(self.gate(x))
        return x * g


class DecoderTft(nn.Module):
    """Practical TFT-style block: feature gating -> LSTM -> causal attention -> tanh."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gate = GatedFeatureBlock(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.attn = DecoderTransformer(
            input_size=hidden_size,
            d_model=hidden_size,
            num_heads=num_heads,
            num_layers=1,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xg = self.gate(x)
        h, _ = self.lstm(xg)
        return self.attn(h)
