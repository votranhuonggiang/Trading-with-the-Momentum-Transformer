"""Decoder-only Transformer model."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
        self.tanh = nn.Tanh()

    def _causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        m = torch.triu(torch.ones(t, t, device=device), diagonal=1).bool()
        return m

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        z = self.pos(z)
        mask = self._causal_mask(z.size(1), z.device)
        z = self.enc(z, mask=mask)
        return z[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_last = self.encode(x)
        out = self.head(z_last)
        return self.tanh(out).squeeze(-1)
