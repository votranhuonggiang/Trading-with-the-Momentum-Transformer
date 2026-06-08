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
        multitask: bool = False,
        dual_position_heads: bool = False,
    ) -> None:
        super().__init__()
        self.multitask = multitask
        self.dual_position_heads = dual_position_heads
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
        self.position_head = nn.Linear(hidden_size, 1)
        self.fast_position_head = nn.Linear(hidden_size, 1)
        self.slow_position_head = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.vol_head = nn.Linear(hidden_size, 1)
        self.regime_head = nn.Linear(hidden_size, 1)
        self.downside_semivariance_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        xg = self.gate(x)
        h, _ = self.lstm(xg)
        attn_hidden = self.attn.encode(h)
        regime_logit = self.regime_head(h[:, -1, :]).squeeze(-1)
        if self.dual_position_heads:
            fast_position = self.tanh(self.fast_position_head(attn_hidden)).squeeze(-1)
            slow_position = self.tanh(self.slow_position_head(attn_hidden)).squeeze(-1)
            p_high_vol = torch.sigmoid(regime_logit)
            position = (1.0 - p_high_vol) * slow_position + p_high_vol * fast_position
        else:
            position = self.tanh(self.position_head(attn_hidden)).squeeze(-1)
        if not self.multitask:
            return position
        return {
            "position": position,
            "future_vol": self.vol_head(h[:, -1, :]).squeeze(-1),
            "future_vol_regime_logit": regime_logit,
            "future_downside_semivariance": self.downside_semivariance_head(h[:, -1, :]).squeeze(-1),
            "fast_position": fast_position if self.dual_position_heads else position,
            "slow_position": slow_position if self.dual_position_heads else position,
        }
