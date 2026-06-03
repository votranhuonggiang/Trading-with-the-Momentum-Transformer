"""Training utilities for deep learning models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.decoder_tft import DecoderTft
from models.decoder_transformer import DecoderTransformer
from models.lstm_dmn import LstmDmn


@dataclass
class TrainConfig:
    model_name: str
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 128
    hidden_size: int = 64
    num_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.2
    turnover_penalty_lambda: float = 0.0
    valid_selection_turnover_lambda: float = 0.0
    device: str = "cpu"


def sharpe_loss(
    positions: torch.Tensor,
    future_returns: torch.Tensor,
    turnover_penalty_lambda: float = 0.0,
) -> torch.Tensor:
    captured = positions * future_returns
    mean = captured.mean()
    var = captured.var(unbiased=False)
    sharpe = mean / torch.sqrt(var + 1e-9)
    loss = -sharpe
    if turnover_penalty_lambda > 0 and positions.numel() > 1:
        turnover_proxy = torch.mean(torch.abs(positions[1:] - positions[:-1]))
        loss = loss + turnover_penalty_lambda * turnover_proxy
    return loss


def model_factory(input_size: int, cfg: TrainConfig) -> nn.Module:
    if cfg.model_name == "lstm_dmn":
        return LstmDmn(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    if cfg.model_name == "decoder_transformer":
        return DecoderTransformer(
            input_size=input_size,
            d_model=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_layers=max(cfg.num_layers, 1),
            dim_feedforward=cfg.hidden_size * 4,
            dropout=cfg.dropout,
        )
    if cfg.model_name == "decoder_tft":
        return DecoderTft(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown model name: {cfg.model_name}")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    turnover_penalty_lambda: float = 0.0,
) -> float:
    losses = []
    model.train(optimizer is not None)
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = sharpe_loss(pred, yb, turnover_penalty_lambda=turnover_penalty_lambda)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def _validation_score(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    turnover_penalty_lambda: float = 0.0,
) -> tuple[float, float, float]:
    losses = []
    turnover_terms = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = sharpe_loss(pred, yb, turnover_penalty_lambda=0.0)
            turnover_proxy = 0.0
            if pred.numel() > 1:
                turnover_proxy = float(torch.mean(torch.abs(pred[1:] - pred[:-1])).detach().cpu().item())
            losses.append(float(loss.detach().cpu().item()))
            turnover_terms.append(turnover_proxy)
    valid_loss = float(np.mean(losses)) if losses else 0.0
    valid_turnover_proxy = float(np.mean(turnover_terms)) if turnover_terms else 0.0
    valid_score = valid_loss + turnover_penalty_lambda * valid_turnover_proxy
    return valid_loss, valid_turnover_proxy, valid_score


def fit_model(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    input_size: int,
    cfg: TrainConfig,
) -> tuple[nn.Module, Dict[str, Iterable[float]]]:
    model = model_factory(input_size, cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_state = None
    best_valid_score = float("inf")
    history = {"train_loss": [], "valid_loss": [], "valid_turnover_proxy": [], "valid_score": []}

    for _ in range(cfg.epochs):
        tr = _run_epoch(
            model,
            train_loader,
            optimizer,
            cfg.device,
            turnover_penalty_lambda=cfg.turnover_penalty_lambda,
        )
        va, va_turnover_proxy, va_score = _validation_score(
            model,
            valid_loader,
            cfg.device,
            turnover_penalty_lambda=cfg.valid_selection_turnover_lambda,
        )
        history["train_loss"].append(tr)
        history["valid_loss"].append(va)
        history["valid_turnover_proxy"].append(va_turnover_proxy)
        history["valid_score"].append(va_score)
        if va_score < best_valid_score:
            best_valid_score = va_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
