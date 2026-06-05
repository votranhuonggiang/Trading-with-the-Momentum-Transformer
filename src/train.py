"""Training utilities for deep learning models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
    valid_selection_buy_turnover_lambda: float = 0.0
    valid_selection_sell_turnover_lambda: float = 0.0
    multitask_aux_loss_weight: float = 0.0
    multitask_regime_loss_weight: float = 0.0
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


def _split_prediction(model_out: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(model_out, dict):
        return model_out["position"], model_out
    return model_out, {}


def _total_loss(
    model_out: torch.Tensor | dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    cfg: TrainConfig,
    apply_turnover_penalty: bool,
) -> torch.Tensor:
    positions, extras = _split_prediction(model_out)
    loss = sharpe_loss(
        positions,
        target["main"],
        turnover_penalty_lambda=cfg.turnover_penalty_lambda if apply_turnover_penalty else 0.0,
    )
    aux_target = target.get("aux")
    if aux_target is not None and extras:
        if cfg.multitask_aux_loss_weight > 0.0 and "future_vol" in extras:
            loss = loss + cfg.multitask_aux_loss_weight * F.mse_loss(extras["future_vol"], aux_target[:, 0])
        if cfg.multitask_regime_loss_weight > 0.0 and "future_vol_regime_logit" in extras:
            loss = loss + cfg.multitask_regime_loss_weight * F.binary_cross_entropy_with_logits(
                extras["future_vol_regime_logit"],
                aux_target[:, 1],
            )
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
            multitask=(cfg.multitask_aux_loss_weight > 0.0 or cfg.multitask_regime_loss_weight > 0.0),
        )
    raise ValueError(f"Unknown model name: {cfg.model_name}")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    cfg: TrainConfig,
) -> float:
    losses = []
    model.train(optimizer is not None)
    for xb, yb in loader:
        xb = xb.to(cfg.device)
        target = {"main": yb["main"].to(cfg.device)}
        if "aux" in yb:
            target["aux"] = yb["aux"].to(cfg.device)
        pred = model(xb)
        loss = _total_loss(
            pred,
            target,
            cfg=cfg,
            apply_turnover_penalty=True,
        )
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
    cfg: TrainConfig,
    turnover_penalty_lambda: float = 0.0,
    buy_turnover_penalty_lambda: float = 0.0,
    sell_turnover_penalty_lambda: float = 0.0,
) -> tuple[float, float, float, float, float]:
    losses = []
    turnover_terms = []
    buy_turnover_terms = []
    sell_turnover_terms = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            target = {"main": yb["main"].to(cfg.device)}
            if "aux" in yb:
                target["aux"] = yb["aux"].to(cfg.device)
            pred = model(xb)
            loss = _total_loss(
                pred,
                target,
                cfg=cfg,
                apply_turnover_penalty=False,
            )
            positions, _ = _split_prediction(pred)
            turnover_proxy = 0.0
            buy_turnover_proxy = 0.0
            sell_turnover_proxy = 0.0
            if positions.numel() > 1:
                delta = positions[1:] - positions[:-1]
                turnover_proxy = float(torch.mean(torch.abs(delta)).detach().cpu().item())
                buy_turnover_proxy = float(torch.mean(torch.clamp(delta, min=0.0)).detach().cpu().item())
                sell_turnover_proxy = float(torch.mean(torch.clamp(-delta, min=0.0)).detach().cpu().item())
            losses.append(float(loss.detach().cpu().item()))
            turnover_terms.append(turnover_proxy)
            buy_turnover_terms.append(buy_turnover_proxy)
            sell_turnover_terms.append(sell_turnover_proxy)
    valid_loss = float(np.mean(losses)) if losses else 0.0
    valid_turnover_proxy = float(np.mean(turnover_terms)) if turnover_terms else 0.0
    valid_buy_turnover_proxy = float(np.mean(buy_turnover_terms)) if buy_turnover_terms else 0.0
    valid_sell_turnover_proxy = float(np.mean(sell_turnover_terms)) if sell_turnover_terms else 0.0
    valid_score = valid_loss + turnover_penalty_lambda * valid_turnover_proxy
    valid_score = (
        valid_score
        + buy_turnover_penalty_lambda * valid_buy_turnover_proxy
        + sell_turnover_penalty_lambda * valid_sell_turnover_proxy
    )
    return (
        valid_loss,
        valid_turnover_proxy,
        valid_buy_turnover_proxy,
        valid_sell_turnover_proxy,
        valid_score,
    )


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
    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_turnover_proxy": [],
        "valid_buy_turnover_proxy": [],
        "valid_sell_turnover_proxy": [],
        "valid_score": [],
    }

    for _ in range(cfg.epochs):
        tr = _run_epoch(
            model,
            train_loader,
            optimizer,
            cfg,
        )
        va, va_turnover_proxy, va_buy_turnover_proxy, va_sell_turnover_proxy, va_score = _validation_score(
            model,
            valid_loader,
            cfg,
            turnover_penalty_lambda=cfg.valid_selection_turnover_lambda,
            buy_turnover_penalty_lambda=cfg.valid_selection_buy_turnover_lambda,
            sell_turnover_penalty_lambda=cfg.valid_selection_sell_turnover_lambda,
        )
        history["train_loss"].append(tr)
        history["valid_loss"].append(va)
        history["valid_turnover_proxy"].append(va_turnover_proxy)
        history["valid_buy_turnover_proxy"].append(va_buy_turnover_proxy)
        history["valid_sell_turnover_proxy"].append(va_sell_turnover_proxy)
        history["valid_score"].append(va_score)
        if va_score < best_valid_score:
            best_valid_score = va_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
