"""Backtest engine for intraday VN30F1M strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from metrics import summarize


@dataclass
class ThresholdConfig:
    long_entry_threshold: float
    short_entry_threshold: float
    long_exit_threshold: float
    short_exit_threshold: float
    long_to_short_reverse_threshold: float
    short_to_long_reverse_threshold: float


def _round_away_from_zero(values: pd.Series) -> pd.Series:
    arr = values.astype(float).to_numpy()
    rounded = np.sign(arr) * np.floor(np.abs(arr) + 0.5)
    return pd.Series(rounded, index=values.index, dtype=float)


def _quantize_to_integer_contracts(position: pd.Series, trading_cfg: dict) -> pd.Series:
    if not bool(trading_cfg.get("integer_contract_execution", False)):
        return position.astype(float)

    max_contracts = int(trading_cfg.get("max_contracts", 1))
    if max_contracts <= 0:
        raise ValueError("trading.max_contracts must be positive when integer_contract_execution is enabled.")

    rounding_mode = str(trading_cfg.get("contract_rounding_mode", "nearest")).lower()
    scaled = position.astype(float) * max_contracts

    if rounding_mode == "nearest":
        quantized = _round_away_from_zero(scaled)
    elif rounding_mode == "floor":
        quantized = pd.Series(np.sign(scaled) * np.floor(np.abs(scaled)), index=position.index, dtype=float)
    elif rounding_mode == "ceil":
        quantized = pd.Series(np.sign(scaled) * np.ceil(np.abs(scaled)), index=position.index, dtype=float)
    else:
        raise ValueError(f"Unsupported contract_rounding_mode: {rounding_mode}")

    return quantized.clip(-max_contracts, max_contracts)


def _threshold_position(signal: pd.Series, allowed: pd.Series, cfg: ThresholdConfig) -> pd.Series:
    out = np.zeros(len(signal), dtype=float)
    current = 0.0
    for i, val in enumerate(signal.fillna(0.0).values):
        can_open = bool(allowed.iloc[i])
        if current == 0.0:
            if can_open and val > cfg.long_entry_threshold:
                current = float(np.clip(val, 0.0, 1.0))
            elif can_open and val < -cfg.short_entry_threshold:
                current = float(np.clip(val, -1.0, 0.0))
        elif current > 0.0:
            if val < cfg.long_exit_threshold:
                current = 0.0
            elif val < -cfg.long_to_short_reverse_threshold and can_open:
                current = float(np.clip(val, -1.0, 0.0))
            else:
                current = float(np.clip(val, 0.0, 1.0))
        else:
            if val > -cfg.short_exit_threshold:
                current = 0.0
            elif val > cfg.short_to_long_reverse_threshold and can_open:
                current = float(np.clip(val, 0.0, 1.0))
            else:
                current = float(np.clip(val, -1.0, 0.0))
        out[i] = current
    return pd.Series(out, index=signal.index)


def apply_intraday_close(position: pd.Series, trade_date: pd.Series, is_last_5min: pd.Series) -> pd.Series:
    pos = position.copy()
    must_flat = (is_last_5min == 1) | (trade_date != trade_date.shift(-1))
    pos[must_flat] = 0.0
    return pos


def run_strategy(
    df: pd.DataFrame,
    raw_signal: pd.Series,
    thresholds: ThresholdConfig,
    trading_cfg: dict,
) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = raw_signal.clip(-1.0, 1.0).fillna(0.0)
    out["position"] = _threshold_position(out["signal"], out["trade_allowed"].astype(bool), thresholds)
    out["position"] = apply_intraday_close(out["position"], out["trade_date"], out["is_last_5min"])
    out["position"] = _quantize_to_integer_contracts(out["position"], trading_cfg)

    contract_multiplier = float(trading_cfg.get("contract_multiplier", 100000.0))
    fee_vsdc = float(trading_cfg.get("fee_vsdc_vnd_per_contract_per_side", 5000.0))
    fee_hnx = float(trading_cfg.get("fee_hnx_vnd_per_contract_per_side", 2700.0))
    fee_ctck = float(trading_cfg.get("fee_ctck_vnd_per_contract_per_side", 2700.0))
    margin_rate = float(trading_cfg.get("margin_rate", 0.1848))
    transfer_tax_rate = float(trading_cfg.get("transfer_tax_rate", 0.001))

    out["position_prev"] = out["position"].shift(1).fillna(0.0)
    out["position_delta"] = out["position"] - out["position_prev"]
    out["turnover"] = out["position_delta"].abs()

    out["buy_units"] = (out["position"] - out["position_prev"]).clip(lower=0.0)
    out["sell_units"] = (out["position_prev"] - out["position"]).clip(lower=0.0)
    per_side_fixed_fee = fee_vsdc + fee_hnx + fee_ctck

    # Tax applies on sell/short-side turnover only. Buy/long-side turnover
    # pays fixed fees without tax.
    out["tax_vnd_per_contract"] = (
        margin_rate * contract_multiplier * transfer_tax_rate * out["close"]
    )
    out["buy_cost_vnd"] = out["buy_units"] * per_side_fixed_fee
    out["sell_fixed_cost_vnd"] = out["sell_units"] * per_side_fixed_fee
    out["sell_tax_cost_vnd"] = out["sell_units"] * out["tax_vnd_per_contract"]
    out["sell_cost_vnd"] = out["sell_fixed_cost_vnd"] + out["sell_tax_cost_vnd"]
    out["cost_vnd"] = out["buy_cost_vnd"] + out["sell_cost_vnd"]
    out["cost_points"] = out["cost_vnd"] / contract_multiplier

    out["gross_pnl_points"] = out["position"].shift(1).fillna(0.0) * out["price_change"]
    out["net_pnl_points"] = out["gross_pnl_points"] - out["cost_points"]
    out["gross_return"] = out["position"].shift(1).fillna(0.0) * out["simple_return"]
    out["cost_return"] = out["cost_vnd"] / (out["close"].replace(0, np.nan) * contract_multiplier)
    out["net_return"] = (out["gross_return"] - out["cost_return"]).fillna(0.0)
    return out


def baseline_long_only(df: pd.DataFrame) -> pd.Series:
    # True long-only intent: always long signal; execution engine still enforces
    # no-trade/open filters and forced end-of-day flattening.
    return pd.Series(1.0, index=df.index)


def baseline_tsmom(df: pd.DataFrame, lookback: int, threshold: float = 0.0) -> pd.Series:
    r = df[f"ret_{lookback}"]
    sig = np.where(r > threshold, 1.0, np.where(r < -threshold, -1.0, 0.0))
    return pd.Series(sig, index=df.index)


def baseline_ema_cross(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    fast_col = f"ema_{fast}"
    slow_col = f"ema_{slow}"
    sig = np.where(df[fast_col] > df[slow_col], 1.0, np.where(df[fast_col] < df[slow_col], -1.0, 0.0))
    return pd.Series(sig, index=df.index)


def baseline_macd(df: pd.DataFrame, short_w: int, long_w: int) -> pd.Series:
    col = f"macd_{short_w}_{long_w}"
    sig = np.where(df[col] > 0.0, 1.0, np.where(df[col] < 0.0, -1.0, 0.0))
    return pd.Series(sig, index=df.index)


def baseline_signals(df: pd.DataFrame) -> Dict[str, pd.Series]:
    signals: Dict[str, pd.Series] = {"long_only": baseline_long_only(df)}
    for lb in [12, 24, 48, 78]:
        signals[f"tsmom_{lb}"] = baseline_tsmom(df, lb)
    for fast, slow in [(8, 24), (16, 48), (32, 96)]:
        signals[f"ema_cross_{fast}_{slow}"] = baseline_ema_cross(df, fast, slow)
        signals[f"macd_{fast}_{slow}"] = baseline_macd(df, fast, slow)
    return signals


def cost_sensitivity(
    strat_df: pd.DataFrame,
    cost_scenarios: Iterable[float],
) -> pd.DataFrame:
    if "cost_vnd" in strat_df.columns:
        metrics = summarize(strat_df["net_return"], strat_df["net_pnl_points"], strat_df["turnover"])
        return pd.DataFrame(
            [
                {
                    "round_trip_cost_points": np.nan,
                    "net_total_return": metrics["total_return"],
                    "net_sharpe": metrics["sharpe_ratio"],
                    "net_max_drawdown": metrics["max_drawdown"],
                    "net_profit_factor": metrics["profit_factor"],
                    "net_average_trade_pnl": metrics["average_trade_pnl"],
                }
            ]
        )

    # Legacy fallback for flat-point cost model.
    from costs import cost_points, net_pnl_points

    rows = []
    for c in cost_scenarios:
        net_points = net_pnl_points(strat_df["position"], strat_df["price_change"], c)
        net_ret = (
            strat_df["position"].shift(1).fillna(0.0) * strat_df["simple_return"]
            - cost_points(strat_df["position"], c) / strat_df["close"].replace(0, np.nan)
        ).fillna(0.0)
        metrics = summarize(net_ret, net_points, strat_df["turnover"])
        rows.append(
            {
                "round_trip_cost_points": c,
                "net_total_return": metrics["total_return"],
                "net_sharpe": metrics["sharpe_ratio"],
                "net_max_drawdown": metrics["max_drawdown"],
                "net_profit_factor": metrics["profit_factor"],
                "net_average_trade_pnl": metrics["average_trade_pnl"],
            }
        )
    return pd.DataFrame(rows)
