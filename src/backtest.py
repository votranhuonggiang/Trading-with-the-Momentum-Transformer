"""Backtest engine for intraday VN30F1M strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from costs import cost_points, net_pnl_points
from metrics import summarize


@dataclass
class ThresholdConfig:
    entry_threshold: float
    exit_threshold: float
    reverse_threshold: float


def _threshold_position(signal: pd.Series, allowed: pd.Series, cfg: ThresholdConfig) -> pd.Series:
    out = np.zeros(len(signal), dtype=float)
    current = 0.0
    for i, val in enumerate(signal.fillna(0.0).values):
        can_open = bool(allowed.iloc[i])
        if current == 0.0:
            if can_open and val > cfg.entry_threshold:
                current = float(np.clip(val, 0.0, 1.0))
            elif can_open and val < -cfg.entry_threshold:
                current = float(np.clip(val, -1.0, 0.0))
        elif current > 0.0:
            if abs(val) < cfg.exit_threshold:
                current = 0.0
            elif val < -cfg.reverse_threshold and can_open:
                current = float(np.clip(val, -1.0, 0.0))
            else:
                current = float(np.clip(val, 0.0, 1.0))
        else:
            if abs(val) < cfg.exit_threshold:
                current = 0.0
            elif val > cfg.reverse_threshold and can_open:
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
    round_trip_cost_points: float,
) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = raw_signal.clip(-1.0, 1.0).fillna(0.0)
    out["position"] = _threshold_position(out["signal"], out["trade_allowed"].astype(bool), thresholds)
    out["position"] = apply_intraday_close(out["position"], out["trade_date"], out["is_last_5min"])
    out["turnover"] = (out["position"] - out["position"].shift(1).fillna(0.0)).abs()
    out["cost_points"] = cost_points(out["position"], round_trip_cost_points)
    out["gross_pnl_points"] = out["position"].shift(1).fillna(0.0) * out["price_change"]
    out["net_pnl_points"] = net_pnl_points(out["position"], out["price_change"], round_trip_cost_points)
    out["gross_return"] = out["position"].shift(1).fillna(0.0) * out["simple_return"]
    out["cost_return"] = out["cost_points"] / out["close"].replace(0, np.nan)
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
