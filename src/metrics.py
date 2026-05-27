"""Performance metric utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) else 0.0


def annualization_factor_from_5min() -> float:
    return float(np.sqrt(243 * 252))


def sharpe(returns: pd.Series) -> float:
    r = returns.fillna(0.0)
    if r.std() == 0:
        return 0.0
    return float((r.mean() / r.std()) * annualization_factor_from_5min())


def sortino(returns: pd.Series) -> float:
    r = returns.fillna(0.0)
    downside = r[r < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float((r.mean() / downside_std) * annualization_factor_from_5min())


def max_drawdown(cum_returns: pd.Series) -> float:
    peak = cum_returns.cummax()
    dd = cum_returns / peak - 1.0
    return float(dd.min())


def summarize(returns: pd.Series, pnl_points: pd.Series, trades: pd.Series) -> dict:
    r = returns.fillna(0.0)
    cum = (1.0 + r).cumprod()
    ann_factor = 243 * 252
    ann_return = float((1.0 + r.mean()) ** ann_factor - 1.0)
    ann_vol = float(r.std() * np.sqrt(ann_factor))
    mdd = max_drawdown(cum)
    trade_pnls = pnl_points[trades > 0]
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    profit_factor = _safe_div(wins.sum(), abs(losses.sum()))

    return {
        "total_return": float(cum.iloc[-1] - 1.0 if len(cum) else 0.0),
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe(r),
        "sortino_ratio": sortino(r),
        "max_drawdown": mdd,
        "calmar_ratio": _safe_div(ann_return, abs(mdd)),
        "hit_rate": float((r > 0).mean()),
        "profit_factor": float(profit_factor),
        "average_trade_pnl": float(trade_pnls.mean() if len(trade_pnls) else 0.0),
        "average_win": float(wins.mean() if len(wins) else 0.0),
        "average_loss": float(losses.mean() if len(losses) else 0.0),
        "win_loss_ratio": _safe_div(wins.mean() if len(wins) else 0.0, abs(losses.mean()) if len(losses) else 0.0),
        "number_of_trades": int((trades > 0).sum()),
        "turnover": float(trades.sum()),
    }
