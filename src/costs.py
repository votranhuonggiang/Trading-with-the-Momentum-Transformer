"""Transaction cost utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def one_way_cost(round_trip_cost_points: float) -> float:
    return round_trip_cost_points / 2.0


def turnover(position: pd.Series) -> pd.Series:
    prev = position.shift(1).fillna(0.0)
    return (position - prev).abs()


def cost_points(position: pd.Series, round_trip_cost_points: float) -> pd.Series:
    return turnover(position) * one_way_cost(round_trip_cost_points)


def net_pnl_points(
    position: pd.Series, price_change: pd.Series, round_trip_cost_points: float
) -> pd.Series:
    gross = position.shift(1).fillna(0.0) * price_change.fillna(0.0)
    costs = cost_points(position, round_trip_cost_points)
    return gross - costs


def net_return(
    position: pd.Series,
    simple_return: pd.Series,
    round_trip_cost_points: float,
    close_price: pd.Series,
) -> pd.Series:
    gross = position.shift(1).fillna(0.0) * simple_return.fillna(0.0)
    costs = cost_points(position, round_trip_cost_points) / close_price.replace(0, np.nan)
    return (gross - costs.fillna(0.0)).fillna(0.0)
