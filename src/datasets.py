"""Dataset utilities for sequence modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class FeaturePack:
    feature_cols: List[str]
    target_col: str = "target_return_next"
    aux_target_cols: List[str] | None = None
    date_col: str = "trade_date"


def _all_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "timestamp",
        "trade_date",
        "target_return_next",
        "target_future_realized_vol_12",
        "target_future_vol_regime_12",
        "target_future_downside_semivariance_12",
        "future_return_sign",
        "simple_return",
        "price_change",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def default_feature_columns(df: pd.DataFrame, cfg: dict | None = None) -> List[str]:
    feature_cols = _all_numeric_feature_columns(df)
    subset_mode = str((cfg or {}).get("features", {}).get("model_feature_subset", "all")).lower()
    if subset_mode in {"", "all"}:
        return feature_cols

    if subset_mode != "clean_trend":
        raise ValueError(f"Unknown features.model_feature_subset: {subset_mode}")

    keep_exact = {
        "log_return",
        "ewm_vol_78",
        "ewm_vol_390",
        "ema_12",
        "ema_24",
        "ema_32",
        "ema_78",
        "ema_96",
        "price_to_sma_24",
        "price_to_sma_78",
        "ema_slope_12",
        "ema_slope_24",
        "ema_slope_78",
        "macd_32_96",
        "macd_32_96_norm",
        "macd_32_96_signal",
        "macd_32_96_hist",
        "macd_32_96_hist_norm",
        "macd_32_96_slope",
        "macd_32_96_slope_norm",
        "macd_32_96_abs_norm",
        "macd_32_96_sign",
        "macd_32_96_cross_up",
        "macd_32_96_cross_down",
        "atr_14",
        "atr_78",
        "sin_time",
        "cos_time",
        "sin_day_of_week",
        "cos_day_of_week",
        "time_to_close",
        "trade_allowed",
    }
    keep_prefixes = (
        "ret_",
        "rolling_vol_",
    )
    selected = [c for c in feature_cols if c in keep_exact or c.startswith(keep_prefixes)]
    if not selected:
        raise ValueError("Feature subset 'clean_trend' selected zero columns.")
    return selected


class SequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features: FeaturePack,
        sequence_length: int,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = features.feature_cols
        self.target_col = features.target_col
        self.aux_target_cols = list(features.aux_target_cols or [])
        self.sequence_length = sequence_length

        x = self.df[self.feature_cols].astype(np.float32).values
        y = self.df[self.target_col].astype(np.float32).values
        aux_y = (
            self.df[self.aux_target_cols].astype(np.float32).values
            if self.aux_target_cols
            else np.empty((len(self.df), 0), dtype=np.float32)
        )
        self.x = x
        self.y = y
        self.aux_y = aux_y
        self.indices = list(range(sequence_length - 1, len(self.df) - 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        end_idx = self.indices[idx]
        start_idx = end_idx - self.sequence_length + 1
        x = self.x[start_idx : end_idx + 1]
        target_idx = end_idx + 1
        target = {
            "main": torch.tensor(self.y[target_idx], dtype=torch.float32),
        }
        if self.aux_target_cols:
            target["aux"] = torch.from_numpy(self.aux_y[target_idx])
        return torch.from_numpy(x), target
