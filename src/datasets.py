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
    date_col: str = "trade_date"


def default_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "timestamp",
        "trade_date",
        "target_return_next",
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
        self.sequence_length = sequence_length

        x = self.df[self.feature_cols].astype(np.float32).values
        y = self.df[self.target_col].astype(np.float32).values
        self.x = x
        self.y = y
        self.indices = list(range(sequence_length - 1, len(self.df) - 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end_idx = self.indices[idx]
        start_idx = end_idx - self.sequence_length + 1
        x = self.x[start_idx : end_idx + 1]
        y = self.y[end_idx + 1]  # next-step return target
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
