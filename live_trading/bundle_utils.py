"""Utilities for packaging and loading decoder_tft live-trading bundles."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class BundleManifest:
    bundle_name: str
    bundle_version: str
    created_at_utc: str
    model_name: str
    config_file: str
    checkpoint_file: str | None
    feature_columns_file: str
    scaler_file: str
    execution_policy_file: str
    training_summary_file: str | None
    timeframe: str
    timestamp_col: str
    sequence_length: int
    hidden_size: int
    num_heads: int
    dropout: float
    target_horizon_bars: int
    multitask_aux_vol_loss_weight: float
    multitask_regime_loss_weight: float
    multitask_downside_loss_weight: float
    dual_position_heads: bool
    integer_contract_execution: bool
    max_contracts: int
    contract_rounding_mode: str
    notes: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: BundleManifest) -> None:
    save_json(path, asdict(manifest))


def load_manifest(path: Path) -> BundleManifest:
    return BundleManifest(**load_json(path))


def scaler_stats_from_frame(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    mu = df[feature_cols].mean()
    sigma = df[feature_cols].std().replace(0, 1.0).fillna(1.0)
    return {
        col: {
            "mean": float(mu[col]),
            "std": float(sigma[col]),
        }
        for col in feature_cols
    }


def scale_feature_frame(df: pd.DataFrame, scaler_stats: dict[str, dict[str, float]], feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        stats = scaler_stats[col]
        out[col] = (out[col] - stats["mean"]) / stats["std"]
    return out.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)


def copy_if_present(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_execution_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    trading = cfg.get("trading", {})
    return {
        "long_entry_threshold": float(trading["long_entry_threshold"]),
        "short_entry_threshold": float(trading["short_entry_threshold"]),
        "long_exit_threshold": float(trading["long_exit_threshold"]),
        "short_exit_threshold": float(trading["short_exit_threshold"]),
        "long_to_short_reverse_threshold": float(trading["long_to_short_reverse_threshold"]),
        "short_to_long_reverse_threshold": float(trading["short_to_long_reverse_threshold"]),
        "decoder_tft_regime_conditioned_exposure": bool(
            trading.get("decoder_tft_regime_conditioned_exposure", False)
        ),
        "decoder_tft_low_vol_scale": float(trading.get("decoder_tft_low_vol_scale", 1.0)),
        "decoder_tft_high_vol_scale": float(trading.get("decoder_tft_high_vol_scale", 1.0)),
        "decoder_tft_confidence_scaled_exposure": bool(
            trading.get("decoder_tft_confidence_scaled_exposure", False)
        ),
        "decoder_tft_confidence_gate_tau": float(trading.get("decoder_tft_confidence_gate_tau", 0.0)),
        "decoder_tft_confidence_scale_alpha": float(trading.get("decoder_tft_confidence_scale_alpha", 1.0)),
        "decoder_tft_confidence_scaled_max_position": float(
            trading.get("decoder_tft_confidence_scaled_max_position", 1.0)
        ),
        "decoder_tft_asymmetric_regime_scaling": bool(
            trading.get("decoder_tft_asymmetric_regime_scaling", False)
        ),
        "decoder_tft_low_vol_long_scale": float(trading.get("decoder_tft_low_vol_long_scale", 1.0)),
        "decoder_tft_high_vol_long_scale": float(trading.get("decoder_tft_high_vol_long_scale", 1.0)),
        "decoder_tft_low_vol_short_scale": float(trading.get("decoder_tft_low_vol_short_scale", 1.0)),
        "decoder_tft_high_vol_short_scale": float(trading.get("decoder_tft_high_vol_short_scale", 1.0)),
        "integer_contract_execution": bool(trading.get("integer_contract_execution", False)),
        "max_contracts": int(trading.get("max_contracts", 1)),
        "contract_rounding_mode": str(trading.get("contract_rounding_mode", "nearest")),
        "intraday_only": bool(trading.get("intraday_only", True)),
        "close_before_end_of_day": bool(trading.get("close_before_end_of_day", True)),
        "no_overnight": bool(trading.get("no_overnight", True)),
    }
