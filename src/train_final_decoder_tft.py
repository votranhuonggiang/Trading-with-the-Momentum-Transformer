"""Train one final deployable decoder_tft model and write a live bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import abs_path, ensure_parent, load_config
from datasets import FeaturePack, SequenceDataset, default_feature_columns
from train import TrainConfig, fit_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_TRADING_DIR = PROJECT_ROOT / "live_trading"
if str(LIVE_TRADING_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_TRADING_DIR))

from bundle_utils import BundleManifest, build_execution_policy, save_json, save_manifest, utc_now_iso  # noqa: E402


def _scale_with_train_stats(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    mu = train_df[feature_cols].mean()
    sigma = train_df[feature_cols].std().replace(0, 1.0).fillna(1.0)

    tr = train_df.copy()
    va = valid_df.copy()
    tr[feature_cols] = (tr[feature_cols] - mu) / sigma
    va[feature_cols] = (va[feature_cols] - mu) / sigma
    tr[feature_cols] = tr[feature_cols].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    va[feature_cols] = va[feature_cols].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    scaler = {
        col: {"mean": float(mu[col]), "std": float(sigma[col])}
        for col in feature_cols
    }
    return tr, va, scaler


def _load_summary(outputs_root: Path, model_name: str) -> dict:
    summary: dict[str, object] = {"model": model_name}
    file_map = {
        "headline": outputs_root / "tables" / "model_comparison_metrics.csv",
        "trade_profile": outputs_root / "tables" / "trade_statistics.csv",
        "daily_profile": outputs_root / "tables" / "daily_metrics_summary.csv",
    }
    for key, path in file_map.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "model" not in df.columns:
            continue
        hit = df[df["model"] == model_name]
        if not hit.empty:
            summary[key] = hit.iloc[0].to_dict()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundle-name", default="decoder_tft_main_branch_5m")
    parser.add_argument("--bundle-root", default="live_trading/bundles")
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--outputs-run-label", default="vn30f1m_outputs_version1")
    args = parser.parse_args()

    if not 0.0 < args.valid_fraction < 0.5:
        raise ValueError("--valid-fraction must be between 0 and 0.5.")

    cfg = load_config(args.config)
    feat = pd.read_parquet(abs_path(cfg["data"]["features_path"])).sort_values(cfg["data"]["timestamp_col"]).reset_index(drop=True)
    feature_cols = default_feature_columns(feat, cfg)

    split_idx = int(len(feat) * (1.0 - args.valid_fraction))
    split_idx = max(split_idx, int(cfg["models"]["sequence_lengths"][0]) + 2)
    train_df = feat.iloc[:split_idx].copy()
    valid_df = feat.iloc[split_idx:].copy()
    if valid_df.empty:
        raise ValueError("Validation split is empty. Increase data or lower --valid-fraction.")

    aux_target_cols: list[str] = []
    requested_aux_target_cols = [
        "target_future_realized_vol_12",
        "target_future_vol_regime_12",
        "target_future_downside_semivariance_12",
    ]
    if (
        float(cfg.get("training", {}).get("decoder_tft_aux_vol_loss_weight", 0.0)) > 0.0
        or float(cfg.get("training", {}).get("decoder_tft_aux_regime_loss_weight", 0.0)) > 0.0
        or float(cfg.get("training", {}).get("decoder_tft_aux_downside_loss_weight", 0.0)) > 0.0
    ):
        if all(col in train_df.columns and col in valid_df.columns for col in requested_aux_target_cols):
            aux_target_cols = requested_aux_target_cols

    train_scaled, valid_scaled, scaler = _scale_with_train_stats(train_df, valid_df, feature_cols)

    target_col = "target_return_horizon" if "target_return_horizon" in train_scaled.columns else "target_return_next"
    fp = FeaturePack(feature_cols=feature_cols, target_col=target_col, aux_target_cols=aux_target_cols)
    seq_len = int(cfg["models"]["sequence_lengths"][0])
    train_ds = SequenceDataset(train_scaled, fp, sequence_length=seq_len)
    valid_ds = SequenceDataset(valid_scaled, fp, sequence_length=seq_len)
    if len(train_ds) == 0 or len(valid_ds) == 0:
        raise RuntimeError("Final training split produced an empty sequence dataset.")

    batch_size = int(cfg.get("training", {}).get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "auto" else "cpu"
    train_cfg = TrainConfig(
        model_name="decoder_tft",
        lr=float(cfg.get("training", {}).get("learning_rate", 1e-3)),
        epochs=int(cfg.get("training", {}).get("max_epochs", 100)),
        batch_size=batch_size,
        hidden_size=int(cfg.get("models", {}).get("hidden_size", 64)),
        num_layers=1,
        num_heads=4,
        dropout=0.2,
        turnover_penalty_lambda=float(cfg.get("training", {}).get("decoder_tft_turnover_penalty_lambda", 0.0)),
        train_buy_turnover_lambda=float(cfg.get("training", {}).get("decoder_tft_train_buy_turnover_lambda", 0.0)),
        train_sell_turnover_lambda=float(cfg.get("training", {}).get("decoder_tft_train_sell_turnover_lambda", 0.0)),
        valid_selection_turnover_lambda=float(
            cfg.get("training", {}).get("decoder_tft_valid_selection_turnover_lambda", 0.0)
        ),
        valid_selection_buy_turnover_lambda=float(
            cfg.get("training", {}).get("decoder_tft_valid_selection_buy_turnover_lambda", 0.0)
        ),
        valid_selection_sell_turnover_lambda=float(
            cfg.get("training", {}).get("decoder_tft_valid_selection_sell_turnover_lambda", 0.0)
        ),
        multitask_aux_loss_weight=float(cfg.get("training", {}).get("decoder_tft_aux_vol_loss_weight", 0.0)),
        multitask_regime_loss_weight=float(cfg.get("training", {}).get("decoder_tft_aux_regime_loss_weight", 0.0)),
        multitask_downside_loss_weight=float(cfg.get("training", {}).get("decoder_tft_aux_downside_loss_weight", 0.0)),
        dual_position_heads=bool(cfg.get("training", {}).get("decoder_tft_dual_position_heads", False)),
        device=device,
    )

    model, history = fit_model(
        train_loader=train_loader,
        valid_loader=valid_loader,
        input_size=len(feature_cols),
        cfg=train_cfg,
    )

    bundle_dir = abs_path(args.bundle_root) / args.bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = bundle_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    config_src = abs_path(args.config)
    config_dst = bundle_dir / "config_snapshot.yaml"
    shutil.copy2(config_src, config_dst)

    save_json(bundle_dir / "feature_columns.json", feature_cols)
    save_json(bundle_dir / "scaler.json", scaler)
    save_json(bundle_dir / "execution_policy.json", build_execution_policy(cfg))

    outputs_root = PROJECT_ROOT / "outputs" / args.outputs_run_label
    if not outputs_root.exists():
        outputs_root = PROJECT_ROOT / "outputs"
    training_summary = _load_summary(outputs_root, "decoder_tft")
    training_summary["final_training"] = {
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "sequence_length": seq_len,
        "device": device,
        "best_valid_score": float(min(history.get("valid_score", [0.0]))),
        "train_start": str(train_df[cfg["data"]["timestamp_col"]].min()),
        "train_end": str(train_df[cfg["data"]["timestamp_col"]].max()),
        "valid_start": str(valid_df[cfg["data"]["timestamp_col"]].min()),
        "valid_end": str(valid_df[cfg["data"]["timestamp_col"]].max()),
    }
    save_json(bundle_dir / "training_summary.json", training_summary)
    save_json(bundle_dir / "final_training_history.json", history)

    manifest = BundleManifest(
        bundle_name=args.bundle_name,
        bundle_version="1.0",
        created_at_utc=utc_now_iso(),
        model_name="decoder_tft",
        config_file=config_dst.name,
        checkpoint_file="model.pt",
        feature_columns_file="feature_columns.json",
        scaler_file="scaler.json",
        execution_policy_file="execution_policy.json",
        training_summary_file="training_summary.json",
        timeframe=str(cfg["data"]["timeframe"]),
        timestamp_col=str(cfg["data"]["timestamp_col"]),
        sequence_length=seq_len,
        hidden_size=int(cfg["models"]["hidden_size"]),
        num_heads=4,
        dropout=0.2,
        target_horizon_bars=int(cfg["training"]["target_horizon_bars"]),
        multitask_aux_vol_loss_weight=float(cfg["training"].get("decoder_tft_aux_vol_loss_weight", 0.0)),
        multitask_regime_loss_weight=float(cfg["training"].get("decoder_tft_aux_regime_loss_weight", 0.0)),
        multitask_downside_loss_weight=float(cfg["training"].get("decoder_tft_aux_downside_loss_weight", 0.0)),
        dual_position_heads=bool(cfg["training"].get("decoder_tft_dual_position_heads", False)),
        integer_contract_execution=bool(cfg["trading"].get("integer_contract_execution", False)),
        max_contracts=int(cfg["trading"].get("max_contracts", 1)),
        contract_rounding_mode=str(cfg["trading"].get("contract_rounding_mode", "nearest")),
        notes="Final deployable decoder_tft model trained from the current config and exported as a live bundle.",
    )
    save_manifest(bundle_dir / "manifest.json", manifest)

    report = {
        "bundle_dir": str(bundle_dir),
        "checkpoint_path": str(checkpoint_path),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "device": device,
    }
    report_path = bundle_dir / "final_training_report.json"
    ensure_parent(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
