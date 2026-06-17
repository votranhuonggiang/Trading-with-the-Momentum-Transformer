"""Export a deployable 5-minute decoder_tft live-trading bundle."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import abs_path, load_config  # noqa: E402
from datasets import default_feature_columns  # noqa: E402

from bundle_utils import (  # noqa: E402
    BundleManifest,
    build_execution_policy,
    copy_if_present,
    save_json,
    save_manifest,
    scaler_stats_from_frame,
    utc_now_iso,
)


def _load_training_summary(outputs_root: Path, model_name: str) -> dict:
    summary: dict[str, object] = {"model": model_name}
    files = {
        "headline": outputs_root / "tables" / "model_comparison_metrics.csv",
        "trade_profile": outputs_root / "tables" / "trade_statistics.csv",
        "daily_profile": outputs_root / "tables" / "daily_metrics_summary.csv",
    }
    for section, path in files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "model" not in df.columns:
            continue
        rows = df[df["model"] == model_name]
        if rows.empty:
            continue
        summary[section] = rows.iloc[0].to_dict()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bundle-name", default="decoder_tft_main_branch_5m")
    parser.add_argument("--bundle-root", default="live_trading/bundles")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--outputs-run-label", default="vn30f1m_outputs_version1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    features_path = abs_path(cfg["data"]["features_path"])
    feat = pd.read_parquet(features_path).sort_values(cfg["data"]["timestamp_col"]).reset_index(drop=True)

    if args.train_end:
        cutoff = pd.Timestamp(args.train_end)
        feat = feat[feat[cfg["data"]["timestamp_col"]] <= cutoff].copy()
        if feat.empty:
            raise ValueError(f"No feature rows remain after applying --train-end={args.train_end}.")

    feature_cols = default_feature_columns(feat, cfg)
    scaler_stats = scaler_stats_from_frame(feat, feature_cols)
    execution_policy = build_execution_policy(cfg)

    bundle_root = abs_path(args.bundle_root) / args.bundle_name
    bundle_root.mkdir(parents=True, exist_ok=True)

    config_src = abs_path(args.config)
    config_dst = bundle_root / "config_snapshot.yaml"
    shutil.copy2(config_src, config_dst)

    checkpoint_file = None
    if args.checkpoint:
        checkpoint_src = Path(args.checkpoint)
        if not checkpoint_src.is_absolute():
            checkpoint_src = PROJECT_ROOT / checkpoint_src
        checkpoint_dst = bundle_root / "model.pt"
        copied = copy_if_present(checkpoint_src, checkpoint_dst)
        checkpoint_file = checkpoint_dst.name if copied else None

    save_json(bundle_root / "feature_columns.json", feature_cols)
    save_json(bundle_root / "scaler.json", scaler_stats)
    save_json(bundle_root / "execution_policy.json", execution_policy)

    outputs_root = PROJECT_ROOT / "outputs" / args.outputs_run_label
    training_summary = _load_training_summary(outputs_root, model_name="decoder_tft")
    save_json(bundle_root / "training_summary.json", training_summary)

    manifest = BundleManifest(
        bundle_name=args.bundle_name,
        bundle_version="1.0",
        created_at_utc=utc_now_iso(),
        model_name="decoder_tft",
        config_file=config_dst.name,
        checkpoint_file=checkpoint_file,
        feature_columns_file="feature_columns.json",
        scaler_file="scaler.json",
        execution_policy_file="execution_policy.json",
        training_summary_file="training_summary.json",
        timeframe=str(cfg["data"]["timeframe"]),
        timestamp_col=str(cfg["data"]["timestamp_col"]),
        sequence_length=int(cfg["models"]["sequence_lengths"][0]),
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
        notes=(
            "Config, feature schema, scaler, and execution policy exported from the current main-branch "
            "decoder_tft setup. Supply --checkpoint to make the bundle directly loadable for inference."
        ),
    )
    save_manifest(bundle_root / "manifest.json", manifest)

    next_steps = (
        "# Bundle status\n\n"
        f"- bundle_dir: `{bundle_root}`\n"
        f"- config_snapshot: `{config_dst.name}`\n"
        f"- checkpoint_present: `{checkpoint_file is not None}`\n"
        f"- timeframe: `{cfg['data']['timeframe']}`\n"
        f"- sequence_length: `{cfg['models']['sequence_lengths'][0]}`\n"
        f"- max_contracts: `{cfg['trading'].get('max_contracts', 1)}`\n\n"
        "## What to do next\n\n"
        "1. Export or save a final `decoder_tft` checkpoint as `model.pt` into this bundle.\n"
        "2. Run `DecoderTftLiveTrade.py` against this bundle in `dry_run=True` first.\n"
        "3. Compare bundle replay output to backtest output before enabling real orders.\n"
    )
    (bundle_root / "NEXT_STEPS.md").write_text(next_steps, encoding="utf-8")
    safe_bundle_path = str(bundle_root).encode("ascii", errors="backslashreplace").decode("ascii")
    print(f"Bundle prepared at: {safe_bundle_path}")


if __name__ == "__main__":
    main()
