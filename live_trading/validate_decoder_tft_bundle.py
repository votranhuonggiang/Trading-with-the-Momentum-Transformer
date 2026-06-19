"""Preflight validation for the current decoder_tft live bundle."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.decoder_tft import DecoderTft  # noqa: E402

from bundle_utils import load_json, load_manifest  # noqa: E402


def main() -> None:
    load_dotenv(PROJECT_ROOT / "live_trading" / ".env")
    bundle_dir = Path(
        os.environ.get(
            "DECODER_TFT_LIVE_BUNDLE",
            str(PROJECT_ROOT / "live_trading" / "bundles" / "decoder_tft_main_branch_5m"),
        )
    )
    required_files = [
        "manifest.json",
        "config_snapshot.yaml",
        "feature_columns.json",
        "scaler.json",
        "execution_policy.json",
        "training_summary.json",
        "model.pt",
    ]
    missing = [name for name in required_files if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle missing files: {missing} | bundle_dir={bundle_dir}")

    manifest = load_manifest(bundle_dir / "manifest.json")
    feature_cols = load_json(bundle_dir / manifest.feature_columns_file)
    checkpoint_path = bundle_dir / manifest.checkpoint_file
    state = torch.load(checkpoint_path, map_location="cpu")

    model = DecoderTft(
        input_size=len(feature_cols),
        hidden_size=int(manifest.hidden_size),
        num_heads=int(manifest.num_heads),
        dropout=float(manifest.dropout),
        multitask=(
            float(manifest.multitask_aux_vol_loss_weight) > 0.0
            or float(manifest.multitask_regime_loss_weight) > 0.0
            or float(manifest.multitask_downside_loss_weight) > 0.0
        ),
        dual_position_heads=bool(manifest.dual_position_heads),
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    required_env = [
        "QUESTDB_HOST",
        "DNSE_API_KEY",
        "DNSE_API_SECRET",
    ]
    missing_env = [name for name in required_env if not os.environ.get(name)]

    print("Bundle validation OK")
    print(f"bundle_dir = {bundle_dir}")
    print(f"model_name = {manifest.model_name}")
    print(f"timeframe = {manifest.timeframe}")
    print(f"sequence_length = {manifest.sequence_length}")
    print(f"max_contracts = {manifest.max_contracts}")
    print(f"missing_env = {missing_env}")


if __name__ == "__main__":
    main()
