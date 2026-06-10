"""Walk-forward training and test orchestration entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from backtest import ThresholdConfig, baseline_signals, cost_sensitivity, run_strategy
from common import abs_path, ensure_parent, load_config
from datasets import FeaturePack, SequenceDataset, default_feature_columns
from metrics import summarize
from train import TrainConfig, fit_model


@dataclass
class Window:
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str


def _windows_from_config(
    feat: pd.DataFrame,
    ts_col: str,
    validation_months: int,
    test_months: int,
    step_months: int | None = None,
    method: str = "expanding",
    train_months: int | None = None,
    initial_train_end: str = "2020-12-31",
) -> List[Window]:
    if validation_months <= 0 or test_months <= 0:
        raise ValueError("validation_months and test_months must be positive integers.")
    if step_months is None:
        step_months = validation_months + test_months
    if step_months <= 0:
        raise ValueError("step_months must be a positive integer.")
    if feat.empty:
        return []

    max_ts = pd.Timestamp(feat[ts_col].max()).normalize()
    train_end = pd.Timestamp(initial_train_end).normalize()
    windows: list[Window] = []

    while True:
        valid_start = (train_end + pd.Timedelta(days=1)).normalize()
        valid_end = (valid_start + pd.DateOffset(months=validation_months) - pd.Timedelta(days=1)).normalize()
        test_start = (valid_end + pd.Timedelta(days=1)).normalize()
        test_end = (test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)).normalize()

        if valid_start > max_ts or test_start > max_ts:
            break
        if test_end > max_ts:
            test_end = max_ts
        if test_end < test_start:
            break

        if method == "rolling":
            if not train_months or train_months <= 0:
                raise ValueError("walk_forward.train_months must be set for rolling mode.")
            train_start = (valid_start - pd.DateOffset(months=train_months)).normalize()
        else:
            train_start = pd.Timestamp(feat[ts_col].min()).normalize()

        windows.append(
            Window(
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                valid_start=valid_start.strftime("%Y-%m-%d"),
                valid_end=valid_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
            )
        )
        train_end = (train_end + pd.DateOffset(months=step_months)).normalize()
        if train_end >= max_ts:
            break

    return windows


def split_df(df: pd.DataFrame, ts_col: str, w: Window) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df[ts_col] >= pd.Timestamp(w.train_start)) & (df[ts_col] <= pd.Timestamp(w.train_end))]
    valid = df[(df[ts_col] >= pd.Timestamp(w.valid_start)) & (df[ts_col] <= pd.Timestamp(w.valid_end))]
    test = df[(df[ts_col] >= pd.Timestamp(w.test_start)) & (df[ts_col] <= pd.Timestamp(w.test_end))]
    return train, valid, test


def evaluate_on_split(
    test_df: pd.DataFrame, cfg: dict, window_name: str, out_root: Path, signals: dict[str, pd.Series]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tc = ThresholdConfig(
        long_entry_threshold=cfg["trading"]["long_entry_threshold"],
        short_entry_threshold=cfg["trading"]["short_entry_threshold"],
        long_exit_threshold=cfg["trading"]["long_exit_threshold"],
        short_exit_threshold=cfg["trading"]["short_exit_threshold"],
        long_to_short_reverse_threshold=cfg["trading"]["long_to_short_reverse_threshold"],
        short_to_long_reverse_threshold=cfg["trading"]["short_to_long_reverse_threshold"],
    )
    rows = []
    cost_tables = []
    preds = []
    for model_name, sig in signals.items():
        strat = run_strategy(test_df, sig, tc, cfg["trading"])
        m = summarize(strat["net_return"], strat["net_pnl_points"], strat["turnover"])
        m["window"] = window_name
        m["model"] = model_name
        rows.append(m)

        cs = cost_sensitivity(strat, cfg["trading"]["cost_scenarios_points"])
        cs["window"] = window_name
        cs["model"] = model_name
        cost_tables.append(cs)

        p = strat[[cfg["data"]["timestamp_col"], "signal", "position", "turnover", "net_return", "net_pnl_points"]].copy()
        p["window"] = window_name
        p["model"] = model_name
        preds.append(p)

    metrics_df = pd.DataFrame(rows)
    cost_df = pd.concat(cost_tables, ignore_index=True)
    preds_df = pd.concat(preds, ignore_index=True)

    metrics_file = out_root / "outputs" / "metrics" / f"metrics_{window_name}.csv"
    preds_file = out_root / "outputs" / "predictions" / f"predictions_{window_name}.csv"
    cost_file = out_root / "outputs" / "tables" / f"cost_sensitivity_{window_name}.csv"
    ensure_parent(metrics_file)
    ensure_parent(preds_file)
    ensure_parent(cost_file)
    metrics_df.to_csv(metrics_file, index=False)
    preds_df.to_csv(preds_file, index=False)
    cost_df.to_csv(cost_file, index=False)
    return metrics_df, cost_df, preds_df


def _scale_with_train_stats(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mu = train_df[feature_cols].mean()
    sigma = train_df[feature_cols].std().replace(0, 1.0)
    tr = train_df.copy()
    va = valid_df.copy()
    te = test_df.copy()
    tr[feature_cols] = (tr[feature_cols] - mu) / sigma
    va[feature_cols] = (va[feature_cols] - mu) / sigma
    te[feature_cols] = (te[feature_cols] - mu) / sigma
    tr[feature_cols] = tr[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    va[feature_cols] = va[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    te[feature_cols] = te[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    return tr, va, te


def _infer_signals_from_model(
    model: torch.nn.Module,
    data_df: pd.DataFrame,
    fp: FeaturePack,
    seq_len: int,
    device: str,
    model_name: str,
    cfg: dict,
) -> pd.Series:
    ds = SequenceDataset(data_df, fp, sequence_length=seq_len)
    if len(ds) == 0:
        return pd.Series(0.0, index=data_df.index)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    model.eval()
    preds: list[float] = []
    regime_probs: list[float] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb)
            if isinstance(out, dict):
                preds.extend(out["position"].detach().cpu().numpy().tolist())
                if (
                    model_name == "decoder_tft"
                    and cfg.get("trading", {}).get("decoder_tft_regime_conditioned_exposure", False)
                    and "future_vol_regime_logit" in out
                ):
                    regime_probs.extend(torch.sigmoid(out["future_vol_regime_logit"]).detach().cpu().numpy().tolist())
            else:
                preds.extend(out.detach().cpu().numpy().tolist())

    signal = pd.Series(0.0, index=data_df.index)
    valid_indices = ds.indices
    for i, end_idx in enumerate(valid_indices):
        signal.iloc[end_idx] = float(np.clip(preds[i], -1.0, 1.0))
    if model_name == "decoder_tft" and bool(
        cfg.get("trading", {}).get("decoder_tft_confidence_scaled_exposure", False)
    ):
        tau = float(cfg.get("trading", {}).get("decoder_tft_confidence_gate_tau", 0.0))
        alpha = float(cfg.get("trading", {}).get("decoder_tft_confidence_scale_alpha", 1.0))
        max_position = float(cfg.get("trading", {}).get("decoder_tft_confidence_scaled_max_position", 1.0))
        if alpha <= 0.0:
            raise ValueError("decoder_tft_confidence_scale_alpha must be positive.")
        scaled_signal = signal.copy()
        active_mask = scaled_signal.abs() >= tau
        scaled_signal.loc[~active_mask] = 0.0
        if active_mask.any():
            active_values = scaled_signal.loc[active_mask]
            scaled_signal.loc[active_mask] = np.sign(active_values) * np.power(active_values.abs(), alpha)
        signal = scaled_signal.clip(-max_position, max_position)
    if regime_probs:
        regime_scale = pd.Series(1.0, index=data_df.index)
        low_scale = float(cfg.get("trading", {}).get("decoder_tft_low_vol_scale", 1.0))
        high_scale = float(cfg.get("trading", {}).get("decoder_tft_high_vol_scale", 0.6))
        use_asymmetric_scaling = bool(
            cfg.get("trading", {}).get("decoder_tft_asymmetric_regime_scaling", False)
        )
        low_long_scale = float(cfg.get("trading", {}).get("decoder_tft_low_vol_long_scale", low_scale))
        high_long_scale = float(cfg.get("trading", {}).get("decoder_tft_high_vol_long_scale", high_scale))
        low_short_scale = float(cfg.get("trading", {}).get("decoder_tft_low_vol_short_scale", low_scale))
        high_short_scale = float(cfg.get("trading", {}).get("decoder_tft_high_vol_short_scale", high_scale))
        for i, end_idx in enumerate(valid_indices):
            p_high = float(np.clip(regime_probs[i], 0.0, 1.0))
            if use_asymmetric_scaling:
                if signal.iloc[end_idx] >= 0.0:
                    regime_scale.iloc[end_idx] = low_long_scale * (1.0 - p_high) + high_long_scale * p_high
                else:
                    regime_scale.iloc[end_idx] = low_short_scale * (1.0 - p_high) + high_short_scale * p_high
            else:
                regime_scale.iloc[end_idx] = low_scale * (1.0 - p_high) + high_scale * p_high
        signal = signal * regime_scale
    return signal


def model_signals_for_split(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    max_tr = int(cfg.get("training", {}).get("max_train_rows_per_window", 0) or 0)
    max_va = int(cfg.get("training", {}).get("max_valid_rows_per_window", 0) or 0)
    if max_tr > 0 and len(train) > max_tr:
        train = train.tail(max_tr).copy()
    if max_va > 0 and len(valid) > max_va:
        valid = valid.tail(max_va).copy()

    feature_cols = default_feature_columns(train, cfg)
    tr, va, te = _scale_with_train_stats(train, valid, test, feature_cols)
    aux_target_cols: list[str] = []
    requested_aux_target_cols = [
        "target_future_realized_vol_12",
        "target_future_vol_regime_12",
        "target_future_downside_semivariance_12",
    ]
    if float(cfg.get("training", {}).get("decoder_tft_aux_vol_loss_weight", 0.0)) > 0.0 or float(
        cfg.get("training", {}).get("decoder_tft_aux_regime_loss_weight", 0.0)
    ) > 0.0 or float(
        cfg.get("training", {}).get("decoder_tft_aux_downside_loss_weight", 0.0)
    ) > 0.0:
        if all(col in tr.columns and col in va.columns and col in te.columns for col in requested_aux_target_cols):
            aux_target_cols = requested_aux_target_cols
        else:
            missing = [col for col in requested_aux_target_cols if col not in tr.columns or col not in va.columns or col not in te.columns]
            print(
                "Auxiliary targets requested for decoder_tft but missing from feature data; "
                f"disabling multitask auxiliary targets for this run. Missing: {missing}"
            )
    target_col = "target_return_horizon" if "target_return_horizon" in tr.columns else "target_return_next"
    fp = FeaturePack(feature_cols=feature_cols, target_col=target_col, aux_target_cols=aux_target_cols)
    seq_len = int(cfg["models"]["sequence_lengths"][0])
    hidden_size = int(cfg.get("models", {}).get("hidden_size", 64))
    batch_size = int(cfg.get("training", {}).get("batch_size", 256))
    epochs = int(cfg.get("training", {}).get("max_epochs_per_window", cfg.get("training", {}).get("max_epochs", 100)))
    epochs = min(epochs, 8)
    lr = float(cfg.get("training", {}).get("learning_rate", 1e-3))
    device = "cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "auto" else "cpu"

    tr_ds = SequenceDataset(tr, fp, sequence_length=seq_len)
    va_ds = SequenceDataset(va, fp, sequence_length=seq_len)
    if len(tr_ds) == 0 or len(va_ds) == 0:
        return {}
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    out: dict[str, pd.Series] = {}
    for model_name in cfg["models"]["main_models"]:
        turnover_penalty_lambda = 0.0
        if model_name == "decoder_tft":
            turnover_penalty_lambda = float(cfg.get("training", {}).get("decoder_tft_turnover_penalty_lambda", 0.0))
        tr_loader = DataLoader(
            tr_ds,
            batch_size=batch_size,
            shuffle=turnover_penalty_lambda <= 0,
        )
        tcfg = TrainConfig(
            model_name=model_name,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_layers=2 if model_name == "lstm_dmn" else 1,
            num_heads=4,
            dropout=0.2,
            turnover_penalty_lambda=turnover_penalty_lambda,
            train_buy_turnover_lambda=float(
                cfg.get("training", {}).get(
                    "decoder_tft_train_buy_turnover_lambda" if model_name == "decoder_tft" else "train_buy_turnover_lambda",
                    0.0,
                )
            ),
            train_sell_turnover_lambda=float(
                cfg.get("training", {}).get(
                    "decoder_tft_train_sell_turnover_lambda" if model_name == "decoder_tft" else "train_sell_turnover_lambda",
                    0.0,
                )
            ),
            valid_selection_turnover_lambda=float(
                cfg.get("training", {}).get(
                    "decoder_tft_valid_selection_turnover_lambda" if model_name == "decoder_tft" else "valid_selection_turnover_lambda",
                    0.0,
                )
            ),
            valid_selection_buy_turnover_lambda=float(
                cfg.get("training", {}).get(
                    "decoder_tft_valid_selection_buy_turnover_lambda" if model_name == "decoder_tft" else "valid_selection_buy_turnover_lambda",
                    0.0,
                )
            ),
            valid_selection_sell_turnover_lambda=float(
                cfg.get("training", {}).get(
                    "decoder_tft_valid_selection_sell_turnover_lambda" if model_name == "decoder_tft" else "valid_selection_sell_turnover_lambda",
                    0.0,
                )
            ),
            multitask_aux_loss_weight=float(
                cfg.get("training", {}).get(
                    "decoder_tft_aux_vol_loss_weight" if model_name == "decoder_tft" else "multitask_aux_loss_weight",
                    0.0,
                )
            ),
            multitask_regime_loss_weight=float(
                cfg.get("training", {}).get(
                    "decoder_tft_aux_regime_loss_weight" if model_name == "decoder_tft" else "multitask_regime_loss_weight",
                    0.0,
                )
            ),
            multitask_downside_loss_weight=float(
                cfg.get("training", {}).get(
                    "decoder_tft_aux_downside_loss_weight"
                    if model_name == "decoder_tft"
                    else "multitask_downside_loss_weight",
                    0.0,
                )
            ),
            dual_position_heads=bool(
                cfg.get("training", {}).get(
                    "decoder_tft_dual_position_heads" if model_name == "decoder_tft" else "dual_position_heads",
                    False,
                )
            ),
            device=device,
        )
        model, _ = fit_model(tr_loader, va_loader, input_size=len(feature_cols), cfg=tcfg)
        out[model_name] = _infer_signals_from_model(
            model,
            te,
            fp,
            seq_len=seq_len,
            device=device,
            model_name=model_name,
            cfg=cfg,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = abs_path(".")
    ts_col = cfg["data"]["timestamp_col"]
    feat = pd.read_parquet(abs_path(cfg["data"]["features_path"])).sort_values(ts_col)

    all_metrics = []
    all_cost = []
    all_preds = []
    wf_cfg = cfg.get("walk_forward", {})
    validation_months = int(wf_cfg.get("validation_months", 6))
    test_months = int(wf_cfg.get("test_months", 6))
    step_months = int(wf_cfg.get("step_months", validation_months + test_months))
    method = str(wf_cfg.get("method", "expanding")).lower()
    train_months = wf_cfg.get("train_months")
    train_months = int(train_months) if train_months is not None else None
    windows = _windows_from_config(
        feat,
        ts_col,
        validation_months,
        test_months,
        step_months=step_months,
        method=method,
        train_months=train_months,
    )
    max_windows = int(cfg.get("walk_forward", {}).get("max_windows", 0) or 0)
    if max_windows > 0:
        windows = windows[:max_windows]

    for i, w in enumerate(windows, start=1):
        train, valid, test = split_df(feat, ts_col, w)
        if len(train) == 0 or len(valid) == 0 or len(test) == 0:
            continue
        window_name = f"wf_{i}_{w.test_start}_{w.test_end}"
        signals = baseline_signals(test)
        signals.update(model_signals_for_split(train, valid, test, cfg))
        m, c, p = evaluate_on_split(test, cfg, window_name, out_root, signals=signals)
        all_metrics.append(m)
        all_cost.append(c)
        all_preds.append(p)

    if not all_metrics:
        raise RuntimeError("No valid walk-forward windows found with non-empty train/valid/test.")

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    cost_all = pd.concat(all_cost, ignore_index=True)
    preds_all = pd.concat(all_preds, ignore_index=True)

    tables_dir = out_root / "outputs" / "tables"
    metrics_dir = out_root / "outputs" / "metrics"
    ensure_parent(tables_dir / "x.csv")
    ensure_parent(metrics_dir / "x.csv")

    metrics_all.to_csv(metrics_dir / "walk_forward_results.csv", index=False)
    cost_all.to_csv(tables_dir / "cost_sensitivity_table.csv", index=False)
    preds_all.to_csv(out_root / "outputs" / "predictions" / "all_predictions.csv", index=False)

    # Instruction-aligned model comparison: compute metrics from the full
    # concatenated out-of-sample return series per model, not mean of window stats.
    rows = []
    for model_name, g in preds_all.groupby("model"):
        sm = summarize(g["net_return"], g["net_pnl_points"], g["turnover"])
        rows.append(
            {
                "model": model_name,
                "total_return": sm["total_return"],
                "sharpe_ratio": sm["sharpe_ratio"],
                "max_drawdown": sm["max_drawdown"],
                "profit_factor": sm["profit_factor"],
            }
        )
    model_cmp = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
    model_cmp.to_csv(tables_dir / "model_comparison_metrics.csv", index=False)
    print("Saved walk-forward metrics, predictions, and cost sensitivity tables.")


if __name__ == "__main__":
    main()
