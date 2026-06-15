from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_action(position_before: float, position_after: float) -> str:
    if position_before == 0.0 and position_after > 0.0:
        return "open_long"
    if position_before == 0.0 and position_after < 0.0:
        return "open_short"
    if position_before > 0.0 and position_after == 0.0:
        return "close_long"
    if position_before < 0.0 and position_after == 0.0:
        return "close_short"
    if position_before > 0.0 and position_after < 0.0:
        return "reverse_long_to_short"
    if position_before < 0.0 and position_after > 0.0:
        return "reverse_short_to_long"
    if position_before >= 0.0 and position_after > position_before:
        return "increase_long"
    if position_before > 0.0 and 0.0 < position_after < position_before:
        return "reduce_long"
    if position_before <= 0.0 and position_after < position_before:
        return "increase_short"
    if position_before < 0.0 and position_after < 0.0 and position_after > position_before:
        return "reduce_short"
    return "rebalance"


def action_side(position_delta: float) -> str:
    if position_delta > 0.0:
        return "buy"
    if position_delta < 0.0:
        return "sell"
    return "flat"


def build_transaction_blotter(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    trading_cfg: dict,
    model_name: str,
    num_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = predictions.loc[predictions["model"] == model_name].copy()
    if preds.empty:
        raise ValueError(f"No predictions found for model '{model_name}'.")

    preds["timestamp"] = pd.to_datetime(preds["timestamp"])
    preds = preds.sort_values("timestamp").reset_index(drop=True)
    preds["position"] = pd.to_numeric(preds["position"], errors="coerce").fillna(0.0)
    preds["turnover"] = pd.to_numeric(preds["turnover"], errors="coerce").fillna(0.0)
    preds["net_return"] = pd.to_numeric(preds["net_return"], errors="coerce").fillna(0.0)
    preds["net_pnl_points"] = pd.to_numeric(preds["net_pnl_points"], errors="coerce").fillna(0.0)
    preds["position_before"] = preds["position"].shift(1).fillna(0.0)
    preds["position_after"] = preds["position"]
    preds["position_delta"] = preds["position_after"] - preds["position_before"]
    preds["trade_day"] = preds["timestamp"].dt.date

    last_days = preds["trade_day"].drop_duplicates().tail(num_days).tolist()
    preds = preds.loc[preds["trade_day"].isin(last_days)].copy()

    feat = features.copy()
    feat["timestamp"] = pd.to_datetime(feat["timestamp"])
    keep_cols = ["timestamp", "close"]
    if "trade_date" in feat.columns:
        keep_cols.append("trade_date")
    feat = feat[keep_cols].drop_duplicates(subset=["timestamp"])

    merged = preds.merge(feat, on="timestamp", how="left", validate="many_to_one")
    if merged["close"].isna().any():
        missing = int(merged["close"].isna().sum())
        raise ValueError(f"Missing close price for {missing} prediction rows after merge.")

    contract_multiplier = float(trading_cfg["contract_multiplier"])
    fee_vsdc = float(trading_cfg["fee_vsdc_vnd_per_contract_per_side"])
    fee_hnx = float(trading_cfg["fee_hnx_vnd_per_contract_per_side"])
    fee_ctck = float(trading_cfg["fee_ctck_vnd_per_contract_per_side"])
    margin_rate = float(trading_cfg["margin_rate"])
    transfer_tax_rate = float(trading_cfg["transfer_tax_rate"])
    per_side_fixed_fee = fee_vsdc + fee_hnx + fee_ctck

    merged["buy_units"] = merged["position_delta"].clip(lower=0.0)
    merged["sell_units"] = (-merged["position_delta"]).clip(lower=0.0)
    merged["tax_vnd_per_contract"] = margin_rate * contract_multiplier * transfer_tax_rate * merged["close"]
    merged["buy_cost_vnd"] = merged["buy_units"] * per_side_fixed_fee
    merged["sell_fixed_cost_vnd"] = merged["sell_units"] * per_side_fixed_fee
    merged["sell_tax_cost_vnd"] = merged["sell_units"] * merged["tax_vnd_per_contract"]
    merged["sell_cost_vnd"] = merged["sell_fixed_cost_vnd"] + merged["sell_tax_cost_vnd"]
    merged["cost_vnd"] = merged["buy_cost_vnd"] + merged["sell_cost_vnd"]
    merged["cost_points"] = merged["cost_vnd"] / contract_multiplier
    merged["action"] = [
        classify_action(before, after)
        for before, after in zip(merged["position_before"], merged["position_after"])
    ]
    merged["side"] = merged["position_delta"].map(action_side)

    transactions = merged.loc[merged["turnover"] > 0.0].copy()
    transactions["trade_day"] = pd.to_datetime(transactions["trade_day"])

    tx_cols = [
        "trade_day",
        "timestamp",
        "model",
        "window",
        "action",
        "side",
        "position_before",
        "position_after",
        "position_delta",
        "buy_units",
        "sell_units",
        "turnover",
        "close",
        "buy_cost_vnd",
        "sell_fixed_cost_vnd",
        "sell_tax_cost_vnd",
        "cost_vnd",
        "cost_points",
        "net_pnl_points",
        "net_return",
    ]
    transactions = transactions[tx_cols].rename(columns={"trade_day": "day", "net_pnl_points": "bar_net_pnl_points"})

    daily_summary = (
        merged.groupby("trade_day", as_index=False)
        .agg(
            model=("model", "last"),
            number_of_transactions=("turnover", lambda s: int((s > 0.0).sum())),
            buy_transactions=("buy_units", lambda s: int((s > 0.0).sum())),
            sell_transactions=("sell_units", lambda s: int((s > 0.0).sum())),
            buy_turnover=("buy_units", "sum"),
            sell_turnover=("sell_units", "sum"),
            total_turnover=("turnover", "sum"),
            total_cost_points=("cost_points", "sum"),
            total_cost_vnd=("cost_vnd", "sum"),
            daily_net_pnl_points=("net_pnl_points", "sum"),
            daily_net_return=("net_return", "sum"),
        )
        .rename(columns={"trade_day": "day"})
    )
    daily_summary["day"] = pd.to_datetime(daily_summary["day"])

    return transactions, daily_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-label", default="vn30f1m_outputs_version1")
    parser.add_argument("--model", default="decoder_tft")
    parser.add_argument("--num-days", type=int, default=10)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    cfg = load_config(project_root / args.config)

    predictions_path = project_root / "outputs" / args.run_label / "predictions" / "all_predictions.csv"
    features_path = project_root / cfg["data"]["features_path"]
    output_dir = project_root / "outputs" / args.run_label / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_path)
    features = pd.read_parquet(features_path)
    transactions, summary = build_transaction_blotter(
        predictions=predictions,
        features=features,
        trading_cfg=cfg["trading"],
        model_name=args.model,
        num_days=args.num_days,
    )

    tx_path = output_dir / "daily_transactions_last_10_days.csv"
    summary_path = output_dir / "daily_transaction_summary_last_10_days.csv"
    transactions.to_csv(tx_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved {tx_path.name}")
    print(f"Saved {summary_path.name}")


if __name__ == "__main__":
    main()
