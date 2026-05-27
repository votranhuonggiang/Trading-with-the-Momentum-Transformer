"""Evaluation entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import abs_path, ensure_parent, load_config
from visualization import plot_model_cumulative_returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    _ = load_config(args.config)
    root = abs_path(".")
    wf_file = root / "outputs" / "metrics" / "walk_forward_results.csv"
    if not wf_file.exists():
        raise FileNotFoundError("walk_forward_results.csv not found. Run walk_forward.py first.")

    wf = pd.read_csv(wf_file)
    ts = pd.read_csv(root / "outputs" / "predictions" / "all_predictions.csv", parse_dates=["timestamp"])
    ts["year"] = ts["timestamp"].dt.year
    ts["month"] = ts["timestamp"].dt.to_period("M").astype(str)

    yearly = ts.groupby(["model", "year"])["net_return"].sum().reset_index(name="net_return_sum")
    monthly = ts.groupby(["model", "month"])["net_return"].sum().reset_index(name="net_return_sum")

    tables = root / "outputs" / "tables"
    ensure_parent(tables / "x.csv")
    yearly.to_csv(tables / "yearly_performance.csv", index=False)
    monthly.to_csv(tables / "monthly_performance.csv", index=False)

    # A simple trade stats output from predictions table.
    trade_stats = (
        ts.assign(is_trade=(ts["turnover"] > 0).astype(int))
        .groupby("model")
        .agg(
            number_of_trades=("is_trade", "sum"),
            turnover=("turnover", "sum"),
            average_net_pnl_points=("net_pnl_points", "mean"),
            percentage_time_long=("position", lambda x: (x > 0).mean()),
            percentage_time_short=("position", lambda x: (x < 0).mean()),
            percentage_time_flat=("position", lambda x: (x == 0).mean()),
        )
        .reset_index()
    )
    trade_stats.to_csv(tables / "trade_statistics.csv", index=False)

    # Mirror existing files requested in instruction with a baseline mapping.
    wf.to_csv(tables / "walk_forward_results.csv", index=False)
    plot_model_cumulative_returns(ts, root / "outputs" / "figures" / "model_cumulative_returns.png")
    print("Saved evaluation tables: yearly, monthly, trade stats, and model line chart.")


if __name__ == "__main__":
    main()
