"""Purged combinatorial cross-validation robustness check.

This runner is intentionally separate from walk_forward.py. It is used to
stress-test overfit/leakage risk by evaluating the same model settings across
multiple purged, embargoed chronological group holdouts.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import ThresholdConfig, baseline_signals, run_strategy
from common import abs_path, ensure_parent, load_config
from metrics import summarize
from walk_forward import model_signals_for_split


@dataclass
class CpcvSplit:
    split_id: str
    test_group_ids: tuple[int, ...]
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    valid_rows: int
    test_rows: int
    purged_rows: int
    embargoed_rows: int


def _contiguous_date_groups(df: pd.DataFrame, ts_col: str, n_groups: int) -> list[pd.Index]:
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")

    dates = pd.Series(pd.to_datetime(df[ts_col]).dt.normalize().unique()).sort_values().reset_index(drop=True)
    if len(dates) < n_groups:
        raise ValueError(f"Not enough trading dates ({len(dates)}) for n_groups={n_groups}.")

    date_chunks = np.array_split(dates.to_numpy(), n_groups)
    groups: list[pd.Index] = []
    normalized_ts = pd.to_datetime(df[ts_col]).dt.normalize()
    for chunk in date_chunks:
        mask = normalized_ts.isin(chunk)
        groups.append(df.index[mask])
    return groups


def _split_train_valid(train_valid_df: pd.DataFrame, ts_col: str, valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be between 0.0 and 0.5.")

    dates = pd.Series(pd.to_datetime(train_valid_df[ts_col]).dt.normalize().unique()).sort_values().reset_index(drop=True)
    valid_days = max(1, int(round(len(dates) * valid_fraction)))
    if valid_days >= len(dates):
        valid_days = max(1, len(dates) // 5)

    valid_dates = set(dates.tail(valid_days).tolist())
    normalized_ts = pd.to_datetime(train_valid_df[ts_col]).dt.normalize()
    valid = train_valid_df[normalized_ts.isin(valid_dates)]
    train = train_valid_df[~normalized_ts.isin(valid_dates)]
    return train, valid


def _make_purged_split(
    feat: pd.DataFrame,
    ts_col: str,
    groups: list[pd.Index],
    test_group_ids: tuple[int, ...],
    purge_bars: int,
    embargo_bars: int,
    valid_fraction: float,
) -> tuple[CpcvSplit, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(feat)
    test_mask = np.zeros(n_rows, dtype=bool)
    blocked_mask = np.zeros(n_rows, dtype=bool)
    purge_mask = np.zeros(n_rows, dtype=bool)
    embargo_mask = np.zeros(n_rows, dtype=bool)

    for gid in test_group_ids:
        group_idx = groups[gid]
        if len(group_idx) == 0:
            continue
        positions = group_idx.to_numpy()
        start_pos = int(positions.min())
        end_pos = int(positions.max())
        test_mask[positions] = True
        blocked_mask[positions] = True

        purge_start = max(0, start_pos - purge_bars)
        if purge_start < start_pos:
            purge_mask[purge_start:start_pos] = True
            blocked_mask[purge_start:start_pos] = True

        embargo_end = min(n_rows, end_pos + 1 + embargo_bars)
        if end_pos + 1 < embargo_end:
            embargo_mask[end_pos + 1 : embargo_end] = True
            blocked_mask[end_pos + 1 : embargo_end] = True

    train_valid = feat.loc[~blocked_mask].copy()
    test = feat.loc[test_mask].copy()
    train, valid = _split_train_valid(train_valid, ts_col, valid_fraction)

    split = CpcvSplit(
        split_id="cpcv_" + "_".join(str(g + 1) for g in test_group_ids),
        test_group_ids=tuple(g + 1 for g in test_group_ids),
        test_start=pd.Timestamp(test[ts_col].min()),
        test_end=pd.Timestamp(test[ts_col].max()),
        train_rows=len(train),
        valid_rows=len(valid),
        test_rows=len(test),
        purged_rows=int(purge_mask.sum()),
        embargoed_rows=int(embargo_mask.sum()),
    )
    return split, train, valid, test


def _threshold_config(cfg: dict) -> ThresholdConfig:
    trading = cfg["trading"]
    return ThresholdConfig(
        long_entry_threshold=trading["long_entry_threshold"],
        short_entry_threshold=trading["short_entry_threshold"],
        long_exit_threshold=trading["long_exit_threshold"],
        short_exit_threshold=trading["short_exit_threshold"],
        long_to_short_reverse_threshold=trading["long_to_short_reverse_threshold"],
        short_to_long_reverse_threshold=trading["short_to_long_reverse_threshold"],
    )


def _evaluate_signals(
    test_df: pd.DataFrame,
    cfg: dict,
    split_id: str,
    signals: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    tc = _threshold_config(cfg)
    ts_col = cfg["data"]["timestamp_col"]

    for model_name, signal in signals.items():
        strat = run_strategy(test_df, signal, tc, cfg["trading"])
        metrics = summarize(strat["net_return"], strat["net_pnl_points"], strat["turnover"])
        metrics["split"] = split_id
        metrics["model"] = model_name
        rows.append(metrics)

        pred = strat[
            [
                ts_col,
                "signal",
                "position",
                "turnover",
                "net_return",
                "net_pnl_points",
                "cost_vnd",
                "cost_points",
            ]
        ].copy()
        pred["split"] = split_id
        pred["model"] = model_name
        preds.append(pred)

    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def _model_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, g in metrics.groupby("model"):
        rows.append(
            {
                "model": model_name,
                "splits": int(g["split"].nunique()),
                "median_total_return": float(g["total_return"].median()),
                "mean_total_return": float(g["total_return"].mean()),
                "worst_total_return": float(g["total_return"].min()),
                "positive_return_rate": float((g["total_return"] > 0.0).mean()),
                "median_sharpe": float(g["sharpe_ratio"].median()),
                "mean_sharpe": float(g["sharpe_ratio"].mean()),
                "worst_sharpe": float(g["sharpe_ratio"].min()),
                "median_max_drawdown": float(g["max_drawdown"].median()),
                "worst_max_drawdown": float(g["max_drawdown"].min()),
                "median_profit_factor": float(g["profit_factor"].median()),
                "profit_factor_gt_1_rate": float((g["profit_factor"] > 1.0).mean()),
                "median_turnover": float(g["turnover"].median()),
                "median_number_of_trades": float(g["number_of_trades"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["median_sharpe", "median_total_return"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-groups", type=int, default=None)
    parser.add_argument("--test-groups", type=int, default=None)
    parser.add_argument("--purge-bars", type=int, default=None)
    parser.add_argument("--embargo-bars", type=int, default=None)
    parser.add_argument("--valid-fraction", type=float, default=None)
    parser.add_argument("--max-splits", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--output-label", default="purged_cpcv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    robust_cfg = cfg.get("robust_validation", {})
    ts_col = cfg["data"]["timestamp_col"]
    seq_len = int(cfg["models"]["sequence_lengths"][0])
    target_horizon = int(cfg.get("training", {}).get("target_horizon_bars", 1))

    n_groups = int(args.n_groups or robust_cfg.get("n_groups", 8))
    test_groups = int(args.test_groups or robust_cfg.get("test_groups", 2))
    purge_bars = int(args.purge_bars or robust_cfg.get("purge_bars", seq_len + target_horizon))
    embargo_bars = int(args.embargo_bars or robust_cfg.get("embargo_bars", 48))
    valid_fraction = float(args.valid_fraction or robust_cfg.get("valid_fraction", 0.20))
    max_splits = int(args.max_splits if args.max_splits is not None else robust_cfg.get("max_splits", 12))
    models = args.models or robust_cfg.get("models", ["decoder_tft"])

    if test_groups <= 0 or test_groups >= n_groups:
        raise ValueError("test_groups must be positive and smaller than n_groups.")

    cfg = dict(cfg)
    cfg["models"] = dict(cfg["models"])
    cfg["models"]["main_models"] = list(models)

    feat = pd.read_parquet(abs_path(cfg["data"]["features_path"])).sort_values(ts_col).reset_index(drop=True)
    groups = _contiguous_date_groups(feat, ts_col, n_groups)
    combos = list(itertools.combinations(range(n_groups), test_groups))
    if max_splits > 0 and max_splits < len(combos):
        combo_indices = np.linspace(0, len(combos) - 1, max_splits, dtype=int)
        combos = [combos[i] for i in sorted(set(combo_indices.tolist()))]

    all_metrics: list[pd.DataFrame] = []
    all_preds: list[pd.DataFrame] = []
    split_rows: list[dict] = []

    for test_group_ids in combos:
        split, train, valid, test = _make_purged_split(
            feat=feat,
            ts_col=ts_col,
            groups=groups,
            test_group_ids=test_group_ids,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            valid_fraction=valid_fraction,
        )
        split_rows.append(
            {
                "split": split.split_id,
                "test_group_ids": ",".join(str(g) for g in split.test_group_ids),
                "test_start": split.test_start,
                "test_end": split.test_end,
                "train_rows": split.train_rows,
                "valid_rows": split.valid_rows,
                "test_rows": split.test_rows,
                "purged_rows": split.purged_rows,
                "embargoed_rows": split.embargoed_rows,
            }
        )

        if len(train) == 0 or len(valid) == 0 or len(test) == 0:
            continue

        signals = {}
        if args.include_baselines:
            signals.update(baseline_signals(test))
        signals.update(model_signals_for_split(train, valid, test, cfg))
        if not signals:
            continue

        metrics, preds = _evaluate_signals(test, cfg, split.split_id, signals)
        all_metrics.append(metrics)
        all_preds.append(preds)
        print(
            f"{split.split_id}: train={len(train)} valid={len(valid)} test={len(test)} "
            f"purge={split.purged_rows} embargo={split.embargoed_rows}"
        )

    if not all_metrics:
        raise RuntimeError("No CPCV splits produced metrics. Check group count, purge, embargo, and data coverage.")

    out_root = abs_path(".")
    metrics_dir = out_root / "outputs" / "metrics"
    tables_dir = out_root / "outputs" / "tables"
    preds_dir = out_root / "outputs" / "predictions"
    ensure_parent(metrics_dir / "x.csv")
    ensure_parent(tables_dir / "x.csv")
    ensure_parent(preds_dir / "x.csv")

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    preds_all = pd.concat(all_preds, ignore_index=True)
    splits_df = pd.DataFrame(split_rows)
    summary = _model_summary(metrics_all)

    label = args.output_label
    metrics_all.to_csv(metrics_dir / f"{label}_results.csv", index=False)
    preds_all.to_csv(preds_dir / f"{label}_predictions.csv", index=False)
    splits_df.to_csv(tables_dir / f"{label}_splits.csv", index=False)
    summary.to_csv(tables_dir / f"{label}_model_summary.csv", index=False)

    print(f"Saved CPCV metrics to outputs/metrics/{label}_results.csv")
    print(f"Saved CPCV summary to outputs/tables/{label}_model_summary.csv")


if __name__ == "__main__":
    main()
