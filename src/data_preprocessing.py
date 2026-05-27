"""Data audit and 5-minute resampling entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from common import abs_path, ensure_parent, load_config


def load_raw(raw_path: Path, ts_col: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=[ts_col])
    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col]).reset_index(drop=True)
    return df


def infer_expected_times(df: pd.DataFrame, ts_col: str) -> set[str]:
    minute_str = df[ts_col].dt.strftime("%H:%M")
    counts = minute_str.value_counts()
    day_count = df[ts_col].dt.date.nunique()
    threshold = max(int(0.95 * day_count), 1)
    return set(counts[counts >= threshold].index.tolist())


def data_quality_summary(df: pd.DataFrame, ts_col: str) -> Dict[str, float | int]:
    px_cols = ["open", "high", "low", "close"]
    checks: Dict[str, float | int] = {}
    checks["rows"] = int(len(df))
    checks["start"] = str(df[ts_col].min())
    checks["end"] = str(df[ts_col].max())
    checks["days"] = int(df[ts_col].dt.date.nunique())
    checks["duplicates_removed"] = 0
    checks["non_positive_ohlc"] = int((df[px_cols] <= 0).any(axis=1).sum())
    checks["high_low_inconsistency"] = int(
        (
            (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
        ).sum()
    )
    checks["extreme_jump_rows"] = int(
        (np.log(df["close"]).diff().abs() > 0.03).sum()
    )
    vol_q1 = df["volume"].quantile(0.25)
    vol_q3 = df["volume"].quantile(0.75)
    iqr = vol_q3 - vol_q1
    vol_hi = vol_q3 + 5.0 * iqr
    checks["extreme_volume_rows"] = int((df["volume"] > vol_hi).sum())

    expected = infer_expected_times(df, ts_col)
    by_day_times = df.groupby(df[ts_col].dt.date)[ts_col].apply(
        lambda s: set(s.dt.strftime("%H:%M").tolist())
    )
    missing_count = int(sum(len(expected - times) for times in by_day_times))
    checks["missing_session_timestamps_vs_mode"] = missing_count
    checks["expected_session_time_count"] = len(expected)
    return checks


def write_report(report_path: Path, checks: Dict[str, float | int]) -> None:
    ensure_parent(report_path)
    lines = [
        "# Data Quality Report",
        "",
        "## Summary",
        f"- Rows: {checks['rows']}",
        f"- Date range: {checks['start']} -> {checks['end']}",
        f"- Trading days: {checks['days']}",
        "",
        "## Integrity Checks",
        f"- Non-positive OHLC rows: {checks['non_positive_ohlc']}",
        f"- High/Low inconsistency rows: {checks['high_low_inconsistency']}",
        f"- Extreme price jump rows (|log ret| > 3%): {checks['extreme_jump_rows']}",
        f"- Extreme volume outlier rows (IQR rule): {checks['extreme_volume_rows']}",
        f"- Missing timestamps vs inferred session grid: {checks['missing_session_timestamps_vs_mode']}",
        f"- Inferred session time points per day: {checks['expected_session_time_count']}",
        "",
        "## Notes",
        "- Input data was sorted by timestamp and duplicate timestamps removed.",
        "- Session grid was inferred from recurring intraday timestamps across days.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def resample_5min(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    frame = df.set_index(ts_col).sort_index()
    bars = (
        frame.resample("5min", label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    bars["date"] = bars[ts_col].dt.date
    first_minute = bars.groupby("date")[ts_col].transform("min")
    bars = bars[bars[ts_col] > first_minute].copy()
    bars = bars.drop(columns=["date"])
    return bars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ts_col = cfg["data"]["timestamp_col"]
    raw_path = abs_path(cfg["data"]["raw_path"])
    processed_path = abs_path(cfg["data"]["processed_path"])
    report_path = abs_path("outputs/data_quality_report.md")

    df = load_raw(raw_path, ts_col)
    checks = data_quality_summary(df, ts_col)
    write_report(report_path, checks)

    bars = resample_5min(df, ts_col)
    ensure_parent(processed_path)
    bars.to_csv(processed_path, index=False)

    print("Saved data quality report.")
    print("Saved 5-minute data.")


if __name__ == "__main__":
    main()
