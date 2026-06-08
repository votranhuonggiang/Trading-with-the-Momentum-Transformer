"""Feature engineering entrypoint."""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from common import abs_path, ensure_parent, load_config


def _session_flags(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    out["trade_date"] = out[ts_col].dt.date
    out["minute_of_day"] = out[ts_col].dt.hour * 60 + out[ts_col].dt.minute
    out["bar_index_in_day"] = out.groupby("trade_date").cumcount()
    out["bars_in_day"] = out.groupby("trade_date")["bar_index_in_day"].transform("max") + 1
    out["day_of_week"] = out[ts_col].dt.weekday
    out["sin_day_of_week"] = np.sin(2.0 * math.pi * out["day_of_week"] / 7.0)
    out["cos_day_of_week"] = np.cos(2.0 * math.pi * out["day_of_week"] / 7.0)
    out["sin_time"] = np.sin(2.0 * math.pi * out["minute_of_day"] / (24.0 * 60.0))
    out["cos_time"] = np.cos(2.0 * math.pi * out["minute_of_day"] / (24.0 * 60.0))
    out["is_first_5min"] = (out["bar_index_in_day"] <= 1).astype(int)
    out["is_last_5min"] = (out["bar_index_in_day"] >= (out["bars_in_day"] - 2)).astype(int)
    out["is_morning_session"] = (out[ts_col].dt.hour < 12).astype(int)
    out["is_afternoon_session"] = 1 - out["is_morning_session"]
    out["time_to_close"] = (out["bars_in_day"] - 1 - out["bar_index_in_day"]).clip(lower=0)
    return out


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ts_col = cfg["data"]["timestamp_col"]
    out = df.copy().sort_values(ts_col).reset_index(drop=True)
    out = _session_flags(out, ts_col)
    out["log_return"] = np.log(out["close"]).diff()

    for h in cfg["features"]["return_horizons"]:
        out[f"ret_{h}"] = np.log(out["close"] / out["close"].shift(h))

    vol_windows = [12, 24, 48, 78]
    for w in vol_windows:
        out[f"rolling_vol_{w}"] = out["log_return"].rolling(w).std()
    out["ewm_vol_78"] = out["log_return"].ewm(span=78, adjust=False).std()
    out["ewm_vol_390"] = out["log_return"].ewm(span=390, adjust=False).std()

    ref_vol = out["ewm_vol_78"].replace(0, np.nan)
    for h in cfg["features"]["return_horizons"]:
        out[f"ret_{h}_norm"] = out[f"ret_{h}"] / (ref_vol * np.sqrt(h) + 1e-9)

    for w in [6, 8, 12, 16, 24, 32, 48, 78, 96]:
        out[f"sma_{w}"] = out["close"].rolling(w).mean()
        out[f"ema_{w}"] = out["close"].ewm(span=w, adjust=False).mean()
    out["price_to_sma_24"] = out["close"] / out["sma_24"] - 1.0
    out["price_to_sma_78"] = out["close"] / out["sma_78"] - 1.0
    out["ema_slope_12"] = out["ema_12"].pct_change()
    out["ema_slope_24"] = out["ema_24"].pct_change()
    out["ema_slope_78"] = out["ema_78"].pct_change()

    for short_w, long_w in cfg["features"]["macd_pairs"]:
        ema_short = out["close"].ewm(span=short_w, adjust=False).mean()
        ema_long = out["close"].ewm(span=long_w, adjust=False).mean()
        macd = ema_short - ema_long
        out[f"macd_{short_w}_{long_w}"] = macd
        out[f"macd_{short_w}_{long_w}_norm"] = macd / (ref_vol * out["close"] + 1e-9)
        if short_w == 32 and long_w == 96:
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            sign = np.sign(macd).fillna(0)
            prev_sign = sign.shift(1).fillna(0)
            out["macd_32_96_signal"] = signal
            out["macd_32_96_hist"] = hist
            out["macd_32_96_hist_norm"] = hist / (ref_vol * out["close"] + 1e-9)
            out["macd_32_96_slope"] = macd.diff()
            out["macd_32_96_slope_norm"] = out["macd_32_96_slope"] / (ref_vol * out["close"] + 1e-9)
            out["macd_32_96_abs_norm"] = out["macd_32_96_norm"].abs()
            out["macd_32_96_sign"] = sign
            out["macd_32_96_cross_up"] = ((sign > 0) & (prev_sign <= 0)).astype(int)
            out["macd_32_96_cross_down"] = ((sign < 0) & (prev_sign >= 0)).astype(int)

    out["log_volume"] = np.log(out["volume"].clip(lower=1.0))
    out["volume_change"] = out["volume"].pct_change()
    out["volume_zscore_24"] = (out["volume"] - out["volume"].rolling(24).mean()) / (
        out["volume"].rolling(24).std() + 1e-9
    )
    out["volume_zscore_78"] = (out["volume"] - out["volume"].rolling(78).mean()) / (
        out["volume"].rolling(78).std() + 1e-9
    )
    minute_avg = out.groupby("minute_of_day")["volume"].transform("mean")
    out["volume_ratio_to_intraday_average"] = out["volume"] / (minute_avg + 1e-9)

    out["high_low_range"] = out["high"] - out["low"]
    out["close_open_range"] = out["close"] - out["open"]
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    prev_close = out["close"].shift(1)
    out["true_range"] = np.maximum.reduce(
        [
            (out["high"] - out["low"]).values,
            (out["high"] - prev_close).abs().values,
            (out["low"] - prev_close).abs().values,
        ]
    )
    out["atr_14"] = pd.Series(out["true_range"]).rolling(14).mean().values
    out["atr_78"] = pd.Series(out["true_range"]).rolling(78).mean().values

    g = out.groupby("trade_date")
    out["intraday_return_from_open"] = out["close"] / g["open"].transform("first") - 1.0
    out["intraday_high_so_far"] = g["high"].cummax()
    out["intraday_low_so_far"] = g["low"].cummin()
    out["distance_from_intraday_high"] = out["close"] / out["intraday_high_so_far"] - 1.0
    out["distance_from_intraday_low"] = out["close"] / out["intraday_low_so_far"] - 1.0
    out["intraday_volume_cumsum"] = g["volume"].cumsum()

    out["simple_return"] = out["close"].pct_change()
    out["price_change"] = out["close"].diff()
    out["future_return_sign"] = np.sign(out["simple_return"].shift(-1)).fillna(0)
    out["target_return_next"] = out["simple_return"].shift(-1)
    future_vol_horizon = int(cfg.get("training", {}).get("decoder_tft_aux_future_vol_horizon", 12))
    future_log_returns = out["log_return"].shift(-1)
    future_realized_vol = future_log_returns.rolling(future_vol_horizon).std().shift(-(future_vol_horizon - 1))
    future_downside_semivariance = (
        future_log_returns.clip(upper=0.0).pow(2).rolling(future_vol_horizon).mean().shift(-(future_vol_horizon - 1))
    )
    out["target_future_realized_vol_12"] = future_realized_vol
    out["target_future_vol_regime_12"] = (future_realized_vol > (out["ewm_vol_78"] * 1.1)).astype(float)
    out["target_future_downside_semivariance_12"] = future_downside_semivariance
    out["trade_allowed"] = ((out["is_first_5min"] == 0) & (out["is_last_5min"] == 0)).astype(int)
    return out.dropna().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ts_col = cfg["data"]["timestamp_col"]
    processed_path = abs_path(cfg["data"]["processed_path"])
    features_path = abs_path(cfg["data"]["features_path"])

    bars = pd.read_csv(processed_path, parse_dates=[ts_col]).sort_values(ts_col)
    feat = build_features(bars, cfg)
    ensure_parent(features_path)
    feat.to_parquet(features_path, index=False)
    print("Saved features.")


if __name__ == "__main__":
    main()
