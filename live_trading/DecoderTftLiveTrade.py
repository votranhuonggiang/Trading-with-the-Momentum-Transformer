"""5-minute live trader for the current decoder_tft stack."""

from __future__ import annotations

import json
import os
import sys
import time
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import numpy as np
import pandas as pd
import torch

try:
    import psycopg
except ImportError:  # pragma: no cover - only needed in live runtime
    psycopg = None

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import load_config  # noqa: E402
from feature_engineering import build_features  # noqa: E402
from models.decoder_tft import DecoderTft  # noqa: E402

from bundle_utils import load_json, load_manifest, scale_feature_frame  # noqa: E402
from dnse_broker import DNSEBroker, MARKET_DERIVATIVE, ORDER_TYPE_LO, Position, TokenExpiredError  # noqa: E402
from simple_logger import SimpleLogger  # noqa: E402


POSITION_LABEL = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
_VN_TZ = "Asia/Ho_Chi_Minh"


@dataclass
class ThresholdState:
    current_normalized_position: float
    current_contracts: int


def _round_away_from_zero(value: float) -> int:
    return int(np.sign(value) * np.floor(abs(value) + 0.5))


def _quantize_contracts(normalized_position: float, execution_policy: dict[str, Any]) -> int:
    if not bool(execution_policy.get("integer_contract_execution", False)):
        max_position = float(execution_policy.get("decoder_tft_confidence_scaled_max_position", 1.0))
        return int(np.clip(normalized_position, -max_position, max_position))

    max_contracts = int(execution_policy.get("max_contracts", 1))
    rounding_mode = str(execution_policy.get("contract_rounding_mode", "nearest")).lower()
    scaled = float(normalized_position) * max_contracts
    if rounding_mode == "nearest":
        contracts = _round_away_from_zero(scaled)
    elif rounding_mode == "floor":
        contracts = int(np.sign(scaled) * np.floor(abs(scaled)))
    elif rounding_mode == "ceil":
        contracts = int(np.sign(scaled) * np.ceil(abs(scaled)))
    else:
        raise ValueError(f"Unsupported contract_rounding_mode: {rounding_mode}")
    return int(np.clip(contracts, -max_contracts, max_contracts))


def _apply_signal_post_processing(
    raw_signal: float,
    regime_probability: float | None,
    execution_policy: dict[str, Any],
) -> float:
    signal = float(np.clip(raw_signal, -1.0, 1.0))

    if bool(execution_policy.get("decoder_tft_confidence_scaled_exposure", False)):
        tau = float(execution_policy.get("decoder_tft_confidence_gate_tau", 0.0))
        alpha = float(execution_policy.get("decoder_tft_confidence_scale_alpha", 1.0))
        max_position = float(execution_policy.get("decoder_tft_confidence_scaled_max_position", 1.0))
        if abs(signal) < tau:
            signal = 0.0
        else:
            signal = float(np.sign(signal) * (abs(signal) ** alpha))
        signal = float(np.clip(signal, -max_position, max_position))

    if bool(execution_policy.get("decoder_tft_regime_conditioned_exposure", False)) and regime_probability is not None:
        p_high = float(np.clip(regime_probability, 0.0, 1.0))
        if bool(execution_policy.get("decoder_tft_asymmetric_regime_scaling", False)):
            if signal >= 0.0:
                low_scale = float(execution_policy.get("decoder_tft_low_vol_long_scale", 1.0))
                high_scale = float(execution_policy.get("decoder_tft_high_vol_long_scale", 1.0))
            else:
                low_scale = float(execution_policy.get("decoder_tft_low_vol_short_scale", 1.0))
                high_scale = float(execution_policy.get("decoder_tft_high_vol_short_scale", 1.0))
        else:
            low_scale = float(execution_policy.get("decoder_tft_low_vol_scale", 1.0))
            high_scale = float(execution_policy.get("decoder_tft_high_vol_scale", 1.0))
        signal = signal * (low_scale * (1.0 - p_high) + high_scale * p_high)

    return float(np.clip(signal, -1.0, 1.0))


def _threshold_transition(signal: float, can_open: bool, state: ThresholdState, execution_policy: dict[str, Any]) -> float:
    current = float(np.clip(state.current_normalized_position, -1.0, 1.0))
    if current == 0.0:
        if can_open and signal > float(execution_policy["long_entry_threshold"]):
            return float(np.clip(signal, 0.0, 1.0))
        if can_open and signal < -float(execution_policy["short_entry_threshold"]):
            return float(np.clip(signal, -1.0, 0.0))
        return 0.0
    if current > 0.0:
        if signal < float(execution_policy["long_exit_threshold"]):
            return 0.0
        if can_open and signal < -float(execution_policy["long_to_short_reverse_threshold"]):
            return float(np.clip(signal, -1.0, 0.0))
        return float(np.clip(signal, 0.0, 1.0))

    if signal > -float(execution_policy["short_exit_threshold"]):
        return 0.0
    if can_open and signal > float(execution_policy["short_to_long_reverse_threshold"]):
        return float(np.clip(signal, 0.0, 1.0))
    return float(np.clip(signal, -1.0, 0.0))


class DecoderTftLiveTrade:
    SESSION_OPEN = dt.time(9, 0)
    SESSION_CLOSE = dt.time(14, 30)
    FORCE_FLAT_AT = dt.time(14, 25)

    def __init__(
        self,
        bundle_dir: str | Path,
        data_symbol: str,
        broker: DNSEBroker,
        connection_params: dict[str, Any],
        poll_interval_s: int = 30,
        slippage_accept: float = 1.5,
        tick_size: float = 0.1,
        device: str = "cpu",
        dry_run: bool = True,
        log_path: str | Path = "live_trading/decoder_tft_live.log",
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest = load_manifest(self.bundle_dir / "manifest.json")
        self.cfg = load_config(str(self.bundle_dir / self.manifest.config_file))
        self.feature_cols: list[str] = load_json(self.bundle_dir / self.manifest.feature_columns_file)
        self.scaler = load_json(self.bundle_dir / self.manifest.scaler_file)
        self.execution_policy = load_json(self.bundle_dir / self.manifest.execution_policy_file)
        self.data_symbol = data_symbol
        self.broker = broker
        self.connection_params = connection_params
        self.poll_interval_s = int(poll_interval_s)
        self.slippage_accept = float(slippage_accept)
        self.tick_size = float(tick_size)
        self.device = torch.device(device)
        self.dry_run = dry_run
        self.logger = SimpleLogger(self.__class__.__name__, log_path=log_path)
        self.last_processed_ts: pd.Timestamp | None = None
        self.prev_close: float | None = None
        self.simulated_contracts: int = 0
        self.cum_net_pnl_points: float = 0.0
        self.cum_net_return: float = 0.0
        self.trading_cfg = dict(self.cfg.get("trading", {}))
        self.contract_multiplier = float(self.trading_cfg.get("contract_multiplier", 100000.0))
        self.fee_vsdc = float(self.trading_cfg.get("fee_vsdc_vnd_per_contract_per_side", 5000.0))
        self.fee_hnx = float(self.trading_cfg.get("fee_hnx_vnd_per_contract_per_side", 2700.0))
        self.fee_ctck = float(self.trading_cfg.get("fee_ctck_vnd_per_contract_per_side", 2700.0))
        self.margin_rate = float(self.trading_cfg.get("margin_rate", 0.1848))
        self.transfer_tax_rate = float(self.trading_cfg.get("transfer_tax_rate", 0.001))
        self.logs_dir = PROJECT_ROOT / "live_trading" / "logs"
        self.daily_summary_path = self.logs_dir / "live_dryrun_daily_summary.csv"
        self.decision_log_path = self.logs_dir / "live_dryrun_decisions.csv"
        self.trade_log_path = self.logs_dir / "live_dryrun_trades.csv"
        self.equity_log_path = self.logs_dir / "live_dryrun_equity_curve.csv"
        self.daily_summary: dict[str, dict[str, float]] = {}
        if self.dry_run:
            self.logs_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_rel = self.manifest.checkpoint_file
        if not checkpoint_rel:
            raise FileNotFoundError(
                "Bundle has no checkpoint_file in manifest.json. Export or copy a decoder_tft checkpoint first."
            )
        checkpoint_path = self.bundle_dir / checkpoint_rel
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: Path) -> DecoderTft:
        model = DecoderTft(
            input_size=len(self.feature_cols),
            hidden_size=int(self.manifest.hidden_size),
            num_heads=int(self.manifest.num_heads),
            dropout=float(self.manifest.dropout),
            multitask=(
                self.manifest.multitask_aux_vol_loss_weight > 0.0
                or self.manifest.multitask_regime_loss_weight > 0.0
                or self.manifest.multitask_downside_loss_weight > 0.0
            ),
            dual_position_heads=bool(self.manifest.dual_position_heads),
        ).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def _append_csv_row(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _compute_costs(self, prev_contracts: int, next_contracts: int, close_price: float) -> dict[str, float]:
        buy_units = float(max(next_contracts - prev_contracts, 0))
        sell_units = float(max(prev_contracts - next_contracts, 0))
        per_side_fixed_fee = self.fee_vsdc + self.fee_hnx + self.fee_ctck
        tax_vnd_per_contract = self.margin_rate * self.contract_multiplier * self.transfer_tax_rate * close_price
        buy_cost_vnd = buy_units * per_side_fixed_fee
        sell_fixed_cost_vnd = sell_units * per_side_fixed_fee
        sell_tax_cost_vnd = sell_units * tax_vnd_per_contract
        sell_cost_vnd = sell_fixed_cost_vnd + sell_tax_cost_vnd
        cost_vnd = buy_cost_vnd + sell_cost_vnd
        cost_points = cost_vnd / self.contract_multiplier
        return {
            "buy_units": buy_units,
            "sell_units": sell_units,
            "turnover": abs(float(next_contracts - prev_contracts)),
            "tax_vnd_per_contract": tax_vnd_per_contract,
            "buy_cost_vnd": buy_cost_vnd,
            "sell_fixed_cost_vnd": sell_fixed_cost_vnd,
            "sell_tax_cost_vnd": sell_tax_cost_vnd,
            "sell_cost_vnd": sell_cost_vnd,
            "cost_vnd": cost_vnd,
            "cost_points": cost_points,
        }

    def _update_daily_summary(
        self,
        timestamp: pd.Timestamp,
        turnover: float,
        cost_vnd: float,
        gross_pnl_points: float,
        net_pnl_points: float,
        trade_count_increment: int,
    ) -> None:
        trade_day = str(timestamp.date())
        day = self.daily_summary.setdefault(
            trade_day,
            {
                "trade_day": trade_day,
                "bars_processed": 0.0,
                "number_of_transactions": 0.0,
                "total_turnover": 0.0,
                "total_cost_vnd": 0.0,
                "gross_pnl_points": 0.0,
                "net_pnl_points": 0.0,
                "cum_net_pnl_points": 0.0,
            },
        )
        day["bars_processed"] += 1.0
        day["number_of_transactions"] += float(trade_count_increment)
        day["total_turnover"] += float(turnover)
        day["total_cost_vnd"] += float(cost_vnd)
        day["gross_pnl_points"] += float(gross_pnl_points)
        day["net_pnl_points"] += float(net_pnl_points)
        day["cum_net_pnl_points"] = float(self.cum_net_pnl_points)
        pd.DataFrame(sorted(self.daily_summary.values(), key=lambda x: x["trade_day"])).to_csv(
            self.daily_summary_path,
            index=False,
        )

    def _record_dry_run(
        self,
        decision: dict[str, Any],
        current_contracts_before: int,
        target_contracts: int,
        delta_contracts: int,
    ) -> None:
        ts = pd.Timestamp(decision["timestamp"])
        close_price = float(decision["close"])
        price_change = 0.0 if self.prev_close is None else close_price - float(self.prev_close)
        gross_pnl_points = float(current_contracts_before) * float(price_change)
        costs = self._compute_costs(current_contracts_before, target_contracts, close_price)
        net_pnl_points = gross_pnl_points - float(costs["cost_points"])
        cost_return = 0.0 if close_price == 0 else float(costs["cost_vnd"]) / (close_price * self.contract_multiplier)
        gross_return = 0.0 if close_price == 0 else (float(current_contracts_before) * price_change) / (
            close_price * self.contract_multiplier
        )
        net_return = gross_return - cost_return
        self.cum_net_pnl_points += net_pnl_points
        self.cum_net_return += net_return

        decision_row = {
            "timestamp": ts.isoformat(),
            "close": close_price,
            "raw_signal": float(decision["raw_signal"]),
            "processed_signal": float(decision["processed_signal"]),
            "regime_probability": decision["regime_probability"],
            "target_normalized_position": float(decision["target_normalized_position"]),
            "current_contracts_before": int(current_contracts_before),
            "target_contracts": int(target_contracts),
            "delta_contracts": int(delta_contracts),
            "trade_allowed": bool(decision["trade_allowed"]),
            "dry_run": True,
        }
        self._append_csv_row(self.decision_log_path, decision_row)

        if delta_contracts != 0:
            action = "buy" if delta_contracts > 0 else "sell"
            trade_row = {
                "timestamp": ts.isoformat(),
                "close": close_price,
                "action": action,
                "contracts_before": int(current_contracts_before),
                "contracts_after": int(target_contracts),
                "delta_contracts": int(delta_contracts),
                "turnover": float(costs["turnover"]),
                "buy_units": float(costs["buy_units"]),
                "sell_units": float(costs["sell_units"]),
                "buy_cost_vnd": float(costs["buy_cost_vnd"]),
                "sell_fixed_cost_vnd": float(costs["sell_fixed_cost_vnd"]),
                "sell_tax_cost_vnd": float(costs["sell_tax_cost_vnd"]),
                "total_cost_vnd": float(costs["cost_vnd"]),
                "cost_points": float(costs["cost_points"]),
            }
            self._append_csv_row(self.trade_log_path, trade_row)

        equity_row = {
            "timestamp": ts.isoformat(),
            "close": close_price,
            "previous_close": self.prev_close,
            "price_change": float(price_change),
            "contracts_held_during_bar": int(current_contracts_before),
            "contracts_after_decision": int(target_contracts),
            "gross_pnl_points": float(gross_pnl_points),
            "cost_points": float(costs["cost_points"]),
            "cost_vnd": float(costs["cost_vnd"]),
            "net_pnl_points": float(net_pnl_points),
            "gross_return": float(gross_return),
            "net_return": float(net_return),
            "cum_net_pnl_points": float(self.cum_net_pnl_points),
            "cum_net_return": float(self.cum_net_return),
        }
        self._append_csv_row(self.equity_log_path, equity_row)
        self._update_daily_summary(
            timestamp=ts,
            turnover=float(costs["turnover"]),
            cost_vnd=float(costs["cost_vnd"]),
            gross_pnl_points=float(gross_pnl_points),
            net_pnl_points=float(net_pnl_points),
            trade_count_increment=1 if delta_contracts != 0 else 0,
        )
        self.simulated_contracts = int(target_contracts)
        self.prev_close = close_price

    def fetch_latest_data(self, cursor) -> pd.DataFrame:
        query = """
        SELECT
            dateadd('h', 7, timestamp) AS ts_vn,
            first(open)  AS open,
            max(high)    AS high,
            min(low)     AS low,
            last(close)  AS close,
            last(volume) AS volume
        FROM derivative_ohcl_krx
        WHERE symbol = %s
          AND timestamp >= dateadd('d', -20, now())
        SAMPLE BY 5m;
        """
        cursor.execute(query, (self.data_symbol,))
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return (
            df.sort_values("timestamp")
            .drop_duplicates(subset="timestamp", keep="last")
            .reset_index(drop=True)
        )

    def get_current_contracts(self) -> tuple[int, list[Position]]:
        positions = self.broker.get_open_positions()
        net_contracts = 0
        for p in positions:
            if p.side.upper() == "BUY":
                net_contracts += int(p.open_qty)
            elif p.side.upper() == "SELL":
                net_contracts -= int(p.open_qty)
        return net_contracts, positions

    def infer_target_contracts(self, bars: pd.DataFrame, current_contracts: int) -> dict[str, Any]:
        if len(bars) < max(int(self.manifest.sequence_length) + 5, 300):
            raise ValueError(f"Need more bars before inference. Current rows={len(bars)}")

        features = build_features(bars.copy(), self.cfg, include_targets=False)
        features = features.sort_values(self.manifest.timestamp_col).reset_index(drop=True)
        latest_feature_row = features.iloc[-1]
        trade_allowed = bool(int(latest_feature_row["trade_allowed"]) == 1)
        features_scaled = scale_feature_frame(features, self.scaler, self.feature_cols)
        window = features_scaled.tail(int(self.manifest.sequence_length))
        if len(window) != int(self.manifest.sequence_length):
            raise ValueError("Feature window shorter than required sequence_length.")

        latest_bar = latest_feature_row
        latest_ts = pd.Timestamp(latest_bar[self.manifest.timestamp_col])
        x = torch.tensor(
            window[self.feature_cols].astype(np.float32).to_numpy(),
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(x)
        if isinstance(out, dict):
            raw_signal = float(out["position"].item())
            regime_probability = None
            if "future_vol_regime_logit" in out:
                regime_probability = float(torch.sigmoid(out["future_vol_regime_logit"]).item())
        else:
            raw_signal = float(out.item())
            regime_probability = None

        processed_signal = _apply_signal_post_processing(raw_signal, regime_probability, self.execution_policy)
        max_contracts = int(self.execution_policy.get("max_contracts", 1))
        current_normalized = float(current_contracts / max_contracts) if max_contracts > 0 else 0.0
        target_normalized = _threshold_transition(
            signal=processed_signal,
            can_open=trade_allowed,
            state=ThresholdState(
                current_normalized_position=current_normalized,
                current_contracts=current_contracts,
            ),
            execution_policy=self.execution_policy,
        )

        bar_time = latest_ts.time()
        if bar_time >= self.FORCE_FLAT_AT or bar_time < self.SESSION_OPEN or bar_time >= self.SESSION_CLOSE:
            target_normalized = 0.0

        target_contracts = _quantize_contracts(target_normalized, self.execution_policy)
        return {
            "timestamp": latest_ts,
            "raw_signal": raw_signal,
            "processed_signal": processed_signal,
            "regime_probability": regime_probability,
            "target_normalized_position": target_normalized,
            "target_contracts": target_contracts,
            "close": float(latest_bar["close"]),
            "trade_allowed": trade_allowed,
        }

    def _place_delta_order(self, delta_contracts: int, ref_price: float) -> None:
        if delta_contracts == 0:
            return
        target_pos = 1 if delta_contracts > 0 else -1
        if target_pos == 1:
            limit_px = ref_price + self.slippage_accept
        else:
            limit_px = ref_price - self.slippage_accept
        limit_px = round(round(limit_px / self.tick_size) * self.tick_size, 2)

        ack = self.broker.open_position(
            target_pos=target_pos,
            quantity=abs(delta_contracts),
            price=limit_px,
            order_type=ORDER_TYPE_LO,
            dry_run=self.dry_run,
        )
        self.logger.log(
            f"Submitted {'BUY' if target_pos > 0 else 'SELL'} {abs(delta_contracts)} @ {limit_px:.2f} | "
            f"ack_status={ack.status}"
        )

    def run(self) -> None:
        if psycopg is None:
            raise ImportError("psycopg is required for live data access but is not installed.")

        self.logger.log(
            f"Starting decoder_tft live trader | data_symbol={self.data_symbol} | broker_symbol={self.broker.symbol} | "
            f"bundle={self.bundle_dir.name} | dry_run={self.dry_run}"
        )
        conn = psycopg.connect(**self.connection_params)
        cursor = conn.cursor()
        try:
            while True:
                try:
                    bars = self.fetch_latest_data(cursor)
                    if bars.empty:
                        self.logger.log("No bars returned from QuestDB.")
                        time.sleep(self.poll_interval_s)
                        continue

                    if self.dry_run:
                        current_contracts = int(self.simulated_contracts)
                    else:
                        current_contracts, _ = self.get_current_contracts()
                    decision = self.infer_target_contracts(bars, current_contracts)
                    ts = decision["timestamp"]
                    if self.last_processed_ts is not None and ts <= self.last_processed_ts:
                        time.sleep(self.poll_interval_s)
                        continue

                    self.last_processed_ts = ts
                    delta_contracts = int(decision["target_contracts"]) - int(current_contracts)
                    self.logger.log(
                        f"{ts} | close={decision['close']:.2f} | raw={decision['raw_signal']:+.4f} | "
                        f"processed={decision['processed_signal']:+.4f} | current={current_contracts} | "
                        f"target={decision['target_contracts']} | delta={delta_contracts}"
                    )
                    if self.dry_run:
                        self._record_dry_run(
                            decision=decision,
                            current_contracts_before=int(current_contracts),
                            target_contracts=int(decision["target_contracts"]),
                            delta_contracts=int(delta_contracts),
                        )
                    if delta_contracts != 0:
                        self._place_delta_order(delta_contracts, ref_price=float(decision["close"]))

                    time.sleep(self.poll_interval_s)
                except TokenExpiredError as exc:
                    self.logger.log(f"Trading token expired: {exc}")
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    self.logger.log(f"Loop error: {exc}")
                    time.sleep(self.poll_interval_s)
        except KeyboardInterrupt:
            self.logger.log("Stopped by user.")
        finally:
            try:
                cursor.close()
                conn.close()
            except Exception:
                pass
            self.logger.log("Done.")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    bundle_dir = os.environ.get(
        "DECODER_TFT_LIVE_BUNDLE",
        str(PROJECT_ROOT / "live_trading" / "bundles" / "decoder_tft_main_branch_5m"),
    )
    connection = {
        "host": os.environ["QUESTDB_HOST"],
        "port": int(os.environ.get("QUESTDB_PORT", 8812)),
        "user": os.environ.get("QUESTDB_USER", "admin"),
        "password": os.environ.get("QUESTDB_PASSWORD", "quest"),
        "dbname": os.environ.get("QUESTDB_DBNAME", "qdb"),
    }
    broker = DNSEBroker(
        api_key=os.environ["DNSE_API_KEY"],
        api_secret=os.environ["DNSE_API_SECRET"],
        symbol=os.environ.get("DNSE_BROKER_SYMBOL", "41I1G6000"),
        market_type=MARKET_DERIVATIVE,
    )
    token = os.environ.get("DNSE_TRADING_TOKEN")
    if token:
        broker.set_trading_token(token)

    trader = DecoderTftLiveTrade(
        bundle_dir=bundle_dir,
        data_symbol=os.environ.get("DECODER_TFT_DATA_SYMBOL", "VN30F1M"),
        broker=broker,
        connection_params=connection,
        poll_interval_s=int(os.environ.get("DECODER_TFT_POLL_INTERVAL_S", 30)),
        slippage_accept=float(os.environ.get("DECODER_TFT_SLIPPAGE_ACCEPT", 1.5)),
        tick_size=float(os.environ.get("DECODER_TFT_TICK_SIZE", 0.1)),
        device=os.environ.get("DECODER_TFT_DEVICE", "cpu"),
        dry_run=os.environ.get("DECODER_TFT_DRY_RUN", "true").lower() == "true",
    )
    trader.run()
