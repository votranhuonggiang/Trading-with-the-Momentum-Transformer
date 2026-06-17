"""
SharpeHeadLiveTrade.py — Real-time paper trading with the JEPA Sharpe-head policy,
backed by the official DNSE OpenAPI SDK via ``dnse_broker.DNSEBroker``.

Surgical port of the original ``live_trade.py``. Broker interaction, position
reconciliation, force-flat rule, cooldown, flip-as-netting-order, token-expiry
handling, and shutdown flattening are unchanged. The only differences:

  • Model:   SharpePolicyHead (continuous tanh) instead of JEPAClassifier (softmax)
  • Decision: |z| < θ_flat → flat;  z > 0 → long;  z < 0 → short
  • Logging:  z + target in the heartbeat instead of class probabilities

The discretisation threshold ``θ_flat`` is loaded from the checkpoint metadata
(set by ``train_sharpe_head.ipynb``) and can be overridden via the ``config``
dict for production tuning without retraining.

Operator workflow (once per session):
    # 1. Mint a trading token using your own OTP script:
    #        export DNSE_TRADING_TOKEN=<token>
    #
    # 2. Start the strategy:
    #        python SharpeHeadLiveTrade.py
    #
    # For dry-run validation (no orders sent), the trading token is not needed.
"""
from __future__ import annotations

import os
import time
import datetime as dt
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import psycopg
import torch

from logger import LoggerService

from dataset      import build_features
from jepa         import JEPA
from head_sharpe  import SharpePolicyHead

from dnse_broker import (
    DNSEBroker,
    Position,
    BrokerError,
    TokenExpiredError,
    MARKET_DERIVATIVE,
    ORDER_TYPE_LO,
)

_VN_TZ = "Asia/Ho_Chi_Minh"  # UTC+7


POSITION_LABEL = {-1: "SHORT", 0: "FLAT", +1: "LONG"}


class SharpeHeadLiveTrade:
    """
    Live trading strategy backed by the JEPA Sharpe head and DNSE OpenAPI.

    The strategy maintains *no* local position state. Each cycle it queries
    the broker for currently open deals on the configured symbol, decides the
    target position from the model, and submits the transition order(s).
    """

    # -- Session boundaries -------------------------------------------------
    SESSION_OPEN   = dt.time(9, 0)
    SESSION_CLOSE  = dt.time(14, 30)
    FORCE_FLAT_AT  = dt.time(14, 29)        # mirrors backtest.run_backtest

    # -- Default hyperparameters -------------------------------------------
    DEFAULTS = {
        "window_size":     60,    # bars fed to the encoder
        "min_history":    200,    # minimum bars in df before we'll act
        "poll_interval_s": 30,    # seconds between polls
        "cooldown_bars":    1,    # bars to wait after a trade
        "position_size":    1,    # contracts per fresh entry
        "slippage_accept": 0.5,   # price offset applied to LO limit (cross the spread)
        "tick_size":       0.1,   # price grid (VN30F1M = 0.1 index points)
        # theta_flat is loaded from the checkpoint metadata by default. Set this
        # key in the config dict to override (e.g. trade more aggressively with
        # 0.3 or more selectively with 0.7 without retraining).
        "theta_flat":     None,
        "log_path": "live_trade_log.txt",
    }

    def __init__(
        self,
        symbol:            str,
        connection_params: dict,
        jepa_ckpt:         str,
        head_ckpt:         str,
        broker:            DNSEBroker,
        config:            Optional[dict] = None,
        device:            str = "cpu",
        dry_run:           bool = False,
    ):
        self.logger = LoggerService(self.__class__.__name__)
        self.symbol = symbol
        self.connection_params = connection_params
        self.broker = broker
        self.dry_run = dry_run
        self.device = torch.device(device)

        cfg = {**self.DEFAULTS, **(config or {})}
        self.window_size: int       = int(cfg["window_size"])
        self.min_history: int       = int(cfg["min_history"])
        self.poll_interval_s: int   = int(cfg["poll_interval_s"])
        self.cooldown_bars: int     = int(cfg["cooldown_bars"])
        self.position_size: int     = int(cfg["position_size"])
        self.slippage_accept: float = float(cfg["slippage_accept"])
        self.tick_size: float       = float(cfg["tick_size"])
        self.log_path: str          = cfg["log_path"]

        # -- Per-cycle state (broker is the source of truth for positions) --
        self.last_processed_ts: Optional[pd.Timestamp] = None
        self.cooldown_until:    Optional[pd.Timestamp] = None

        # -- Local position cache (same semantics as the original) ----------
        self._local_position: Optional[Position] = None
        self._reconciled_position_id: Optional[int] = None
        self._last_seen_price: float = 0.0

        # -- Load model -----------------------------------------------------
        # _load_model returns the head plus the θ_flat value baked into the
        # checkpoint by train_sharpe_head.ipynb. Config can override.
        self.jepa, self.head, theta_flat_ckpt = self._load_model(jepa_ckpt, head_ckpt)
        self.theta_flat: float = (
            float(cfg["theta_flat"]) if cfg["theta_flat"] is not None
            else float(theta_flat_ckpt)
        )
        self.logger.log(
            f"Loaded JEPA from {jepa_ckpt} + Sharpe head from {head_ckpt}  "
            f"|  θ_flat={self.theta_flat:.3f}"
            + (" (from config override)" if cfg["theta_flat"] is not None
               else " (from checkpoint)")
        )

    # =======================================================================
    # Model loading
    # =======================================================================

    def _load_model(self, jepa_ckpt: str, head_ckpt: str) -> Tuple[JEPA, SharpePolicyHead, float]:
        """
        Reconstruct JEPA + SharpePolicyHead from saved checkpoints.

        The head checkpoint carries the architecture config under "cfg" and the
        validation-tuned discretisation threshold under "theta_flat" — both are
        written by train_sharpe_head.ipynb.
        """
        jepa_state = torch.load(jepa_ckpt, map_location=self.device)
        head_state = torch.load(head_ckpt, map_location=self.device)
        arch = head_state.get("cfg", {})

        jepa = JEPA(
            in_features    = arch.get("F", 14),
            window_size    = arch.get("W", 60),
            patch_size     = arch.get("patch_size", 5),
            d_model        = arch.get("d_model", 192),
            d_z            = arch.get("d_z", 192),
            n_heads        = arch.get("n_heads", 6),
            n_enc_layers   = arch.get("n_enc_layers", 6),
            n_pred_layers  = arch.get("n_pred_layers", 3),
        ).to(self.device)
        jepa.load_state_dict(jepa_state["model"])
        jepa.eval()
        for p in jepa.parameters():
            p.requires_grad_(False)

        head = SharpePolicyHead(
            jepa,
            hidden  = arch.get("shp_hidden", 128),
            dropout = arch.get("shp_dropout", 0.1),
        ).to(self.device).eval()
        head.load_state_dict(head_state["state_dict"], strict=False)
        for p in head.parameters():
            p.requires_grad_(False)

        theta_flat = float(head_state.get("theta_flat", 0.20))
        return jepa, head, theta_flat

    # =======================================================================
    # Data fetch
    # =======================================================================

    def fetch_latest_data(self, cursor) -> pd.DataFrame:
        """
        Pull 3 days of 1-minute OHLCV bars from QuestDB. The 3-day buffer
        ensures we have enough history for the 60-bar model window plus any
        feature lookbacks (rolling volatility, MAs).
        """
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
          AND timestamp >= dateadd('d', -3, now())
        SAMPLE BY 1m;
        """
        try:
            cursor.execute(query, (self.symbol,))
            rows = cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(
                rows,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = (
                df.sort_values("timestamp")
                  .drop_duplicates(subset="timestamp", keep="last")
                  .set_index("timestamp")
            )
            return df
        except Exception as e:
            self.logger.log(f"fetch_latest_data error: {e}")
            return pd.DataFrame()

    # =======================================================================
    # Position state (read from broker)
    # =======================================================================

    def get_current_position(self) -> Tuple[int, List[Position]]:
        """
        Return the current position state on our symbol as ``(direction, positions)``.

        Truth model:
          * dry_run: ``self._local_position`` is the only source — broker has
            no real positions in this mode.
          * live: the broker is preferred when it returns a non-empty list
            (and on a hit we sync ``self._local_position`` to match). When
            the broker returns empty, we fall back to ``self._local_position``
            — this guards against API sync lag after a freshly-accepted
            order, and against undocumented broker-side filters that have
            hidden our position before.
        """
        if self.dry_run:
            p = self._local_position
            if p is None or p.open_qty <= 0:
                return 0, []
            direction = +1 if p.side.upper() == "BUY" else -1
            return direction, [p]

        try:
            # CRITICAL: filter by the BROKER's trading symbol (e.g. "41I1G6000"),
            # NOT by self.symbol (e.g. "VN30F1M" — the rolling alias used for
            # QuestDB data only). They differ when trading a specific expiry
            # contract while pulling data from a rolling-alias series.
            # Passing no symbol arg lets broker default to broker.symbol.
            positions = self.broker.get_open_positions()
        except BrokerError as e:
            self.logger.log(f"get_open_positions failed: {e}; falling back to local cache")
            positions = []

        buys  = [p for p in positions if p.side.upper() == "BUY"  and p.open_qty > 0]
        sells = [p for p in positions if p.side.upper() == "SELL" and p.open_qty > 0]
        if buys and sells:
            self.logger.log(
                f"WARN: both BUY ({len(buys)}) and SELL ({len(sells)}) positions "
                f"open simultaneously for {self.symbol}"
            )
        if buys:
            if self._reconciled_position_id is not None:
                self.logger.log(
                    f"[reconcile] broker now confirms BUY position — "
                    f"local cache validated, resuming broker as primary"
                )
                self._reconciled_position_id = None
            self._local_position = buys[0]
            return +1, buys
        if sells:
            if self._reconciled_position_id is not None:
                self.logger.log(
                    f"[reconcile] broker now confirms SELL position — "
                    f"local cache validated, resuming broker as primary"
                )
                self._reconciled_position_id = None
            self._local_position = sells[0]
            return -1, sells

        if self._local_position is not None and self._local_position.open_qty > 0:
            p = self._local_position
            direction = +1 if p.side.upper() == "BUY" else -1
            if self._reconciled_position_id != p.id:
                self._reconciled_position_id = p.id
                self.logger.log(
                    f"[reconcile] broker shows FLAT but local cache says "
                    f"{POSITION_LABEL[direction]} (id={p.id}, qty={p.open_qty}, "
                    f"cost={p.cost_price}) — trusting local until broker confirms"
                )
            return direction, [p]

        return 0, []

    # =======================================================================
    # Signal detection (model forward + decision rule)
    # =======================================================================

    def detect_signal(
        self, df: pd.DataFrame, current_pos: int
    ) -> Tuple[Optional[str], str, Dict[str, Any]]:
        """
        Run the Sharpe head on the most recent fully-closed bar and decide
        the transition action given the broker-reported current position.

        Decision rule (mirrors ContinuousModelPolicy in head_sharpe.py):
            z = head(window)              # continuous in (-1, 1)
            |z| < θ_flat  → target = 0    (flat)
            z >=  θ_flat  → target = +1   (long)
            z <= -θ_flat  → target = -1   (short)
        """
        if len(df) < self.min_history:
            return None, f"insufficient data ({len(df)}<{self.min_history})", {}

        # Last fully-closed bar (the last row may still be the active minute).
        curr_ts = df.index[-2]
        curr_close = float(df.iloc[-2]["close"])

        info: Dict[str, Any] = {
            "ts": curr_ts, "price": curr_close, "n_bars": len(df),
            "current_pos": current_pos,
        }

        # ---- Deduplicate ---------------------------------------------------
        if self.last_processed_ts is not None and curr_ts <= self.last_processed_ts:
            return None, "already processed", info

        # ---- Force-flat rule ----------------------------------------------
        bar_time = curr_ts.time()
        if bar_time >= self.FORCE_FLAT_AT or bar_time < self.SESSION_OPEN \
                or bar_time >= self.SESSION_CLOSE:
            if current_pos != 0:
                return "force_exit", "outside session / force-flat", info
            return None, "outside session", info

        # ---- Cooldown ------------------------------------------------------
        if self.cooldown_until is not None and curr_ts < self.cooldown_until:
            return None, f"cooldown until {self.cooldown_until}", info

        # ---- Features -----------------------------------------------------
        try:
            df_for_features = df.reset_index().rename(columns={"index": "timestamp"})
            features = build_features(df_for_features, train_mask=None)
        except TypeError:
            features = build_features(df.reset_index())
        except Exception as e:
            return None, f"feature computation failed: {e}", info

        if features is None or len(features) < self.window_size + 1:
            return None, "insufficient features", info

        end_i = len(features) - 1
        x = features[end_i - self.window_size : end_i]
        if x.shape[0] != self.window_size:
            return None, f"window shape mismatch ({x.shape})", info

        # ---- Inference: Sharpe head -> continuous z -> discrete action -----
        try:
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
                z = float(self.head(xt).item())     # continuous in (-1, 1)
                if abs(z) < self.theta_flat:
                    target_pos = 0
                elif z > 0:
                    target_pos = +1
                else:
                    target_pos = -1
        except Exception as e:
            return None, f"inference failed: {e}", info

        info["z"]          = round(z, 4)
        info["target_pos"] = target_pos

        # ---- Transition decision (unchanged from classifier version) -------
        if target_pos == current_pos:
            return None, f"hold {POSITION_LABEL[current_pos]}", info

        if current_pos == 0 and target_pos == 1:
            return "long_entry",  f"long, z={z:+.3f}",  info
        if current_pos == 0 and target_pos == -1:
            return "short_entry", f"short, z={z:+.3f}", info
        if current_pos != 0 and target_pos == 0:
            return "signal_exit", f"flat, z={z:+.3f}",  info
        if current_pos == 1 and target_pos == -1:
            return "flip_short",  f"long->short, z={z:+.3f}", info
        if current_pos == -1 and target_pos == 1:
            return "flip_long",   f"short->long, z={z:+.3f}", info

        return None, "no transition", info

    # =======================================================================
    # Order execution (broker calls) — UNCHANGED from live_trade.py
    # =======================================================================

    def execute_signal(
        self,
        signal:    str,
        reason:    str,
        info:      Dict[str, Any],
        positions: List[Position],
    ) -> None:
        """
        Translate a signal into broker calls.

        Transition strategy (unchanged from the classifier version):
          * long_entry / short_entry: open from flat with ``position_size``.
          * signal_exit / force_exit: close every open position via the
            dedicated close_position endpoint.
          * flip_long / flip_short: SINGLE netting order of size
            ``existing_qty + position_size`` so the existing position is
            closed and the opposite opened in one submission. The platform
            nets positions on the same symbol, so a BUY 2 against a 1-SHORT
            ends with a 1-LONG.
        """
        ts = info.get("ts", pd.Timestamp.now(_VN_TZ))
        px = info.get("price", 0.0)

        # Mark bar processed + start cooldown regardless of order outcome.
        self.last_processed_ts = ts
        self.cooldown_until = ts + pd.Timedelta(minutes=self.cooldown_bars)

        if signal == "long_entry":
            self._open(target_pos=+1, ts=ts, px=px, reason=reason)

        elif signal == "short_entry":
            self._open(target_pos=-1, ts=ts, px=px, reason=reason)

        elif signal in ("signal_exit", "force_exit"):
            self._close_all(positions, ts=ts, px=px, reason=reason)

        elif signal == "flip_short":
            flip_qty = sum(p.open_qty for p in positions) + self.position_size
            self._log_flip_estimate(positions, px, reason)
            self._open(target_pos=-1, ts=ts, px=px,
                       quantity=flip_qty, reason=f"flip {reason}")

        elif signal == "flip_long":
            flip_qty = sum(p.open_qty for p in positions) + self.position_size
            self._log_flip_estimate(positions, px, reason)
            self._open(target_pos=+1, ts=ts, px=px,
                       quantity=flip_qty, reason=f"flip {reason}")

        self._log_state(info, signal, reason)

    def _log_flip_estimate(self, positions: List[Position],
                           px: float, reason: str) -> None:
        """
        Log the estimated PnL of the existing position being implicitly closed
        by the upcoming netting order. The actual close is handled by the
        single 2x order; this is for log readability only.
        """
        for p in positions:
            if p.cost_price is None:
                continue
            direction = +1 if p.side.upper() == "BUY" else -1
            est_pnl = direction * (px - p.cost_price) * p.open_qty
            self.logger.log(
                f"FLIP-CLOSE {p.side} id={p.id} qty={p.open_qty} "
                f"cost={p.cost_price} @ ~{px:.2f} | est_pnl={est_pnl:+.2f} pts | {reason}"
            )

    def _open(self, target_pos: int, ts: pd.Timestamp, px: float,
              reason: str, quantity: Optional[int] = None) -> None:
        """
        Place an LO order with slippage tolerance applied to the limit price.
        Unchanged from the classifier version.
        """
        qty = quantity if quantity is not None else self.position_size
        side_word = "LONG" if target_pos == 1 else "SHORT"

        if target_pos == +1:
            limit_px = px + self.slippage_accept
        else:
            limit_px = px - self.slippage_accept
        limit_px = round(round(limit_px / self.tick_size) * self.tick_size, 2)

        self.logger.log(
            f"OPEN {side_word} qty={qty} ref={px:.2f} limit={limit_px:.2f} | {reason}"
        )
        try:
            ack = self.broker.open_position(
                target_pos = target_pos,
                quantity   = qty,
                price      = limit_px,
                order_type = ORDER_TYPE_LO,
                dry_run    = self.dry_run,
            )
            self.logger.log(
                f"  -> order accepted: id={ack.order_id} status={ack.status}"
            )
            self._local_position = Position(
                id           = ack.order_id if ack.order_id is not None
                                 else int(ts.timestamp()),
                account_no   = self.broker.account_no or "DRY",
                symbol       = self.broker.symbol,
                side         = "BUY" if target_pos == +1 else "SELL",
                status       = "OPEN",
                open_qty     = self.position_size,
                cost_price   = limit_px,
                market_price = limit_px,
                raw          = {},
            )
        except TokenExpiredError as e:
            self.logger.log(f"  -> TOKEN EXPIRED: {e}")
            raise
        except BrokerError as e:
            self.logger.log(f"  -> ORDER FAILED: {e}")

    def _close_all(self, positions: List[Position], ts: pd.Timestamp,
                   px: float, reason: str) -> None:
        """
        Close every position in the list by sending a netting LO order in
        the opposite direction. Unchanged from the classifier version.

        We deliberately do NOT use the broker's dedicated close_position
        endpoint because that endpoint requires a real DNSE position_id which
        only arrives when get_positions returns the position; in our setup
        the broker has been returning empty for get_positions, so Position.id
        in the local cache holds an order_id, not a position_id, and
        close_position responds with HTTP 403 when passed an order_id.
        """
        if not positions:
            self.logger.log(f"CLOSE skipped — no open positions | {reason}")
            return

        for p in positions:
            sim_pnl = 0.0
            if p.cost_price is not None:
                direction = +1 if p.side.upper() == "BUY" else -1
                sim_pnl = direction * (px - p.cost_price) * p.open_qty
            self.logger.log(
                f"CLOSE {p.side} position id={p.id} qty={p.open_qty} "
                f"cost={p.cost_price} @ {px:.2f} | pnl={sim_pnl:+.2f} pts | {reason}"
            )

            opposite_target = -1 if p.side.upper() == "BUY" else +1
            if opposite_target == +1:
                limit_px = px + self.slippage_accept
            else:
                limit_px = px - self.slippage_accept
            limit_px = round(round(limit_px / self.tick_size) * self.tick_size, 2)
            order_side = "SELL" if opposite_target == -1 else "BUY"

            try:
                ack = self.broker.open_position(
                    target_pos = opposite_target,
                    quantity   = p.open_qty,
                    price      = limit_px,
                    order_type = ORDER_TYPE_LO,
                    dry_run    = self.dry_run,
                )
                self.logger.log(
                    f"  -> close-netting {order_side} qty={p.open_qty} "
                    f"limit={limit_px:.2f} accepted: id={ack.order_id} "
                    f"status={ack.status}"
                )
                self._local_position = None
                self._reconciled_position_id = None
            except TokenExpiredError as e:
                self.logger.log(f"  -> TOKEN EXPIRED: {e}")
                raise
            except BrokerError as e:
                self.logger.log(f"  -> CLOSE FAILED for position {p.id}: {e}")

    # =======================================================================
    # Logging
    # =======================================================================

    def _log_state(self, info: Dict[str, Any],
                   signal: Optional[str], reason: str) -> None:
        """
        Append a structured line to the log file.

        Format (one line per decision, parseable by tail-script):
            ts | symbol | current_pos | px=... | z=... | target=... | signal=... | reason
        """
        ts = info.get("ts")
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts is not None else "?"
        px = info.get("price", 0.0)
        z = info.get("z")
        target_pos = info.get("target_pos", 0)
        current_pos = info.get("current_pos", 0)

        z_str = f"z={z:+.4f}" if z is not None else "z=N/A"
        line = (
            f"{ts_str} | {self.symbol} | {POSITION_LABEL[current_pos]:<5} | "
            f"px={px:.2f} | {z_str} | target={POSITION_LABEL[target_pos]:<5} | "
            f"signal={signal} | {reason}\n"
        )
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            self.logger.log(f"log write failed: {e}")

    # =======================================================================
    # Main loop
    # =======================================================================

    def run(self) -> None:
        """Polling loop. Blocks until KeyboardInterrupt, fatal error, or token expiry."""
        self.logger.log(
            f"Starting Sharpe-head live trade | data_symbol={self.symbol} -> "
            f"broker_symbol={self.broker.symbol} | "
            f"device={self.device} | dry_run={self.dry_run}"
        )
        self.logger.log(
            f"  window={self.window_size}  poll={self.poll_interval_s}s  "
            f"cooldown={self.cooldown_bars}bars  size={self.position_size}  "
            f"θ_flat={self.theta_flat:.3f}"
        )
        self.logger.log(
            f"  broker=DNSE  account={self.broker.account_no}  "
            f"market={self.broker.market_type}  "
            f"has_token={self.broker.has_trading_token}"
        )

        if not self.broker.has_trading_token and not self.dry_run:
            self.logger.log(
                "FATAL: no trading token. Mint one with your own OTP script "
                "and export DNSE_TRADING_TOKEN before restarting, or set "
                "dry_run=True to run without placing orders."
            )
            return

        try:
            conn = psycopg.connect(**self.connection_params)
            cursor = conn.cursor()
        except Exception as e:
            self.logger.log(f"FATAL: cannot connect to QuestDB: {e}")
            return

        consecutive_errors = 0
        MAX_CONSEC_ERRORS = 10

        try:
            while True:
                try:
                    # Refresh broker view of positions every cycle
                    current_pos, positions = self.get_current_position()

                    df = self.fetch_latest_data(cursor)
                    if df.empty:
                        self.logger.log("empty DataFrame; retrying")
                        time.sleep(self.poll_interval_s)
                        continue

                    signal, reason, info = self.detect_signal(df, current_pos)

                    # Heartbeat log every cycle. Shows the continuous z and the
                    # discretised target so the operator can see exactly how
                    # the decision rule fired. Both are absent for early exits
                    # (insufficient data, outside session) where inference
                    # never ran.
                    ts = info.get("ts")
                    ts_str = ts.strftime("%H:%M:%S") if ts is not None else "?"
                    px = info.get("price", 0.0)
                    if px > 0:
                        self._last_seen_price = px
                    z = info.get("z")
                    target_pos = info.get("target_pos")
                    if z is not None:
                        decision_str = (
                            f"z={z:+.3f} | target={POSITION_LABEL[target_pos]:<5} | "
                            f"θ={self.theta_flat:.2f} | "
                        )
                    else:
                        decision_str = ""
                    self.logger.log(
                        f"{ts_str} | {POSITION_LABEL[current_pos]:<5} | "
                        f"px={px:.2f} | {decision_str}{reason}"
                    )

                    if signal is not None:
                        self.execute_signal(signal, reason, info, positions)

                    consecutive_errors = 0
                    time.sleep(self.poll_interval_s)

                except TokenExpiredError as e:
                    # Fatal — operator must re-mint a token. Don't keep
                    # spinning and burning rate limit.
                    self.logger.log(
                        f"FATAL: trading token expired: {e}. "
                        "Re-mint a token (your own OTP script), "
                        "update DNSE_TRADING_TOKEN, and restart."
                    )
                    break
                except Exception as e:
                    consecutive_errors += 1
                    self.logger.log(
                        f"loop error ({consecutive_errors}/{MAX_CONSEC_ERRORS}): {e}"
                    )
                    if consecutive_errors >= MAX_CONSEC_ERRORS:
                        self.logger.log("too many consecutive errors; stopping")
                        break
                    time.sleep(self.poll_interval_s)

        except KeyboardInterrupt:
            self.logger.log("stopped by user")
        finally:
            # Shutdown safety: flatten any open position if we still hold one
            if not self.dry_run:
                try:
                    pos, positions = self.get_current_position()
                    if pos != 0 and positions:
                        close_px = self._last_seen_price
                        if close_px <= 0 and positions[0].cost_price:
                            close_px = float(positions[0].cost_price)
                        if close_px > 0:
                            self.logger.log(
                                f"flattening open position on shutdown @ ref={close_px:.2f}"
                            )
                            self._close_all(
                                positions,
                                ts=pd.Timestamp.now(_VN_TZ),
                                px=close_px,
                                reason="shutdown",
                            )
                        else:
                            self.logger.log(
                                "WARN: no price reference for shutdown close — "
                                "skipping. Manually close on DNSE app."
                            )
                except Exception as e:
                    self.logger.log(f"shutdown close failed: {e}")
            try:
                cursor.close()
                conn.close()
            except Exception:
                pass
            self.logger.log("done")


# ==============================================================================
# Example entry point
# ==============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    CONNECTION = {
        "host":     os.environ["QUESTDB_HOST"],
        "port":     int(os.environ.get("QUESTDB_PORT", 8812)),
        "user":     os.environ.get("QUESTDB_USER", "admin"),
        "password": os.environ.get("QUESTDB_PASSWORD", "quest"),
        "dbname":   os.environ.get("QUESTDB_DBNAME", "qdb"),
    }

    # Trading symbol (broker) vs data symbol (strategy):
    #   - broker.symbol  = "41I1G6000" — the specific expiring contract code,
    #                       used in DNSE order placement and position records.
    #   - strategy.symbol = "VN30F1M"  — the rolling alias used to query
    #                       QuestDB (which stores prices under this name).
    broker = DNSEBroker(
        api_key     = os.environ["DNSE_API_KEY"],
        api_secret  = os.environ["DNSE_API_SECRET"],
        symbol      = "41I1G6000",
        market_type = MARKET_DERIVATIVE,
    )

    # Trading token is loaded from env var; obtain it by running your own OTP
    # script once per session. Not needed for dry_run.
    token = os.environ.get("DNSE_TRADING_TOKEN")
    if token:
        broker.set_trading_token(token)

    strategy = SharpeHeadLiveTrade(
        symbol            = "VN30F1M",
        connection_params = CONNECTION,
        jepa_ckpt         = "models/jepa_v5/jepa_pretrain.pt",
        head_ckpt         = "models/head_sharpe/sharpe_chosen.pt",
        broker            = broker,
        config = {
            "window_size":     60,
            "poll_interval_s": 30,
            "cooldown_bars":    1,
            "position_size":    1,
            "slippage_accept": 1.5,
            "tick_size":       0.1,
            # theta_flat is auto-loaded from the checkpoint (set by
            # train_sharpe_head.ipynb). Uncomment to override without retraining:
            # "theta_flat":    0.50,
        },
        device  = "cpu",
        dry_run = False,
    )

    strategy.run()
