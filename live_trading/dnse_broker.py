"""
dnse_broker.py — Thin adapter over the official DNSE OpenAPI SDK.

Two-tier auth model (matches DNSE's design):
  * api_key + api_secret  -> all reads (get_accounts, get_positions,
                              get_balances, market data, etc.). HMAC-SHA256
                              signed per request.
  * trading_token         -> writes only (post_order, close_position,
                              cancel_order). Obtained via OTP exchange
                              (smart_otp or email_otp) once per session.

Account number is auto-discovered: at construction time you do NOT need to
know your derivative sub-account number. The first time the broker performs an
action that needs it, it calls ``get_accounts()`` and picks the first sub-account
with ``derivativeAccount: true`` and ``derivative.status == "ACTIVE"``. If you
have multiple derivative sub-accounts, pass ``account_no`` explicitly.

Authentication flow on first use of trading endpoints:
    1.  Build the broker with api_key + api_secret.
    2.  Call ``request_otp()`` once — DNSE emails an OTP.
    3.  Call ``authenticate(otp)`` with the OTP — broker stores the trading
        token. Until the token expires, the broker is ready to trade.

For pure dry-run / validation, no trading token is needed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from dnse.client import DNSEClient


# DNSE side codes (camelCase JSON):
#   NB = Normal Buy   (open long / close short)
#   NS = Normal Sell  (open short / close long)
DNSE_SIDE_BUY  = "NB"
DNSE_SIDE_SELL = "NS"

# Order type codes. Market-style orders (MTL/MOK/MAK) must NOT include the
# `price` field in the payload — DNSE rejects them with STOCK_PRICE_UNDEFINED
# when both `orderType: MTL` and `price` are present. The broker handles this
# branching automatically in `_post_order`.
ORDER_TYPE_LO  = "LO"      # Limit order (price required)
ORDER_TYPE_MTL = "MTL"     # Market-to-limit
ORDER_TYPE_MOK = "MOK"     # Market or kill (immediate or cancel)
ORDER_TYPE_MAK = "MAK"     # Market and kill (partial fill, cancel rest)
ORDER_TYPE_ATO = "ATO"     # At-the-open auction
ORDER_TYPE_ATC = "ATC"     # At-the-close auction

_MARKET_ORDER_TYPES = {ORDER_TYPE_MTL, ORDER_TYPE_MOK, ORDER_TYPE_MAK,
                       ORDER_TYPE_ATO, ORDER_TYPE_ATC}

# Tick size and aggression knob (used only for LO orders — ignored for market
# orders). With LO, a 1-tick aggression crosses the spread for near-immediate
# fill at the cost of half a tick of slippage.
TICK_SIZE_DEFAULT        = 0.1
AGGRESSION_TICKS_DEFAULT = 1

# Market types
MARKET_DERIVATIVE = "DERIVATIVE"
MARKET_STOCK      = "STOCK"

# Default base URL for DNSE OpenAPI
DEFAULT_BASE_URL = "https://openapi.dnse.com.vn"


# ===========================================================================
# Lightweight response containers — keep strategy code free of dict-digging
# ===========================================================================

@dataclass
class Position:
    """A single open position at the broker."""
    id:           int
    account_no:   str
    symbol:       str
    side:         str               # "BUY" or "SELL" — DNSE's response side
    status:       str               # e.g. "OPEN"
    open_qty:     int
    cost_price:   Optional[float]
    market_price: Optional[float]
    raw:          dict              # original JSON for any field we missed

    @classmethod
    def from_json(cls, j: dict) -> "Position":
        return cls(
            id           = int(j.get("id") or 0),
            account_no   = str(j.get("accountNo") or ""),
            symbol       = str(j.get("symbol") or ""),
            side         = str(j.get("side") or ""),
            status       = str(j.get("status") or ""),
            open_qty     = int(j.get("openQuantity") or 0),
            cost_price   = _to_float(j.get("costPrice")),
            market_price = _to_float(j.get("marketPrice")),
            raw          = j,
        )


@dataclass
class Account:
    """A DNSE sub-account entry from ``get_accounts``."""
    id:                 str           # the accountNo used in API calls
    deal_account:       bool
    derivative_account: bool
    derivative_active:  bool
    raw:                dict

    @classmethod
    def from_json(cls, j: dict) -> "Account":
        derivative = j.get("derivative") or {}
        return cls(
            id                 = str(j.get("id") or ""),
            deal_account       = bool(j.get("dealAccount") or False),
            derivative_account = bool(j.get("derivativeAccount") or False),
            derivative_active  = derivative.get("status") == "ACTIVE",
            raw                = j,
        )


@dataclass
class OrderAck:
    """Response from a successful order placement."""
    order_id: Optional[int]
    status:   Optional[str]
    side:     Optional[str]
    quantity: Optional[int]
    price:    Optional[float]
    raw:      dict

    @classmethod
    def from_json(cls, j: dict) -> "OrderAck":
        return cls(
            order_id = _to_int(j.get("id")),
            status   = j.get("orderStatus"),
            side     = j.get("side"),
            quantity = _to_int(j.get("quantity")),
            price    = _to_float(j.get("price")),
            raw      = j,
        )


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ===========================================================================
# Exceptions
# ===========================================================================

class BrokerError(Exception):
    """Generic broker error — non-2xx HTTP status or response parse failure."""


class AuthRequiredError(BrokerError):
    """Raised when a trading action is attempted without a trading token."""


class TokenExpiredError(BrokerError):
    """Raised when the trading token has been rejected by the API.

    The strategy should catch this, halt, and trigger a fresh OTP flow.
    """


class AccountResolutionError(BrokerError):
    """Raised when auto-discovery of the derivative account fails."""


# ===========================================================================
# Broker adapter
# ===========================================================================

class DNSEBroker:
    """
    Wraps ``DNSEClient`` and exposes a small surface for the trading strategy.

    Construct once per session. The broker auto-discovers your derivative
    sub-account on first use, so you only need to provide ``account_no``
    if you have multiple derivative accounts.

    Read endpoints (``get_open_positions``, ``get_accounts``) work with just
    api_key + api_secret. Write endpoints (``open_position``, ``close_position``)
    additionally require a trading token, obtained via the OTP flow.

    Parameters
    ----------
    api_key, api_secret
        DNSE OpenAPI credentials. Issued via the DNSE web console.
    symbol
        Trading instrument (e.g. ``"VN30F1M"``).
    account_no
        Optional. If omitted, auto-discovered from ``get_accounts()`` by
        picking the first account with ``derivativeAccount=True`` and
        ``derivative.status="ACTIVE"``. Pass explicitly if you have more
        than one derivative sub-account.
    market_type
        ``"DERIVATIVE"`` or ``"STOCK"``. Defaults to derivative.
    loan_package_id
        Optional. If omitted, auto-fetched on first write via
        ``get_loan_packages``. Pass explicitly to pin a specific margin package.
    base_url
        DNSE OpenAPI base URL. Override for sandbox if applicable.
    default_order_type
        ``"MTL"`` (market-to-limit) or ``"LO"`` (limit).
    """

    def __init__(
        self,
        api_key:            str,
        api_secret:         str,
        symbol:             str,
        account_no:         Optional[str] = None,
        market_type:        str = MARKET_DERIVATIVE,
        loan_package_id:    Optional[int] = None,
        base_url:           str = DEFAULT_BASE_URL,
        default_order_type: str = ORDER_TYPE_MTL,
        tick_size:          float = TICK_SIZE_DEFAULT,
        aggression_ticks:   int   = AGGRESSION_TICKS_DEFAULT,
    ) -> None:
        self._client = DNSEClient(
            api_key    = api_key,
            api_secret = api_secret,
            base_url   = base_url,
        )
        self.symbol             = symbol
        self.market_type        = market_type
        self.default_order_type = default_order_type
        self.tick_size          = tick_size
        self.aggression_ticks   = aggression_ticks
        self._account_no:      Optional[str] = account_no
        self._loan_package_id: Optional[int] = loan_package_id
        self._trading_token:   Optional[str] = None
        # Cached user info from get_accounts (filled on first resolve)
        self._investor_id:   Optional[str] = None
        self._custody_code:  Optional[str] = None

    # ---------------------------------------------------------------- Auth

    def request_otp(self, dry_run: bool = False) -> None:
        """
        Trigger DNSE to email an OTP to the account's registered address.

        Only applies to the **email OTP** flow. For ``smart_otp`` (DNSE mobile
        app), the passcode is generated on-device and there is no preceding
        request step — go directly to ``authenticate(otp, otp_type='smart_otp')``.

        Call this once at the start of a trading session, retrieve the OTP from
        email, then call ``authenticate(otp)`` to obtain a trading token.
        """
        status, body = self._client.send_email_otp(dry_run=dry_run)
        if dry_run:
            return
        self._raise_if_error("send_email_otp", status, body)

    def authenticate(
        self, otp: str, otp_type: Literal["smart_otp", "email_otp"] = "smart_otp"
    ) -> str:
        """
        Exchange an OTP for a trading token. Stores the token internally so
        subsequent ``open_position`` / ``close_position`` calls can use it.

        This is a convenience for callers that want to mint inline. Most
        deployments mint the token in a separate operator script and pass it
        in via ``set_trading_token()`` instead.

        Returns the trading token for caller-side persistence if desired.
        """
        status, body = self._client.create_trading_token(
            otp_type=otp_type, passcode=otp,
        )
        data = self._raise_if_error("create_trading_token", status, body)
        token = data.get("tradingToken") or data.get("trading_token")
        if not token:
            raise BrokerError(
                f"create_trading_token returned no token: {data}"
            )
        self._trading_token = token
        return token

    def set_trading_token(self, token: str) -> None:
        """Set a previously-obtained trading token directly (e.g. from env var)."""
        self._trading_token = token

    @property
    def has_trading_token(self) -> bool:
        return self._trading_token is not None

    def _require_token(self) -> str:
        if not self._trading_token:
            raise AuthRequiredError(
                "no trading token; call request_otp() then authenticate(otp), "
                "or set_trading_token(<token from env>)"
            )
        return self._trading_token

    # ---------------------------------------------------------------- Reads

    def get_accounts_payload(self) -> dict:
        """
        Return the raw ``get_accounts`` response dict, including top-level
        fields like ``investorId`` and ``custodyCode``.
        """
        status, body = self._client.get_accounts()
        return self._raise_if_error("get_accounts", status, body)

    def get_accounts(self) -> List[Account]:
        """Return all sub-accounts as parsed ``Account`` instances."""
        payload = self.get_accounts_payload()
        # Cache user-info fields on the broker for diagnostic display
        self._investor_id  = payload.get("investorId")
        self._custody_code = payload.get("custodyCode")
        items = payload.get("accounts") or payload.get("data") or []
        return [Account.from_json(a) for a in items if isinstance(a, dict)]

    def discover_derivative_account(self) -> Account:
        """
        Find and return the first derivative-enabled, active sub-account.
        Raises ``AccountResolutionError`` if none is found.
        """
        accounts = self.get_accounts()
        for a in accounts:
            if a.derivative_account and a.derivative_active:
                return a
        raise AccountResolutionError(
            "no active derivative sub-account found. "
            f"Available: {[(a.id, a.derivative_account, a.derivative_active) for a in accounts]}"
        )

    @property
    def account_no(self) -> str:
        """
        The derivative sub-account number used for all API calls. Auto-resolved
        on first access; cached thereafter.
        """
        if self._account_no is None:
            account = self.discover_derivative_account()
            self._account_no = account.id
        return self._account_no

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Return open positions for our account+market, optionally filtered by
        symbol.
        """
        status, body = self._client.get_positions(
            account_no  = self.account_no,
            market_type = self.market_type,
        )
        data = self._raise_if_error("get_positions", status, body)
        # Response shape: handle several common wrappers
        items = (
            data.get("positions")
            or data.get("deals")
            or data.get("data")
            or []
        )
        positions = [Position.from_json(d) for d in items if isinstance(d, dict)]
        sym = symbol or self.symbol
        if sym:
            positions = [p for p in positions if p.symbol == sym]
        return positions

    def ensure_loan_package(self) -> int:
        """
        Resolve and cache a loan package id for this account+market+symbol.
        Returns the cached id on subsequent calls.
        """
        if self._loan_package_id is not None:
            return self._loan_package_id
        status, body = self._client.get_loan_packages(
            account_no  = self.account_no,
            market_type = self.market_type,
            symbol      = self.symbol,
        )
        data = self._raise_if_error("get_loan_packages", status, body)
        items = data.get("loanPackages") or data.get("data") or []
        if not items:
            raise BrokerError(
                f"no loan packages available for {self.symbol} on "
                f"account={self.account_no} market={self.market_type}"
            )
        first = items[0]
        if isinstance(first, dict):
            self._loan_package_id = int(first.get("id") or first.get("loanPackageId") or 0)
        return self._loan_package_id

    # ---------------------------------------------------------------- Writes

    def open_position(
        self,
        target_pos: int,
        quantity:   int = 1,
        price:      float = 0.0,
        order_type: Optional[str] = None,
        dry_run:    bool = False,
    ) -> OrderAck:
        """
        Place an order to open or flip a directional position.

        Parameters
        ----------
        target_pos
            +1 for long (buy), -1 for short (sell).
        quantity
            Contract quantity. For a fresh entry from flat this equals the
            desired position size. For a flip from an opposite position this
            is ``existing_qty + new_position_size`` so a single netting order
            both closes the old and opens the new.
        price
            Limit price for LO orders. The caller is responsible for any
            slippage offset and tick rounding — the broker sends this value
            as-is. Ignored for market orders.
        order_type
            Override ``default_order_type``.
        dry_run
            If True, prints the request and returns an empty ``OrderAck``
            without sending or requiring a trading token.
        """
        if target_pos not in (-1, 1):
            raise ValueError(f"target_pos must be +1 or -1, got {target_pos}")

        side = DNSE_SIDE_BUY if target_pos == 1 else DNSE_SIDE_SELL
        ot   = order_type or self.default_order_type

        return self._post_order(
            side       = side,
            quantity   = quantity,
            price      = price,
            order_type = ot,
            dry_run    = dry_run,
        )

    def close_position(
        self,
        position: Position,
        dry_run:  bool = False,
    ) -> OrderAck:
        """
        Close an open position by ID. The SDK handles side derivation and
        order construction internally, so the broker only needs the position
        reference and a valid trading token.

        Note that ``close_position`` on the underlying SDK is *not* a
        traditional order — it submits a closing instruction that the
        platform fulfills against the position record. Price and order-type
        are determined by DNSE's close-position semantics; we do not pass
        them here.
        """
        if dry_run:
            print(f"[DRY] close_position id={position.id} "
                  f"side={position.side} qty={position.open_qty}")
            return OrderAck(None, "DRY", None, position.open_qty, 0.0, raw={})

        token = self._require_token()
        status, body = self._client.close_position(
            position_id   = str(position.id),
            market_type   = self.market_type,
            trading_token = token,
        )
        data = self._raise_if_error("close_position", status, body)
        return OrderAck.from_json(data if isinstance(data, dict) else {"raw": data})

    def close_all_for_symbol(self, dry_run: bool = False) -> List[OrderAck]:
        """Close every open position at this account+market+symbol."""
        positions = self.get_open_positions(symbol=self.symbol)
        return [self.close_position(p, dry_run=dry_run) for p in positions]

    # ------------------------------------------------------------- Internal

    def _post_order(
        self,
        side:       str,
        quantity:   int,
        price:      float,
        order_type: str,
        dry_run:    bool,
    ) -> OrderAck:
        is_market = order_type in _MARKET_ORDER_TYPES

        if dry_run:
            if is_market:
                print(f"[DRY] post_order side={side} qty={quantity} "
                      f"(market, no price) type={order_type} symbol={self.symbol}")
                return OrderAck(None, "DRY", side, quantity, None, raw={})
            print(f"[DRY] post_order side={side} qty={quantity} px={price} "
                  f"type={order_type} symbol={self.symbol}")
            return OrderAck(None, "DRY", side, quantity, price, raw={})

        token = self._require_token()
        loan_package_id = self.ensure_loan_package()
        payload: dict = {
            "accountNo":     self.account_no,
            "symbol":        self.symbol,
            "side":          side,
            "orderType":     order_type,
            "quantity":      quantity,
            "loanPackageId": loan_package_id,
        }
        # DNSE rejects market orders with STOCK_PRICE_UNDEFINED if a price
        # is present. The `price` field is included only for LO and other
        # limit-style types.
        if not is_market:
            payload["price"] = price

        status, body = self._client.post_order(
            market_type    = self.market_type,
            payload        = payload,
            trading_token  = token,
            order_category = "NORMAL",
        )
        data = self._raise_if_error("post_order", status, body)
        return OrderAck.from_json(data if isinstance(data, dict) else {"raw": data})

    @staticmethod
    def _raise_if_error(name: str, status: Optional[int], body: Optional[str]):
        """Validate ``(status, body)``. Returns parsed JSON on success."""
        if status is None:
            return {}  # dry-run path
        if not 200 <= status < 300:
            text = body or ""
            if "INVALID_TRADING_TOKEN" in text:
                raise TokenExpiredError(f"{name}: trading token rejected: {text}")
            raise BrokerError(f"{name}: HTTP {status}: {text}")
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise BrokerError(f"{name}: invalid JSON: {e}; body={body!r}")


# ===========================================================================
# Operator notes
# ===========================================================================
#
# Trading-token minting is left to the operator's own script (smart_otp or
# email_otp). The broker only consumes a pre-minted token:
#
#     # In your own minting script (one-off, per session):
#     from dnse import DNSEClient
#     client = DNSEClient(api_key=..., api_secret=...)
#     status, body = client.create_trading_token(
#         otp_type="smart_otp", passcode="<your-passcode>",
#     )
#     # parse body, copy the tradingToken value
#     # then:  export DNSE_TRADING_TOKEN=<value>
#
#     # In live_trade.py (or any other consumer):
#     broker = DNSEBroker(api_key=..., api_secret=..., symbol="VN30F1M")
#     broker.set_trading_token(os.environ["DNSE_TRADING_TOKEN"])
#     # broker is now ready to place orders
#
# The broker also exposes `authenticate(otp, otp_type=...)` for callers that
# prefer to mint inline, but that path is optional.