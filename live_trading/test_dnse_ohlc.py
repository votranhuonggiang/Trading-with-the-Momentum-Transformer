from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from DNSE.client import DNSEClient  # noqa: E402


def _default_from_to() -> tuple[str, str]:
    now = datetime.now()
    start = now - timedelta(days=2)
    return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return repr(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test DNSE OHLC endpoint")
    parser.add_argument("--mode", choices=["ohlc", "secdef", "latest_trade"], default="ohlc")
    parser.add_argument("--bar-type", default="1m")
    parser.add_argument("--symbol", default="VN30F1M")
    parser.add_argument("--board-id", default=None)
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / "live_trading" / ".env")

    from_date, to_date = _default_from_to()
    if args.from_date:
        from_date = args.from_date
    if args.to_date:
        to_date = args.to_date

    client = DNSEClient(
        api_key=os.environ["DNSE_API_KEY"],
        api_secret=os.environ["DNSE_API_SECRET"],
    )

    query: dict[str, Any] = {
        "symbol": args.symbol,
        "from": from_date,
        "to": to_date,
    }
    if args.board_id:
        query["boardId"] = args.board_id

    if args.mode == "ohlc":
        print("Calling DNSE OHLC endpoint with:")
        print(json.dumps({"bar_type": args.bar_type, "query": query}, indent=2))
        result = client.get_ohlc(args.bar_type, query=query, dry_run=False)
    elif args.mode == "secdef":
        print("Calling DNSE security definition endpoint with:")
        print(json.dumps({"symbol": args.symbol, "board_id": args.board_id}, indent=2))
        result = client.get_security_definition(args.symbol, board_id=args.board_id, dry_run=False)
    else:
        print("Calling DNSE latest trade endpoint with:")
        print(json.dumps({"symbol": args.symbol, "board_id": args.board_id}, indent=2))
        result = client.get_latest_trade(args.symbol, board_id=args.board_id, dry_run=False)

    print("\nResponse type:", type(result).__name__)
    print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
