"""Very small logger used by live-trading scripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class SimpleLogger:
    def __init__(self, name: str, log_path: str | Path | None = None) -> None:
        self.name = name
        self.log_path = Path(log_path) if log_path else None

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] [{self.name}] {message}"
        print(line)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
