"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_model_cumulative_returns(ts: pd.DataFrame, out_file: Path) -> None:
    """Plot cumulative net return lines for all models."""
    df = ts.copy()
    df = df.sort_values(["model", "timestamp"])
    df["cum_net_return"] = df.groupby("model")["net_return"].cumsum()

    fig, ax = plt.subplots(figsize=(14, 7))
    for model_name, g in df.groupby("model"):
        ax.plot(g["timestamp"], g["cum_net_return"], label=model_name, linewidth=1.6)

    ax.set_title("Cumulative Net Return by Model")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative Net Return")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2, fontsize=9)
    fig.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
