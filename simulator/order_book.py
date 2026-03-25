"""Lightweight L2 order book from bookTicker snapshots."""
import pandas as pd
import numpy as np
from pathlib import Path


class OrderBook:
    """
    Maintains current best bid/ask from bookTicker data.
    Provides lookup by timestamp via merge_asof.
    """

    def __init__(self):
        self._data: pd.DataFrame = pd.DataFrame()

    def load(self, df: pd.DataFrame) -> None:
        """Load bookTicker dataframe sorted by timestamp_ms."""
        self._data = df.sort_values("timestamp_ms").reset_index(drop=True)

    def get_at(self, timestamp_ms: int) -> dict:
        """Return best bid/ask at or before the given timestamp."""
        if self._data.empty:
            return {"best_bid_price": np.nan, "best_bid_qty": np.nan,
                    "best_ask_price": np.nan, "best_ask_qty": np.nan}
        idx = self._data["timestamp_ms"].searchsorted(timestamp_ms, side="right") - 1
        idx = max(0, min(idx, len(self._data) - 1))
        row = self._data.iloc[idx]
        return {
            "best_bid_price": float(row["best_bid_price"]),
            "best_bid_qty": float(row["best_bid_qty"]),
            "best_ask_price": float(row["best_ask_price"]),
            "best_ask_qty": float(row["best_ask_qty"]),
        }

    def get_window(self, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Return bookTicker rows in [start_ms, end_ms]."""
        mask = (self._data["timestamp_ms"] >= start_ms) & (self._data["timestamp_ms"] <= end_ms)
        return self._data[mask]
