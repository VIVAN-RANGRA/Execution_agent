"""Replays parsed Parquet data window by window, returning MarketState objects."""
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import List, Tuple

import numpy as np
import pandas as pd

from simulator.data_classes import MarketState
from simulator.order_book import OrderBook


class MarketDataStream:
    """
    Loads aggTrades and bookTicker for a date range.
    Advances in fixed-duration windows, building MarketState objects.

    Optimized with:
    - column-pruned parquet reads
    - thread-parallel day loading
    - numpy arrays and prefix sums for hot-path queries
    - pre-computed 1-second rolling features
    - LRU cache on build_market_state for repeated timestamp queries
    """

    _RANGE_BUNDLE_CACHE = OrderedDict()
    _RANGE_CACHE_LOCK = Lock()
    _MAX_RANGE_BUNDLES = 4

    def __init__(self, parsed_data_dir: Path, volume_profile_path: Path):
        self.parsed_data_dir = Path(parsed_data_dir)
        self._volume_profile_path = Path(volume_profile_path)
        self.volume_profile, self._vol_profile_dict = self._load_volume_profile_cached(
            str(self._volume_profile_path.resolve())
        )
        self._trades: pd.DataFrame = pd.DataFrame()
        self._book: OrderBook = OrderBook()

        # Trade arrays and prefix sums.
        self._trade_ts: np.ndarray = np.array([], dtype=np.int64)
        self._trade_prices: np.ndarray = np.array([], dtype=np.float64)
        self._trade_qtys: np.ndarray = np.array([], dtype=np.float64)
        self._trade_is_buyer_maker: np.ndarray = np.array([], dtype=np.bool_)
        self._trade_qty_prefix: np.ndarray = np.array([0.0], dtype=np.float64)
        self._trade_buy_qty_prefix: np.ndarray = np.array([0.0], dtype=np.float64)
        self._trade_sell_qty_prefix: np.ndarray = np.array([0.0], dtype=np.float64)

        # Book arrays.
        self._book_ts: np.ndarray = np.array([], dtype=np.int64)
        self._book_bid_prices: np.ndarray = np.array([], dtype=np.float64)
        self._book_bid_qtys: np.ndarray = np.array([], dtype=np.float64)
        self._book_ask_prices: np.ndarray = np.array([], dtype=np.float64)
        self._book_ask_qtys: np.ndarray = np.array([], dtype=np.float64)

        # Pre-computed 1-second resolution data.
        self._second_timestamps: np.ndarray = np.array([], dtype=np.int64)
        self._second_last_price: np.ndarray = np.array([], dtype=np.float64)
        self._second_log_returns: np.ndarray = np.array([], dtype=np.float64)
        self._sec_vol_60s: np.ndarray = np.array([], dtype=np.float64)
        self._sec_vol_300s: np.ndarray = np.array([], dtype=np.float64)
        self._sec_ofi_10s: np.ndarray = np.array([], dtype=np.float64)
        self._sec_volume_30s: np.ndarray = np.array([], dtype=np.float64)
        self._sec_index_by_start_ms: dict[int, int] = {}
        self._available_starts_cache: dict[int, List[int]] = {}

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_volume_profile_cached(volume_profile_path: str):
        volume_profile = pd.read_parquet(volume_profile_path)
        vol_profile_dict = dict(
            zip(
                volume_profile["minute_of_day"].astype(int),
                volume_profile["fraction"].astype(float),
            )
        )
        return volume_profile, vol_profile_dict

    @staticmethod
    def _read_parquet(parquet_path: Path, columns: List[str]) -> pd.DataFrame:
        return pd.read_parquet(parquet_path, columns=columns)

    @staticmethod
    def _default_load_workers() -> int:
        """
        Keep dashboard/env startup conservative so loading replay data does not
        saturate the whole machine.
        """
        cpu_count = os.cpu_count() or 4
        reserved = 2 if cpu_count > 2 else 1
        return max(1, min(2, cpu_count - reserved))

    def _range_cache_key(self, dates: List[str]) -> Tuple[str, str, Tuple[str, ...]]:
        return (
            str(self.parsed_data_dir.resolve()),
            str(self._volume_profile_path.resolve()),
            tuple(dates),
        )

    @classmethod
    def _get_cached_range_bundle(
        cls,
        cache_key: Tuple[str, str, Tuple[str, ...]],
        *,
        trade_paths: List[Path],
        book_paths: List[Path],
        trade_columns: List[str],
        book_columns: List[str],
        max_workers: int,
    ) -> dict:
        with cls._RANGE_CACHE_LOCK:
            bundle = cls._RANGE_BUNDLE_CACHE.get(cache_key)
            if bundle is None:
                bundle = cls._build_range_bundle(
                    trade_paths=trade_paths,
                    book_paths=book_paths,
                    trade_columns=trade_columns,
                    book_columns=book_columns,
                    max_workers=max_workers,
                )
                cls._RANGE_BUNDLE_CACHE[cache_key] = bundle
                while len(cls._RANGE_BUNDLE_CACHE) > cls._MAX_RANGE_BUNDLES:
                    cls._RANGE_BUNDLE_CACHE.popitem(last=False)
            else:
                cls._RANGE_BUNDLE_CACHE.move_to_end(cache_key)
            return bundle

    @classmethod
    def _build_range_bundle(
        cls,
        *,
        trade_paths: List[Path],
        book_paths: List[Path],
        trade_columns: List[str],
        book_columns: List[str],
        max_workers: int,
    ) -> dict:
        def _read_many(paths: List[Path], columns: List[str]) -> List[pd.DataFrame]:
            if not paths:
                return []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(executor.map(lambda path: cls._read_parquet(path, columns), paths))

        trade_dfs = _read_many(trade_paths, trade_columns)
        book_dfs = _read_many(book_paths, book_columns)

        trades = (
            pd.concat(trade_dfs, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
            if trade_dfs else pd.DataFrame()
        )
        book_df = (
            pd.concat(book_dfs, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
            if book_dfs else pd.DataFrame()
        )

        bundle = {
            "trades": trades,
            "book_df": book_df,
            "trade_ts": np.array([], dtype=np.int64),
            "trade_prices": np.array([], dtype=np.float64),
            "trade_qtys": np.array([], dtype=np.float64),
            "trade_is_buyer_maker": np.array([], dtype=np.bool_),
            "trade_qty_prefix": np.array([0.0], dtype=np.float64),
            "trade_buy_qty_prefix": np.array([0.0], dtype=np.float64),
            "trade_sell_qty_prefix": np.array([0.0], dtype=np.float64),
            "second_timestamps": np.array([], dtype=np.int64),
            "second_last_price": np.array([], dtype=np.float64),
            "second_log_returns": np.array([], dtype=np.float64),
            "sec_vol_60s": np.array([], dtype=np.float64),
            "sec_vol_300s": np.array([], dtype=np.float64),
            "sec_ofi_10s": np.array([], dtype=np.float64),
            "sec_volume_30s": np.array([], dtype=np.float64),
            "sec_index_by_start_ms": {},
            "book_ts": np.array([], dtype=np.int64),
            "book_bid_prices": np.array([], dtype=np.float64),
            "book_bid_qtys": np.array([], dtype=np.float64),
            "book_ask_prices": np.array([], dtype=np.float64),
            "book_ask_qtys": np.array([], dtype=np.float64),
        }

        if not trades.empty:
            trade_ts = trades["timestamp_ms"].to_numpy(dtype=np.int64, copy=False)
            trade_prices = trades["price"].to_numpy(dtype=np.float64, copy=False)
            trade_qtys = trades["qty"].to_numpy(dtype=np.float64, copy=False)
            trade_is_buyer_maker = trades["is_buyer_maker"].to_numpy(dtype=np.bool_, copy=False)

            buyer_qtys = np.where(~trade_is_buyer_maker, trade_qtys, 0.0)
            seller_qtys = np.where(trade_is_buyer_maker, trade_qtys, 0.0)

            seconds = trade_ts // 1000
            unique_seconds, first_indices = np.unique(seconds, return_index=True)
            n_seconds = len(unique_seconds)
            last_indices = np.empty(n_seconds, dtype=np.int64)
            last_indices[:-1] = first_indices[1:] - 1
            last_indices[-1] = len(trade_ts) - 1

            second_timestamps = unique_seconds * 1000
            second_last_price = trade_prices[last_indices]
            second_log_returns = (
                np.diff(np.log(second_last_price))
                if len(second_last_price) >= 2
                else np.array([], dtype=np.float64)
            )

            sec_vol_60s = np.full(n_seconds, 1e-6, dtype=np.float64)
            sec_vol_300s = np.full(n_seconds, 1e-6, dtype=np.float64)
            sec_ofi_10s = np.zeros(n_seconds, dtype=np.float64)
            sec_volume_30s = np.full(n_seconds, 1.0, dtype=np.float64)
            sec_seconds = second_timestamps // 1000
            sec_index_by_start_ms = {
                int(timestamp_ms): idx for idx, timestamp_ms in enumerate(second_timestamps.tolist())
            }

            n_returns = len(second_log_returns)
            if n_returns > 0:
                return_starts = sec_seconds[:-1]
                return_prefix = np.concatenate(([0.0], np.cumsum(second_log_returns, dtype=np.float64)))
                return_sq_prefix = np.concatenate(([0.0], np.cumsum(second_log_returns ** 2, dtype=np.float64)))

                for window_seconds, target, scale in (
                    (60, sec_vol_60s, np.sqrt(60.0)),
                    (300, sec_vol_300s, np.sqrt(300.0)),
                ):
                    start_idx = np.searchsorted(return_starts, sec_seconds - window_seconds, side="left")
                    end_idx = np.arange(n_seconds, dtype=np.int64)
                    counts = end_idx - start_idx
                    valid = counts >= 2
                    if np.any(valid):
                        sums = return_prefix[end_idx[valid]] - return_prefix[start_idx[valid]]
                        sumsq = return_sq_prefix[end_idx[valid]] - return_sq_prefix[start_idx[valid]]
                        means = sums / counts[valid]
                        variances = np.maximum((sumsq / counts[valid]) - (means ** 2), 0.0)
                        target[valid] = np.sqrt(variances) * scale

            trade_seconds = trade_ts // 1000
            trade_qty_prefix = np.concatenate(([0.0], np.cumsum(trade_qtys, dtype=np.float64)))
            trade_buy_qty_prefix = np.concatenate(([0.0], np.cumsum(buyer_qtys, dtype=np.float64)))
            trade_sell_qty_prefix = np.concatenate(([0.0], np.cumsum(seller_qtys, dtype=np.float64)))

            window_end = np.searchsorted(trade_seconds, sec_seconds, side="left")
            volume_start = np.searchsorted(trade_seconds, sec_seconds - 30, side="left")
            volume_values = trade_qty_prefix[window_end] - trade_qty_prefix[volume_start]
            positive_volume = volume_values > 0
            sec_volume_30s[positive_volume] = volume_values[positive_volume]

            ofi_start = np.searchsorted(trade_seconds, sec_seconds - 10, side="left")
            total_window = trade_qty_prefix[window_end] - trade_qty_prefix[ofi_start]
            buyer_window = trade_buy_qty_prefix[window_end] - trade_buy_qty_prefix[ofi_start]
            seller_window = trade_sell_qty_prefix[window_end] - trade_sell_qty_prefix[ofi_start]
            valid_ofi = total_window > 0
            sec_ofi_10s[valid_ofi] = (
                (buyer_window[valid_ofi] - seller_window[valid_ofi]) / total_window[valid_ofi]
            )

            bundle.update(
                {
                    "trade_ts": trade_ts,
                    "trade_prices": trade_prices,
                    "trade_qtys": trade_qtys,
                    "trade_is_buyer_maker": trade_is_buyer_maker,
                    "trade_qty_prefix": trade_qty_prefix,
                    "trade_buy_qty_prefix": trade_buy_qty_prefix,
                    "trade_sell_qty_prefix": trade_sell_qty_prefix,
                    "second_timestamps": second_timestamps,
                    "second_last_price": second_last_price,
                    "second_log_returns": second_log_returns,
                    "sec_vol_60s": sec_vol_60s,
                    "sec_vol_300s": sec_vol_300s,
                    "sec_ofi_10s": sec_ofi_10s,
                    "sec_volume_30s": sec_volume_30s,
                    "sec_index_by_start_ms": sec_index_by_start_ms,
                }
            )

        if not book_df.empty:
            bundle.update(
                {
                    "book_ts": book_df["timestamp_ms"].to_numpy(dtype=np.int64, copy=False),
                    "book_bid_prices": book_df["best_bid_price"].to_numpy(dtype=np.float64, copy=False),
                    "book_bid_qtys": book_df["best_bid_qty"].to_numpy(dtype=np.float64, copy=False),
                    "book_ask_prices": book_df["best_ask_price"].to_numpy(dtype=np.float64, copy=False),
                    "book_ask_qtys": book_df["best_ask_qty"].to_numpy(dtype=np.float64, copy=False),
                }
            )

        return bundle

    def _apply_range_bundle(self, bundle: dict) -> None:
        self._trades = bundle["trades"]
        self._book = OrderBook()
        self._book._data = bundle["book_df"]
        self._trade_ts = bundle["trade_ts"]
        self._trade_prices = bundle["trade_prices"]
        self._trade_qtys = bundle["trade_qtys"]
        self._trade_is_buyer_maker = bundle["trade_is_buyer_maker"]
        self._trade_qty_prefix = bundle["trade_qty_prefix"]
        self._trade_buy_qty_prefix = bundle["trade_buy_qty_prefix"]
        self._trade_sell_qty_prefix = bundle["trade_sell_qty_prefix"]
        self._second_timestamps = bundle["second_timestamps"]
        self._second_last_price = bundle["second_last_price"]
        self._second_log_returns = bundle["second_log_returns"]
        self._sec_vol_60s = bundle["sec_vol_60s"]
        self._sec_vol_300s = bundle["sec_vol_300s"]
        self._sec_ofi_10s = bundle["sec_ofi_10s"]
        self._sec_volume_30s = bundle["sec_volume_30s"]
        self._sec_index_by_start_ms = bundle["sec_index_by_start_ms"]
        self._book_ts = bundle["book_ts"]
        self._book_bid_prices = bundle["book_bid_prices"]
        self._book_bid_qtys = bundle["book_bid_qtys"]
        self._book_ask_prices = bundle["book_ask_prices"]
        self._book_ask_qtys = bundle["book_ask_qtys"]
        self._available_starts_cache = {}

    def load_date_range(self, dates: List[str]) -> None:
        """Load aggTrades and bookTicker for the given list of date strings."""
        trade_paths = []
        book_paths = []
        for date_str in dates:
            trade_path = self.parsed_data_dir / "aggTrades" / f"{date_str}.parquet"
            book_path = self.parsed_data_dir / "bookTicker" / f"{date_str}.parquet"
            if trade_path.exists():
                trade_paths.append(trade_path)
            if book_path.exists():
                book_paths.append(book_path)

        trade_columns = ["timestamp_ms", "price", "qty", "is_buyer_maker"]
        book_columns = ["timestamp_ms", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]
        max_workers = min(
            self._default_load_workers(),
            max(len(trade_paths), len(book_paths), 1),
        )
        bundle = self._get_cached_range_bundle(
            self._range_cache_key(dates),
            trade_paths=trade_paths,
            book_paths=book_paths,
            trade_columns=trade_columns,
            book_columns=book_columns,
            max_workers=max_workers,
        )
        self._apply_range_bundle(bundle)
        self._build_market_state_cached.cache_clear()

    def _precompute_second_prices(self) -> None:
        """Pre-compute last price per second for fast volatility calculation."""
        seconds = self._trade_ts // 1000
        unique_seconds, first_indices = np.unique(seconds, return_index=True)

        n_seconds = len(unique_seconds)
        last_indices = np.empty(n_seconds, dtype=np.int64)
        last_indices[:-1] = first_indices[1:] - 1
        last_indices[-1] = len(self._trade_ts) - 1

        self._second_timestamps = unique_seconds * 1000
        self._second_last_price = self._trade_prices[last_indices]
        if len(self._second_last_price) >= 2:
            self._second_log_returns = np.diff(np.log(self._second_last_price))
        else:
            self._second_log_returns = np.array([], dtype=np.float64)

    def _precompute_rolling_features(self) -> None:
        """Pre-compute vol_60s, vol_300s, ofi_10s, and volume_30s at 1-second resolution."""
        n_seconds = len(self._second_timestamps)
        if n_seconds == 0:
            self._sec_vol_60s = np.array([], dtype=np.float64)
            self._sec_vol_300s = np.array([], dtype=np.float64)
            self._sec_ofi_10s = np.array([], dtype=np.float64)
            self._sec_volume_30s = np.array([], dtype=np.float64)
            self._sec_index_by_start_ms = {}
            return

        sec_seconds = self._second_timestamps // 1000
        self._sec_index_by_start_ms = {
            int(timestamp_ms): idx for idx, timestamp_ms in enumerate(self._second_timestamps.tolist())
        }
        self._sec_vol_60s = np.full(n_seconds, 1e-6, dtype=np.float64)
        self._sec_vol_300s = np.full(n_seconds, 1e-6, dtype=np.float64)

        n_returns = len(self._second_log_returns)
        if n_returns > 0:
            return_starts = sec_seconds[:-1]
            return_prefix = np.concatenate(([0.0], np.cumsum(self._second_log_returns, dtype=np.float64)))
            return_sq_prefix = np.concatenate(([0.0], np.cumsum(self._second_log_returns ** 2, dtype=np.float64)))

            for window_seconds, target, scale in (
                (60, self._sec_vol_60s, np.sqrt(60.0)),
                (300, self._sec_vol_300s, np.sqrt(300.0)),
            ):
                start_idx = np.searchsorted(return_starts, sec_seconds - window_seconds, side="left")
                end_idx = np.arange(n_seconds, dtype=np.int64)
                counts = end_idx - start_idx
                valid = counts >= 2
                if np.any(valid):
                    sums = return_prefix[end_idx[valid]] - return_prefix[start_idx[valid]]
                    sumsq = return_sq_prefix[end_idx[valid]] - return_sq_prefix[start_idx[valid]]
                    means = sums / counts[valid]
                    variances = np.maximum((sumsq / counts[valid]) - (means ** 2), 0.0)
                    target[valid] = np.sqrt(variances) * scale

        self._sec_volume_30s = np.full(n_seconds, 1.0, dtype=np.float64)
        self._sec_ofi_10s = np.zeros(n_seconds, dtype=np.float64)
        if len(self._trade_ts) == 0:
            return

        trade_seconds = self._trade_ts // 1000
        window_end = np.searchsorted(trade_seconds, sec_seconds, side="left")

        volume_start = np.searchsorted(trade_seconds, sec_seconds - 30, side="left")
        volume_values = self._trade_qty_prefix[window_end] - self._trade_qty_prefix[volume_start]
        positive_volume = volume_values > 0
        self._sec_volume_30s[positive_volume] = volume_values[positive_volume]

        ofi_start = np.searchsorted(trade_seconds, sec_seconds - 10, side="left")
        total_window = self._trade_qty_prefix[window_end] - self._trade_qty_prefix[ofi_start]
        buyer_window = self._trade_buy_qty_prefix[window_end] - self._trade_buy_qty_prefix[ofi_start]
        seller_window = self._trade_sell_qty_prefix[window_end] - self._trade_sell_qty_prefix[ofi_start]
        valid_ofi = total_window > 0
        self._sec_ofi_10s[valid_ofi] = (
            (buyer_window[valid_ofi] - seller_window[valid_ofi]) / total_window[valid_ofi]
        )

    def get_available_starts(self, window_seconds: int) -> List[int]:
        """Return valid episode start timestamps (ms) ensuring enough future data."""
        cached = self._available_starts_cache.get(int(window_seconds))
        if cached is not None:
            return cached
        if len(self._trade_ts) == 0:
            return []
        cutoff = int(self._trade_ts[-1]) - (window_seconds * 1000)
        end_idx = np.searchsorted(self._trade_ts, cutoff, side="right")
        if end_idx == 0:
            return []
        starts = sorted(np.unique(self._trade_ts[:end_idx]).tolist())
        self._available_starts_cache[int(window_seconds)] = starts
        return starts

    def build_market_state(self, start_ms: int, window_s: int = 60) -> MarketState:
        """Build a MarketState for the timestamp using pre-computed rolling features."""
        return self._build_market_state_cached(start_ms, window_s)

    @lru_cache(maxsize=4096)
    def _build_market_state_cached(self, start_ms: int, window_s: int = 60) -> MarketState:
        """Cached implementation of build_market_state."""
        bid, ask, bid_qty, ask_qty = self._get_book_snapshot(start_ms)
        if np.isnan(bid) or np.isnan(ask):
            mid = self._get_fallback_mid(start_ms)
            bid = mid * (1 - 0.0001)
            ask = mid * (1 + 0.0001)
            bid_qty = 1.0
            ask_qty = 1.0

        mid_price = (bid + ask) / 2.0
        spread = ask - bid
        vol_60s, vol_300s, ofi_10s, volume_30s = self._lookup_precomputed_features(start_ms)
        minute_of_day = int((start_ms // 1000 // 60) % 1440)

        return MarketState(
            timestamp_ms=start_ms,
            mid_price=mid_price,
            bid_price=bid,
            ask_price=ask,
            spread=spread,
            recent_volume_30s=max(volume_30s, 1e-8),
            recent_volatility_60s=vol_60s,
            recent_volatility_300s=vol_300s,
            ofi_10s=ofi_10s,
            bid_qty=max(float(bid_qty), 1e-8),
            ask_qty=max(float(ask_qty), 1e-8),
            minute_of_day=minute_of_day,
        )

    def _get_book_snapshot(self, timestamp_ms: int) -> Tuple[float, float, float, float]:
        """Return the book snapshot at or before timestamp_ms."""
        if len(self._book_ts) == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        idx = np.searchsorted(self._book_ts, timestamp_ms, side="right") - 1
        if idx < 0:
            return (np.nan, np.nan, np.nan, np.nan)
        return (
            float(self._book_bid_prices[idx]),
            float(self._book_ask_prices[idx]),
            float(self._book_bid_qtys[idx]),
            float(self._book_ask_qtys[idx]),
        )

    def _get_fallback_mid(self, start_ms: int) -> float:
        """Get fallback mid price using the last seen trade at or before start_ms."""
        if len(self._trade_ts) == 0:
            return 50000.0
        idx = np.searchsorted(self._trade_ts, start_ms, side="right") - 1
        if idx >= 0:
            return float(self._trade_prices[idx])
        return 50000.0

    def _lookup_precomputed_features(self, start_ms: int) -> Tuple[float, float, float, float]:
        """Look up pre-computed rolling features at the nearest second before start_ms."""
        if len(self._second_timestamps) == 0:
            return (1e-6, 1e-6, 0.0, 1.0)

        idx = self._sec_index_by_start_ms.get(int(start_ms))
        if idx is None:
            idx = np.searchsorted(self._second_timestamps, start_ms, side="right") - 1
        if idx < 0:
            return (1e-6, 1e-6, 0.0, 1.0)

        idx = min(idx, len(self._second_timestamps) - 1)
        return (
            float(self._sec_vol_60s[idx]),
            float(self._sec_vol_300s[idx]),
            float(self._sec_ofi_10s[idx]),
            float(self._sec_volume_30s[idx]),
        )

    def get_window_trades(self, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Return trades in [start_ms, end_ms). Uses numpy searchsorted for O(log n)."""
        if len(self._trade_ts) == 0:
            return pd.DataFrame()
        i_start = np.searchsorted(self._trade_ts, start_ms, side="left")
        i_end = np.searchsorted(self._trade_ts, end_ms, side="left")
        return self._trades.iloc[i_start:i_end]

    def get_mid_price_at(self, timestamp_ms: int) -> float:
        """Get mid price at or before timestamp."""
        bid, ask, _, _ = self._get_book_snapshot(timestamp_ms)
        if not np.isnan(bid) and not np.isnan(ask):
            return (bid + ask) / 2.0
        return self._get_fallback_mid(timestamp_ms)

    def _compute_realized_vol(self, trades: pd.DataFrame, window_s: int) -> float:
        """Backward-compatible realized volatility helper for tests and diagnostics."""
        if trades.empty or len(trades) < 2:
            return 1e-6
        trades = trades.copy()
        trades["second"] = trades["timestamp_ms"] // 1000
        prices_1s = trades.groupby("second")["price"].last()
        if len(prices_1s) < 2:
            return 1e-6
        log_returns = np.diff(np.log(prices_1s.values))
        return float(np.std(log_returns) * np.sqrt(window_s)) if len(log_returns) > 0 else 1e-6
