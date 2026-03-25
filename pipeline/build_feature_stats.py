"""Build feature normalization stats from parsed training data."""
from __future__ import annotations

import argparse
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "default_config.yaml"
CALIBRATION_PATH = BASE_DIR / "config" / "calibration_params.json"
PARSED_DIR = BASE_DIR / "data" / "parsed"
VOLUME_PROFILE_PATH = BASE_DIR / "data" / "volume_profile.parquet"
DEFAULT_OUTPUT_PATH = BASE_DIR / "config" / "feature_stats.json"

FEATURE_NAMES = [
    "spread_to_vol_ratio",
    "order_flow_imbalance",
    "volume_participation_rate",
    "price_momentum_bps",
    "inventory_fraction",
    "time_fraction",
    "urgency_ratio",
    "realized_vol_60s",
    "realized_vol_300s",
    "vol_ratio_short_long",
    "bid_qty_normalized",
    "ask_qty_normalized",
    "OFI_x_momentum",
    "spread_vol_x_inventory",
    "vol_ratio_x_time",
    "bid_qty_x_ask_qty",
    "urgency_x_vol_participation",
    "momentum_x_time",
]

INTERACTION_TERMS = [
    (1, 3),
    (0, 4),
    (9, 5),
    (10, 11),
    (6, 2),
    (3, 5),
]


class RunningStats:
    """Online mean/std accumulator to avoid storing the full training corpus."""

    def __init__(self, dim: int):
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, row: np.ndarray) -> None:
        values = np.asarray(row, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("RunningStats.update expects a 1-D row vector")
        if not np.all(np.isfinite(values)):
            raise ValueError("RunningStats.update received non-finite values")

        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        delta2 = values - self.mean
        self.m2 += delta * delta2

    def update_batch(self, rows: np.ndarray) -> None:
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("RunningStats.update_batch expects a 2-D array")
        if values.shape[0] == 0:
            return
        if not np.all(np.isfinite(values)):
            raise ValueError("RunningStats.update_batch received non-finite values")

        batch_count = int(values.shape[0])
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        self.merge(batch_count, batch_mean, batch_m2)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("Cannot finalize empty running statistics")
        variances = self.m2 / self.count
        stds = np.sqrt(np.maximum(variances, 0.0))
        stds = np.where(stds < 1e-8, 1.0, stds)
        return self.mean.astype(np.float32), stds.astype(np.float32)

    def merge(self, count: int, mean: np.ndarray, m2: np.ndarray) -> None:
        if count <= 0:
            return

        other_mean = np.asarray(mean, dtype=np.float64)
        other_m2 = np.asarray(m2, dtype=np.float64)
        if self.count == 0:
            self.count = int(count)
            self.mean = other_mean.copy()
            self.m2 = other_m2.copy()
            return

        total = self.count + int(count)
        delta = other_mean - self.mean
        self.mean = self.mean + delta * (count / total)
        self.m2 = self.m2 + other_m2 + (delta * delta) * (self.count * count / total)
        self.count = total


def _default_worker_count() -> int:
    cpu_count = os.cpu_count() or 4
    reserved = 2 if cpu_count > 2 else 1
    return max(1, min(4, cpu_count - reserved))


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f) or {}


def _training_dates(parsed_dir: Path, train_split: float) -> list[str]:
    agg_dir = parsed_dir / "aggTrades"
    all_dates = sorted(path.stem for path in agg_dir.glob("*.parquet"))
    if not all_dates:
        raise FileNotFoundError(f"No parsed aggTrades parquet files found in {agg_dir}")

    n_train = max(1, int(len(all_dates) * train_split))
    return all_dates[:n_train]


def _read_day_trades(trade_path: Path) -> pd.DataFrame:
    trades = pd.read_parquet(
        trade_path,
        columns=["timestamp_ms", "price", "qty", "is_buyer_maker"],
    )
    if trades.empty:
        return trades
    if not trades["timestamp_ms"].is_monotonic_increasing:
        trades = trades.sort_values("timestamp_ms").reset_index(drop=True)
    return trades


def _realized_vol(
    second_ts: np.ndarray,
    second_prices: np.ndarray,
    timestamp_ms: int,
    window_seconds: int,
) -> float | None:
    start_ms = timestamp_ms - (window_seconds * 1000)
    start_idx = np.searchsorted(second_ts, start_ms, side="left")
    end_idx = np.searchsorted(second_ts, timestamp_ms, side="right")
    prices = second_prices[start_idx:end_idx]
    if len(prices) < 2:
        return None
    returns = np.diff(np.log(prices))
    if len(returns) == 0:
        return None
    realized = float(np.std(returns) * np.sqrt(window_seconds))
    if not np.isfinite(realized) or realized <= 0.0:
        return None
    return realized


def _build_day_trade_state(trades: pd.DataFrame) -> dict[str, np.ndarray]:
    """Prepare per-day trade arrays and rolling-stat helpers."""
    trade_ts = trades["timestamp_ms"].to_numpy(dtype=np.int64, copy=False)
    trade_prices = trades["price"].to_numpy(dtype=np.float64, copy=False)
    trade_qty = trades["qty"].to_numpy(dtype=np.float64, copy=False)
    trade_is_buyer_maker = trades["is_buyer_maker"].to_numpy(dtype=np.bool_, copy=False)

    buyer_qty = np.where(~trade_is_buyer_maker, trade_qty, 0.0)
    seller_qty = np.where(trade_is_buyer_maker, trade_qty, 0.0)
    qty_prefix = np.concatenate(([0.0], np.cumsum(trade_qty, dtype=np.float64)))
    buy_prefix = np.concatenate(([0.0], np.cumsum(buyer_qty, dtype=np.float64)))
    sell_prefix = np.concatenate(([0.0], np.cumsum(seller_qty, dtype=np.float64)))

    trade_seconds = trade_ts // 1000
    second_prices = trades.groupby(trade_seconds)["price"].last()
    second_ts = (second_prices.index.to_numpy(dtype=np.int64) * 1000).astype(np.int64)
    second_last_prices = second_prices.to_numpy(dtype=np.float64, copy=False)

    sec_seconds = second_ts // 1000
    n_seconds = len(second_ts)
    sec_vol_60s = np.full(n_seconds, np.nan, dtype=np.float64)
    sec_vol_300s = np.full(n_seconds, np.nan, dtype=np.float64)

    if n_seconds >= 2:
        second_log_returns = np.diff(np.log(second_last_prices))
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

    sec_volume_30s = np.zeros(n_seconds, dtype=np.float64)
    sec_ofi_10s = np.zeros(n_seconds, dtype=np.float64)
    sec_has_ofi_10s = np.zeros(n_seconds, dtype=np.bool_)

    if n_seconds > 0:
        window_end = np.searchsorted(trade_seconds, sec_seconds, side="left")
        volume_start = np.searchsorted(trade_seconds, sec_seconds - 30, side="left")
        sec_volume_30s = qty_prefix[window_end] - qty_prefix[volume_start]

        ofi_start = np.searchsorted(trade_seconds, sec_seconds - 10, side="left")
        total_window = qty_prefix[window_end] - qty_prefix[ofi_start]
        buyer_window = buy_prefix[window_end] - buy_prefix[ofi_start]
        seller_window = sell_prefix[window_end] - sell_prefix[ofi_start]
        valid_ofi = total_window > 0
        sec_has_ofi_10s[valid_ofi] = True
        sec_ofi_10s[valid_ofi] = (
            (buyer_window[valid_ofi] - seller_window[valid_ofi]) / total_window[valid_ofi]
        )

    return {
        "trade_ts": trade_ts,
        "trade_prices": trade_prices,
        "qty_prefix": qty_prefix,
        "buy_prefix": buy_prefix,
        "sell_prefix": sell_prefix,
        "second_ts": second_ts,
        "second_last_prices": second_last_prices,
        "sec_vol_60s": sec_vol_60s,
        "sec_vol_300s": sec_vol_300s,
        "sec_volume_30s": sec_volume_30s,
        "sec_ofi_10s": sec_ofi_10s,
        "sec_has_ofi_10s": sec_has_ofi_10s,
    }


def _compute_interactions(normalized_base: np.ndarray) -> np.ndarray:
    return np.array(
        [normalized_base[idx_a] * normalized_base[idx_b] for idx_a, idx_b in INTERACTION_TERMS],
        dtype=np.float64,
    )


def _rolling_mean_with_history(
    history: np.ndarray,
    values: np.ndarray,
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.array([], dtype=np.float64), history

    prior = np.asarray(history, dtype=np.float64)
    extended = np.concatenate([prior, values]) if prior.size else values.copy()
    prefix = np.concatenate(([0.0], np.cumsum(extended, dtype=np.float64)))

    positions = np.arange(prior.size, extended.size, dtype=np.int64)
    starts = np.maximum(0, positions + 1 - window)
    sums = prefix[positions + 1] - prefix[starts]
    counts = (positions + 1) - starts
    means = sums / counts

    keep = max(window - 1, 0)
    new_history = extended[-keep:].copy() if keep > 0 else np.array([], dtype=np.float64)
    return means, new_history


def _iter_book_second_batches(
    *,
    book_path: Path,
    min_timestamp_ms: int,
) -> Iterator[dict[str, np.ndarray]]:
    columns = [
        "timestamp_ms",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
    ]
    parquet = pq.ParquetFile(book_path)
    carry_df: pd.DataFrame | None = None
    for batch in parquet.iter_batches(columns=columns, batch_size=250_000):
        batch_df = batch.to_pandas()
        if batch_df.empty:
            continue
        batch_df = batch_df.loc[
            batch_df["timestamp_ms"].to_numpy(dtype=np.int64, copy=False) >= min_timestamp_ms,
            columns,
        ]
        if batch_df.empty:
            continue
        batch_df = batch_df.sort_values("timestamp_ms").reset_index(drop=True)
        batch_df["second"] = batch_df["timestamp_ms"] // 1000

        if carry_df is not None:
            batch_df = pd.concat([carry_df, batch_df], ignore_index=True)
            carry_df = None

        collapsed = batch_df.groupby("second", sort=True).tail(1).sort_values("timestamp_ms")
        if collapsed.empty:
            continue

        carry_df = collapsed.tail(1)[columns + ["second"]].copy()
        emit_df = collapsed.iloc[:-1]
        if emit_df.empty:
            continue

        yield {
            "timestamp_ms": emit_df["timestamp_ms"].to_numpy(dtype=np.int64, copy=False),
            "best_bid_price": emit_df["best_bid_price"].to_numpy(dtype=np.float64, copy=False),
            "best_bid_qty": emit_df["best_bid_qty"].to_numpy(dtype=np.float64, copy=False),
            "best_ask_price": emit_df["best_ask_price"].to_numpy(dtype=np.float64, copy=False),
            "best_ask_qty": emit_df["best_ask_qty"].to_numpy(dtype=np.float64, copy=False),
        }

    if carry_df is not None and not carry_df.empty:
        yield {
            "timestamp_ms": carry_df["timestamp_ms"].to_numpy(dtype=np.int64, copy=False),
            "best_bid_price": carry_df["best_bid_price"].to_numpy(dtype=np.float64, copy=False),
            "best_bid_qty": carry_df["best_bid_qty"].to_numpy(dtype=np.float64, copy=False),
            "best_ask_price": carry_df["best_ask_price"].to_numpy(dtype=np.float64, copy=False),
            "best_ask_qty": carry_df["best_ask_qty"].to_numpy(dtype=np.float64, copy=False),
        }


def _iter_raw_base_batches(
    *,
    book_path: Path,
    trade_state: dict[str, np.ndarray],
    vol_profile: dict[int, float],
    adv_btc: float,
    num_slices: int,
    window_seconds: int,
) -> Iterator[np.ndarray]:
    history_seconds = max(int(window_seconds), 300, 30, 10, 1)
    min_timestamp_ms = int(trade_state["trade_ts"][0]) + (history_seconds * 1000)
    vol_profile_arr = np.zeros(1440, dtype=np.float64)
    for minute, fraction in vol_profile.items():
        minute_idx = int(minute)
        if 0 <= minute_idx < 1440:
            vol_profile_arr[minute_idx] = float(fraction)

    bid_history = np.array([], dtype=np.float64)
    ask_history = np.array([], dtype=np.float64)
    observed_index = 0

    for batch in _iter_book_second_batches(book_path=book_path, min_timestamp_ms=min_timestamp_ms):
        timestamps = batch["timestamp_ms"]
        bid_prices = batch["best_bid_price"]
        bid_qtys = batch["best_bid_qty"]
        ask_prices = batch["best_ask_price"]
        ask_qtys = batch["best_ask_qty"]

        if timestamps.size == 0:
            continue

        second_idx = np.searchsorted(trade_state["second_ts"], timestamps, side="right") - 1
        prev_trade_idx = np.searchsorted(trade_state["trade_ts"], timestamps - 1000, side="right") - 1

        valid = (
            (second_idx >= 0)
            & (prev_trade_idx >= 0)
            & np.isfinite(bid_prices)
            & np.isfinite(ask_prices)
            & np.isfinite(bid_qtys)
            & np.isfinite(ask_qtys)
            & (bid_prices > 0.0)
            & (ask_prices > 0.0)
            & (bid_qtys > 0.0)
            & (ask_qtys > 0.0)
            & (ask_prices >= bid_prices)
        )
        if not np.any(valid):
            continue

        timestamps = timestamps[valid]
        bid_prices = bid_prices[valid]
        bid_qtys = bid_qtys[valid]
        ask_prices = ask_prices[valid]
        ask_qtys = ask_qtys[valid]
        second_idx = second_idx[valid]
        prev_trade_idx = prev_trade_idx[valid]

        vol_60 = trade_state["sec_vol_60s"][second_idx]
        vol_300 = trade_state["sec_vol_300s"][second_idx]
        recent_volume_30s = trade_state["sec_volume_30s"][second_idx]
        ofi = trade_state["sec_ofi_10s"][second_idx]
        has_ofi = trade_state["sec_has_ofi_10s"][second_idx]
        prev_mid = trade_state["trade_prices"][prev_trade_idx]

        minute_of_day = ((timestamps // 1000 // 60) % 1440).astype(np.int64)
        expected_volume_30s = vol_profile_arr[minute_of_day] * adv_btc * 0.5

        valid = (
            has_ofi
            & np.isfinite(vol_60)
            & np.isfinite(vol_300)
            & (vol_60 > 0.0)
            & (vol_300 > 0.0)
            & np.isfinite(recent_volume_30s)
            & (recent_volume_30s >= 0.0)
            & np.isfinite(ofi)
            & np.isfinite(prev_mid)
            & (prev_mid > 0.0)
            & np.isfinite(expected_volume_30s)
            & (expected_volume_30s > 0.0)
        )
        if not np.any(valid):
            continue

        timestamps = timestamps[valid]
        bid_prices = bid_prices[valid]
        bid_qtys = bid_qtys[valid]
        ask_prices = ask_prices[valid]
        ask_qtys = ask_qtys[valid]
        vol_60 = vol_60[valid]
        vol_300 = vol_300[valid]
        recent_volume_30s = recent_volume_30s[valid]
        ofi = ofi[valid]
        prev_mid = prev_mid[valid]
        expected_volume_30s = expected_volume_30s[valid]

        mid = (bid_prices + ask_prices) / 2.0
        momentum_bps = (mid - prev_mid) / prev_mid * 10000.0
        volume_participation = recent_volume_30s / expected_volume_30s
        spread_to_vol = (ask_prices - bid_prices) / vol_60
        vol_ratio = vol_60 / vol_300

        bid_means, bid_history = _rolling_mean_with_history(bid_history, bid_qtys, window=60)
        ask_means, ask_history = _rolling_mean_with_history(ask_history, ask_qtys, window=60)
        bid_qty_norm = bid_qtys / bid_means
        ask_qty_norm = ask_qtys / ask_means

        batch_count = timestamps.size
        state_indices = observed_index + np.arange(batch_count, dtype=np.int64)
        time_step = state_indices % max(num_slices, 1)
        time_fraction = (np.maximum(0, num_slices - time_step) / max(num_slices, 1)).astype(np.float64)
        inventory_fraction = np.maximum(0.0, 1.0 - (time_step / max(num_slices - 1, 1)))
        urgency_ratio = np.clip(inventory_fraction / (time_fraction + 1e-8), 0.0, 5.0)

        raw_batch = np.column_stack(
            [
                spread_to_vol,
                ofi,
                volume_participation,
                momentum_bps,
                inventory_fraction,
                time_fraction,
                urgency_ratio,
                vol_60,
                vol_300,
                vol_ratio,
                bid_qty_norm,
                ask_qty_norm,
            ]
        ).astype(np.float64, copy=False)

        finite_mask = np.all(np.isfinite(raw_batch), axis=1)
        if not np.any(finite_mask):
            continue

        raw_batch = raw_batch[finite_mask]
        observed_index += int(raw_batch.shape[0])
        yield raw_batch


def _base_day_worker(
    *,
    date_str: str,
    parsed_dir_str: str,
    vol_profile: dict[int, float],
    adv_btc: float,
    num_slices: int,
    window_seconds: int,
) -> dict:
    started = time.perf_counter()
    try:
        parsed_dir = Path(parsed_dir_str)
        trade_path = parsed_dir / "aggTrades" / f"{date_str}.parquet"
        book_path = parsed_dir / "bookTicker" / f"{date_str}.parquet"

        if not trade_path.exists() or not book_path.exists():
            return {
                "date": date_str,
                "status": "skipped",
                "reason": "missing parquet input",
                "count": 0,
                "elapsed_s": time.perf_counter() - started,
            }

        trades = _read_day_trades(trade_path)
        if trades.empty:
            return {
                "date": date_str,
                "status": "skipped",
                "reason": "empty aggTrades",
                "count": 0,
                "elapsed_s": time.perf_counter() - started,
            }

        trade_state = _build_day_trade_state(trades)
        stats = RunningStats(dim=12)
        for raw_batch in _iter_raw_base_batches(
            book_path=book_path,
            trade_state=trade_state,
            vol_profile=vol_profile,
            adv_btc=adv_btc,
            num_slices=num_slices,
            window_seconds=window_seconds,
        ):
            stats.update_batch(raw_batch)

        elapsed_s = time.perf_counter() - started
        if stats.count == 0:
            return {
                "date": date_str,
                "status": "skipped",
                "reason": "no valid observed states",
                "count": 0,
                "elapsed_s": elapsed_s,
            }

        return {
            "date": date_str,
            "status": "ok",
            "count": int(stats.count),
            "mean": stats.mean,
            "m2": stats.m2,
            "elapsed_s": elapsed_s,
        }
    except Exception as exc:
        return {
            "date": date_str,
            "status": "error",
            "reason": str(exc),
            "count": 0,
            "elapsed_s": time.perf_counter() - started,
        }


def _interaction_day_worker(
    *,
    date_str: str,
    parsed_dir_str: str,
    vol_profile: dict[int, float],
    adv_btc: float,
    num_slices: int,
    window_seconds: int,
    base_means: np.ndarray,
    base_stds: np.ndarray,
) -> dict:
    started = time.perf_counter()
    try:
        parsed_dir = Path(parsed_dir_str)
        trade_path = parsed_dir / "aggTrades" / f"{date_str}.parquet"
        book_path = parsed_dir / "bookTicker" / f"{date_str}.parquet"

        if not trade_path.exists() or not book_path.exists():
            return {
                "date": date_str,
                "status": "skipped",
                "reason": "missing parquet input",
                "count": 0,
                "elapsed_s": time.perf_counter() - started,
            }

        trades = _read_day_trades(trade_path)
        if trades.empty:
            return {
                "date": date_str,
                "status": "skipped",
                "reason": "empty aggTrades",
                "count": 0,
                "elapsed_s": time.perf_counter() - started,
            }

        trade_state = _build_day_trade_state(trades)
        stats = RunningStats(dim=len(INTERACTION_TERMS))
        for raw_batch in _iter_raw_base_batches(
            book_path=book_path,
            trade_state=trade_state,
            vol_profile=vol_profile,
            adv_btc=adv_btc,
            num_slices=num_slices,
            window_seconds=window_seconds,
        ):
            normalized_base = (raw_batch - base_means) / base_stds
            normalized_base[:, 4] = raw_batch[:, 4]
            normalized_base[:, 5] = raw_batch[:, 5]
            interactions = np.column_stack(
                [
                    normalized_base[:, idx_a] * normalized_base[:, idx_b]
                    for idx_a, idx_b in INTERACTION_TERMS
                ]
            ).astype(np.float64, copy=False)
            if interactions.size == 0:
                continue
            finite_mask = np.all(np.isfinite(interactions), axis=1)
            if not np.any(finite_mask):
                continue
            stats.update_batch(interactions[finite_mask])

        elapsed_s = time.perf_counter() - started
        if stats.count == 0:
            return {
                "date": date_str,
                "status": "skipped",
                "reason": "no valid interaction states",
                "count": 0,
                "elapsed_s": elapsed_s,
            }

        return {
            "date": date_str,
            "status": "ok",
            "count": int(stats.count),
            "mean": stats.mean,
            "m2": stats.m2,
            "elapsed_s": elapsed_s,
        }
    except Exception as exc:
        return {
            "date": date_str,
            "status": "error",
            "reason": str(exc),
            "count": 0,
            "elapsed_s": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build feature normalization stats from parsed training data."
    )
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--calibration", default=str(CALIBRATION_PATH))
    parser.add_argument("--parsed-dir", default=str(PARSED_DIR))
    parser.add_argument("--volume-profile", default=str(VOLUME_PROFILE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_worker_count(),
        help="Number of worker processes to use. Default leaves some CPU headroom.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Deprecated and ignored. Feature stats are computed from all valid observed rows.",
    )
    parser.add_argument("--window-seconds", type=int, default=300)
    args = parser.parse_args()

    config_path = Path(args.config)
    calibration_path = Path(args.calibration)
    parsed_dir = Path(args.parsed_dir)
    volume_profile_path = Path(args.volume_profile)
    output_path = Path(args.output)

    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found at {calibration_path}. Run pipeline/calibrate_params.py first."
        )
    if not volume_profile_path.exists():
        raise FileNotFoundError(
            f"Volume profile not found at {volume_profile_path}. Run pipeline/build_volume_profile.py first."
        )

    cfg = _load_yaml(config_path)
    calibration = _load_json(calibration_path)
    volume_profile = pd.read_parquet(volume_profile_path)
    vol_profile = dict(
        zip(
            volume_profile["minute_of_day"].astype(int),
            volume_profile["fraction"].astype(float),
        )
    )

    train_dates = _training_dates(parsed_dir, float(cfg["data"]["train_split"]))
    d_features = int(cfg.get("features", {}).get("d_features", 12))
    d_features = 18 if d_features >= 18 else 12
    num_slices = int(cfg.get("execution", {}).get("num_slices", 60))
    adv_btc = float(calibration["adv_btc"])
    worker_count = max(1, int(args.workers))

    print(
        f"Building feature stats from observed market data over {len(train_dates)} training days...",
        flush=True,
    )
    print(
        f"Using {worker_count} worker process(es) for per-day feature-stat aggregation.",
        flush=True,
    )

    base_stats = RunningStats(dim=12)
    n_observed_rows = 0
    base_errors: list[str] = []
    print("Submitting base-pass day jobs...", flush=True)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _base_day_worker,
                date_str=date_str,
                parsed_dir_str=str(parsed_dir),
                vol_profile=vol_profile,
                adv_btc=adv_btc,
                num_slices=num_slices,
                window_seconds=int(args.window_seconds),
            )
            for date_str in train_dates
        ]
        print(f"Base pass: queued {len(futures)} day tasks.", flush=True)

        for completed_idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["status"] == "ok":
                base_stats.merge(result["count"], result["mean"], result["m2"])
                n_observed_rows += int(result["count"])
                print(
                    f"[Base pass {completed_idx}/{len(futures)}] {result['date']}: "
                    f"{result['count']} observed states in {result['elapsed_s']:.1f}s "
                    f"(running total {n_observed_rows})",
                    flush=True,
                )
            elif result["status"] == "skipped":
                print(
                    f"[Base pass {completed_idx}/{len(futures)}] {result['date']}: "
                    f"skipped ({result['reason']}) after {result['elapsed_s']:.1f}s",
                    flush=True,
                )
            else:
                base_errors.append(f"{result['date']}: {result['reason']}")
                print(
                    f"[Base pass {completed_idx}/{len(futures)}] {result['date']}: "
                    f"ERROR ({result['reason']}) after {result['elapsed_s']:.1f}s",
                    flush=True,
                )

    if base_errors:
        raise RuntimeError(
            "Feature-stat base pass failed for "
            f"{len(base_errors)} day(s): " + "; ".join(base_errors[:5])
        )

    if n_observed_rows == 0:
        raise RuntimeError("Feature-stat fitting could not build any valid market states.")

    base_means, base_stds = base_stats.finalize()
    means = base_means
    stds = base_stds

    if d_features == 18:
        print("Starting interaction pass...", flush=True)
        interaction_stats = RunningStats(dim=len(INTERACTION_TERMS))
        n_interaction_rows = 0
        interaction_errors: list[str] = []
        print("Submitting interaction-pass day jobs...", flush=True)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _interaction_day_worker,
                    date_str=date_str,
                    parsed_dir_str=str(parsed_dir),
                    vol_profile=vol_profile,
                    adv_btc=adv_btc,
                    num_slices=num_slices,
                    window_seconds=int(args.window_seconds),
                    base_means=base_means,
                    base_stds=base_stds,
                )
                for date_str in train_dates
            ]
            print(f"Interaction pass: queued {len(futures)} day tasks.", flush=True)

            for completed_idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result["status"] == "ok":
                    interaction_stats.merge(result["count"], result["mean"], result["m2"])
                    n_interaction_rows += int(result["count"])
                    print(
                        f"[Interaction pass {completed_idx}/{len(futures)}] {result['date']}: "
                        f"{result['count']} observed states in {result['elapsed_s']:.1f}s "
                        f"(running total {n_interaction_rows})",
                        flush=True,
                    )
                else:
                    if result["status"] == "skipped":
                        print(
                            f"[Interaction pass {completed_idx}/{len(futures)}] {result['date']}: "
                            f"skipped ({result['reason']}) after {result['elapsed_s']:.1f}s",
                            flush=True,
                        )
                    else:
                        interaction_errors.append(f"{result['date']}: {result['reason']}")
                        print(
                            f"[Interaction pass {completed_idx}/{len(futures)}] {result['date']}: "
                            f"ERROR ({result['reason']}) after {result['elapsed_s']:.1f}s",
                            flush=True,
                        )

        if interaction_errors:
            raise RuntimeError(
                "Feature-stat interaction pass failed for "
                f"{len(interaction_errors)} day(s): " + "; ".join(interaction_errors[:5])
            )

        if n_interaction_rows == 0:
            raise RuntimeError("Feature-stat fitting could not build any valid interaction states.")

        interaction_means, interaction_stds = interaction_stats.finalize()
        means = np.concatenate([base_means, interaction_means])
        stds = np.concatenate([base_stds, interaction_stds])

    output = {
        "means": means.tolist(),
        "stds": stds.tolist(),
        "feature_names": FEATURE_NAMES[:d_features],
        "metadata": {
            "train_dates": train_dates,
            "n_samples": int(n_observed_rows),
            "d_features": d_features,
            "window_seconds": int(args.window_seconds),
            "adv_btc": adv_btc,
            "sampling": "disabled",
            "source": "observed market data only",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Feature stats saved to {output_path}")
    print(f"Training dates: {train_dates[0]} -> {train_dates[-1]}")
    print(f"Observed states used: {n_observed_rows}")
    print(f"Feature width: {d_features}")


if __name__ == "__main__":
    freeze_support()
    main()
