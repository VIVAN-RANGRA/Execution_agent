"""Gym-like execution environment: reset() and step()."""
import json
import numpy as np
from typing import Optional, Tuple, Dict, List

from config import load_calibration, load_config, resolve_project_paths
from simulator.data_classes import MarketState, ExecutionConfig, Fill
from simulator.market_data_stream import MarketDataStream
from simulator.fill_engine import compute_fill
from simulator.episode_state import EpisodeState
from simulator.reward_calculator import compute_step_reward, compute_terminal_reward
from features.feature_engineer import FeatureEngineer

# Urgency level -> fraction of uniform pace
URGENCY_MULTIPLIERS = {0: 0.20, 1: 0.60, 2: 1.00, 3: 1.50, 4: 2.00}


class ExecutionEnv:
    """
    Gym-like environment for order execution simulation.
    """

    def __init__(
        self,
        config_path: str = None,
        parsed_data_dir: str = None,
        volume_profile_path: str = None,
        calibration_params_path: str = None,
        feature_stats_path: str = None,
        split: str = "train",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_days: Optional[int] = None,
    ):
        self._paths = resolve_project_paths(
            config_path=config_path,
            parsed_data_dir=parsed_data_dir,
            volume_profile_path=volume_profile_path,
            calibration_params_path=calibration_params_path,
            feature_stats_path=feature_stats_path,
        )
        self._cfg = load_config(str(self._paths.config_path))
        self._cal = load_calibration(str(self._paths.calibration_params_path))

        self._split = split
        self._start_date = start_date
        self._end_date = end_date
        self._max_days = int(max_days) if max_days is not None else None
        self._parsed_data_dir = self._paths.parsed_data_dir
        self._volume_profile_path = self._paths.volume_profile_path

        # Config values
        exec_cfg = self._cfg.execution
        self.min_slice_fraction = exec_cfg.min_slice_fraction
        self.max_slice_fraction = exec_cfg.max_slice_fraction
        self.configure_execution(
            total_quantity=exec_cfg.total_quantity_btc,
            time_horizon_seconds=exec_cfg.time_horizon_seconds,
            num_slices=exec_cfg.num_slices,
        )

        # Impact model params
        self._eta = self._cal.eta
        self._gamma = self._cal.gamma
        self._alpha = self._cal.alpha
        self._adv = self._cal.adv_btc

        # Feature engineer
        self._feature_dim = self._resolve_feature_dimension()
        self._feature_eng = FeatureEngineer(
            d_features=self._feature_dim,
            feature_stats_path=self._paths.feature_stats_path,
        )

        # Determine available dates for this split
        self._dates = self._get_split_dates()

        # Market data stream
        self._stream = MarketDataStream(self._parsed_data_dir, self._volume_profile_path)
        self._stream.load_date_range(self._dates)

        # Episode state (initialized in reset)
        self._episode: Optional[EpisodeState] = None
        self._current_mid: float = 0.0
        self._episode_start_ms: int = 0
        self._current_state: Optional[MarketState] = None
        self._prev_mid: Optional[float] = None

    @property
    def total_quantity(self) -> float:
        return self._total_quantity

    @total_quantity.setter
    def total_quantity(self, value: float) -> None:
        self._total_quantity = float(value)

    @property
    def time_horizon_seconds(self) -> int:
        return self._time_horizon_seconds

    @time_horizon_seconds.setter
    def time_horizon_seconds(self, value: int) -> None:
        self._time_horizon_seconds = max(1, int(value))
        self._refresh_slice_duration()

    @property
    def num_slices(self) -> int:
        return self._num_slices

    @num_slices.setter
    def num_slices(self, value: int) -> None:
        self._num_slices = max(1, int(value))
        self._refresh_slice_duration()

    def configure_execution(
        self,
        *,
        total_quantity: float = None,
        time_horizon_seconds: int = None,
        num_slices: int = None,
    ) -> None:
        """Update the mutable execution settings in one place."""
        if total_quantity is not None:
            self.total_quantity = total_quantity
        if time_horizon_seconds is not None:
            self.time_horizon_seconds = time_horizon_seconds
        if num_slices is not None:
            self.num_slices = num_slices

    def _refresh_slice_duration(self) -> None:
        num_slices = getattr(self, "_num_slices", 1)
        horizon = getattr(self, "_time_horizon_seconds", 1)
        self.slice_duration_s = horizon / max(num_slices, 1)

    def _resolve_feature_dimension(self) -> int:
        """
        Prefer the configured feature dimension, but stay compatible with older
        on-disk feature stats files when they only contain 12 base features.
        """
        configured_dim = self._cfg.feature_dimension
        stats_path = self._paths.feature_stats_path
        if not stats_path.exists():
            return configured_dim

        try:
            with open(stats_path) as f:
                stats = json.load(f)
            stats_dim = len(stats.get("means", []))
        except (OSError, json.JSONDecodeError, TypeError):
            return configured_dim

        return stats_dim if stats_dim in (12, 18) else configured_dim

    def _build_execution_config(self, arrival_price: float) -> ExecutionConfig:
        return ExecutionConfig(
            total_quantity=self.total_quantity,
            time_horizon_seconds=self.time_horizon_seconds,
            num_slices=self.num_slices,
            arrival_price=arrival_price,
        )

    def _build_context(
        self,
        state: MarketState,
        *,
        inventory_fraction: float,
        time_step: int,
        total_steps: int,
        prev_mid_price: Optional[float],
    ) -> np.ndarray:
        expected_vol = self._get_expected_volume_30s(state.minute_of_day)
        return self._feature_eng.transform(
            state,
            inventory=inventory_fraction,
            time_step=time_step,
            total_steps=total_steps,
            expected_volume_30s=expected_vol,
            prev_mid_price=prev_mid_price,
        )

    # ------------------------------------------------------------------
    # Public properties that delegate to EpisodeState for backward compat
    # ------------------------------------------------------------------
    @property
    def inventory(self) -> float:
        return self._episode.inventory if self._episode else 0.0

    @inventory.setter
    def inventory(self, value: float) -> None:
        if self._episode:
            self._episode.inventory = value

    @property
    def time_step(self) -> int:
        return self._episode.time_step if self._episode else 0

    @time_step.setter
    def time_step(self, value: int) -> None:
        if self._episode:
            self._episode.time_step = value

    @property
    def arrival_price(self) -> float:
        return self._episode.arrival_price if self._episode else 0.0

    @arrival_price.setter
    def arrival_price(self, value: float) -> None:
        if self._episode:
            self._episode.arrival_price = value

    @property
    def current_config(self) -> Optional[ExecutionConfig]:
        return self._episode.current_config if self._episode else None

    @current_config.setter
    def current_config(self, value: ExecutionConfig) -> None:
        if self._episode:
            self._episode.current_config = value

    @property
    def fills(self) -> List[Fill]:
        return self._episode.fills if self._episode else []

    @fills.setter
    def fills(self, value: List[Fill]) -> None:
        if self._episode:
            self._episode.fills = value

    def _get_split_dates(self) -> List[str]:
        """Get date strings for the requested split."""
        agg_dir = self._parsed_data_dir / "aggTrades"
        all_dates = sorted([f.stem for f in agg_dir.glob("*.parquet")]) if agg_dir.exists() else []
        n = len(all_dates)
        if n == 0:
            return []
        train_end = int(n * self._cfg.data.train_split)
        val_end = train_end + int(n * self._cfg.data.val_split)
        if self._split == "train":
            split_dates = all_dates[:train_end]
        elif self._split == "val":
            split_dates = all_dates[train_end:val_end]
        else:
            split_dates = all_dates[val_end:]

        if self._start_date:
            split_dates = [date for date in split_dates if date >= self._start_date]
        if self._end_date:
            split_dates = [date for date in split_dates if date <= self._end_date]
        if self._max_days is not None and self._max_days > 0:
            split_dates = split_dates[-self._max_days :]
        return split_dates

    def reset(self, episode_seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for a new episode.
        Returns (context_vector, info_dict).
        """
        rng = np.random.RandomState(episode_seed)

        # Select a valid start timestamp
        valid_starts = self._stream.get_available_starts(self.time_horizon_seconds)
        if not valid_starts:
            raise RuntimeError("No valid episode starts available in data")
        start_ms = int(rng.choice(valid_starts))

        self._episode_start_ms = start_ms
        self._prev_mid = None

        # Arrival price = mid-price at start
        self._current_state = self._stream.build_market_state(start_ms, window_s=60)
        arr_price = self._current_state.mid_price
        self._current_mid = arr_price

        config = self._build_execution_config(arr_price)

        # Create fresh episode state
        self._episode = EpisodeState(
            total_quantity=self.total_quantity,
            num_slices=self.num_slices,
            arrival_price=arr_price,
            config=config,
        )

        context = self._build_context(
            self._current_state,
            inventory_fraction=1.0,
            time_step=0,
            total_steps=self.num_slices,
            prev_mid_price=None,
        )

        return context, {
            "arrival_price": arr_price,
            "episode_seed": episode_seed,
            "start_timestamp": start_ms,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one slice with the given urgency action.
        Returns (context, reward, done, info).
        """
        assert 0 <= action <= 4, f"Invalid action {action}"
        ep = self._episode

        # Compute lot size from urgency action
        steps_remaining = max(1, ep.num_slices - ep.time_step)
        uniform_pace = ep.inventory / steps_remaining
        urgency_mult = URGENCY_MULTIPLIERS[action]
        target_qty = uniform_pace * urgency_mult

        # Clip to bounds
        min_qty = self.min_slice_fraction * ep.inventory
        max_qty = self.max_slice_fraction * ep.inventory
        v = float(np.clip(target_qty, min_qty, max_qty))

        # Current slice window
        slice_start_ms = self._episode_start_ms + ep.time_step * int(self.slice_duration_s * 1000)
        slice_end_ms = slice_start_ms + int(self.slice_duration_s * 1000)

        # Get market state for this window
        state = self._stream.build_market_state(slice_start_ms, window_s=int(self.slice_duration_s))
        mid = state.mid_price + ep.permanent_impact_accumulated

        # Compute fill via fill_engine
        fill_price, impact_cost_usd, delta_permanent = compute_fill(
            v=v,
            mid_price=mid,
            spread=state.spread,
            eta=self._eta,
            adv=self._adv,
            alpha=self._alpha,
            gamma=self._gamma,
            slice_duration_s=self.slice_duration_s,
            side=ep.current_config.side,
            permanent_impact_accumulated=ep.permanent_impact_accumulated,
        )

        # Update permanent impact
        ep.permanent_impact_accumulated += delta_permanent

        # Update inventory
        ep.update_inventory(v)

        # Record fill
        fill = ep.record_fill(
            timestamp_ms=slice_start_ms,
            quantity_filled=v,
            fill_price=fill_price,
            impact_cost_usd=impact_cost_usd,
            slice_index=ep.time_step,
        )

        # Step reward via reward_calculator
        reward = compute_step_reward(fill_price, ep.arrival_price, v, ep.total_quantity)

        ep.advance_time()
        done = ep.is_done()

        # Terminal: force-execute remaining inventory
        if done and ep.inventory > 1e-6:
            terminal_fill = ep.force_execute_remaining(
                timestamp_ms=slice_end_ms,
                mid_price=state.mid_price,
                spread=state.spread,
                eta=self._eta,
                adv=self._adv,
                alpha=self._alpha,
                gamma=self._gamma,
                slice_duration_s=self.slice_duration_s,
            )
            if terminal_fill is not None:
                reward_rem = compute_step_reward(
                    terminal_fill.fill_price,
                    ep.arrival_price,
                    terminal_fill.quantity_filled,
                    ep.total_quantity,
                )
                reward += reward_rem
        else:
            terminal_fill = None

        # Terminal bonus (vectorized path)
        if done:
            terminal_reward = compute_terminal_reward(
                ep.fills, ep.arrival_price,
                fill_prices=ep.get_fill_prices(),
                fill_quantities=ep.get_fill_quantities(),
            )
            reward += terminal_reward

        # Next context
        self._prev_mid = state.mid_price
        self._current_state = state

        next_context = self._build_context(
            state,
            inventory_fraction=ep.inventory_fraction,
            time_step=ep.time_step,
            total_steps=ep.num_slices,
            prev_mid_price=self._prev_mid,
        )

        info = {
            "fill": fill,
            "terminal_fill": terminal_fill,
            "executed_quantity": float(
                fill.quantity_filled + (terminal_fill.quantity_filled if terminal_fill else 0.0)
            ),
            "market_state": state,
        }
        return next_context, float(reward), done, info

    def _get_expected_volume_30s(self, minute_of_day: int) -> float:
        """Get expected volume for a 30-second window at this time of day."""
        fraction = self._stream._vol_profile_dict.get(minute_of_day, 0.0)
        # Expected volume in 30s = fraction of daily ADV * 30/86400 * ADV
        # Simplified: use fraction * adv * (30/60) for 30s window within a minute
        return max(fraction * self._adv * 0.5, 1e-6)
