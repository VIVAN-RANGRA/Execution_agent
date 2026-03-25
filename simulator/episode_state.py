"""Episode state tracking extracted from ExecutionEnv."""
import numpy as np
from typing import List, Optional
from simulator.data_classes import ExecutionConfig, Fill
from simulator.fill_engine import compute_fill


class EpisodeState:
    """
    Tracks the mutable state of a single execution episode.

    Encapsulates inventory, time progression, fill records, and permanent
    impact accumulation so that the main environment only orchestrates
    market data and feature computation.

    Optimized: uses pre-allocated numpy arrays for fill data instead of
    list.append with Fill objects. Fill objects are constructed lazily
    for backward compatibility.
    """

    def __init__(
        self,
        total_quantity: float,
        num_slices: int,
        arrival_price: float,
        config: ExecutionConfig,
    ):
        self.inventory: float = total_quantity
        self.time_step: int = 0
        self.permanent_impact_accumulated: float = 0.0
        self.arrival_price: float = arrival_price
        self.current_config: ExecutionConfig = config

        self._total_quantity = total_quantity
        self._num_slices = num_slices

        # Pre-allocated fill arrays (+2 for potential force-execute)
        max_fills = num_slices + 2
        self._fill_timestamps = np.zeros(max_fills, dtype=np.int64)
        self._fill_quantities = np.zeros(max_fills, dtype=np.float64)
        self._fill_prices = np.zeros(max_fills, dtype=np.float64)
        self._fill_impacts = np.zeros(max_fills, dtype=np.float64)
        self._fill_slice_indices = np.zeros(max_fills, dtype=np.int64)
        self._fill_count: int = 0

        # Cached Fill objects (lazily constructed)
        self._fills_cache: Optional[List[Fill]] = None
        self.terminal_fill: Optional[Fill] = None

    @property
    def fills(self) -> List[Fill]:
        """Backward-compatible property: returns list of Fill objects.

        Lazily constructed from pre-allocated arrays.
        """
        if self._fills_cache is not None and len(self._fills_cache) == self._fill_count:
            return self._fills_cache
        self._fills_cache = [
            Fill(
                timestamp_ms=int(self._fill_timestamps[i]),
                quantity_filled=float(self._fill_quantities[i]),
                fill_price=float(self._fill_prices[i]),
                market_impact_cost=float(self._fill_impacts[i]),
                slice_index=int(self._fill_slice_indices[i]),
            )
            for i in range(self._fill_count)
        ]
        return self._fills_cache

    @fills.setter
    def fills(self, value: List[Fill]) -> None:
        """Backward-compatible setter: repopulates arrays from Fill list."""
        self._fill_count = len(value)
        # Ensure arrays are large enough
        if self._fill_count > len(self._fill_timestamps):
            new_size = self._fill_count + 2
            self._fill_timestamps = np.zeros(new_size, dtype=np.int64)
            self._fill_quantities = np.zeros(new_size, dtype=np.float64)
            self._fill_prices = np.zeros(new_size, dtype=np.float64)
            self._fill_impacts = np.zeros(new_size, dtype=np.float64)
            self._fill_slice_indices = np.zeros(new_size, dtype=np.int64)
        for i, f in enumerate(value):
            self._fill_timestamps[i] = f.timestamp_ms
            self._fill_quantities[i] = f.quantity_filled
            self._fill_prices[i] = f.fill_price
            self._fill_impacts[i] = f.market_impact_cost
            self._fill_slice_indices[i] = f.slice_index
        self._fills_cache = None

    @property
    def fill_count(self) -> int:
        """Number of fills recorded so far."""
        return self._fill_count

    def record_fill(
        self,
        timestamp_ms: int,
        quantity_filled: float,
        fill_price: float,
        impact_cost_usd: float,
        slice_index: int,
    ) -> Fill:
        """Record a fill using pre-allocated arrays and return the Fill object."""
        idx = self._fill_count
        self._fill_timestamps[idx] = timestamp_ms
        self._fill_quantities[idx] = quantity_filled
        self._fill_prices[idx] = fill_price
        self._fill_impacts[idx] = impact_cost_usd
        self._fill_slice_indices[idx] = slice_index
        self._fill_count += 1
        self._fills_cache = None  # Invalidate cache

        return Fill(
            timestamp_ms=timestamp_ms,
            quantity_filled=quantity_filled,
            fill_price=fill_price,
            market_impact_cost=impact_cost_usd,
            slice_index=slice_index,
        )

    def update_inventory(self, qty_executed: float) -> None:
        """Reduce inventory by the executed quantity, floor at zero."""
        self.inventory -= qty_executed
        self.inventory = max(0.0, self.inventory)

    def advance_time(self) -> None:
        """Move to the next time step."""
        self.time_step += 1

    def is_done(self) -> bool:
        """Check whether the episode is complete."""
        return (self.time_step >= self._num_slices) or (self.inventory <= 1e-6)

    def force_execute_remaining(
        self,
        timestamp_ms: int,
        mid_price: float,
        spread: float,
        eta: float,
        adv: float,
        alpha: float,
        gamma: float,
        slice_duration_s: float,
    ) -> Optional[Fill]:
        """
        Force-execute any remaining inventory at terminal step.

        Returns the terminal Fill if there was remaining inventory, else None.
        """
        if self.inventory <= 1e-6:
            return None

        v_rem = self.inventory
        side = self.current_config.side

        # Use current permanent impact for the terminal fill's mid
        effective_mid = mid_price + self.permanent_impact_accumulated

        fill_price, impact_cost, delta_perm = compute_fill(
            v=v_rem,
            mid_price=effective_mid,
            spread=spread,
            eta=eta,
            adv=adv,
            alpha=alpha,
            gamma=gamma,
            slice_duration_s=slice_duration_s,
            side=side,
            permanent_impact_accumulated=self.permanent_impact_accumulated,
        )

        self.permanent_impact_accumulated += delta_perm
        fill = self.record_fill(
            timestamp_ms=timestamp_ms,
            quantity_filled=v_rem,
            fill_price=fill_price,
            impact_cost_usd=impact_cost,
            slice_index=self.time_step,
        )
        self.terminal_fill = fill
        self.inventory = 0.0
        return fill

    # --- Vectorized accessors for reward computation ---

    def get_fill_quantities(self) -> np.ndarray:
        """Return numpy view of fill quantities (no copy)."""
        return self._fill_quantities[:self._fill_count]

    def get_fill_prices(self) -> np.ndarray:
        """Return numpy view of fill prices (no copy)."""
        return self._fill_prices[:self._fill_count]

    def get_fill_impacts(self) -> np.ndarray:
        """Return numpy view of fill impact costs (no copy)."""
        return self._fill_impacts[:self._fill_count]

    @property
    def inventory_fraction(self) -> float:
        """Fraction of original quantity still remaining."""
        if self._total_quantity <= 0:
            return 0.0
        return self.inventory / self._total_quantity

    @property
    def total_quantity(self) -> float:
        return self._total_quantity

    @property
    def num_slices(self) -> int:
        return self._num_slices
