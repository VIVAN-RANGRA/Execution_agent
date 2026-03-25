"""Feature engineering: fit(), transform(), fit_transform()."""
from collections import deque
import numpy as np
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from simulator.data_classes import MarketState


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_STATS_PATH = BASE_DIR / "config" / "feature_stats.json"
CONFIG_PATH = BASE_DIR / "config" / "default_config.yaml"

# Interaction term definitions: (name, index_a, index_b)
INTERACTION_TERMS = [
    ("OFI_x_momentum", 1, 3),
    ("spread_vol_x_inventory", 0, 4),
    ("vol_ratio_x_time", 9, 5),
    ("bid_qty_x_ask_qty", 10, 11),
    ("urgency_x_vol_participation", 6, 2),
    ("momentum_x_time", 3, 5),
]


class FeatureEngineer:
    """
    Converts MarketState into a normalized context vector.
    Supports 12-dim (base features only) or 18-dim (base + 6 interaction terms).
    Must call fit() on training data before transform().
    """

    BASE_FEATURE_NAMES = [
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
    ]

    INTERACTION_FEATURE_NAMES = [name for name, _, _ in INTERACTION_TERMS]

    FEATURE_NAMES = BASE_FEATURE_NAMES + INTERACTION_FEATURE_NAMES

    N_BASE = 12
    N_INTERACTIONS = 6

    def __init__(self, d_features: int = 12, feature_stats_path: Optional[Path] = None):
        if d_features not in (12, 18):
            raise ValueError(f"d_features must be 12 or 18, got {d_features}")
        self.d_features = d_features
        self.use_interactions = d_features == 18
        self.feature_stats_path = Path(feature_stats_path) if feature_stats_path else FEATURE_STATS_PATH
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._fitted = False
        self._rolling_bid_window = 60  # steps
        # Rolling accumulators for normalization
        self._rolling_bid_qty = deque()
        self._rolling_ask_qty = deque()
        self._rolling_bid_sum = 0.0
        self._rolling_ask_sum = 0.0

        # Try loading existing stats
        if self.feature_stats_path.exists():
            self._load_stats()

    @staticmethod
    @lru_cache(maxsize=8)
    def _read_stats_file(path_str: str):
        with open(path_str) as f:
            stats = json.load(f)
        means = np.array(stats["means"], dtype=np.float32)
        stds = np.array(stats["stds"], dtype=np.float32)
        return means, stds

    def _load_stats(self) -> None:
        means, stds = self._read_stats_file(str(self.feature_stats_path.resolve()))
        self._means = means.copy()
        self._stds = stds.copy()
        self._fitted = True

    def _save_stats(self) -> None:
        names = self.FEATURE_NAMES[:self.d_features]
        stats = {
            "means": self._means.tolist(),
            "stds": self._stds.tolist(),
            "feature_names": names,
        }
        self.feature_stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feature_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Keep the in-process cache hot after writing fresh stats.
        self._read_stats_file.cache_clear()

    def _extract_raw_base(
        self,
        state: MarketState,
        inventory: float,
        time_step: int,
        total_steps: int,
        volume_profile_dict: Optional[dict] = None,
        expected_volume_30s: Optional[float] = None,
        prev_mid_price: Optional[float] = None,
    ) -> np.ndarray:
        """Extract raw (un-normalized) 12-dim base feature vector."""
        # Feature 0: Spread-to-volatility ratio
        vol_60s = state.recent_volatility_60s if state.recent_volatility_60s > 1e-10 else 1e-10
        spread_to_vol = state.spread / vol_60s

        # Feature 1: OFI -- already in [-1, 1], no normalization needed
        ofi = state.ofi_10s

        # Feature 2: Volume participation rate
        expected_vol = expected_volume_30s if (expected_volume_30s and expected_volume_30s > 0) else (state.recent_volume_30s + 1e-8)
        vol_participation = state.recent_volume_30s / expected_vol

        # Feature 3: Short-term price momentum in bps
        if prev_mid_price and prev_mid_price > 0:
            momentum_bps = (state.mid_price - prev_mid_price) / prev_mid_price * 10000.0
        else:
            momentum_bps = 0.0

        # Feature 4: Inventory fraction remaining [0, 1]
        inv_fraction = float(np.clip(inventory, 0.0, 1.0))

        # Feature 5: Time fraction remaining [0, 1]
        steps_remaining = max(0, total_steps - time_step)
        time_fraction = float(steps_remaining / total_steps) if total_steps > 0 else 0.0

        # Feature 6: Urgency ratio
        urgency_ratio = inv_fraction / (time_fraction + 1e-8)
        urgency_ratio = float(np.clip(urgency_ratio, 0.0, 5.0))

        # Feature 7: Realized vol short (60s)
        vol_short = state.recent_volatility_60s

        # Feature 8: Realized vol long (300s)
        vol_long = state.recent_volatility_300s

        # Feature 9: Vol ratio short/long
        vol_ratio = vol_short / (vol_long + 1e-10)

        # Feature 10 & 11: Bid/ask qty normalized by rolling mean
        if len(self._rolling_bid_qty) >= self._rolling_bid_window:
            self._rolling_bid_sum -= self._rolling_bid_qty.popleft()
            self._rolling_ask_sum -= self._rolling_ask_qty.popleft()
        self._rolling_bid_qty.append(state.bid_qty)
        self._rolling_ask_qty.append(state.ask_qty)
        self._rolling_bid_sum += state.bid_qty
        self._rolling_ask_sum += state.ask_qty
        window_len = max(1, len(self._rolling_bid_qty))
        mean_bid_qty = self._rolling_bid_sum / window_len
        mean_ask_qty = self._rolling_ask_sum / window_len
        bid_qty_norm = state.bid_qty / (mean_bid_qty + 1e-10)
        ask_qty_norm = state.ask_qty / (mean_ask_qty + 1e-10)

        raw = np.array([
            spread_to_vol,
            ofi,
            vol_participation,
            momentum_bps,
            inv_fraction,
            time_fraction,
            urgency_ratio,
            vol_short,
            vol_long,
            vol_ratio,
            bid_qty_norm,
            ask_qty_norm,
        ], dtype=np.float64)

        return raw

    # Keep backward-compatible alias
    def _extract_raw(self, *args, **kwargs) -> np.ndarray:
        return self._extract_raw_base(*args, **kwargs)

    @staticmethod
    def _compute_interactions(normalized_base: np.ndarray) -> np.ndarray:
        """
        Compute 6 interaction features from the z-score normalized base features.
        Returns a 6-dim array.
        """
        interactions = np.array([
            normalized_base[idx_a] * normalized_base[idx_b]
            for _, idx_a, idx_b in INTERACTION_TERMS
        ], dtype=np.float64)
        return interactions

    def fit(self, states: list, inventories: list = None, time_steps: list = None, total_steps: int = 60) -> None:
        """
        Compute z-score normalization stats from training data.
        Must be called on training split only.

        For 18-dim mode:
          1. Compute means/stds for the 12 base features.
          2. Z-score normalize base features.
          3. Compute interaction terms from normalized base features.
          4. Compute means/stds for the 6 interaction features.
          5. Store all 18 means/stds (base stats + interaction stats).
        """
        if inventories is None:
            inventories = [0.5] * len(states)
        if time_steps is None:
            time_steps = list(range(len(states)))

        raw_base_features = []
        for i, state in enumerate(states):
            inv = inventories[i] if i < len(inventories) else 0.5
            t = time_steps[i] if i < len(time_steps) else i
            prev_mid = states[i - 1].mid_price if i > 0 else None
            raw = self._extract_raw_base(state, inv, t, total_steps, prev_mid_price=prev_mid)
            raw_base_features.append(raw)

        arr_base = np.array(raw_base_features, dtype=np.float64)
        base_means = arr_base.mean(axis=0).astype(np.float32)
        base_stds = arr_base.std(axis=0).astype(np.float32)
        base_stds = np.where(base_stds < 1e-8, 1.0, base_stds)

        if not self.use_interactions:
            # 12-dim mode: just base features
            self._means = base_means
            self._stds = base_stds
        else:
            # 18-dim mode: normalize base, compute interactions, then fit interactions
            normalized_base = (arr_base - base_means) / base_stds
            # Override inventory_fraction and time_fraction with raw values
            normalized_base[:, 4] = arr_base[:, 4]
            normalized_base[:, 5] = arr_base[:, 5]

            # Compute interaction features for all samples
            interaction_features = np.array([
                self._compute_interactions(normalized_base[i])
                for i in range(normalized_base.shape[0])
            ], dtype=np.float64)

            interaction_means = interaction_features.mean(axis=0).astype(np.float32)
            interaction_stds = interaction_features.std(axis=0).astype(np.float32)
            interaction_stds = np.where(interaction_stds < 1e-8, 1.0, interaction_stds)

            self._means = np.concatenate([base_means, interaction_means])
            self._stds = np.concatenate([base_stds, interaction_stds])

        self._fitted = True
        self._save_stats()

    def transform(
        self,
        state: MarketState,
        inventory: float,
        time_step: int,
        total_steps: int,
        expected_volume_30s: Optional[float] = None,
        prev_mid_price: Optional[float] = None,
    ) -> np.ndarray:
        """
        Returns normalized context vector (12-dim or 18-dim float32).
        Raises RuntimeError if called before fit().
        """
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.fit() must be called before transform()")

        raw = self._extract_raw_base(
            state, inventory, time_step, total_steps,
            expected_volume_30s=expected_volume_30s,
            prev_mid_price=prev_mid_price,
        )

        # Z-score normalize the 12 base features
        base_means = self._means[:self.N_BASE]
        base_stds = self._stds[:self.N_BASE]
        normalized_base = (raw - base_means) / base_stds

        # Features 4 (inventory_fraction) and 5 (time_fraction) stay in [0, 1]
        normalized_base[4] = raw[4]
        normalized_base[5] = raw[5]

        if not self.use_interactions:
            result = normalized_base.astype(np.float32)
        else:
            # Compute interactions from normalized base features
            raw_interactions = self._compute_interactions(normalized_base)

            # Z-score normalize the interaction features
            interaction_means = self._means[self.N_BASE:]
            interaction_stds = self._stds[self.N_BASE:]
            normalized_interactions = (raw_interactions - interaction_means) / interaction_stds

            result = np.concatenate([normalized_base, normalized_interactions]).astype(np.float32)

        # Safety: replace NaN/Inf with 0
        bad_mask = ~np.isfinite(result)
        if np.any(bad_mask):
            result[bad_mask] = 0.0

        # Assertions
        assert len(result) == self.d_features, f"Feature length {len(result)} != {self.d_features}"
        assert 0.0 <= result[4] <= 1.0, f"Inventory fraction {result[4]} out of [0,1]"
        assert 0.0 <= result[5] <= 1.0, f"Time fraction {result[5]} out of [0,1]"

        return result

    def fit_transform(self, states, inventories=None, time_steps=None, total_steps=60) -> np.ndarray:
        """Calls fit then transforms all states."""
        self.fit(states, inventories, time_steps, total_steps)
        results = []
        for i, state in enumerate(states):
            inv = inventories[i] if inventories and i < len(inventories) else 0.5
            t = time_steps[i] if time_steps and i < len(time_steps) else i
            prev_mid = states[i - 1].mid_price if i > 0 else None
            results.append(self.transform(state, inv, t, total_steps, prev_mid_price=prev_mid))
        return np.array(results)
