"""Unit tests for the feature engineer."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.data_classes import MarketState
from features.feature_engineer import FeatureEngineer


def make_market_state(mid=50000.0, spread=10.0, vol_30s=10.0, vol_60s=0.0001,
                      vol_300s=0.0002, ofi=0.1, bid_qty=5.0, ask_qty=4.0,
                      minute=600, timestamp=1_700_000_000_000):
    return MarketState(
        timestamp_ms=timestamp,
        mid_price=mid,
        bid_price=mid - spread/2,
        ask_price=mid + spread/2,
        spread=spread,
        recent_volume_30s=vol_30s,
        recent_volatility_60s=vol_60s,
        recent_volatility_300s=vol_300s,
        ofi_10s=ofi,
        bid_qty=bid_qty,
        ask_qty=ask_qty,
        minute_of_day=minute,
    )


def make_random_states(n=100, seed=42):
    rng = np.random.RandomState(seed)
    states = []
    for i in range(n):
        mid = rng.uniform(40000, 60000)
        spread = rng.uniform(1, 20)
        states.append(make_market_state(
            mid=mid, spread=spread,
            vol_30s=rng.uniform(1, 100),
            vol_60s=rng.uniform(1e-5, 1e-3),
            vol_300s=rng.uniform(1e-5, 1e-3),
            ofi=rng.uniform(-1, 1),
            bid_qty=rng.uniform(0.1, 10),
            ask_qty=rng.uniform(0.1, 10),
            minute=int(rng.uniform(0, 1440)),
        ))
    return states


class TestFeatureEngineer:
    def test_output_length(self):
        fe = FeatureEngineer(d_features=12, feature_stats_path=Path("/tmp/test_feat_stats.json"))
        states = make_random_states(50)
        fe.fit(states)
        state = make_market_state()
        vec = fe.transform(state, inventory=0.5, time_step=30, total_steps=60)
        assert len(vec) == 12

    def test_no_nan_inf(self):
        fe = FeatureEngineer(d_features=12, feature_stats_path=Path("/tmp/test_feat_stats.json"))
        states = make_random_states(100)
        fe.fit(states)
        for i, state in enumerate(states):
            vec = fe.transform(state, inventory=0.5, time_step=i % 60, total_steps=60)
            assert np.all(np.isfinite(vec)), f"NaN/Inf in features at state {i}"

    def test_inventory_time_fraction_bounds(self):
        fe = FeatureEngineer(d_features=12, feature_stats_path=Path("/tmp/test_feat_stats.json"))
        states = make_random_states(50)
        fe.fit(states)
        for inv in [0.0, 0.5, 1.0]:
            for t in [0, 30, 59]:
                state = make_market_state()
                vec = fe.transform(state, inventory=inv, time_step=t, total_steps=60)
                assert 0.0 <= vec[4] <= 1.0, f"Inventory fraction {vec[4]} out of [0,1]"
                assert 0.0 <= vec[5] <= 1.0, f"Time fraction {vec[5]} out of [0,1]"

    def test_raises_before_fit(self):
        fe = FeatureEngineer(d_features=12, feature_stats_path=Path("/tmp/nonexistent_stats_xyz.json"))
        state = make_market_state()
        with pytest.raises(RuntimeError):
            fe.transform(state, inventory=0.5, time_step=0, total_steps=60)

    def test_fit_transform_shape(self):
        fe = FeatureEngineer(d_features=12, feature_stats_path=Path("/tmp/test_feat_stats.json"))
        states = make_random_states(50)
        result = fe.fit_transform(states, total_steps=60)
        assert result.shape == (50, 12)

    def test_output_length_with_interactions(self, tmp_path):
        stats_path = tmp_path / "feature_stats_18.json"
        fe = FeatureEngineer(d_features=18, feature_stats_path=stats_path)
        states = make_random_states(50)
        fe.fit(states)
        vec = fe.transform(make_market_state(), inventory=0.5, time_step=30, total_steps=60)
        assert len(vec) == 18
        assert np.all(np.isfinite(vec))

    def test_interaction_features_change_with_inputs(self, tmp_path):
        stats_path = tmp_path / "feature_stats_interactions.json"
        fe = FeatureEngineer(d_features=18, feature_stats_path=stats_path)
        states = make_random_states(80)
        fe.fit(states)

        low_signal = fe.transform(
            make_market_state(ofi=0.1, bid_qty=5.0, ask_qty=5.0),
            inventory=0.2,
            time_step=10,
            total_steps=60,
            prev_mid_price=50000.0,
        )
        high_signal = fe.transform(
            make_market_state(ofi=0.8, bid_qty=9.0, ask_qty=1.5),
            inventory=0.9,
            time_step=50,
            total_steps=60,
            prev_mid_price=49500.0,
        )

        assert not np.allclose(low_signal[12:], high_signal[12:]), (
            "Interaction features should respond to meaningful input changes"
        )
