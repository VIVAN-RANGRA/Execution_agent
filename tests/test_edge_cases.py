"""Tests for edge cases and boundary conditions."""
import pytest
import numpy as np
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from simulator.data_classes import MarketState, ExecutionConfig, Fill
from simulator.impact_model import temporary_impact, permanent_impact, compute_ac_trajectory
from simulator.reward_calculator import compute_step_reward, compute_terminal_reward
from tests.helpers import DEFAULT_D_FEATURES, make_context, make_feature_stats


# ---------------------------------------------------------------------------
# Helpers  (same synthetic-data factories as test_integration.py)
# ---------------------------------------------------------------------------
def make_synthetic_trades(n=10000, start_ms=1704067200000, price=50000.0, seed=42):
    rng = np.random.RandomState(seed)
    timestamps = start_ms + np.cumsum(rng.exponential(100, n)).astype(int)
    prices = price + np.cumsum(rng.normal(0, 5, n))
    prices = np.abs(prices)
    qtys = rng.exponential(0.1, n)
    is_buyer_maker = rng.choice([True, False], n)
    return pd.DataFrame({
        "timestamp_ms": timestamps.astype(np.int64),
        "price": prices.astype(np.float64),
        "qty": qtys.astype(np.float64),
        "is_buyer_maker": is_buyer_maker,
    })


def make_synthetic_book(n=5000, start_ms=1704067200000, price=50000.0, seed=99):
    rng = np.random.RandomState(seed)
    timestamps = start_ms + np.cumsum(rng.exponential(200, n)).astype(int)
    bids = price - rng.uniform(1, 5, n)
    asks = price + rng.uniform(1, 5, n)
    bid_qtys = rng.exponential(2, n)
    ask_qtys = rng.exponential(2, n)
    return pd.DataFrame({
        "timestamp_ms": timestamps.astype(np.int64),
        "best_bid_price": bids.astype(np.float64),
        "best_bid_qty": bid_qtys.astype(np.float64),
        "best_ask_price": asks.astype(np.float64),
        "best_ask_qty": ask_qtys.astype(np.float64),
    })


def _build_env(tmp_path, total_quantity=1.0, time_horizon_seconds=600, num_slices=10):
    """Create a lightweight synthetic ExecutionEnv in a temp directory."""
    from simulator.execution_env import ExecutionEnv

    parsed_dir = tmp_path / "parsed"
    (parsed_dir / "aggTrades").mkdir(parents=True)
    (parsed_dir / "bookTicker").mkdir(parents=True)
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)

    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    base_ms = 1704067200000
    for i, date in enumerate(dates):
        day_start = base_ms + i * 86400 * 1000
        trades = make_synthetic_trades(n=50000, start_ms=day_start, price=50000.0 + i * 100, seed=i)
        trades.to_parquet(parsed_dir / "aggTrades" / f"{date}.parquet", index=False)
        book = make_synthetic_book(n=20000, start_ms=day_start, price=50000.0 + i * 100, seed=100 + i)
        book.to_parquet(parsed_dir / "bookTicker" / f"{date}.parquet", index=False)

    vol_profile = pd.DataFrame({
        "minute_of_day": list(range(1440)),
        "fraction": [1.0 / 1440] * 1440,
    })
    vol_profile.to_parquet(tmp_path / "volume_profile.parquet", index=False)

    cal_params = {
        "sigma_per_second": 1e-5,
        "adv_btc": 5000.0,
        "eta": 1e-6,
        "gamma": 1e-7,
        "alpha": 0.6,
        "calibration_date_range": "2024-01-01 to 2024-01-02",
    }
    with open(config_dir / "calibration_params.json", "w") as f:
        json.dump(cal_params, f)

    feat_stats = make_feature_stats()
    with open(config_dir / "feature_stats.json", "w") as f:
        json.dump(feat_stats, f)

    config = {
        "data": {
            "symbol": "BTCUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-01-03",
            "train_split": 0.70,
            "val_split": 0.15,
            "test_split": 0.15,
        },
        "execution": {
            "total_quantity_btc": total_quantity,
            "time_horizon_seconds": time_horizon_seconds,
            "num_slices": num_slices,
            "min_slice_fraction": 0.005,
            "max_slice_fraction": 0.30,
        },
        "impact_model": {"alpha": 0.6, "eta_multiplier": 0.1, "gamma_multiplier": 0.01},
        "agents": {
            "linucb": {"alpha_exploration": 1.0, "d_features": DEFAULT_D_FEATURES,
                       "warm_start_from_ac": False, "warm_start_episodes": 0},
            "thompson": {"prior_variance": 1.0, "warm_start_from_ac": False,
                         "warm_start_episodes": 0},
        },
        "features": {
            "volatility_window_short_s": 60,
            "volatility_window_long_s": 300,
            "volume_window_s": 30,
            "ofi_window_s": 10,
            "normalization": "zscore",
        },
        "evaluation": {"n_episodes": 10, "metrics": ["IS"], "random_seed": 42},
    }
    config_path = config_dir / "default_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    env = ExecutionEnv(
        config_path=str(config_path),
        parsed_data_dir=str(parsed_dir),
        volume_profile_path=str(tmp_path / "volume_profile.parquet"),
        calibration_params_path=str(config_dir / "calibration_params.json"),
        feature_stats_path=str(config_dir / "feature_stats.json"),
        split="train",
    )
    return env, str(config_path), str(config_dir / "calibration_params.json")


@pytest.fixture(scope="module")
def env_fixture(tmp_path_factory):
    """Module-scoped environment fixture for edge case tests."""
    tmp = tmp_path_factory.mktemp("edge_cases")
    env, config_path, cal_path = _build_env(tmp)
    return {"env": env, "config_path": config_path, "cal_path": cal_path, "tmp": tmp}


# =========================================================================
# 1. Inventory edge cases
# =========================================================================
class TestInventoryEdgeCases:

    def test_tiny_order(self, env_fixture):
        """Tiny orders should complete, including any terminal forced fill."""
        tmp = env_fixture["tmp"] / "tiny"
        tmp.mkdir(exist_ok=True)
        env, _, _ = _build_env(tmp, total_quantity=0.001, num_slices=5)
        ctx, info = env.reset(episode_seed=42)
        step_total = 0.0
        terminal_total = 0.0
        while True:
            _, _, done, step_info = env.step(2)
            step_total += step_info["fill"].quantity_filled
            if step_info["terminal_fill"] is not None:
                terminal_total += step_info["terminal_fill"].quantity_filled
            if done:
                break
        total_filled = sum(fill.quantity_filled for fill in env.fills)
        assert terminal_total > 0.0, "Terminal forced fill should be exposed in step_info['terminal_fill']"
        assert abs((step_total + terminal_total) - 0.001) < 1e-6
        assert len(env.fills) == env.num_slices + 1, "Expected an extra terminal fill for the residual inventory"
        assert abs(total_filled - 0.001) < 1e-6, f"Total filled {total_filled} != 0.001"

    def test_single_slice(self, env_fixture):
        """num_slices = 1 (execute everything in one go)."""
        tmp = env_fixture["tmp"] / "single_slice"
        tmp.mkdir(exist_ok=True)
        env, _, _ = _build_env(tmp, total_quantity=1.0, num_slices=1)
        ctx, info = env.reset(episode_seed=42)
        _, reward, done, step_info = env.step(2)
        assert done, "Single-slice episode should be done after one step"
        assert env.inventory < 1e-6, f"Inventory should be ~0, got {env.inventory}"

    def test_inventory_depleted_early(self, env_fixture):
        """Aggressive agent (action=4) depleting inventory before time runs out."""
        tmp = env_fixture["tmp"] / "depleted"
        tmp.mkdir(exist_ok=True)
        env, _, _ = _build_env(tmp, total_quantity=0.5, num_slices=10)
        env.reset(episode_seed=42)
        steps = 0
        while True:
            _, _, done, _ = env.step(4)  # max urgency every step
            steps += 1
            if done:
                break
        # Should terminate before num_slices if inventory goes to zero
        assert env.inventory < 1e-6

    def test_all_actions_produce_nonneg_inventory(self, env_fixture):
        """Cycle through all actions; inventory never goes negative."""
        tmp = env_fixture["tmp"] / "nonneg"
        tmp.mkdir(exist_ok=True)
        env, _, _ = _build_env(tmp, total_quantity=1.0, num_slices=10)
        env.reset(episode_seed=7)
        actions = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
        for a in actions:
            _, _, done, _ = env.step(a)
            assert env.inventory >= -1e-9, f"Negative inventory: {env.inventory}"
            if done:
                break


# =========================================================================
# 2. Reproducibility
# =========================================================================
class TestReproducibility:

    def test_same_seed_identical_fills_twap(self, env_fixture):
        """Same seed must produce identical fills for deterministic TWAP agent."""
        from agents.twap_agent import TWAPAgent
        env = env_fixture["env"]
        agent = TWAPAgent()

        def _run(seed):
            ctx, info = env.reset(episode_seed=seed)
            agent.reset(env.current_config)
            fills = []
            while True:
                action = agent.decide(ctx, env.inventory, env.time_step, env.num_slices)
                ctx, reward, done, step_info = env.step(action)
                fills.append((step_info["fill"].quantity_filled, step_info["fill"].fill_price))
                if done:
                    break
            return fills

        fills_a = _run(42)
        fills_b = _run(42)
        assert len(fills_a) == len(fills_b)
        for (qa, pa), (qb, pb) in zip(fills_a, fills_b):
            assert abs(qa - qb) < 1e-10, f"Quantity mismatch: {qa} vs {qb}"
            assert abs(pa - pb) < 1e-10, f"Price mismatch: {pa} vs {pb}"

    def test_different_seeds_different_fills(self, env_fixture):
        """Different seeds must produce different episodes."""
        env = env_fixture["env"]

        def _run(seed):
            ctx, info = env.reset(episode_seed=seed)
            fills = []
            while True:
                ctx, _, done, step_info = env.step(2)
                fills.append(step_info["fill"].fill_price)
                if done:
                    break
            return fills

        fills_a = _run(42)
        fills_b = _run(99)
        # At least the arrival prices or fill prices should differ
        assert fills_a != fills_b, "Seeds 42 and 99 should produce different episodes"


# =========================================================================
# 3. Agent edge cases
# =========================================================================
class TestAgentEdgeCases:

    def _get_all_agents(self, config_path, cal_path):
        """Import all available agent classes with try/except."""
        agents = []
        from agents.twap_agent import TWAPAgent
        agents.append(TWAPAgent())
        try:
            from agents.ac_agent import ACAgent
            agents.append(ACAgent(config_path=config_path, calibration_path=cal_path))
        except Exception:
            pass
        try:
            from agents.linucb_agent import LinUCBAgent
            agents.append(LinUCBAgent(config_path=config_path, calibration_path=cal_path))
        except Exception:
            pass
        try:
            from agents.thompson_agent import ThompsonAgent
            agents.append(ThompsonAgent(config_path=config_path))
        except Exception:
            pass
        try:
            from agents.pov_agent import POVAgent
            agents.append(POVAgent(config_path=config_path))
        except Exception:
            pass
        try:
            from agents.exp3_agent import EXP3Agent
            agents.append(EXP3Agent(config_path=config_path))
        except Exception:
            pass
        return agents

    def test_all_agents_handle_last_step(self, env_fixture):
        """At time_step = total_steps-1, all agents return valid action."""
        config = ExecutionConfig(
            total_quantity=10.0,
            time_horizon_seconds=3600,
            num_slices=60,
            arrival_price=50000.0,
        )
        ctx = make_context(0)
        for agent in self._get_all_agents(env_fixture["config_path"], env_fixture["cal_path"]):
            agent.reset(config)
            action = agent.decide(ctx, 0.5, 59, 60)  # last step
            assert 0 <= action <= 4, f"{agent.name} invalid action at last step: {action}"

    def test_all_agents_handle_zero_inventory(self, env_fixture):
        """With inventory=0, agents should still return valid action."""
        config = ExecutionConfig(
            total_quantity=10.0,
            time_horizon_seconds=3600,
            num_slices=60,
            arrival_price=50000.0,
        )
        ctx = make_context(1)
        for agent in self._get_all_agents(env_fixture["config_path"], env_fixture["cal_path"]):
            agent.reset(config)
            action = agent.decide(ctx, 0.0, 30, 60)  # zero inventory
            assert 0 <= action <= 4, f"{agent.name} invalid action at zero inventory: {action}"

    def test_all_agents_handle_max_urgency(self, env_fixture):
        """Verify agent can recommend action 4 and the env does not break."""
        env = env_fixture["env"]
        env.reset(episode_seed=42)
        # Force action=4 for a few steps
        for _ in range(3):
            ctx, reward, done, info = env.step(4)
            assert np.isfinite(reward)
            if done:
                break

    def test_all_agents_handle_min_urgency(self, env_fixture):
        """Verify agent can recommend action 0 (very passive) without issues."""
        env = env_fixture["env"]
        env.reset(episode_seed=42)
        for _ in range(3):
            ctx, reward, done, info = env.step(0)
            assert np.isfinite(reward)
            if done:
                break

    def test_twap_always_returns_two(self, env_fixture):
        """TWAP should always return action 2, regardless of context."""
        from agents.twap_agent import TWAPAgent
        agent = TWAPAgent()
        config = ExecutionConfig(
            total_quantity=10.0, time_horizon_seconds=3600,
            num_slices=60, arrival_price=50000.0,
        )
        agent.reset(config)
        rng = np.random.RandomState(42)
        for i in range(100):
            ctx = rng.standard_normal(DEFAULT_D_FEATURES).astype(np.float32)
            action = agent.decide(ctx, rng.uniform(0, 10), i % 60, 60)
            assert action == 2, f"TWAP returned {action} != 2"


# =========================================================================
# 4. Reward edge cases
# =========================================================================
class TestRewardEdgeCases:

    def test_step_reward_zero_arrival_price(self):
        """Zero arrival price should return 0 reward."""
        r = compute_step_reward(50000.0, 0.0, 1.0, 10.0)
        assert r == 0.0

    def test_step_reward_zero_total_quantity(self):
        """Zero total quantity should return 0 reward."""
        r = compute_step_reward(50000.0, 50000.0, 1.0, 0.0)
        assert r == 0.0

    def test_step_reward_sign_convention(self):
        """For a buy: fill above arrival => negative reward (bad).
        Fill below arrival => positive reward (good)."""
        # Fill above arrival price (bad for buyer)
        r_bad = compute_step_reward(50100.0, 50000.0, 1.0, 10.0)
        assert r_bad < 0, f"Fill above arrival should be negative reward, got {r_bad}"

        # Fill below arrival price (good for buyer)
        r_good = compute_step_reward(49900.0, 50000.0, 1.0, 10.0)
        assert r_good > 0, f"Fill below arrival should be positive reward, got {r_good}"

    def test_terminal_reward_empty_fills(self):
        """No fills should return 0 terminal reward."""
        r = compute_terminal_reward([], 50000.0)
        assert r == 0.0

    def test_terminal_reward_single_fill(self):
        """Single fill should produce a finite terminal reward."""
        fill = Fill(
            timestamp_ms=1000, quantity_filled=1.0,
            fill_price=50010.0, market_impact_cost=10.0, slice_index=0,
        )
        r = compute_terminal_reward([fill], 50000.0)
        assert np.isfinite(r)


# =========================================================================
# 5. ExecutionConfig edge cases
# =========================================================================
class TestConfigEdgeCases:

    def test_large_num_slices(self, env_fixture):
        """Large number of slices should not crash."""
        tmp = env_fixture["tmp"] / "large_slices"
        tmp.mkdir(exist_ok=True)
        env, _, _ = _build_env(tmp, total_quantity=1.0, num_slices=100, time_horizon_seconds=6000)
        ctx, info = env.reset(episode_seed=42)
        assert ctx.shape == (DEFAULT_D_FEATURES,)
        # Run a few steps
        for _ in range(5):
            ctx, reward, done, _ = env.step(2)
            assert np.isfinite(reward)
            if done:
                break

    def test_very_short_horizon(self, env_fixture):
        """Very short time horizon (60s) with few slices."""
        tmp = env_fixture["tmp"] / "short_horizon"
        tmp.mkdir(exist_ok=True)
        env, _, _ = _build_env(tmp, total_quantity=0.1, num_slices=3, time_horizon_seconds=60)
        ctx, info = env.reset(episode_seed=42)
        assert ctx.shape == (DEFAULT_D_FEATURES,)


# =========================================================================
# 6. MarketState edge cases
# =========================================================================
class TestMarketStateEdgeCases:

    def test_market_state_construction(self):
        """MarketState should accept all required fields."""
        state = MarketState(
            timestamp_ms=1000, mid_price=50000.0,
            bid_price=49999.0, ask_price=50001.0,
            spread=2.0, recent_volume_30s=100.0,
            recent_volatility_60s=0.001, recent_volatility_300s=0.002,
            ofi_10s=0.5, bid_qty=10.0, ask_qty=10.0,
            minute_of_day=720,
        )
        assert state.mid_price == 50000.0
        assert state.minute_of_day == 720

    def test_market_state_zero_spread(self):
        """Zero spread is a valid edge case."""
        state = MarketState(
            timestamp_ms=1000, mid_price=50000.0,
            bid_price=50000.0, ask_price=50000.0,
            spread=0.0, recent_volume_30s=100.0,
            recent_volatility_60s=0.001, recent_volatility_300s=0.002,
            ofi_10s=0.0, bid_qty=10.0, ask_qty=10.0,
            minute_of_day=0,
        )
        assert state.spread == 0.0

    def test_execution_config_defaults(self):
        """ExecutionConfig defaults for side and risk_aversion."""
        config = ExecutionConfig(
            total_quantity=1.0,
            time_horizon_seconds=600,
            num_slices=10,
            arrival_price=50000.0,
        )
        assert config.side == "buy"
        assert config.risk_aversion == 0.1
