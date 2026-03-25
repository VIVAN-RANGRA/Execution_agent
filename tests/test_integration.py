"""End-to-end integration test (uses synthetic data, no real Binance data required)."""
import pytest
import numpy as np
import pandas as pd
import json
import sys
import tempfile
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tests.helpers import DEFAULT_D_FEATURES, make_feature_stats


def make_synthetic_trades(n=10000, start_ms=1704067200000, price=50000.0, seed=42):
    """Generate synthetic aggTrades parquet data."""
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
    """Generate synthetic bookTicker parquet data."""
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


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory):
    """
    Create a fully synthetic environment for integration testing.
    Uses 3 days of synthetic data.
    """
    tmp = tmp_path_factory.mktemp("integration")

    # Directory structure
    parsed_dir = tmp / "parsed"
    (parsed_dir / "aggTrades").mkdir(parents=True)
    (parsed_dir / "bookTicker").mkdir(parents=True)
    config_dir = tmp / "config"
    config_dir.mkdir()
    data_dir = tmp
    results_dir = tmp / "results"
    results_dir.mkdir()

    # 3 days of data (2024-01-01, 2024-01-02, 2024-01-03)
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    base_ms = 1704067200000  # 2024-01-01 00:00:00 UTC

    for i, date in enumerate(dates):
        day_start = base_ms + i * 86400 * 1000
        # Each day: 50000 trades spanning 24h
        trades = make_synthetic_trades(n=50000, start_ms=day_start, price=50000.0 + i * 100, seed=i)
        trades.to_parquet(parsed_dir / "aggTrades" / f"{date}.parquet", index=False)
        book = make_synthetic_book(n=20000, start_ms=day_start, price=50000.0 + i * 100, seed=100+i)
        book.to_parquet(parsed_dir / "bookTicker" / f"{date}.parquet", index=False)

    # Volume profile: uniform
    vol_profile = pd.DataFrame({
        "minute_of_day": list(range(1440)),
        "fraction": [1.0 / 1440] * 1440,
    })
    vol_profile.to_parquet(data_dir / "volume_profile.parquet", index=False)

    # Calibration params
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

    # Feature stats: zeros and ones (identity normalization)
    feat_stats = make_feature_stats()
    with open(config_dir / "feature_stats.json", "w") as f:
        json.dump(feat_stats, f)

    # Config YAML
    import yaml
    config = {
        "data": {"symbol": "BTCUSDT", "start_date": "2024-01-01", "end_date": "2024-01-03",
                 "train_split": 0.70, "val_split": 0.15, "test_split": 0.15},
        "execution": {"total_quantity_btc": 1.0, "time_horizon_seconds": 600,
                      "num_slices": 10, "min_slice_fraction": 0.005, "max_slice_fraction": 0.30},
        "impact_model": {"alpha": 0.6, "eta_multiplier": 0.1, "gamma_multiplier": 0.01},
        "agents": {
            "linucb": {"alpha_exploration": 1.0, "d_features": DEFAULT_D_FEATURES,
                       "warm_start_from_ac": False, "warm_start_episodes": 0},
            "thompson": {"prior_variance": 1.0, "warm_start_from_ac": False, "warm_start_episodes": 0},
        },
        "features": {"volatility_window_short_s": 60, "volatility_window_long_s": 300,
                     "volume_window_s": 30, "ofi_window_s": 10, "normalization": "zscore"},
        "evaluation": {"n_episodes": 10, "metrics": ["IS"], "random_seed": 42},
    }
    with open(config_dir / "default_config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    return {
        "tmp": tmp,
        "config_path": str(config_dir / "default_config.yaml"),
        "parsed_data_dir": str(parsed_dir),
        "volume_profile_path": str(data_dir / "volume_profile.parquet"),
        "calibration_params_path": str(config_dir / "calibration_params.json"),
        "feature_stats_path": str(config_dir / "feature_stats.json"),
        "results_dir": str(results_dir),
        "dates": dates,
    }


class TestIntegration:
    def test_env_reset(self, synthetic_env):
        """reset() returns context of the configured shape with no NaN."""
        from simulator.execution_env import ExecutionEnv
        env = ExecutionEnv(
            config_path=synthetic_env["config_path"],
            parsed_data_dir=synthetic_env["parsed_data_dir"],
            volume_profile_path=synthetic_env["volume_profile_path"],
            calibration_params_path=synthetic_env["calibration_params_path"],
            feature_stats_path=synthetic_env["feature_stats_path"],
            split="train",
        )
        context, info = env.reset(episode_seed=42)
        assert context.shape == (DEFAULT_D_FEATURES,), (
            f"Context shape {context.shape} != ({DEFAULT_D_FEATURES},)"
        )
        assert np.all(np.isfinite(context)), "Context has NaN/Inf"
        assert "arrival_price" in info
        assert info["arrival_price"] > 0

    def test_config_mutation_propagates_to_env(self, synthetic_env):
        """A changed config file should be reflected in the constructed environment."""
        from simulator.execution_env import ExecutionEnv
        import yaml

        config_path = Path(synthetic_env["config_path"])
        with open(config_path) as f:
            config = yaml.safe_load(f)

        config["execution"]["total_quantity_btc"] = 2.5
        config["execution"]["num_slices"] = 5
        mutated_config_path = config_path.parent / "mutated_config.yaml"
        with open(mutated_config_path, "w") as f:
            yaml.safe_dump(config, f)

        env = ExecutionEnv(
            config_path=str(mutated_config_path),
            parsed_data_dir=synthetic_env["parsed_data_dir"],
            volume_profile_path=synthetic_env["volume_profile_path"],
            calibration_params_path=synthetic_env["calibration_params_path"],
            feature_stats_path=synthetic_env["feature_stats_path"],
            split="train",
        )

        context, _ = env.reset(episode_seed=17)
        assert env.total_quantity == 2.5
        assert env.num_slices == 5
        assert env.current_config.total_quantity == 2.5
        assert context.shape == (DEFAULT_D_FEATURES,)

    def test_step_returns_correct_types(self, synthetic_env):
        """step() returns (context, float, bool, dict) with correct types."""
        from simulator.execution_env import ExecutionEnv
        env = ExecutionEnv(
            config_path=synthetic_env["config_path"],
            parsed_data_dir=synthetic_env["parsed_data_dir"],
            volume_profile_path=synthetic_env["volume_profile_path"],
            calibration_params_path=synthetic_env["calibration_params_path"],
            feature_stats_path=synthetic_env["feature_stats_path"],
            split="train",
        )
        env.reset(episode_seed=1)
        ctx, reward, done, info = env.step(2)
        assert isinstance(ctx, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_inventory_never_negative(self, synthetic_env):
        """Inventory never goes negative across 5 random episodes."""
        from simulator.execution_env import ExecutionEnv
        env = ExecutionEnv(
            config_path=synthetic_env["config_path"],
            parsed_data_dir=synthetic_env["parsed_data_dir"],
            volume_profile_path=synthetic_env["volume_profile_path"],
            calibration_params_path=synthetic_env["calibration_params_path"],
            feature_stats_path=synthetic_env["feature_stats_path"],
            split="train",
        )
        for seed in range(5):
            env.reset(episode_seed=seed)
            while True:
                action = np.random.randint(0, 5)
                _, _, done, _ = env.step(action)
                assert env.inventory >= -1e-6, f"Negative inventory: {env.inventory}"
                if done:
                    break

    def test_final_inventory_zero(self, synthetic_env):
        """Final inventory is ~0.0 after episode completion."""
        from simulator.execution_env import ExecutionEnv
        env = ExecutionEnv(
            config_path=synthetic_env["config_path"],
            parsed_data_dir=synthetic_env["parsed_data_dir"],
            volume_profile_path=synthetic_env["volume_profile_path"],
            calibration_params_path=synthetic_env["calibration_params_path"],
            feature_stats_path=synthetic_env["feature_stats_path"],
            split="train",
        )
        env.reset(episode_seed=7)
        while True:
            _, _, done, _ = env.step(2)
            if done:
                break
        assert env.inventory <= 1e-6, f"Final inventory {env.inventory} > 1e-6"

    def test_all_rewards_finite(self, synthetic_env):
        """All rewards returned by step() are finite."""
        from simulator.execution_env import ExecutionEnv
        env = ExecutionEnv(
            config_path=synthetic_env["config_path"],
            parsed_data_dir=synthetic_env["parsed_data_dir"],
            volume_profile_path=synthetic_env["volume_profile_path"],
            calibration_params_path=synthetic_env["calibration_params_path"],
            feature_stats_path=synthetic_env["feature_stats_path"],
            split="train",
        )
        env.reset(episode_seed=3)
        while True:
            _, reward, done, _ = env.step(2)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if done:
                break

    def test_batch_evaluation_writes_json(self, synthetic_env):
        """Batch evaluator writes results JSON with finite metrics."""
        from simulator.execution_env import ExecutionEnv
        from agents.twap_agent import TWAPAgent
        from agents.ac_agent import ACAgent
        from evaluation.batch_evaluator import run_batch_evaluation

        env = ExecutionEnv(
            config_path=synthetic_env["config_path"],
            parsed_data_dir=synthetic_env["parsed_data_dir"],
            volume_profile_path=synthetic_env["volume_profile_path"],
            calibration_params_path=synthetic_env["calibration_params_path"],
            feature_stats_path=synthetic_env["feature_stats_path"],
            split="train",
        )

        agents = [TWAPAgent(), ACAgent(
            config_path=synthetic_env["config_path"],
            calibration_path=synthetic_env["calibration_params_path"],
        )]

        result = run_batch_evaluation(
            agents, env, n_episodes=5, random_seed=42,
            results_dir=synthetic_env["results_dir"],
        )

        # Results JSON written
        import glob
        files = glob.glob(str(Path(synthetic_env["results_dir"]) / "*.json"))
        assert len(files) > 0, "No results JSON written"

        # All metrics finite
        summary = result["summary"]
        for agent_name, s in summary.items():
            assert np.isfinite(s["mean_IS_bps"]), f"{agent_name} mean_IS_bps is not finite"
            assert np.isfinite(s["std_IS_bps"]), f"{agent_name} std_IS_bps is not finite"

    def test_metrics_not_catastrophic(self, synthetic_env):
        """Sanity check: IS values are in a reasonable range."""
        from simulator.execution_env import ExecutionEnv
        from agents.twap_agent import TWAPAgent
        from evaluation.batch_evaluator import run_batch_evaluation

        env = ExecutionEnv(
            config_path=synthetic_env["config_path"],
            parsed_data_dir=synthetic_env["parsed_data_dir"],
            volume_profile_path=synthetic_env["volume_profile_path"],
            calibration_params_path=synthetic_env["calibration_params_path"],
            feature_stats_path=synthetic_env["feature_stats_path"],
            split="train",
        )

        result = run_batch_evaluation(
            [TWAPAgent()], env, n_episodes=5, random_seed=42,
            results_dir=synthetic_env["results_dir"],
        )

        twap_is = result["summary"]["TWAP"]["mean_IS_bps"]
        # IS should be within ±1000 bps (not catastrophically wrong)
        assert abs(twap_is) < 1000, f"TWAP IS seems catastrophically wrong: {twap_is} bps"
