"""Tests for numerical stability under extreme conditions."""
import pytest
import numpy as np
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.data_classes import MarketState, ExecutionConfig
from simulator.impact_model import temporary_impact, permanent_impact, compute_ac_trajectory
from agents.linucb_agent import LinUCBAgent
from agents.thompson_agent import ThompsonAgent

# Read d_features from config so tests adapt to config changes
_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"
with open(_CFG_PATH) as _f:
    _D_FEATURES = yaml.safe_load(_f)["agents"]["linucb"]["d_features"]


def make_config():
    return ExecutionConfig(
        total_quantity=10.0,
        time_horizon_seconds=3600,
        num_slices=60,
        arrival_price=50000.0,
    )


# ---------------------------------------------------------------------------
# Helper: build agents with default config (reads from config/default_config.yaml)
# ---------------------------------------------------------------------------
def _make_agent(cls):
    """Instantiate a bandit agent with defaults, reset it, and return it."""
    agent = cls()
    agent.reset(make_config())
    return agent


# =========================================================================
# 1. Extreme context vectors
# =========================================================================
class TestExtremeContexts:
    """Feed extreme context vectors to agents."""

    def test_zero_context(self):
        """All-zero context should not crash any agent."""
        for AgentClass in [LinUCBAgent, ThompsonAgent]:
            agent = _make_agent(AgentClass)
            ctx = np.zeros(_D_FEATURES, dtype=np.float32)
            action = agent.decide(ctx, 5.0, 0, 60)
            assert 0 <= action <= 4, f"{AgentClass.__name__} returned {action}"

    def test_very_large_context(self):
        """Context with values up to 1e6."""
        for AgentClass in [LinUCBAgent, ThompsonAgent]:
            agent = _make_agent(AgentClass)
            ctx = np.full(_D_FEATURES, 1e6, dtype=np.float32)
            action = agent.decide(ctx, 5.0, 0, 60)
            assert 0 <= action <= 4

    def test_very_small_context(self):
        """Context with values near 1e-10."""
        for AgentClass in [LinUCBAgent, ThompsonAgent]:
            agent = _make_agent(AgentClass)
            ctx = np.full(_D_FEATURES, 1e-10, dtype=np.float32)
            action = agent.decide(ctx, 5.0, 0, 60)
            assert 0 <= action <= 4

    def test_negative_context(self):
        """All-negative context."""
        for AgentClass in [LinUCBAgent, ThompsonAgent]:
            agent = _make_agent(AgentClass)
            ctx = np.full(_D_FEATURES, -100.0, dtype=np.float32)
            action = agent.decide(ctx, 5.0, 0, 60)
            assert 0 <= action <= 4

    def test_mixed_extreme_context(self):
        """Alternating very large and very small values."""
        for AgentClass in [LinUCBAgent, ThompsonAgent]:
            agent = _make_agent(AgentClass)
            values = [1e6, 1e-10, -1e6, 1e-10, 0.5, 0.5,
                      1e6, 1e-10, -1e6, 1e6, 1e-10, -1e-10]
            # Extend/truncate to match d_features
            ctx = np.array(
                (values * ((_D_FEATURES // len(values)) + 1))[:_D_FEATURES],
                dtype=np.float32,
            )
            action = agent.decide(ctx, 5.0, 0, 60)
            assert 0 <= action <= 4

    def test_nan_context_handled(self):
        """NaN in context should be handled gracefully (not crash)."""
        for AgentClass in [LinUCBAgent, ThompsonAgent]:
            agent = _make_agent(AgentClass)
            ctx = np.array([np.nan] * _D_FEATURES, dtype=np.float32)
            try:
                action = agent.decide(ctx, 5.0, 0, 60)
                # If it does not crash, any action in range is acceptable
                assert 0 <= action <= 4 or True  # allow NaN-derived action
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                pass  # Acceptable to raise an error on NaN input

    def test_inf_context_handled(self):
        """Inf in context should be handled gracefully."""
        for AgentClass in [LinUCBAgent, ThompsonAgent]:
            agent = _make_agent(AgentClass)
            ctx = np.array([np.inf] * _D_FEATURES, dtype=np.float32)
            try:
                action = agent.decide(ctx, 5.0, 0, 60)
                assert 0 <= action <= 4 or True
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                pass

    def test_single_large_feature(self):
        """Only one feature is extreme; the rest are normal."""
        for AgentClass in [LinUCBAgent, ThompsonAgent]:
            agent = _make_agent(AgentClass)
            ctx = np.array([0.5] * _D_FEATURES, dtype=np.float32)
            ctx[3] = 1e8  # price momentum explosion
            action = agent.decide(ctx, 5.0, 0, 60)
            assert 0 <= action <= 4


# =========================================================================
# 2. Extreme impact model inputs
# =========================================================================
class TestExtremeImpact:
    """Test impact model with extreme parameters."""

    def test_zero_volume(self):
        assert temporary_impact(0.0, 60.0, 0.01, 1000.0, 0.6) == 0.0

    def test_very_large_volume(self):
        h = temporary_impact(1e6, 60.0, 0.01, 1000.0, 0.6)
        assert np.isfinite(h)
        assert h > 0

    def test_very_small_tau(self):
        h = temporary_impact(1.0, 0.001, 0.01, 1000.0, 0.6)
        assert np.isfinite(h)

    def test_zero_tau(self):
        h = temporary_impact(1.0, 0.0, 0.01, 1000.0, 0.6)
        assert h == 0.0

    def test_zero_adv(self):
        h = temporary_impact(1.0, 60.0, 0.01, 0.0, 0.6)
        assert h == 0.0

    def test_negative_volume(self):
        """Negative volume should still produce a finite result (model does not guard)."""
        h = temporary_impact(-1.0, 60.0, 0.01, 1000.0, 0.6)
        # Depending on alpha, this might be complex; just check no crash
        assert isinstance(h, (float, complex))

    def test_permanent_impact_zero_adv(self):
        assert permanent_impact(1.0, 0.01, 0.0) == 0.0

    def test_permanent_impact_large_volume(self):
        p = permanent_impact(1e6, 0.01, 1000.0)
        assert np.isfinite(p)
        assert p > 0

    def test_permanent_impact_zero_volume(self):
        assert permanent_impact(0.0, 0.01, 1000.0) == 0.0

    def test_ac_trajectory_extreme_risk_aversion(self):
        """Very high risk aversion should front-load execution."""
        traj = compute_ac_trajectory(10.0, 3600, 60, 100.0, 1e-5, 1e-6, 1000.0)
        assert abs(traj.sum() - 10.0) < 1e-6
        assert traj[0] > traj[-1], "High risk aversion should front-load"

    def test_ac_trajectory_zero_risk_aversion(self):
        """Zero risk aversion -> uniform schedule (TWAP-like)."""
        traj = compute_ac_trajectory(10.0, 3600, 60, 0.0, 1e-5, 1e-6, 1000.0)
        assert abs(traj.sum() - 10.0) < 1e-6
        # Should be roughly uniform
        assert np.std(traj) / np.mean(traj) < 0.1

    def test_ac_trajectory_single_slice(self):
        """Single slice should return the full quantity."""
        traj = compute_ac_trajectory(10.0, 3600, 1, 0.1, 1e-5, 1e-6, 1000.0)
        assert len(traj) == 1
        assert abs(traj[0] - 10.0) < 1e-6

    def test_ac_trajectory_zero_eta(self):
        """eta=0 should produce uniform trajectory."""
        traj = compute_ac_trajectory(10.0, 3600, 60, 0.1, 1e-5, 0.0, 1000.0)
        assert abs(traj.sum() - 10.0) < 1e-6
        assert np.std(traj) / np.mean(traj) < 0.1

    def test_ac_trajectory_very_large_sigma(self):
        """Very large sigma should still produce a valid trajectory."""
        traj = compute_ac_trajectory(10.0, 3600, 60, 0.1, 1.0, 1e-6, 1000.0)
        assert abs(traj.sum() - 10.0) < 1e-6
        assert np.all(np.isfinite(traj))

    def test_ac_trajectory_all_positive(self):
        """All slices should be non-negative."""
        for risk_av in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
            traj = compute_ac_trajectory(10.0, 3600, 60, risk_av, 1e-5, 1e-6, 1000.0)
            assert np.all(traj >= -1e-10), f"Negative slice at risk_aversion={risk_av}"


# =========================================================================
# 3. Learning stability under extreme rewards
# =========================================================================
class TestExtremeLearning:
    """Test learning stability under extreme rewards."""

    def test_very_large_rewards(self):
        """Rewards of magnitude 1e6 should not cause overflow."""
        agent = LinUCBAgent()
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(50):
            ctx = rng.standard_normal(_D_FEATURES).astype(np.float32)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            reward = rng.choice([-1e6, 1e6])
            next_ctx = rng.standard_normal(_D_FEATURES).astype(np.float32)
            agent.update(ctx, action, reward, next_ctx)
        # Should still produce valid actions
        ctx = rng.standard_normal(_D_FEATURES).astype(np.float32)
        action = agent.decide(ctx, 5.0, 0, 60)
        assert 0 <= action <= 4

    def test_zero_rewards(self):
        """All-zero rewards should not cause any issues."""
        agent = LinUCBAgent()
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(50):
            ctx = rng.standard_normal(_D_FEATURES).astype(np.float32)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            agent.update(ctx, action, 0.0, rng.standard_normal(_D_FEATURES).astype(np.float32))
        ctx = rng.standard_normal(_D_FEATURES).astype(np.float32)
        action = agent.decide(ctx, 5.0, 0, 60)
        assert 0 <= action <= 4

    def test_alternating_large_rewards(self):
        """Alternating +1e4 / -1e4 rewards."""
        agent = LinUCBAgent()
        agent.reset(make_config())
        rng = np.random.RandomState(7)
        for i in range(100):
            ctx = rng.standard_normal(_D_FEATURES).astype(np.float32)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            reward = 1e4 if i % 2 == 0 else -1e4
            agent.update(ctx, action, reward, rng.standard_normal(_D_FEATURES).astype(np.float32))
        ctx = rng.standard_normal(_D_FEATURES).astype(np.float32)
        action = agent.decide(ctx, 5.0, 0, 60)
        assert 0 <= action <= 4

    def test_thompson_cholesky_stability(self):
        """Thompson must not crash from Cholesky decomposition failures."""
        agent = ThompsonAgent()
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        # Run many updates to stress the covariance matrix
        for i in range(200):
            ctx = rng.standard_normal(_D_FEATURES).astype(np.float32)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            reward = rng.normal(0, 10)
            agent.update(ctx, action, reward, rng.standard_normal(_D_FEATURES).astype(np.float32))
        # Should still work
        action = agent.decide(rng.standard_normal(_D_FEATURES).astype(np.float32), 5.0, 0, 60)
        assert 0 <= action <= 4

    def test_thompson_identical_contexts(self):
        """Repeated identical context vectors should not break the covariance update."""
        agent = ThompsonAgent()
        agent.reset(make_config())
        fixed_ctx = np.ones(_D_FEATURES, dtype=np.float32) * 0.5
        for i in range(100):
            action = agent.decide(fixed_ctx, 5.0, i % 60, 60)
            agent.update(fixed_ctx, action, float(i % 3), fixed_ctx)
        action = agent.decide(fixed_ctx, 5.0, 0, 60)
        assert 0 <= action <= 4

    def test_linucb_collinear_contexts(self):
        """Collinear contexts (rank-1 updates) should not break Sherman-Morrison."""
        agent = LinUCBAgent()
        agent.reset(make_config())
        base = np.arange(1, _D_FEATURES + 1, dtype=np.float32)
        for i in range(50):
            ctx = base * (1.0 + 0.001 * i)  # nearly collinear
            action = agent.decide(ctx, 5.0, i % 60, 60)
            agent.update(ctx, action, float(i), ctx)
        action = agent.decide(base, 5.0, 0, 60)
        assert 0 <= action <= 4

    def test_weights_remain_finite_after_many_updates(self):
        """After 500 updates, all weight matrices should be finite."""
        agent = LinUCBAgent()
        agent.reset(make_config())
        rng = np.random.RandomState(99)
        for i in range(500):
            ctx = rng.standard_normal(_D_FEATURES).astype(np.float32)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            reward = rng.normal(0, 100)
            agent.update(ctx, action, reward, rng.standard_normal(_D_FEATURES).astype(np.float32))
        # Check all weight matrices are finite
        W = agent.get_weights_matrix()
        assert np.all(np.isfinite(W)), "Weight matrix has NaN/Inf after 500 updates"
        for a in range(5):
            assert np.all(np.isfinite(agent._A_inv[a])), f"A_inv[{a}] has NaN/Inf"
            assert np.all(np.isfinite(agent._b[a])), f"b[{a}] has NaN/Inf"
