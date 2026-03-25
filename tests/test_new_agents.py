"""Tests for new agent implementations (POV, EXP3, RegimeSwitchAC, KernelUCB,
MetaAgent, ThompsonACHybrid).

Each test uses synthetic data -- no real Binance data required.
Agents are imported with try/except so the test file does not crash if an
agent module has not been created yet; individual tests are skipped instead.
"""
import pytest
import numpy as np
import json
import sys
import yaml
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from simulator.data_classes import ExecutionConfig
from agents.base_agent import BaseAgent
from tests.helpers import DEFAULT_D_FEATURES, constant_context

# ---------------------------------------------------------------------------
# Conditional imports -- skip individual tests if module is missing
# ---------------------------------------------------------------------------
try:
    from agents.pov_agent import POVAgent
    _HAS_POV = True
except ImportError:
    _HAS_POV = False

try:
    from agents.exp3_agent import EXP3Agent
    _HAS_EXP3 = True
except ImportError:
    _HAS_EXP3 = False

try:
    from agents.regime_switching_ac_agent import RegimeSwitchACAgent
    _HAS_REGIME = True
except ImportError:
    _HAS_REGIME = False

try:
    from agents.kernel_ucb_agent import KernelUCBAgent
    _HAS_KERNEL = True
except ImportError:
    _HAS_KERNEL = False

try:
    from agents.meta_agent import MetaAgent
    _HAS_META = True
except ImportError:
    _HAS_META = False

try:
    from agents.thompson_ac_hybrid_agent import ThompsonACHybridAgent
    _HAS_HYBRID = True
except ImportError:
    _HAS_HYBRID = False

try:
    from agents.linucb_agent import LinUCBAgent
    from agents.thompson_agent import ThompsonAgent
    _HAS_BANDITS = True
except ImportError:
    _HAS_BANDITS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CFG_PATH = BASE_DIR / "config" / "default_config.yaml"
_CAL_PATH = BASE_DIR / "config" / "calibration_params.json"


def _config_path():
    return str(_CFG_PATH) if _CFG_PATH.exists() else None


def _cal_path():
    return str(_CAL_PATH) if _CAL_PATH.exists() else None


def make_config(total_quantity=10.0):
    return ExecutionConfig(
        total_quantity=total_quantity,
        time_horizon_seconds=3600,
        num_slices=60,
        arrival_price=50000.0,
    )


def random_context(rng=None, d=None):
    rng = rng or np.random.RandomState(42)
    return rng.standard_normal(d or DEFAULT_D_FEATURES).astype(np.float32)


# =========================================================================
# 1. POV Agent
# =========================================================================
@pytest.mark.skipif(not _HAS_POV, reason="POVAgent not available")
class TestPOVAgent:

    def test_implements_base_agent(self):
        agent = POVAgent(config_path=_config_path())
        assert isinstance(agent, BaseAgent)

    def test_name_property(self):
        agent = POVAgent(config_path=_config_path())
        assert agent.name == "POV"

    def test_returns_valid_actions(self):
        agent = POVAgent(config_path=_config_path())
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(50):
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            assert 0 <= action <= 4, f"POV returned invalid action {action}"

    def test_deterministic_same_input(self):
        """POV is non-learning -> same input should yield same output."""
        agent = POVAgent(config_path=_config_path())
        agent.reset(make_config())
        ctx = constant_context()
        a1 = agent.decide(ctx, 5.0, 10, 60)
        agent.reset(make_config())
        a2 = agent.decide(ctx, 5.0, 10, 60)
        assert a1 == a2, "POV should be deterministic for same input"

    def test_inventory_urgency_adjustment(self):
        """High remaining inventory at the same late step should increase urgency."""
        agent = POVAgent(config_path=_config_path())
        agent.reset(make_config())
        ctx = constant_context()
        action_high_inventory = agent.decide(ctx, 9.0, 55, 60)
        agent.reset(make_config())
        action_low_inventory = agent.decide(ctx, 0.5, 55, 60)
        assert action_high_inventory >= action_low_inventory, (
            "POV should be at least as urgent when more inventory remains late in the schedule"
        )

    def test_zero_inventory(self):
        agent = POVAgent(config_path=_config_path())
        agent.reset(make_config())
        ctx = random_context()
        action = agent.decide(ctx, 0.0, 30, 60)
        assert action == 2, "POV with zero inventory should return default action 2"


# =========================================================================
# 2. EXP3 Agent
# =========================================================================
@pytest.mark.skipif(not _HAS_EXP3, reason="EXP3Agent not available")
class TestEXP3Agent:

    def test_implements_base_agent(self):
        agent = EXP3Agent(config_path=_config_path())
        assert isinstance(agent, BaseAgent)

    def test_name_property(self):
        agent = EXP3Agent(config_path=_config_path())
        assert agent.name == "EXP3"

    def test_returns_valid_actions(self):
        agent = EXP3Agent(config_path=_config_path())
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(50):
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            assert 0 <= action <= 4

    def test_stochastic(self):
        """EXP3 is a stochastic agent; over many calls with the same input,
        it should explore multiple actions."""
        agent = EXP3Agent(config_path=_config_path())
        agent.reset(make_config())
        ctx = constant_context()
        actions_seen = set()
        for _ in range(200):
            action = agent.decide(ctx, 5.0, 10, 60)
            actions_seen.add(action)
        assert len(actions_seen) >= 2, \
            f"EXP3 should explore at least 2 actions, saw {actions_seen}"

    def test_weight_update(self):
        """After updates, at least one weight vector should change."""
        agent = EXP3Agent(config_path=_config_path())
        agent.reset(make_config())
        initial_weights = [w.copy() for w in agent._weights]
        rng = np.random.RandomState(42)
        for i in range(30):
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            agent.update(ctx, action, rng.normal(0, 1), random_context(rng))
        changed = any(
            not np.allclose(initial_weights[a], agent._weights[a])
            for a in range(5)
        )
        assert changed, "EXP3 weights should change after updates"

    def test_weights_clipped(self):
        """Weight elements should be clipped to [-10, 10]."""
        agent = EXP3Agent(config_path=_config_path())
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(200):
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            agent.update(ctx, action, rng.choice([-1000, 1000]), random_context(rng))
        for a in range(5):
            assert np.all(agent._weights[a] >= -10.0)
            assert np.all(agent._weights[a] <= 10.0)

    def test_probability_distribution_valid(self):
        """Internal mixed_probs should be a valid probability distribution."""
        agent = EXP3Agent(config_path=_config_path())
        agent.reset(make_config())
        ctx = random_context()
        agent.decide(ctx, 5.0, 10, 60)
        probs = agent._last_probs
        assert probs is not None
        assert abs(probs.sum() - 1.0) < 1e-6, f"Probabilities sum to {probs.sum()}"
        assert np.all(probs >= 0), "Negative probabilities"


# =========================================================================
# 3. RegimeSwitchAC Agent
# =========================================================================
@pytest.mark.skipif(not _HAS_REGIME, reason="RegimeSwitchACAgent not available")
class TestRegimeSwitchACAgent:

    def test_implements_base_agent(self):
        agent = RegimeSwitchACAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        assert isinstance(agent, BaseAgent)

    def test_name_property(self):
        agent = RegimeSwitchACAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        assert agent.name == "RegimeSwitchAC"

    def test_returns_valid_actions(self):
        agent = RegimeSwitchACAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(50):
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            assert 0 <= action <= 4

    def test_deterministic_same_input(self):
        """Non-learning agent should be deterministic."""
        agent = RegimeSwitchACAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        ctx = constant_context()
        a1 = agent.decide(ctx, 5.0, 10, 60)
        agent.reset(make_config())
        a2 = agent.decide(ctx, 5.0, 10, 60)
        assert a1 == a2

    def test_regime_switching(self):
        """High vol_ratio context should trigger stressed regime."""
        agent = RegimeSwitchACAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        # Feature index 9 = vol_ratio_short_long
        # Calm context
        calm_ctx = constant_context()
        calm_ctx[9] = 0.8  # below threshold (1.5)
        agent.decide(calm_ctx, 5.0, 0, 60)
        assert not agent._in_stressed, "Should be calm when vol_ratio < 1.5"

        # Stressed context
        stressed_ctx = calm_ctx.copy()
        stressed_ctx[9] = 2.5  # above threshold
        agent.decide(stressed_ctx, 5.0, 1, 60)
        assert agent._in_stressed, "Should be stressed when vol_ratio > 1.5"

    def test_hysteresis(self):
        """After entering stressed mode, agent stays stressed for HYSTERESIS_STEPS
        even if vol_ratio drops below threshold."""
        agent = RegimeSwitchACAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())

        # Trigger stressed mode
        stressed_ctx = constant_context()
        stressed_ctx[9] = 2.5
        agent.decide(stressed_ctx, 5.0, 0, 60)
        assert agent._in_stressed

        # Drop vol_ratio below threshold for 1 step (should stay stressed)
        calm_ctx = constant_context()
        calm_ctx[9] = 0.8
        agent.decide(calm_ctx, 5.0, 1, 60)
        assert agent._in_stressed, "Hysteresis should keep agent stressed"

    def test_trajectory_sums_to_quantity(self):
        """Both calm and stressed trajectories should sum to total_quantity."""
        agent = RegimeSwitchACAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        config = make_config(total_quantity=5.0)
        agent.reset(config)
        assert abs(agent._trajectory_calm.sum() - 5.0) < 1e-6
        assert abs(agent._trajectory_stressed.sum() - 5.0) < 1e-6


# =========================================================================
# 4. KernelUCB Agent
# =========================================================================
@pytest.mark.skipif(not _HAS_KERNEL, reason="KernelUCBAgent not available")
class TestKernelUCBAgent:

    def test_implements_base_agent(self):
        agent = KernelUCBAgent(config_path=_config_path())
        assert isinstance(agent, BaseAgent)

    def test_name_property(self):
        agent = KernelUCBAgent(config_path=_config_path())
        assert agent.name == "KernelUCB"

    def test_returns_valid_actions(self):
        agent = KernelUCBAgent(config_path=_config_path())
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(50):
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            assert 0 <= action <= 4

    def test_growing_history(self):
        """After updates, the per-action context history should grow."""
        agent = KernelUCBAgent(config_path=_config_path(), max_history=500)
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(20):
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            agent.update(ctx, action, rng.normal(), random_context(rng))
        total_obs = sum(len(c) for c in agent._contexts)
        assert total_obs == 20, f"Expected 20 observations, got {total_obs}"

    def test_max_history_enforcement(self):
        """History should be capped at max_history per action."""
        max_hist = 10
        agent = KernelUCBAgent(config_path=_config_path(), max_history=max_hist)
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        # Force all updates to the same action (0) by always choosing 0
        for i in range(max_hist + 5):
            ctx = random_context(rng)
            agent.update(ctx, 0, rng.normal(), random_context(rng))
        assert len(agent._contexts[0]) <= max_hist, \
            f"History {len(agent._contexts[0])} > max {max_hist}"

    def test_kernel_cache_invalidation(self):
        """After an update, the K_inv cache for that action should be invalidated."""
        agent = KernelUCBAgent(config_path=_config_path())
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        ctx = random_context(rng)
        agent.update(ctx, 0, 1.0, random_context(rng))
        assert agent._K_inv_cache[0] is None, "Cache should be invalidated after update"

    def test_exploration_with_few_obs(self):
        """With fewer than MIN_OBS_FOR_PREDICTION observations, agent should explore randomly."""
        agent = KernelUCBAgent(config_path=_config_path())
        agent.reset(make_config())
        # No updates yet -> all actions have random scores -> diverse actions
        actions = set()
        for _ in range(100):
            ctx = random_context(np.random.RandomState())
            action = agent.decide(ctx, 5.0, 10, 60)
            actions.add(action)
        assert len(actions) >= 2, "KernelUCB should explore before MIN_OBS_FOR_PREDICTION"

    def test_rbf_kernel_self(self):
        """RBF kernel of a point with itself should be 1.0."""
        agent = KernelUCBAgent(config_path=_config_path())
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        k_val = agent._rbf_kernel(x, x)
        assert abs(k_val - 1.0) < 1e-10


# =========================================================================
# 5. MetaAgent
# =========================================================================
@pytest.mark.skipif(not _HAS_META, reason="MetaAgent not available")
class TestMetaAgent:

    def test_implements_base_agent(self):
        agent = MetaAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        assert isinstance(agent, BaseAgent)

    def test_name_property(self):
        agent = MetaAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        assert agent.name == "MetaAgent"

    def test_returns_valid_actions(self):
        agent = MetaAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(30):
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            assert 0 <= action <= 4

    def test_delegates_to_sub_agents(self):
        """After several decisions, sub-agent selection counts should be non-zero."""
        agent = MetaAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(30):
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            agent.update(ctx, action, rng.normal(), random_context(rng))
        total_selections = agent.sub_agent_selection_counts.sum()
        assert total_selections == 30, f"Expected 30 selections, got {total_selections}"

    def test_sub_agents_reset(self):
        """All sub-agents should be reset when meta-agent resets."""
        agent = MetaAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        config = make_config()
        agent.reset(config)
        for sub in agent._sub_agents:
            # Sub-agents that track _config should have it set
            if hasattr(sub, '_config'):
                assert sub._config is not None, f"Sub-agent {sub.name} not reset"

    def test_meta_level_learning(self):
        """After updates, meta-level theta vectors should change from zero."""
        agent = MetaAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(50):
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            agent.update(ctx, action, rng.normal(), random_context(rng))
        # At least one theta should be non-zero
        any_nonzero = any(np.linalg.norm(t) > 1e-10 for t in agent._theta)
        assert any_nonzero, "Meta-level theta should be non-zero after learning"

    def test_selection_history_tracked(self):
        """Selection history should record which sub-agent was chosen."""
        agent = MetaAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        for i in range(10):
            ctx = random_context(np.random.RandomState(i))
            agent.decide(ctx, 5.0, i, 60)
        assert len(agent.sub_agent_selection_history) == 10


# =========================================================================
# 6. ThompsonACHybrid Agent
# =========================================================================
@pytest.mark.skipif(not _HAS_HYBRID, reason="ThompsonACHybridAgent not available")
class TestThompsonACHybridAgent:

    def test_implements_base_agent(self):
        agent = ThompsonACHybridAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        assert isinstance(agent, BaseAgent)

    def test_name_property(self):
        agent = ThompsonACHybridAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        assert agent.name == "ThompsonACHybrid"

    def test_returns_valid_actions(self):
        agent = ThompsonACHybridAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(50):
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            assert 0 <= action <= 4

    def test_clips_within_one_of_ac(self):
        """The hybrid action must be within +/-1 of what AC would recommend."""
        agent = ThompsonACHybridAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(100):
            ctx = random_context(rng)
            hybrid_action = agent.decide(ctx, 5.0, i % 60, 60)
            # Get AC's recommendation independently
            ac_action = agent._ac_agent.decide(ctx, 5.0, i % 60, 60)
            lo = max(0, ac_action - 1)
            hi = min(4, ac_action + 1)
            assert lo <= hybrid_action <= hi, \
                f"Hybrid action {hybrid_action} outside [{lo}, {hi}] (AC={ac_action})"

    def test_tracks_clip_count(self):
        """After many decisions, clip_count and total_decisions should be consistent."""
        agent = ThompsonACHybridAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        n_decisions = 50
        for i in range(n_decisions):
            ctx = random_context(rng)
            agent.decide(ctx, 5.0, i % 60, 60)
        assert agent.total_decisions == n_decisions
        assert agent.clip_count >= 0
        assert agent.clip_count <= n_decisions
        assert agent.agreement_count >= 0
        assert agent.agreement_count <= n_decisions

    def test_clip_rate_in_range(self):
        """Clip rate should be in [0, 1]."""
        agent = ThompsonACHybridAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        for i in range(50):
            ctx = random_context(rng)
            agent.decide(ctx, 5.0, i % 60, 60)
        assert 0.0 <= agent.clip_rate <= 1.0
        assert 0.0 <= agent.agreement_rate <= 1.0

    def test_update_propagates_to_thompson(self):
        """Updates should modify the internal Thompson agent's state."""
        agent = ThompsonACHybridAgent(
            config_path=_config_path(), calibration_path=_cal_path()
        )
        agent.reset(make_config())
        rng = np.random.RandomState(42)
        # Record initial state
        initial_m = [m.copy() for m in agent._thompson._m]
        ctx = random_context(rng)
        action = agent.decide(ctx, 5.0, 0, 60)
        agent.update(ctx, action, 10.0, random_context(rng))
        # At least the updated action's m vector should change
        changed = any(
            not np.allclose(initial_m[a], agent._thompson._m[a])
            for a in range(5)
        )
        assert changed, "Thompson internal state should change after update"


# =========================================================================
# 7. Cross-agent consistency checks
# =========================================================================
class TestCrossAgentConsistency:

    def _get_all_agents(self):
        """Collect all available agents."""
        agents = []
        from agents.twap_agent import TWAPAgent
        agents.append(TWAPAgent())
        if _HAS_POV:
            agents.append(POVAgent(config_path=_config_path()))
        if _HAS_EXP3:
            agents.append(EXP3Agent(config_path=_config_path()))
        if _HAS_REGIME:
            agents.append(RegimeSwitchACAgent(
                config_path=_config_path(), calibration_path=_cal_path()))
        if _HAS_KERNEL:
            agents.append(KernelUCBAgent(config_path=_config_path()))
        if _HAS_META:
            agents.append(MetaAgent(
                config_path=_config_path(), calibration_path=_cal_path()))
        if _HAS_HYBRID:
            agents.append(ThompsonACHybridAgent(
                config_path=_config_path(), calibration_path=_cal_path()))
        if _HAS_BANDITS:
            agents.append(LinUCBAgent(config_path=_config_path()))
            agents.append(ThompsonAgent(config_path=_config_path()))
        return agents

    def test_all_agents_have_name(self):
        """Every agent must have a non-empty name."""
        for agent in self._get_all_agents():
            assert isinstance(agent.name, str)
            assert len(agent.name) > 0

    def test_all_agents_accept_reset(self):
        """Every agent's reset() must accept ExecutionConfig without error."""
        config = make_config()
        for agent in self._get_all_agents():
            agent.reset(config)  # should not raise

    def test_all_agents_action_range(self):
        """Every agent must return actions in {0, 1, 2, 3, 4}."""
        config = make_config()
        rng = np.random.RandomState(42)
        for agent in self._get_all_agents():
            agent.reset(config)
            for _ in range(20):
                ctx = random_context(rng)
                action = agent.decide(ctx, 5.0, 10, 60)
                assert action in {0, 1, 2, 3, 4}, \
                    f"{agent.name} returned invalid action {action}"

    def test_all_agents_update_does_not_crash(self):
        """Calling update() on every agent should not raise exceptions."""
        config = make_config()
        rng = np.random.RandomState(42)
        for agent in self._get_all_agents():
            agent.reset(config)
            ctx = random_context(rng)
            action = agent.decide(ctx, 5.0, 10, 60)
            next_ctx = random_context(rng)
            agent.update(ctx, action, 1.0, next_ctx)  # should not raise
