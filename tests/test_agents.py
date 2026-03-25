"""Unit tests for all agent implementations."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.data_classes import ExecutionConfig, MarketState
from agents.base_agent import BaseAgent
from agents.twap_agent import TWAPAgent
from agents.vwap_agent import VWAPAgent
from agents.ac_agent import ACAgent
from agents.linucb_agent import LinUCBAgent
from agents.thompson_agent import ThompsonAgent
from tests.helpers import DEFAULT_D_FEATURES, make_context


def make_config():
    return ExecutionConfig(
        total_quantity=10.0,
        time_horizon_seconds=3600,
        num_slices=60,
        arrival_price=50000.0,
    )


class TestBaseInterface:
    """All agents must implement the BaseAgent interface."""

    def test_twap_is_base_agent(self):
        assert isinstance(TWAPAgent(), BaseAgent)

    def test_ac_is_base_agent(self):
        assert isinstance(ACAgent(), BaseAgent)

    def test_linucb_is_base_agent(self):
        assert isinstance(LinUCBAgent(), BaseAgent)

    def test_thompson_is_base_agent(self):
        assert isinstance(ThompsonAgent(), BaseAgent)

    def test_agents_have_name(self):
        for AgentClass in [TWAPAgent, ACAgent, LinUCBAgent, ThompsonAgent]:
            agent = AgentClass()
            assert isinstance(agent.name, str) and len(agent.name) > 0


class TestTWAP:
    def test_always_returns_2(self):
        agent = TWAPAgent()
        config = make_config()
        agent.reset(config)
        for _ in range(20):
            ctx = make_context(np.random.randint(0, 1000))
            action = agent.decide(ctx, 5.0, 0, 60)
            assert action == 2, f"TWAP returned {action}, expected 2"

    def test_ignores_context(self):
        agent = TWAPAgent()
        agent.reset(make_config())
        ctx_zeros = np.zeros(DEFAULT_D_FEATURES, dtype=np.float32)
        ctx_ones = np.ones(DEFAULT_D_FEATURES, dtype=np.float32)
        assert agent.decide(ctx_zeros, 5.0, 0, 60) == 2
        assert agent.decide(ctx_ones, 5.0, 0, 60) == 2


class TestACAgent:
    def test_deterministic(self):
        """Same trajectory on two resets with same config."""
        agent1 = ACAgent()
        agent2 = ACAgent()
        config = make_config()
        agent1.reset(config)
        agent2.reset(config)
        ctx = make_context(0)
        action1 = agent1.decide(ctx, 10.0, 0, 60)
        action2 = agent2.decide(ctx, 10.0, 0, 60)
        assert action1 == action2

    def test_returns_valid_action(self):
        agent = ACAgent()
        agent.reset(make_config())
        for t in range(60):
            ctx = make_context(t)
            action = agent.decide(ctx, max(0.1, 10.0 - t * 0.16), t, 60)
            assert 0 <= action <= 4


class TestLinUCB:
    def test_valid_actions(self):
        agent = LinUCBAgent()
        agent.reset(make_config())
        for i in range(20):
            ctx = make_context(i)
            action = agent.decide(ctx, 5.0, i, 60)
            assert 0 <= action <= 4

    def test_weight_update_after_100(self):
        """After 100 updates, weights must differ from initialization."""
        agent = LinUCBAgent()
        agent.reset(make_config())
        d = agent._d
        rng = np.random.RandomState(42)
        for i in range(100):
            ctx = rng.standard_normal(d).astype(np.float32)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            reward = rng.uniform(-1, 1)
            next_ctx = rng.standard_normal(d).astype(np.float32)
            agent.update(ctx, action, reward, next_ctx)
        for a in range(5):
            norm = np.linalg.norm(agent._theta[a])
            assert norm > 0.01, f"LinUCB action {a} weights near zero after 100 updates"

    def test_not_degenerate_first_50(self):
        """No single action chosen >90% of the time in first 50 decisions."""
        agent = LinUCBAgent()
        agent.reset(make_config())
        d = agent._d
        rng = np.random.RandomState(7)
        actions = []
        for i in range(50):
            ctx = rng.standard_normal(d).astype(np.float32)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            actions.append(action)
            reward = rng.uniform(-1, 1)
            next_ctx = rng.standard_normal(d).astype(np.float32)
            agent.update(ctx, action, reward, next_ctx)
        from collections import Counter
        counts = Counter(actions)
        most_common_fraction = max(counts.values()) / 50
        assert most_common_fraction <= 0.90, f"LinUCB is degenerate: {counts}"

    def test_reproducible_with_seed(self):
        """With same numpy seed, action selection is reproducible."""
        agent1 = LinUCBAgent()
        agent2 = LinUCBAgent()
        agent1.reset(make_config())
        agent2.reset(make_config())
        np.random.seed(99)
        ctx = make_context(0)
        a1 = agent1.decide(ctx, 5.0, 0, 60)
        np.random.seed(99)
        a2 = agent2.decide(ctx, 5.0, 0, 60)
        assert a1 == a2


class TestThompson:
    def test_valid_actions(self):
        agent = ThompsonAgent()
        agent.reset(make_config())
        for i in range(20):
            ctx = make_context(i)
            action = agent.decide(ctx, 5.0, i, 60)
            assert 0 <= action <= 4

    def test_stochastic(self):
        """Thompson must produce different actions across 3 repeated calls."""
        agent = ThompsonAgent()
        agent.reset(make_config())
        ctx = make_context(0)
        actions = set()
        for _ in range(30):
            action = agent.decide(ctx, 5.0, 0, 60)
            actions.add(action)
        assert len(actions) > 1, "Thompson agent is not stochastic"

    def test_weight_update_after_100(self):
        """After 100 updates, posterior means must differ from init."""
        agent = ThompsonAgent()
        agent.reset(make_config())
        d = agent._d
        rng = np.random.RandomState(42)
        for i in range(100):
            ctx = rng.standard_normal(d).astype(np.float32)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            reward = rng.uniform(-1, 1)
            next_ctx = rng.standard_normal(d).astype(np.float32)
            agent.update(ctx, action, reward, next_ctx)
        for a in range(5):
            mu = agent._Lambda_inv[a] @ agent._m[a]
            assert np.linalg.norm(mu) > 0.01, f"Thompson action {a} posterior mean near zero"

    def test_not_degenerate_first_50(self):
        """No single action chosen >90% in first 50 decisions."""
        agent = ThompsonAgent()
        agent.reset(make_config())
        d = agent._d
        rng = np.random.RandomState(7)
        actions = []
        for i in range(50):
            ctx = rng.standard_normal(d).astype(np.float32)
            action = agent.decide(ctx, 5.0, i % 60, 60)
            actions.append(action)
            reward = rng.uniform(-1, 1)
            next_ctx = rng.standard_normal(d).astype(np.float32)
            agent.update(ctx, action, reward, next_ctx)
        from collections import Counter
        counts = Counter(actions)
        most_common_fraction = max(counts.values()) / 50
        assert most_common_fraction <= 0.90, f"Thompson is degenerate: {counts}"
