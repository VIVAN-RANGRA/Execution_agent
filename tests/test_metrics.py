"""Unit tests for evaluation metrics."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.data_classes import Fill
from evaluation.metrics import (
    implementation_shortfall_bps,
    vwap_slippage_bps,
    participation_rate,
    timing_risk_bps,
    win_rate_vs_baseline,
    mann_whitney_p_value,
)


def make_fills(prices, qtys, arrival=50000.0):
    return [
        Fill(timestamp_ms=i * 60000, quantity_filled=q, fill_price=p,
             market_impact_cost=0.0, slice_index=i)
        for i, (p, q) in enumerate(zip(prices, qtys))
    ]


class TestIS:
    def test_zero_when_fills_at_arrival(self):
        fills = make_fills([50000.0] * 5, [2.0] * 5)
        assert abs(implementation_shortfall_bps(fills, 50000.0)) < 1e-8

    def test_negative_when_below_arrival(self):
        fills = make_fills([49900.0] * 5, [2.0] * 5)
        is_val = implementation_shortfall_bps(fills, 50000.0)
        assert is_val < 0.0  # filled below arrival = good for buy

    def test_positive_when_above_arrival(self):
        fills = make_fills([50100.0] * 5, [2.0] * 5)
        is_val = implementation_shortfall_bps(fills, 50000.0)
        assert is_val > 0.0


class TestVWAPSlippage:
    def test_zero_when_exec_equals_market(self):
        fills = make_fills([50000.0] * 3, [1.0] * 3)
        market_prices = np.array([50000.0, 50000.0, 50000.0])
        market_qtys = np.array([1.0, 1.0, 1.0])
        slip = vwap_slippage_bps(fills, market_prices, market_qtys)
        assert abs(slip) < 1e-6


class TestParticipationRate:
    def test_in_unit_interval(self):
        fills = make_fills([50000.0] * 5, [1.0] * 5)
        rate = participation_rate(fills, 100.0)
        assert 0.0 < rate <= 1.0

    def test_correct_value(self):
        fills = make_fills([50000.0] * 5, [1.0] * 5)
        rate = participation_rate(fills, 50.0)
        assert abs(rate - 0.1) < 1e-8  # 5 / 50 = 0.1


class TestWinRate:
    def test_all_wins(self):
        agent = [1.0, 2.0, 3.0]
        baseline = [5.0, 6.0, 7.0]
        assert win_rate_vs_baseline(agent, baseline) == 1.0

    def test_no_wins(self):
        agent = [5.0, 6.0, 7.0]
        baseline = [1.0, 2.0, 3.0]
        assert win_rate_vs_baseline(agent, baseline) == 0.0


class TestMannWhitney:
    def test_p_value_in_range(self):
        a = list(np.random.normal(0, 1, 50))
        b = list(np.random.normal(1, 1, 50))
        p = mann_whitney_p_value(a, b)
        assert 0.0 <= p <= 1.0

    def test_clearly_different(self):
        a = list(np.random.normal(-5, 0.1, 50))
        b = list(np.random.normal(5, 0.1, 50))
        p = mann_whitney_p_value(a, b)
        assert p < 0.05
