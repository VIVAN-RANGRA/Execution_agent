"""Utilities for making fixed-width bandit contexts robust to mismatched inputs."""
from __future__ import annotations

import numpy as np


def normalize_context(context: np.ndarray, expected_dim: int) -> np.ndarray:
    """Return a finite float64 context vector with exactly ``expected_dim`` values."""
    x = np.asarray(context, dtype=np.float64).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    if expected_dim <= 0:
        return x
    if x.size == expected_dim:
        return x
    if x.size > expected_dim:
        return x[:expected_dim]

    padded = np.zeros(expected_dim, dtype=np.float64)
    padded[:x.size] = x
    return padded
