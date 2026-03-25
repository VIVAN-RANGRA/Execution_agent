"""Shared helpers for deterministic tests."""
import numpy as np
import yaml
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "default_config.yaml"

with open(CONFIG_PATH) as _config_file:
    DEFAULT_CONFIG = yaml.safe_load(_config_file)

DEFAULT_D_FEATURES = DEFAULT_CONFIG["agents"]["linucb"]["d_features"]


def make_context(seed: int = 0, d: int = None) -> np.ndarray:
    """Create a deterministic random context vector."""
    rng = np.random.RandomState(seed)
    return rng.standard_normal(d or DEFAULT_D_FEATURES).astype(np.float32)


def constant_context(value: float = 0.5, d: int = None) -> np.ndarray:
    """Create a constant-valued context vector."""
    return np.full(d or DEFAULT_D_FEATURES, value, dtype=np.float32)


def make_feature_stats(d: int = None) -> dict:
    """Identity normalization stats for the requested feature dimension."""
    width = d or DEFAULT_D_FEATURES
    return {
        "means": [0.0] * width,
        "stds": [1.0] * width,
        "feature_names": [f"f{i}" for i in range(width)],
    }
