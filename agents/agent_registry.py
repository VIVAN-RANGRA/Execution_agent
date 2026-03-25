"""Auto-discovers all agents via BaseAgent.__subclasses__() recursion."""
from __future__ import annotations

import inspect
import importlib
from typing import Dict, List, Optional

from agents.base_agent import BaseAgent


_AGENT_MODULES = [
    "agents.twap_agent",
    "agents.vwap_agent",
    "agents.ac_agent",
    "agents.pov_agent",
    "agents.bandit_base",
    "agents.linucb_agent",
    "agents.thompson_agent",
    "agents.exp3_agent",
    "agents.regime_switching_ac_agent",
    "agents.kernel_ucb_agent",
    "agents.meta_agent",
    "agents.thompson_ac_hybrid_agent",
    "agents.corral_agent",
]

for module_name in _AGENT_MODULES:
    importlib.import_module(module_name)


def _discover_agents(base_cls=BaseAgent):
    """Recursively find all concrete agent subclasses."""
    agents = {}
    for cls in base_cls.__subclasses__():
        if not getattr(cls, "__abstractmethods__", None):
            try:
                instance = cls()
                agents[instance.name] = cls
            except Exception:
                pass
        agents.update(_discover_agents(cls))
    return agents


AGENT_REGISTRY = _discover_agents()

_AGENT_GROUPS = {
    "benchmarks": ["TWAP", "VWAP", "AC_Optimal", "POV", "RegimeSwitchAC"],
    "bandits": ["LinUCB", "Thompson", "EXP3", "KernelUCB"],
    "ensembles": ["MetaAgent", "ThompsonACHybrid", "Corral"],
}

_AGENT_ORDER = _AGENT_GROUPS["benchmarks"] + _AGENT_GROUPS["bandits"] + _AGENT_GROUPS["ensembles"]
_LEARNING_AGENT_NAMES = set(_AGENT_GROUPS["bandits"] + _AGENT_GROUPS["ensembles"])

AGENT_METADATA = {}
for category, names in _AGENT_GROUPS.items():
    for name in names:
        AGENT_METADATA[name] = {
            "category": category,
            "learning": name in _LEARNING_AGENT_NAMES,
        }

for name in AGENT_REGISTRY:
    AGENT_METADATA.setdefault(name, {"category": "other", "learning": False})


def _filter_kwargs(cls, kwargs):
    """Pass only kwargs accepted by the agent constructor."""
    sig = inspect.signature(cls)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def get_agent(name: str, **kwargs):
    """Instantiate an agent by name with optional kwargs."""
    if name not in AGENT_REGISTRY:
        AGENT_REGISTRY.update(_discover_agents())
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{name}'. Available: {list(AGENT_REGISTRY.keys())}")
    cls = AGENT_REGISTRY[name]
    return cls(**_filter_kwargs(cls, kwargs))


def list_agents(category: Optional[str] = None):
    """Return registered agent names in a stable, category-aware order."""
    if category is None:
        return [name for name in _AGENT_ORDER if name in AGENT_REGISTRY]

    normalized = category.lower()
    if normalized in {"all", "any"}:
        return list_agents()

    names = _AGENT_GROUPS.get(normalized, [])
    return [name for name in names if name in AGENT_REGISTRY]


def benchmark_agents() -> List[str]:
    return list_agents("benchmarks")


def learning_agents() -> List[str]:
    return [name for name in list_agents() if AGENT_METADATA.get(name, {}).get("learning", False)]
