"""Single entry point for loading project configuration."""
import copy
import json
import yaml
from functools import lru_cache
from pathlib import Path
from config.config_types import (
    ProjectConfig,
    DataConfig,
    ExecutionParams,
    ImpactModelConfig,
    AgentsConfig,
    LinUCBConfig,
    ThompsonConfig,
    EXP3Config,
    KernelUCBConfig,
    POVConfig,
    RegimeSwitchACConfig,
    MetaAgentConfig,
    ThompsonACHybridConfig,
    CorralConfig,
    FeaturesConfig,
    EvaluationConfig,
    CalibrationConfig,
    ProjectPaths,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "default_config.yaml"
DEFAULT_PARSED_DATA_DIR = BASE_DIR / "data" / "parsed"
DEFAULT_VOLUME_PROFILE_PATH = BASE_DIR / "data" / "volume_profile.parquet"
DEFAULT_CALIBRATION_PATH = BASE_DIR / "config" / "calibration_params.json"
DEFAULT_FEATURE_STATS_PATH = BASE_DIR / "config" / "feature_stats.json"


def _safe_build(cls, raw_dict: dict):
    """Build a dataclass from a dict, ignoring unknown keys and using defaults for missing ones."""
    if raw_dict is None:
        return cls()
    known_fields = set(cls.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw_dict.items() if k in known_fields}
    return cls(**filtered)


@lru_cache(maxsize=8)
def _load_config_cached(path_str: str) -> ProjectConfig:
    path = Path(path_str)

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    agents_raw = raw.get("agents", {}) or {}

    return ProjectConfig(
        data=_safe_build(DataConfig, raw.get("data")),
        execution=_safe_build(ExecutionParams, raw.get("execution")),
        impact_model=_safe_build(ImpactModelConfig, raw.get("impact_model")),
        agents=AgentsConfig(
            linucb=_safe_build(LinUCBConfig, agents_raw.get("linucb")),
            thompson=_safe_build(ThompsonConfig, agents_raw.get("thompson")),
            exp3=_safe_build(EXP3Config, agents_raw.get("exp3")),
            kernel_ucb=_safe_build(KernelUCBConfig, agents_raw.get("kernel_ucb")),
            pov=_safe_build(POVConfig, agents_raw.get("pov")),
            regime_switch_ac=_safe_build(RegimeSwitchACConfig, agents_raw.get("regime_switch_ac")),
            meta_agent=_safe_build(MetaAgentConfig, agents_raw.get("meta_agent")),
            thompson_ac_hybrid=_safe_build(ThompsonACHybridConfig, agents_raw.get("thompson_ac_hybrid")),
            corral=_safe_build(CorralConfig, agents_raw.get("corral")),
        ),
        features=_safe_build(FeaturesConfig, raw.get("features")),
        evaluation=_safe_build(EvaluationConfig, raw.get("evaluation")),
    )


def load_config(config_path: str = None) -> ProjectConfig:
    """
    Load config from YAML and return a typed ProjectConfig.

    If a section is missing from the YAML, dataclass defaults are used.
    If extra keys exist in any section, they are silently ignored.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return copy.deepcopy(_load_config_cached(str(path.resolve())))


@lru_cache(maxsize=8)
def _load_calibration_cached(path_str: str) -> CalibrationConfig:
    path = Path(path_str)
    with open(path) as f:
        raw = json.load(f) or {}
    return _safe_build(CalibrationConfig, raw)


def load_calibration(calibration_params_path: str = None) -> CalibrationConfig:
    """Load typed calibration parameters from JSON."""
    path = Path(calibration_params_path) if calibration_params_path else DEFAULT_CALIBRATION_PATH
    return copy.deepcopy(_load_calibration_cached(str(path.resolve())))


def resolve_project_paths(
    config_path: str = None,
    parsed_data_dir: str = None,
    volume_profile_path: str = None,
    calibration_params_path: str = None,
    feature_stats_path: str = None,
) -> ProjectPaths:
    """Resolve all runtime paths in one place for env and pipeline consumers."""
    return ProjectPaths(
        config_path=Path(config_path) if config_path else DEFAULT_CONFIG_PATH,
        parsed_data_dir=Path(parsed_data_dir) if parsed_data_dir else DEFAULT_PARSED_DATA_DIR,
        volume_profile_path=Path(volume_profile_path) if volume_profile_path else DEFAULT_VOLUME_PROFILE_PATH,
        calibration_params_path=Path(calibration_params_path) if calibration_params_path else DEFAULT_CALIBRATION_PATH,
        feature_stats_path=Path(feature_stats_path) if feature_stats_path else DEFAULT_FEATURE_STATS_PATH,
    )
