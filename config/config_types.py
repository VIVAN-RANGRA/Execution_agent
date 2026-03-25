"""Typed configuration dataclasses for the execution engine project."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-03-31"
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class ExecutionParams:
    total_quantity_btc: float = 10.0
    time_horizon_seconds: int = 3600
    num_slices: int = 60
    min_slice_fraction: float = 0.005
    max_slice_fraction: float = 0.30


@dataclass
class ImpactModelConfig:
    alpha: float = 0.6
    eta_multiplier: float = 0.1
    gamma_multiplier: float = 0.01


@dataclass
class LinUCBConfig:
    alpha_exploration: float = 1.0
    d_features: int = 18
    warm_start_from_ac: bool = True
    warm_start_episodes: int = 20


@dataclass
class ThompsonConfig:
    prior_variance: float = 1.0
    d_features: int = 18
    warm_start_from_ac: bool = True
    warm_start_episodes: int = 20


@dataclass
class EXP3Config:
    gamma: float = 0.1
    d_features: int = 18
    warm_start_from_ac: bool = True
    warm_start_episodes: int = 20


@dataclass
class KernelUCBConfig:
    alpha_exploration: float = 1.0
    d_features: int = 18
    kernel_bandwidth: float = 1.0
    max_history: int = 500
    warm_start_from_ac: bool = True
    warm_start_episodes: int = 20


@dataclass
class POVConfig:
    target_participation_rate: float = 0.10


@dataclass
class RegimeSwitchACConfig:
    calm_risk_aversion: float = 0.05
    stressed_risk_aversion: float = 0.50
    vol_ratio_threshold: float = 1.5
    hysteresis_steps: int = 3


@dataclass
class MetaAgentConfig:
    alpha_meta: float = 0.5
    warm_start_episodes: int = 10


@dataclass
class ThompsonACHybridConfig:
    prior_variance: float = 1.0
    warm_start_from_ac: bool = True
    warm_start_episodes: int = 20


@dataclass
class CorralConfig:
    gamma: Optional[float] = None
    n_episodes: int = 200


@dataclass
class AgentsConfig:
    linucb: LinUCBConfig = field(default_factory=LinUCBConfig)
    thompson: ThompsonConfig = field(default_factory=ThompsonConfig)
    exp3: EXP3Config = field(default_factory=EXP3Config)
    kernel_ucb: KernelUCBConfig = field(default_factory=KernelUCBConfig)
    pov: POVConfig = field(default_factory=POVConfig)
    regime_switch_ac: RegimeSwitchACConfig = field(default_factory=RegimeSwitchACConfig)
    meta_agent: MetaAgentConfig = field(default_factory=MetaAgentConfig)
    thompson_ac_hybrid: ThompsonACHybridConfig = field(default_factory=ThompsonACHybridConfig)
    corral: CorralConfig = field(default_factory=CorralConfig)


@dataclass
class FeaturesConfig:
    volatility_window_short_s: int = 60
    volatility_window_long_s: int = 300
    volume_window_s: int = 30
    ofi_window_s: int = 10
    normalization: str = "zscore"
    use_interaction_features: bool = True
    d_features: int = 18  # 12 base + 6 interaction


@dataclass
class EvaluationConfig:
    n_episodes: int = 200
    metrics: List[str] = field(default_factory=lambda: [
        "IS", "vwap_slippage", "participation_rate", "shortfall_std", "timing_risk_bps"
    ])
    random_seed: int = 42


@dataclass
class CalibrationConfig:
    sigma_per_second: float = 1e-5
    adv_btc: float = 5000.0
    eta: float = 1e-6
    gamma: float = 1e-7
    alpha: float = 0.6
    calibration_date_range: str = ""


@dataclass
class ProjectPaths:
    config_path: Path
    parsed_data_dir: Path
    volume_profile_path: Path
    calibration_params_path: Path
    feature_stats_path: Path


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionParams = field(default_factory=ExecutionParams)
    impact_model: ImpactModelConfig = field(default_factory=ImpactModelConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def feature_dimension(self) -> int:
        return int(max(
            self.features.d_features,
            self.agents.linucb.d_features,
            self.agents.thompson.d_features,
            self.agents.exp3.d_features,
            self.agents.kernel_ucb.d_features,
        ))
