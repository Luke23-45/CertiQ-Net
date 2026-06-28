"""Hydra/OmegaConf structured config schemas for CertiQ-Net."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

__all__ = [
    "EnvConfig",
    "TrainerConfig",
    "LossConfig",
    "ProgressConfig",
    "SweepConfig",
    "DatatypeTrainerConfig",
    "DatatypeProfileConfig",
    "RootConfig",
]


@dataclass
class EnvConfig:
    N: int = MISSING
    lam: float = MISSING
    mu_mode: str = MISSING
    mu_fixed: list[float] | None = None
    mu_lognormal_sigma: float = 0.5
    horizon_T: float = 1000.0
    rho_target: float | None = None


@dataclass
class TrainerConfig:
    """Shared PyTorch Lightning Trainer settings — NOT per-datatype."""
    max_epochs: int = 200
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 0.0
    val_check_interval: float = 0.25
    log_every_n_steps: int = 10


@dataclass
class DatatypeTrainerConfig:
    """Training hyperparameters that differ per datatype."""
    lr: float = MISSING
    weight_decay: float = MISSING
    rollout_horizon: int = MISSING
    imitation_warmup_epochs: int = MISSING
    expert_mode: str | None = None
    gamma: float = MISSING
    val_horizon_max: int = MISSING


@dataclass
class DatatypeProfileConfig:
    """Complete per-datatype configuration block."""
    data: Any = MISSING
    trainer: DatatypeTrainerConfig = MISSING
    loss: LossConfig = MISSING
    input_normalization: str = "none"


@dataclass
class LossConfig:
    omega_action: float = 1.5
    omega_margin: float = 0.1
    omega_roll: float = 1.0
    omega_ent: float = 0.001
    omega_kl: float = 0.05


@dataclass
class ProgressConfig:
    new_line_after_iteration: bool = True
    mininterval: float = 0.1
    maxinterval: float = 1.0
    miniters: int | None = None
    smoothing: float = 0.3
    dynamic_ncols: bool = True
    leave: bool = False
    position: int = 0
    unit: str = "it"
    bar_format: str | None = None
    ascii: bool = True
    ncols: int | None = None
    force_show: bool = False


@dataclass
class SweepConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    models: tuple[str, ...] = ("certiq_index",)
    envs: tuple[str, ...] = ()


@dataclass
class RootConfig:
    project: dict[str, Any] = MISSING
    model: Any = MISSING
    env: EnvConfig | None = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    experiment_family: str = "main_queueing"
    datatype: str = MISSING           # MANDATORY: must be "qgym"
    qgym: DatatypeProfileConfig | None = None
