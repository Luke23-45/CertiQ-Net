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
    max_epochs: int = 200
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 1.0
    val_check_interval: float = 0.25
    log_every_n_steps: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-5
    rollout_horizon: int = 16
    pretrain_epochs: int = 20
    finetune_epochs: int = 40
    entropy_warmup_epochs: int = 20
    imitation_warmup_epochs: int = 20
    policy_buffer_max: int = 4096
    policy_mix_fraction: float = 0.25
    teacher_mix_fraction: float = 0.25
    synthetic_mix_fraction: float = 0.50


@dataclass
class LossConfig:
    omega_bc: float = 1.0
    omega_action: float = 1.5
    omega_margin: float = 0.1
    omega_usage: float = 0.1
    omega_certificate: float = 5.0
    omega_correction: float = 0.01
    omega_ent: float = 0.0
    rollout_weight: float = 1.0
    policy_kl_weight: float = 0.05
    value_weight: float = 1.0
    entropy_weight: float = 0.001
    distillation_temperature: float = 1.0


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


@dataclass
class SweepConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    models: tuple[str, ...] = ("certiq_index",)
    envs: tuple[str, ...] = ("family_a", "family_b", "family_c", "family_e")


@dataclass
class RootConfig:
    project: dict[str, Any] = MISSING
    model: Any = MISSING
    env: EnvConfig = MISSING
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    experiment_family: str = "main_queueing"
