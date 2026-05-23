"""Hydra/OmegaConf structured config schemas for CertiQ-Net."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class BackboneConfig:
    """Analytic backbone parameter constraints and initial values."""

    alpha_min: float = 1e-3
    beta_min: float = 1e-3
    alpha_init: float = 1.0
    beta_init: float = 1.0
    gamma_max: float = 2.0
    gamma_init: float = 0.0
    c_init: float = 0.0


@dataclass
class EncoderConfig:
    """Permutation-equivariant encoder configuration."""

    d_local: int = 64
    d_global: int = 64
    n_layers_local: int = 2
    n_layers_res: int = 2
    hidden_dim: int = 128
    pooling: str = "attention"


@dataclass
class ResidualConfig:
    """Bounded residual configuration."""

    R_max: float = 2.0


@dataclass
class GateConfig:
    """Raw scalar gate configuration."""

    eta_max: float = 1.0


@dataclass
class CertiQNetSConfig:
    """Hard-tail fallback model configuration."""

    _target_: str = "certiqnet.models.certiqnet_s.CertiQNetS"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    R_cert: float = MISSING
    beta: float = MISSING
    tau_smooth: float = 10.0


@dataclass
class CertiQNetPConfig:
    """Drift-envelope projection model configuration."""

    _target_: str = "certiqnet.models.certiqnet_p.CertiQNetP"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    C_B: float = MISSING
    beta: float = MISSING


@dataclass
class EnvConfig:
    """CTMC environment configuration."""

    N: int = MISSING
    lam: float = MISSING
    mu_mode: str = MISSING
    mu_fixed: list[float] | None = None
    mu_lognormal_sigma: float = 0.5
    horizon_T: float = 1000.0
    rho_target: float | None = None


@dataclass
class TrainerConfig:
    """Lightning trainer and optimizer configuration."""

    max_epochs: int = 200
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 1.0
    val_check_interval: float = 0.25
    log_every_n_steps: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-5


@dataclass
class LossConfig:
    """Loss weights. ``omega_ent < 0`` means entropy bonus."""

    omega_bc: float = 1.0
    omega_gate: float = 0.1
    omega_drift: float = 5.0
    omega_res: float = 0.01
    omega_ent: float = -0.001
    rollout_weight: float = 1.0


@dataclass
class RootConfig:
    """Top-level Hydra configuration."""

    project: dict[str, Any] = MISSING
    model: Any = MISSING
    env: EnvConfig = MISSING
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
