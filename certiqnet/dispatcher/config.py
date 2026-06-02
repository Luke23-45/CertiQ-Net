"""Configuration dataclasses for the z3 CertiQ Dispatcher."""

from dataclasses import dataclass, field

from certiqnet.dispatcher.types import CertificateMode


@dataclass
class GeometryConfig:
    alpha_min: float = 1e-3
    beta_min: float = 1e-3
    gamma_max: float = 2.0
    alpha_init: float = 1.0
    beta_init: float = 1.0
    gamma_init: float = 0.0
    c_init: float = 0.0
    C: float = 2.0


@dataclass
class ProposalConfig:
    d_local: int = 64
    d_global: int = 64
    hidden_dim: int = 128
    local_layers: int = 2
    update_layers: int = 2
    pooling: str = "attention"
    correction_bound: float = 2.0
    usage_max: float = 1.0


@dataclass
class CertificateConfig:
    mode: CertificateMode = "projection"
    fallback_radius: float = 50.0
    projection_tolerance: float = 1e-5


@dataclass
class PressureConfig:
    rho: float = 0.75
    step_size: float = 0.15
    decay: float = 0.05


@dataclass
class CertiQDispatcherConfig:
    _target_: str = "certiqnet.dispatcher.model.CertiQDispatcher"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    certificate: CertificateConfig = field(default_factory=CertificateConfig)
    pressure: PressureConfig = field(default_factory=PressureConfig)
