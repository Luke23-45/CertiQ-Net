"""Shared config-to-model helpers for experiment entrypoints."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

import torch
from omegaconf import DictConfig, OmegaConf

from certiqnet.models.baselines import (
    AnalyticBackbonePolicy,
    CertiQNetSAblation_LearnedPhiOnly,
    CertiQNetSAblation_NoGate,
    CertiQNetXAblation,
    JoinShortestWeightedQueue,
    RandomPolicy,
)
from certiqnet.models.certiqnet_p import CertiQNetP
from certiqnet.models.certiqnet_s import CertiQNetS
from certiqnet.utils.config_schemas import CertiQNetPConfig, CertiQNetSConfig

T = TypeVar("T")


def _dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Construct nested dataclasses from a resolved config dictionary."""
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value = data[f.name]
        target = f.type
        if is_dataclass(target) and isinstance(value, dict):
            kwargs[f.name] = _dataclass_from_dict(target, value)  # type: ignore[arg-type]
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def build_mu(cfg: DictConfig) -> tuple[torch.Tensor, float]:
    """Build the service-rate vector and arrival rate from the resolved config."""
    env = cfg.env
    if env.mu_mode == "fixed":
        if env.mu_fixed is None:
            raise ValueError("env.mu_fixed must be set when mu_mode=fixed")
        mu = torch.tensor(list(env.mu_fixed), dtype=torch.float32)
    elif env.mu_mode == "lognormal":
        mu = torch.distributions.LogNormal(0.0, float(env.mu_lognormal_sigma)).sample((int(env.N),))
    else:
        raise ValueError(f"Unsupported mu_mode: {env.mu_mode}")
    lam = float(env.lam)
    if env.rho_target is not None:
        lam = float(env.rho_target) * mu.sum().item()
    if lam >= mu.sum().item():
        raise ValueError("lambda must remain subcritical.")
    return mu, lam


def build_model(cfg: DictConfig, N: int, d_xi: int = 0) -> torch.nn.Module:
    """Instantiate the configured model class from the OmegaConf node."""
    model_data = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_data, dict):
        raise TypeError("cfg.model must resolve to a mapping")

    target = str(model_data.get("_target_", ""))
    if target.endswith("CertiQNetP"):
        return CertiQNetP(_dataclass_from_dict(CertiQNetPConfig, model_data), N=N, d_xi=d_xi)
    if target.endswith("AnalyticBackbonePolicy"):
        return AnalyticBackbonePolicy(N=N, beta=float(model_data.get("beta", 1.0)))
    if target.endswith("RandomPolicy"):
        return RandomPolicy(N=N, beta=float(model_data.get("beta", 1.0)))
    if target.endswith("JoinShortestWeightedQueue"):
        return JoinShortestWeightedQueue(N=N, beta=float(model_data.get("beta", 1.0)))
    if target.endswith("CertiQNetSAblation_NoGate"):
        return CertiQNetSAblation_NoGate(
            _dataclass_from_dict(CertiQNetSConfig, model_data), N=N, d_xi=d_xi
        )
    if target.endswith("CertiQNetXAblation"):
        return CertiQNetXAblation(
            _dataclass_from_dict(CertiQNetSConfig, model_data), N=N, d_xi=d_xi
        )
    if target.endswith("CertiQNetSAblation_LearnedPhiOnly"):
        return CertiQNetSAblation_LearnedPhiOnly(N=N, beta=float(model_data.get("beta", 1.0)))
    return CertiQNetS(_dataclass_from_dict(CertiQNetSConfig, model_data), N=N, d_xi=d_xi)
