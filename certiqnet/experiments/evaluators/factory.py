"""Shared config-to-model helpers for experiment entrypoints.

.. deprecated::
    Prefer ``LightningCLI`` with ``class_path`` resolution (see
    ``certiqnet.cli``).  This manual registry is retained only for
    backward compatibility.
"""

from __future__ import annotations

import warnings
from typing import TypeVar

import torch
from omegaconf import DictConfig, OmegaConf

from certiqnet.dispatcher.certiq.index_model import CertiQIndexModel
from certiqnet.models.baselines import (
    AnalyticBackbonePolicy,
    QuadraticMinDrift,
    RandomPolicy,
    ShortestExpectedDelay,
)

T = TypeVar("T")




def build_mu(cfg: DictConfig) -> tuple[torch.Tensor, float]:
    """Build the service-rate vector and arrival rate from the resolved config.
    
    .. deprecated::
        Pass ``mu`` and ``lam`` directly to data module / lightning module
        constructor arguments instead.
    """
    warnings.warn("factory.build_mu is deprecated; pass mu/lam as direct args.", DeprecationWarning, stacklevel=2)
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
    """Instantiate the configured model class from the OmegaConf node.
    
    .. deprecated::
        Use ``class_path`` resolution via ``LightningCLI`` (``certiqnet.cli``)
        instead of this manual if-elif chain.
    """
    warnings.warn("factory.build_model is deprecated; use class_path resolution instead.", DeprecationWarning, stacklevel=2)
    model_data = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_data, dict):
        raise TypeError("cfg.model must resolve to a mapping")

    target = str(model_data.get("_target_", ""))
    if target.endswith("AnalyticBackbonePolicy"):
        return AnalyticBackbonePolicy(
            N=N, beta=float(model_data.get("beta", 1.0)), C=float(model_data.get("C", float("inf")))
        )
    if target.endswith("RandomPolicy"):
        return RandomPolicy(
            N=N, beta=float(model_data.get("beta", 1.0)), C=float(model_data.get("C", float("inf")))
        )
    if target.endswith("ShortestExpectedDelay"):
        return ShortestExpectedDelay(
            N=N, beta=float(model_data.get("beta", 1.0)), C=float(model_data.get("C", float("inf")))
        )
    if target.endswith("QuadraticMinDrift"):
        return QuadraticMinDrift(
            N=N, beta=float(model_data.get("beta", 1.0)), C=float(model_data.get("C", float("inf")))
        )
    if target.endswith("CertiQIndexModel"):
        token_layers = int(model_data.get("token_layers", model_data.get("encoder_layers", 2)))
        global_layers = int(model_data.get("global_layers", model_data.get("encoder_layers", 2)))
        candidate_top_k = model_data.get("candidate_top_k", None)
        gate_hidden_dim = int(model_data.get("gate_hidden_dim", 32))
        return CertiQIndexModel(
            N=N,
            hidden_dim=int(model_data.get("hidden_dim", 64)),
            tau=float(model_data.get("tau", 1.0)),
            C=float(model_data.get("C", 2.0)),
            exploration_temperature=float(model_data.get("exploration_temperature", 1.5)),
            beta=float(model_data.get("beta", 1.0)),
            d_xi=d_xi,
            token_layers=token_layers,
            global_layers=global_layers,
            dropout=float(model_data.get("dropout", 0.0)),
            candidate_top_k=None if candidate_top_k is None else int(candidate_top_k),
            gate_hidden_dim=gate_hidden_dim,
            cost_fn=str(model_data.get("cost_fn", "qmd")),
            certificate_mode=str(model_data.get("certificate_mode", "exact")),
            constraint_mode=str(model_data.get("constraint_mode", "exact")),
        )
    raise ValueError(f"Unsupported model target: {target}")
