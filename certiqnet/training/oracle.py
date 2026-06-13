"""Oracle and heuristic supervision utilities for CertiQ-Net."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class OracleBundle:
    """One oracle configuration bundle exported by the queueing DP script."""

    N: int
    K: int
    lam: float
    gamma: float
    mu: Tensor
    states: Tensor
    pi_oracle: Tensor
    delta_V: Tensor
    pi_sed: Tensor
    pi_qmd: Tensor
    pi_jswq: Tensor


def _to_tensor_list(value: object) -> list[Tensor]:
    if isinstance(value, list):
        return [torch.as_tensor(v).detach().cpu() for v in value]
    if isinstance(value, Tensor):
        return [value.detach().cpu()]
    raise TypeError(f"Unsupported oracle archive value type: {type(value)!r}")


def load_queueing_oracle_archive(path: str | Path) -> list[OracleBundle]:
    """Load the combined queueing oracle archive written by the oracle script."""
    archive = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(archive, dict):
        raise TypeError("Oracle archive must be a dictionary.")

    states_list = _to_tensor_list(archive["states"])
    mu_list = _to_tensor_list(archive["mu"])
    pi_oracle_list = _to_tensor_list(archive["pi_oracle"])
    delta_v_list = _to_tensor_list(archive["delta_V"])
    pi_sed_list = _to_tensor_list(archive["pi_sed"])
    pi_qmd_list = _to_tensor_list(archive["pi_qmd"])
    pi_jswq_list = _to_tensor_list(archive["pi_jswq"])
    configs = list(archive.get("configs", []))

    bundles: list[OracleBundle] = []
    for idx, states in enumerate(states_list):
        cfg = configs[idx] if idx < len(configs) else None
        if cfg is None:
            N = int(states.shape[-1])
            K = int(states.max().item())
            lam = float("nan")
        else:
            N, K, lam, _mu = cfg
            N = int(N)
            K = int(K)
            lam = float(lam)
        bundles.append(
            OracleBundle(
                N=N,
                K=K,
                lam=lam,
                gamma=float(archive.get("gamma", 0.99)),
                mu=torch.as_tensor(mu_list[idx]).float(),
                states=states.float(),
                pi_oracle=torch.as_tensor(pi_oracle_list[idx]).long(),
                delta_V=torch.as_tensor(delta_v_list[idx]).float(),
                pi_sed=torch.as_tensor(pi_sed_list[idx]).long(),
                pi_qmd=torch.as_tensor(pi_qmd_list[idx]).long(),
                pi_jswq=torch.as_tensor(pi_jswq_list[idx]).long(),
            )
        )
    return bundles


def heuristic_actions(Q: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
    """Return heuristic SED and QMD labels for a batch of states."""
    mu_safe = mu.clamp_min(torch.finfo(mu.dtype).tiny)
    sed = ((Q + 1.0) / mu_safe).argmin(dim=-1)
    qmd = ((2.0 * Q + 1.0) / mu_safe).argmin(dim=-1)
    return sed, qmd
