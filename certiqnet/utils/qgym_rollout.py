"""Helpers for running CertiQ-Net policies inside QGym."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from certiqnet.adapters.qgym.config import QGymEnvConfig
from certiqnet.adapters.qgym.env_loader import compute_effective_mu
from certiqnet.adapters.qgym.env_loader import load_qgym_env


def build_qgym_rollout_env(
    env_config: str | Path | dict,
    *,
    batch: int,
    seed: int,
    device: torch.device | str,
):
    """Build a QGym rollout environment on the requested device."""
    return load_qgym_env(
        env_config=env_config,
        batch=batch,
        seed=seed,
        policy_name="WC",
        device=device,
    )


def as_batch_tensor(value: object, *, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Convert a scalar / vector / batched value to a 2D tensor."""
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.dim() == 0:
        return tensor.reshape(1, 1)
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def extract_qgym_queues(obs: object, info: dict[str, object] | None, *, device: torch.device) -> Tensor:
    """Return the queue tensor from a QGym step."""
    if isinstance(info, dict) and "queues" in info:
        return as_batch_tensor(info["queues"], device=device)
    if isinstance(info, dict) and "Q" in info:
        return as_batch_tensor(info["Q"], device=device)
    return as_batch_tensor(obs, device=device)


def resolve_qgym_effective_mu(env_config: str | Path | dict | QGymEnvConfig, *, device: torch.device) -> Tensor:
    """Resolve the queue-level effective ``mu`` vector used by the models."""
    env = load_qgym_env(
        env_config=env_config,
        batch=1,
        seed=0,
        policy_name="WC",
        device=device,
    )
    network = env.network[0] if env.network.dim() == 3 else env.network
    mu_matrix = env.mu[0] if env.mu.dim() == 3 else env.mu

    pool_size = None
    if isinstance(env_config, QGymEnvConfig):
        pool_size = env_config.server_pool_size
    elif isinstance(env_config, dict):
        pool_size = env_config.get("server_pool_size")

    pool_tensor = None
    if pool_size is not None:
        pool_tensor = torch.as_tensor(pool_size, dtype=torch.float32, device=device)

    return compute_effective_mu(
        torch.as_tensor(network, dtype=torch.float32, device=device),
        torch.as_tensor(mu_matrix, dtype=torch.float32, device=device),
        pool_size=pool_tensor,
    )


def qgym_action_from_pi(
    pi: Tensor,
    network: Tensor,
    queue_lengths: Tensor | None = None,
    *,
    sample: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Map per-queue probabilities to a QGym priority matrix.

    The action is one-hot over queues and expanded across compatible
    servers.  This keeps the rollout consistent with QGym's
    server-queue priority interface while still letting the policy
    choose the queue.
    """
    if pi.dim() == 1:
        pi = pi.unsqueeze(0)
    pi = torch.nan_to_num(pi, nan=0.0).clamp_min(0.0)
    pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    if sample:
        idx = torch.multinomial(pi, num_samples=1, generator=generator).squeeze(-1)
    else:
        idx = pi.argmax(dim=-1)

    if network.dim() == 2:
        network = network.unsqueeze(0)
    if network.shape[0] == 1 and pi.shape[0] > 1:
        network = network.expand(pi.shape[0], -1, -1)

    network = network.to(device=pi.device, dtype=pi.dtype)
    if queue_lengths is None:
        queue_lengths = torch.ones(pi.shape[0], pi.shape[-1], device=pi.device, dtype=pi.dtype)
    else:
        queue_lengths = torch.as_tensor(queue_lengths, device=pi.device, dtype=pi.dtype)
        if queue_lengths.dim() == 1:
            queue_lengths = queue_lengths.unsqueeze(0)
        if queue_lengths.shape[0] == 1 and pi.shape[0] > 1:
            queue_lengths = queue_lengths.expand(pi.shape[0], -1)
        if queue_lengths.shape[0] != pi.shape[0]:
            raise ValueError(
                f"queue_lengths batch {queue_lengths.shape[0]} does not match pi batch {pi.shape[0]}"
            )
    queue_lengths = torch.nan_to_num(queue_lengths, nan=0.0).clamp_min(0.0)
    priority = queue_lengths.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1).clamp_min(1.0)
    selected = F.one_hot(idx, num_classes=pi.shape[-1]).to(dtype=pi.dtype)
    return network * selected.unsqueeze(1) * priority.view(-1, 1, 1)
