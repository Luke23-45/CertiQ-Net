"""Shared context feature construction for QGym-based training.

The helpers in this module turn short trajectory history and topology
signals into a fixed per-queue context tensor that can be consumed by
the regime-aware policy encoder.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

QGYM_CONTEXT_DIM = 8


def _as_batch_state(value: Tensor | Sequence[float] | Sequence[Sequence[float]], *, ref: Tensor) -> Tensor:
    """Convert a value to a batch-aligned float tensor."""
    tensor = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    if tensor.dim() == 0:
        tensor = tensor.reshape(1, 1).expand(ref.shape[0], ref.shape[1])
    elif tensor.dim() == 1:
        if tensor.shape[0] == ref.shape[1]:
            tensor = tensor.unsqueeze(0).expand(ref.shape[0], -1)
        else:
            tensor = tensor.unsqueeze(-1).expand(ref.shape[0], ref.shape[1])
    elif tensor.dim() >= 2 and tensor.shape[0] != ref.shape[0]:
        if tensor.shape[-1] == ref.shape[1] and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0).unsqueeze(0).expand(ref.shape[0], -1)
    return tensor.float()


def _stack_history(history: Sequence[Tensor] | None, *, ref: Tensor) -> Tensor:
    if not history:
        return ref.unsqueeze(0)
    tensors = [_as_batch_state(item, ref=ref) for item in history]
    return torch.stack(tensors, dim=0)


def _normalize_rows(x: Tensor, *, eps: float = 1e-9) -> Tensor:
    denom = x.sum(dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


def _topology_degree(network: Tensor | None, *, ref: Tensor) -> Tensor:
    if network is None:
        return torch.ones_like(ref)
    net = torch.as_tensor(network, dtype=ref.dtype, device=ref.device)
    if net.dim() == 3:
        net = net[0]
    if net.dim() != 2:
        return torch.ones_like(ref)
    degree = net.sum(dim=0, keepdim=True).expand(ref.shape[0], -1)
    return degree / degree.max().clamp_min(1.0)


def _action_load(action_history: Sequence[Tensor] | None, *, ref: Tensor) -> Tensor:
    if not action_history:
        return torch.zeros_like(ref)
    action = torch.as_tensor(action_history[-1], dtype=ref.dtype, device=ref.device)
    if action.dim() == 3:
        load = action.sum(dim=-2)
    elif action.dim() == 2:
        load = action.sum(dim=0, keepdim=True)
    else:
        load = torch.zeros_like(ref)
    if load.shape != ref.shape:
        load = _as_batch_state(load, ref=ref)
    return _normalize_rows(load.clamp_min(0.0))


def _event_rate(history_dt: Sequence[Tensor] | None, *, ref: Tensor) -> Tensor:
    if not history_dt:
        return torch.zeros_like(ref)
    dt = torch.as_tensor(history_dt[-1], dtype=ref.dtype, device=ref.device)
    if dt.dim() == 0:
        dt = dt.reshape(1).expand(ref.shape[0])
    dt = dt.reshape(-1).clamp_min(1e-9)
    if dt.shape[0] != ref.shape[0]:
        dt = dt.mean().reshape(1).expand(ref.shape[0])
    event_rate = (1.0 / dt).unsqueeze(-1).expand_as(ref)
    return event_rate / event_rate.mean(dim=-1, keepdim=True).clamp_min(1.0)


def build_qgym_context(
    Q: Tensor,
    mu: Tensor,
    *,
    history_Q: Sequence[Tensor] | None = None,
    history_action: Sequence[Tensor] | None = None,
    history_dt: Sequence[Tensor] | None = None,
    network: Tensor | None = None,
    context_dim: int = QGYM_CONTEXT_DIM,
) -> Tensor:
    """Build a fixed-width per-queue context tensor.

    The context is intentionally compact and history-aware:

    - current queue share,
    - previous queue share,
    - one-step delta,
    - rolling mean share,
    - rolling standard deviation,
    - latest action load,
    - topology degree,
    - inverse event-time rate.
    """

    Q = torch.as_tensor(Q).float()
    if Q.dim() == 1:
        Q = Q.unsqueeze(0)
    mu = torch.as_tensor(mu).float()
    if mu.dim() == 1:
        mu = mu.unsqueeze(0).expand(Q.shape[0], -1)

    hist_Q = _stack_history(history_Q, ref=Q)
    if hist_Q.shape[0] == 1:
        hist_Q = torch.cat([hist_Q, Q.unsqueeze(0)], dim=0)

    current = Q.clamp_min(0.0)
    prev = hist_Q[-2] if hist_Q.shape[0] >= 2 else current
    current_share = _normalize_rows(current)
    prev_share = _normalize_rows(prev)

    delta = current - prev
    delta_scale = (current.abs().mean(dim=-1, keepdim=True) + prev.abs().mean(dim=-1, keepdim=True) + 1.0)
    delta_share = delta / delta_scale

    hist_mean = hist_Q.mean(dim=0)
    hist_std = hist_Q.std(dim=0, unbiased=False) if hist_Q.shape[0] > 1 else torch.zeros_like(current)
    hist_mean_share = _normalize_rows(hist_mean)
    hist_std_share = hist_std / (hist_mean.abs().mean(dim=-1, keepdim=True) + 1.0)

    action_load = _action_load(history_action, ref=current)
    topology_degree = _topology_degree(network, ref=current)
    event_rate = _event_rate(history_dt, ref=current)

    features = [
        current_share,
        prev_share,
        delta_share,
        hist_mean_share,
        hist_std_share,
        action_load,
        topology_degree,
        torch.where(torch.isfinite(event_rate), event_rate, torch.zeros_like(event_rate)),
    ]

    xi = torch.stack(features, dim=-1)
    if xi.shape[-1] > context_dim:
        xi = xi[..., :context_dim]
    elif xi.shape[-1] < context_dim:
        pad = torch.zeros(*xi.shape[:-1], context_dim - xi.shape[-1], device=xi.device, dtype=xi.dtype)
        xi = torch.cat([xi, pad], dim=-1)

    return xi
