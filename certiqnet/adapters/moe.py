"""Mixture-of-experts adapter with empirical certificate status by default."""

import torch
from torch import Tensor

from certiqnet.adapters.base import AdapterBatch, DispatchTuple
from certiqnet.adapters.queueing import QueueingAdapter


class MoEAdapter(QueueingAdapter):
    """Approximate adapter for MoE token routing.

    The adapter keeps the CertiQ-Net dispatch tuple intact while adding
    token-expert context features. The synthetic generator is intentionally
    small and deterministic enough for local experiments; it mirrors common MoE
    routing testbeds where expert load, affinity, and capacity are visible to
    the router, but it does not claim transformer-level CTMC certification.
    """

    CERTIFICATE_STATUS = "approximate"
    context_dim = 4

    def __init__(
        self,
        assumptions_satisfied: bool = False,
        token_group_size: int = 4,
        affinity_scale: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(assumptions_satisfied=assumptions_satisfied, **kwargs)
        assert token_group_size >= 1, "token_group_size must be positive."
        self.token_group_size = int(token_group_size)
        self.affinity_scale = float(affinity_scale)

    def to_dispatch_tuple(self, system_state: dict[str, object]) -> DispatchTuple:
        dispatch = super().to_dispatch_tuple(system_state)
        xi_obj = system_state.get("xi")
        if xi_obj is not None:
            xi = torch.as_tensor(xi_obj).float()
        else:
            Q = dispatch.Q.reshape(1, -1)
            mu = dispatch.mu.reshape(1, -1)
            xi = self._features(Q, mu, generator=None).squeeze(0)
        return DispatchTuple(
            N=dispatch.N,
            lam=dispatch.lam,
            mu=dispatch.mu,
            xi=xi,
            Q=dispatch.Q,
            cost=dispatch.cost,
        )

    def certificate_assumptions(self) -> list[str]:
        return [
            "Arrivals are token groups dispatched irrevocably to one expert",
            "Expert service approximated as exponential",
            "Batched execution effects ignored",
            "Empirical diagnostics required; exact CTMC theorem not claimed",
        ]

    def sample_batch(
        self,
        *,
        n_samples: int,
        N: int,
        mu: Tensor,
        max_queue: int = 100,
        generator: torch.Generator | None = None,
    ) -> AdapterBatch:
        batch = super().sample_batch(
            n_samples=n_samples,
            N=N,
            mu=mu,
            max_queue=max_queue,
            generator=generator,
        )
        xi = self._features(batch.Q, batch.mu, generator=generator)
        affinity = xi[..., 0]
        drop_pressure = (batch.Q / batch.mu.clamp_min(1e-8)).amax(dim=-1)
        cost = batch.Q.sum(dim=-1) + 0.1 * drop_pressure - 0.05 * affinity.max(dim=-1).values
        return AdapterBatch(Q=batch.Q, mu=batch.mu, xi=xi, cost=cost)

    def make_observation(self, Q: Tensor, mu: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        if mu.dim() == 1:
            mu = mu.unsqueeze(0).expand(Q.shape[0], -1)
        return Q, mu, self._features(Q, mu, generator=None)

    def _features(
        self, Q: Tensor, mu: Tensor, generator: torch.Generator | None
    ) -> Tensor:
        batch, n = Q.shape
        capacity_share = mu / mu.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized_load = Q / (Q.sum(dim=-1, keepdim=True).clamp_min(1.0))
        expert_id = torch.linspace(0.0, 1.0, n, device=Q.device, dtype=Q.dtype).expand(batch, -1)
        if generator is None:
            token_topic = torch.sigmoid((expert_id - 0.5) * self.affinity_scale)
        else:
            topic = torch.rand(batch, 1, generator=generator, device=Q.device, dtype=Q.dtype)
            token_topic = torch.exp(-self.affinity_scale * (expert_id - topic).square())
        group = torch.full_like(Q, float(self.token_group_size))
        group = group / group.amax(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.stack([token_topic, capacity_share, normalized_load, group], dim=-1)
