"""Communication-channel dispatch adapter."""

import torch
from torch import Tensor

from certiqnet.adapters.base import AdapterBatch, DispatchTuple
from certiqnet.adapters.queueing import QueueingAdapter


class ChannelAdapter(QueueingAdapter):
    """Approximate wireless/channel scheduling adapter."""

    CERTIFICATE_STATUS = "approximate"
    context_dim = 4

    def __init__(
        self,
        assumptions_satisfied: bool = False,
        fading_mode: str = "block",
        **kwargs: object,
    ) -> None:
        super().__init__(assumptions_satisfied=assumptions_satisfied, **kwargs)
        assert fading_mode in {"block", "fixed"}, "Unsupported fading_mode."
        self.fading_mode = fading_mode

    def to_dispatch_tuple(self, system_state: dict[str, object]) -> DispatchTuple:
        dispatch = super().to_dispatch_tuple(system_state)
        xi_obj = system_state.get("xi")
        xi = (
            self._features(dispatch.Q.reshape(1, -1), dispatch.mu.reshape(1, -1), None).squeeze(0)
            if xi_obj is None
            else torch.as_tensor(xi_obj).float()
        )
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
            "Packet arrivals are Poisson",
            "Channel service rates are fixed during each dispatch decision",
            "Fading is either block-constant or included in the observed Markov state",
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
        xi = self._features(batch.Q, batch.mu, generator)
        snr = xi[..., 0]
        effective_mu = batch.mu * torch.log2(1.0 + snr).clamp_min(0.1)
        cost = batch.Q.sum(dim=-1) + (1.0 / snr.clamp_min(1e-3)).mean(dim=-1)
        return AdapterBatch(Q=batch.Q, mu=effective_mu, xi=xi, cost=cost)

    def make_observation(self, Q: Tensor, mu: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        if mu.dim() == 1:
            mu = mu.unsqueeze(0).expand(Q.shape[0], -1)
        xi = self._features(Q, mu, None)
        return Q, mu * torch.log2(1.0 + xi[..., 0]).clamp_min(0.1), xi

    def _features(
        self, Q: Tensor, mu: Tensor, generator: torch.Generator | None
    ) -> Tensor:
        batch, n = Q.shape
        if generator is None or self.fading_mode == "fixed":
            base = torch.linspace(0.5, 4.0, n, device=Q.device, dtype=Q.dtype).expand(batch, -1)
            snr = base
        else:
            snr = 0.5 + 3.5 * torch.rand(
                batch, n, generator=generator, device=Q.device, dtype=Q.dtype
            )
        reliability = snr / (1.0 + snr)
        backlog_share = Q / Q.sum(dim=-1, keepdim=True).clamp_min(1.0)
        capacity_share = mu / mu.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fading_flag = torch.full_like(Q, 1.0 if self.fading_mode == "block" else 0.0)
        return torch.stack([snr, reliability, backlog_share, capacity_share * fading_flag], dim=-1)
