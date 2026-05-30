"""Queueing adapter for the exact CTMC model."""

import torch

from certiqnet.adapters.base import AdapterBatch, DispatchAdapter, DispatchTuple


class QueueingAdapter(DispatchAdapter):
    """Exact adapter for states already represented as CTMC queues."""

    CERTIFICATE_STATUS = "exact"
    context_dim = 0

    def to_dispatch_tuple(self, system_state: dict[str, object]) -> DispatchTuple:
        Q = torch.as_tensor(system_state["Q"]).float()
        mu = torch.as_tensor(system_state["mu"]).float()
        lam_obj = system_state["lam"]
        assert isinstance(lam_obj, int | float | str), "lambda must be numeric."
        lam = float(lam_obj)
        xi_obj = system_state.get("xi")
        xi = None if xi_obj is None else torch.as_tensor(xi_obj).float()
        return DispatchTuple(N=Q.numel(), lam=lam, mu=mu, xi=xi, Q=Q, cost=float(Q.sum()))

    def apply_action(self, action: int, system_state: dict[str, object]) -> dict[str, object]:
        Q = torch.as_tensor(system_state["Q"]).float().clone()
        Q[action] += 1.0
        out = dict(system_state)
        out["Q"] = Q
        return out

    def certificate_assumptions(self) -> list[str]:
        return ["Poisson arrivals", "Exponential service per resource", "lambda < sum(mu)"]

    def sample_batch(
        self,
        *,
        n_samples: int,
        N: int,
        mu: torch.Tensor,
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
        return AdapterBatch(Q=batch.Q, mu=batch.mu, xi=None, cost=batch.Q.sum(dim=-1))
