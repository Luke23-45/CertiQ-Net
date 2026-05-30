"""Multi-robot task-allocation adapter."""

import torch
from torch import Tensor

from certiqnet.adapters.base import AdapterBatch, DispatchTuple
from certiqnet.adapters.queueing import QueueingAdapter


class RoboticsAdapter(QueueingAdapter):
    """Robotics adapter for spatial task allocation demonstrations."""

    CERTIFICATE_STATUS = "empirical"
    context_dim = 6

    def __init__(
        self,
        assumptions_satisfied: bool = False,
        workspace_size: float = 100.0,
        travel_time_mode: str = "constant",
        **kwargs: object,
    ) -> None:
        super().__init__(assumptions_satisfied=assumptions_satisfied, **kwargs)
        assert workspace_size > 0, "workspace_size must be positive."
        assert travel_time_mode in {"constant", "distance"}, "Unsupported travel_time_mode."
        self.workspace_size = float(workspace_size)
        self.travel_time_mode = travel_time_mode

    def to_dispatch_tuple(self, system_state: dict[str, object]) -> DispatchTuple:
        dispatch = super().to_dispatch_tuple(system_state)
        xi_obj = system_state.get("xi")
        if xi_obj is None:
            Q = dispatch.Q.reshape(1, -1)
            mu = dispatch.mu.reshape(1, -1)
            xi = self._features(Q, mu, generator=None).squeeze(0)
        else:
            xi = torch.as_tensor(xi_obj).float()
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
            "Poisson task arrivals",
            "Exponential task completion per robot",
            "Constant travel-time abstraction or separately certified generator",
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
        distance = xi[..., 5]
        cost = batch.Q.sum(dim=-1) + distance.min(dim=-1).values
        if self.travel_time_mode == "distance":
            effective_mu = batch.mu / (1.0 + distance)
        else:
            effective_mu = batch.mu
        return AdapterBatch(Q=batch.Q, mu=effective_mu.clamp_min(1e-3), xi=xi, cost=cost)

    def make_observation(self, Q: Tensor, mu: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        if mu.dim() == 1:
            mu = mu.unsqueeze(0).expand(Q.shape[0], -1)
        xi = self._features(Q, mu, generator=None)
        if self.travel_time_mode == "distance":
            mu = (mu / (1.0 + xi[..., 5])).clamp_min(1e-3)
        return Q, mu, xi

    def _features(
        self, Q: Tensor, mu: Tensor, generator: torch.Generator | None
    ) -> Tensor:
        batch, n = Q.shape
        if generator is None:
            base = torch.linspace(0.0, 1.0, n, device=Q.device, dtype=Q.dtype)
            robot_xy = torch.stack([base, 1.0 - base], dim=-1).unsqueeze(0).expand(batch, -1, -1)
            task_xy = torch.full((batch, 1, 2), 0.5, device=Q.device, dtype=Q.dtype)
        else:
            robot_xy = torch.rand(batch, n, 2, generator=generator, device=Q.device, dtype=Q.dtype)
            task_xy = torch.rand(batch, 1, 2, generator=generator, device=Q.device, dtype=Q.dtype)
        distance = torch.linalg.vector_norm(robot_xy - task_xy, dim=-1)
        battery = (mu / mu.amax(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(0.0, 1.0)
        return torch.cat(
            [
                robot_xy,
                task_xy.expand(batch, n, -1),
                battery.unsqueeze(-1),
                distance.unsqueeze(-1),
            ],
            dim=-1,
        )
