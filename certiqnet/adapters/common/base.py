"""Abstract domain adapter interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DispatchTuple:
    """Abstract dispatch tuple ``D = (N, lambda, mu, xi, Q, c)``."""

    N: int
    lam: float
    mu: Tensor
    xi: Tensor | None
    Q: Tensor
    cost: float


@dataclass(frozen=True)
class AdapterBatch:
    """Vectorized training/evaluation batch emitted by a domain adapter."""

    Q: Tensor
    mu: Tensor
    xi: Tensor | None
    cost: Tensor


class DispatchAdapter(ABC):
    """Map a real system into the abstract dispatch model."""

    CERTIFICATE_STATUS: str
    context_dim: int = 0

    def __init__(self, assumptions_satisfied: bool = False, **_: object) -> None:
        self.assumptions_satisfied = bool(assumptions_satisfied)

    @abstractmethod
    def to_dispatch_tuple(self, system_state: dict[str, object]) -> DispatchTuple:
        """Convert a domain state to the abstract dispatch tuple."""

    @abstractmethod
    def apply_action(self, action: int, system_state: dict[str, object]) -> dict[str, object]:
        """Apply a resource assignment action to a domain state."""

    @abstractmethod
    def certificate_assumptions(self) -> list[str]:
        """Return assumptions required for exact certificate inheritance."""

    def sample_batch(
        self,
        *,
        n_samples: int,
        N: int,
        mu: Tensor,
        max_queue: int = 100,
        generator: torch.Generator | None = None,
    ) -> AdapterBatch:
        """Generate synthetic adapter-consistent states for training and audit."""
        assert n_samples >= 1, "n_samples must be positive."
        assert N >= 1, "N must be positive."
        assert mu.shape == (N,), "mu must have shape (N,)."
        assert (mu > 0).all(), "All service rates must be positive."
        Q = torch.randint(0, max_queue, (n_samples, N), generator=generator).float()
        mu_b = mu.float().unsqueeze(0).expand(n_samples, -1).clone()
        return AdapterBatch(Q=Q, mu=mu_b, xi=None, cost=Q.sum(dim=-1))

    def make_observation(self, Q: Tensor, mu: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Return tensors ready for a CertiQ-Net forward pass."""
        if mu.dim() == 1:
            mu = mu.unsqueeze(0).expand(Q.shape[0], -1)
        return Q, mu, None
