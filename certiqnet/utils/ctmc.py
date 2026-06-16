"""Vectorized continuous-time Markov chain environment."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CTMCStep:
    """Single vectorized CTMC event result."""

    Q: Tensor
    dt: Tensor
    cost: Tensor
    event_idx: Tensor
    event_type: Tensor


class CTMCEnvironment:
    """Vectorized CTMC environment with Poisson arrivals and exponential service."""

    def __init__(self, N: int, lam: float, mu: Tensor, B: int = 1) -> None:
        assert N >= 1, "N must be positive."
        assert B >= 1, "B must be positive."
        assert lam > 0, "lambda must be positive."
        assert mu.shape == (N,), "mu must have shape (N,)."
        assert (mu > 0).all(), "All service rates must be strictly positive."
        assert lam < mu.sum().item(), (
            f"Load is supercritical: lambda={lam:.4f} >= Lambda={mu.sum().item():.4f}."
        )
        self.N = N
        self.lam = float(lam)
        self.mu = mu.float()
        self.B = B
        self.Q = torch.zeros(B, N, dtype=torch.float32, device=mu.device)

    def reset(self, Q: Tensor | None = None) -> Tensor:
        """Reset the environment state and return a copy of ``Q``."""
        if Q is None:
            self.Q.zero_()
        else:
            assert Q.shape == (self.B, self.N), "Q must have shape (B, N)."
            assert (Q >= 0).all(), "Queue lengths must be non-negative."
            self.Q = Q.float().clone()
        return self.Q.clone()

    def step(self, pi: Tensor) -> dict[str, Tensor]:
        """Advance every parallel environment by one event."""
        assert pi.shape == (self.B, self.N), "pi must have shape (B, N)."
        assert (pi >= 0).all(), "Routing probabilities must be non-negative."
        assert torch.allclose(pi.sum(dim=-1), torch.ones(self.B, device=pi.device), atol=1e-5)
        arrival_rates = self.lam * pi
        service_rates = self.mu.unsqueeze(0) * (self.Q > 0).float()
        all_rates = torch.cat([arrival_rates, service_rates], dim=-1)
        total_rates = all_rates.sum(dim=-1)
        dt = torch.distributions.Exponential(total_rates).sample()
        event_probs = all_rates / total_rates.unsqueeze(-1)
        event_idx = torch.distributions.Categorical(probs=event_probs).sample()
        arrival_mask = event_idx < self.N
        resource_idx = torch.where(arrival_mask, event_idx, event_idx - self.N)
        delta = torch.zeros_like(self.Q)
        delta.scatter_(1, resource_idx.unsqueeze(-1), 1.0)
        self.Q = torch.where(
            arrival_mask.unsqueeze(-1),
            self.Q + delta,
            (self.Q - delta).clamp(min=0.0),
        )
        return {
            "Q": self.Q.clone(),
            "dt": dt,
            "cost": self.Q.sum(dim=-1),
            "event_idx": event_idx,
            "event_type": arrival_mask.to(torch.long),
        }
