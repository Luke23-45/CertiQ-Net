"""State-bank generation for certificate audits."""

from typing import Protocol

import torch
from torch import Tensor

from certiqnet.math.certificate import CertificateDiagnostics
from certiqnet.math.lyapunov import tail_size


class AuditableModel(Protocol):
    """Protocol for models used in adversarial state generation."""

    def __call__(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, CertificateDiagnostics]: ...


def _expand_mu(mu: Tensor, rows: int) -> Tensor:
    return mu.unsqueeze(0).expand(rows, -1) if mu.dim() == 1 else mu[:rows]


def generate_state_bank(
    N: int,
    mu: Tensor,
    beta: float,
    R_cert: float,
    n_random: int = 1000,
    n_grid: int = 500,
    n_boundary: int = 200,
    n_adversarial: int = 200,
) -> Tensor:
    """Generate random, grid, boundary, balanced, tail, and adversarial-shaped states."""
    del n_adversarial
    assert N >= 1, "N must be positive."
    assert (mu > 0).all(), "Service rates must be positive."
    states: list[Tensor] = [torch.randint(0, 100, (n_random, N)).float()]
    if N <= 5 and n_grid > 0:
        side = int(n_grid ** (1.0 / N)) + 1
        vals = torch.linspace(0, 50, side)
        grid = torch.stack(torch.meshgrid(*[vals] * N, indexing="ij"), dim=-1)
        states.append(grid.reshape(-1, N)[:n_grid])

    Q_bound = torch.zeros(n_boundary, N)
    for row in range(n_boundary):
        Q_bound[row, row % N] = float(torch.randint(50, 500, (1,)).item())
    states.append(Q_bound)
    states.append(torch.randint(40, 100, (max(1, n_boundary // 2), N)).float())

    tail_rows = max(1, n_boundary)
    Q_tail = torch.randint(1, 50, (tail_rows, N)).float()
    mu_tail = _expand_mu(mu, tail_rows)
    if R_cert < float("inf"):
        S = tail_size(Q_tail, mu_tail, beta).clamp_min(1e-6)
        scale = (R_cert * 1.5 / S).clamp(min=1.0).unsqueeze(-1)
        Q_tail = (Q_tail * scale).ceil()
    states.append(Q_tail)
    return torch.cat(states, dim=0)


def generate_adversarial_states(
    model: AuditableModel,
    mu: Tensor,
    N: int,
    n_states: int = 100,
    n_steps: int = 50,
) -> Tensor:
    """Use gradient ascent on certificate violation to produce hard audit states."""
    mu_b = _expand_mu(mu, n_states)
    Q = torch.randint(0, 50, (n_states, N)).float().requires_grad_(True)
    opt = torch.optim.Adam([Q], lr=0.5)
    for _ in range(n_steps):
        opt.zero_grad()
        Q_pos = Q.clamp(min=0)
        _, diag = model(Q_pos, mu_b, training_mode=True)
        loss = -diag.drift_slack.mean()
        loss.backward()
        opt.step()
    return Q.detach().clamp(min=0).ceil()
