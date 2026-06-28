"""State-bank generation for certificate audits."""

from typing import Protocol

import torch
from torch import Tensor

from certiqnet.data.common.lyapunov import tail_size
from certiqnet.dispatcher.delay_geometry import quadratic_drift_index, sed_index
from certiqnet.dispatcher.types import DispatcherDiagnostics


class AuditableModel(Protocol):
    """Protocol for models used in adversarial state generation."""

    def __call__(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, DispatcherDiagnostics]: ...


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

    if n_adversarial > 0:
        Q_adv = torch.randint(75, 150, (n_adversarial, N)).float()
        if R_cert < float("inf"):
            mu_adv = _expand_mu(mu, n_adversarial)
            S_adv = tail_size(Q_adv, mu_adv, beta).clamp_min(1e-6)
            scale_adv = (R_cert * 2.0 / S_adv).clamp(min=1.0).unsqueeze(-1)
            Q_adv = (Q_adv * scale_adv).ceil()
        states.append(Q_adv)
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
    device = next((p.device for p in model.parameters() if p is not None), torch.device("cpu"))
    Q = torch.randint(0, 50, (n_states, N), device=device).float().requires_grad_(True)
    mu_b = mu_b.to(device=device)
    opt = torch.optim.Adam([Q], lr=0.5)
    for _ in range(n_steps):
        opt.zero_grad()
        Q_pos = Q.clamp(min=0)
        if hasattr(model, "reset_dispatch_state"):
            model.reset_dispatch_state()
        _, diag = model(Q_pos, mu_b, training_mode=False)
        loss = -diag.certificate_slack.mean()
        loss.backward()
        opt.step()
    return Q.detach().clamp(min=0).ceil().to(device="cpu")


def generate_disagreement_states(
    *,
    mu: Tensor,
    N: int,
    n_states: int = 100,
    bank_size: int = 4096,
    beta: float = 1.0,
    bank: Tensor | None = None,
) -> Tensor:
    """Select states where QMD and SED disagree or are nearly tied.

    The returned states intentionally concentrate training on the
    ambiguous region where the analytic rules are least certain.
    """
    if bank is None:
        bank = generate_state_bank(
            N=N,
            mu=mu,
            beta=beta,
            R_cert=float("inf"),
            n_random=max(bank_size // 2, n_states * 8),
            n_grid=0,
            n_boundary=max(64, n_states * 2),
            n_adversarial=max(64, n_states * 2),
        )
    mu_b = _expand_mu(mu, bank.shape[0])
    qmd = quadratic_drift_index(bank, mu_b)
    sed = sed_index(bank, mu_b)
    qmd_choice = qmd.argmin(dim=-1)
    sed_choice = sed.argmin(dim=-1)
    disagreement = (qmd_choice != sed_choice).float()

    def _margin(x: Tensor) -> Tensor:
        top2 = torch.topk(x, k=min(2, x.shape[-1]), largest=False).values
        if top2.shape[-1] == 1:
            return torch.zeros_like(top2[:, 0])
        return top2[:, 1] - top2[:, 0]

    qmd_margin = _margin(qmd)
    sed_margin = _margin(sed)
    score = disagreement * 10.0 + (1.0 / (1.0 + qmd_margin + sed_margin))
    score = score + 1e-3 * bank.sum(dim=-1).float()
    idx = torch.topk(score, k=min(n_states, bank.shape[0]), largest=True).indices
    selected = bank[idx]
    if selected.shape[0] < n_states:
        selected = selected.repeat((n_states + selected.shape[0] - 1) // selected.shape[0], 1)[:n_states]
    return selected[:n_states]
