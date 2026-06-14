"""Certificate operators for z3 dispatch."""

import torch
from torch import Tensor




def arrival_coordinate(pi: Tensor, y: Tensor) -> Tensor:
    """Return ``sum_i pi_i y_i``."""
    assert pi.shape == y.shape, "pi and y must have identical shape."
    return (pi * y).sum(dim=-1)


def kl_project_linear(
    q: Tensor,
    cost: Tensor,
    budget: Tensor,
    *,
    iterations: int = 48,
    tolerance: float = 1e-10,
) -> tuple[Tensor, Tensor]:
    """Project ``q`` onto ``{p in simplex : E_p[cost] <= budget}`` via KL.

    Uses exponential search for the Lagrange multiplier upper bound, then
    bisection.  The solution is ``p_i = q_i * exp(-nu * c_i) / Z(nu)`` where
    ``nu`` is the Lagrange multiplier returned as the second output.

    Args:
        q: ``(B, N)`` proposal distribution (non-negative, rows sum to 1).
        cost: ``(B, N)`` positive cost vector (clamped to ``finfo.tiny``).
        budget: ``(B,)`` maximum allowable expected cost.
        iterations: bisection iterations (default 48).
        tolerance: feasibility tolerance (default 1e-10).

    Returns:
        p: ``(B, N)`` projected distribution.
        nu: ``(B,)`` Lagrange multiplier (zero when ``q`` is already feasible).
    """
    assert q.dim() == 2, "q must be (B, N)"
    assert q.shape == cost.shape, "q and cost must have the same shape"
    assert budget.dim() == 1 and budget.shape[0] == q.shape[0], "budget must be (B,)"

    work_dtype = torch.float64 if cost.dtype in {torch.float16, torch.bfloat16, torch.float32} else cost.dtype
    q_work = q.to(dtype=work_dtype)
    cost_work = cost.to(dtype=work_dtype)
    budget_work = budget.to(dtype=work_dtype)

    eps = torch.finfo(cost_work.dtype).tiny
    cost_work = cost_work.clamp_min(eps)
    budget_work = budget_work.clamp_min(eps)

    expected = (q_work * cost_work).sum(dim=-1)
    feasible = expected <= budget_work + tolerance

    p = q_work.clone()
    nu = torch.zeros(q.shape[0], device=q.device, dtype=work_dtype)

    mask = ~feasible
    if not mask.any():
        return p.to(dtype=q.dtype), nu.to(dtype=q.dtype)

    q_masked = q_work[mask]
    cost_masked = cost_work[mask]
    budget_masked = budget_work[mask]
    n_infeas = q_masked.shape[0]
    min_cost = cost_masked.min(dim=-1).values

    infeasible_budget = budget_masked < min_cost
    if infeasible_budget.any():
        argmin_idx = cost_masked.argmin(dim=-1)
        p_dirac = torch.zeros_like(q_masked)
        p_dirac.scatter_(1, argmin_idx.unsqueeze(-1), 1.0)
        p_mask = infeasible_budget.unsqueeze(-1).expand_as(q_masked)
        q_masked = torch.where(p_mask, p_dirac, q_masked)
        budget_masked = torch.where(infeasible_budget, min_cost, budget_masked)

    expected_masked = (q_masked * cost_masked).sum(dim=-1)
    feasible_masked = expected_masked <= budget_masked + tolerance
    if feasible_masked.all():
        p[mask] = q_masked
        return p.to(dtype=q.dtype), nu.to(dtype=q.dtype)

    bisect_mask = ~feasible_masked
    if not bisect_mask.any():
        p[mask] = q_masked
        return p.to(dtype=q.dtype), nu.to(dtype=q.dtype)

    q_bisect = q_masked[bisect_mask]
    cost_bisect = cost_masked[bisect_mask]
    budget_bisect = budget_masked[bisect_mask]
    n_bisect = q_bisect.shape[0]

    lo = torch.zeros(n_bisect, device=q.device, dtype=work_dtype)
    hi = torch.full((n_bisect,), 10.0, device=q.device, dtype=work_dtype)

    with torch.no_grad():
        for _ in range(32):
            logits = q_bisect.clamp_min(eps).log() - hi.unsqueeze(-1) * cost_bisect
            p_test = torch.softmax(logits, dim=-1)
            expected_test = (p_test * cost_bisect).sum(dim=-1)
            still_high = expected_test > budget_bisect
            hi = torch.where(still_high, hi * 2.0, hi)
            if not still_high.any():
                break

    for _ in range(iterations):
        nu_mid = (lo + hi) / 2.0
        logits = q_bisect.clamp_min(eps).log() - nu_mid.unsqueeze(-1) * cost_bisect
        p_mid = torch.softmax(logits, dim=-1)
        expected_mid = (p_mid * cost_bisect).sum(dim=-1)
        too_high = expected_mid > budget_bisect
        lo = torch.where(too_high, nu_mid, lo)
        hi = torch.where(too_high, hi, nu_mid)

    nu_final = hi
    logits = q_bisect.clamp_min(eps).log() - nu_final.unsqueeze(-1) * cost_bisect
    p_final = torch.softmax(logits, dim=-1)

    p_result = q_masked.clone()
    nu_result = torch.zeros(n_infeas, device=q.device, dtype=work_dtype)
    p_result[bisect_mask] = p_final
    nu_result[bisect_mask] = nu_final
    p[mask] = p_result
    nu[mask] = nu_result

    expected_final = (p * cost_work).sum(dim=-1)
    min_cost = cost_work.min(dim=-1).values
    effective_budget = torch.max(budget_work, min_cost)
    audit_violation = (expected_final - effective_budget).clamp_min(0.0)
    max_viol = audit_violation.max().item()
    audit_tol = max(tolerance + 1e-5, 1e-4)
    if max_viol > audit_tol:
        raise RuntimeError(
            f"KL projection violated constraint by {max_viol:.3e} > {audit_tol:.3e}"
        )

    return p.to(dtype=q.dtype), nu.to(dtype=q.dtype)


def policy_entropy(pi: Tensor) -> Tensor:
    """Return categorical entropy."""
    return -(pi * pi.clamp_min(1e-9).log()).sum(dim=-1)


def normalize_policy(pi: Tensor) -> Tensor:
    """Return a finite strictly positive simplex row for each batch item."""
    eps = torch.finfo(pi.dtype).tiny
    pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    row_sum = pi.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(pi, 1.0 / pi.shape[-1])
    pi = torch.where(row_sum > 0, pi / row_sum.clamp_min(eps), uniform)
    pi = pi.clamp_min(eps)
    return pi / pi.sum(dim=-1, keepdim=True)



