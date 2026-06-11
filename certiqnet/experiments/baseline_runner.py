"""Baseline comparison runner with persisted dual metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from certiqnet.adapters.base import DispatchAdapter
from certiqnet.adapters.queueing import QueueingAdapter
from certiqnet.experiments.metrics import ExperimentMetrics, aggregate_metrics, save_metrics
from certiqnet.experiments.progress import progress
from certiqnet.models.baselines import (
    AnalyticBackbonePolicy,
    JoinShortestWeightedQueue,
    QuadraticMinDrift,
    RandomPolicy,
    ShortestExpectedDelay,
    SoftQuadraticMinDrift,
    SoftSED,
)
from certiqnet.simulation.ctmc import CTMCEnvironment


@dataclass(frozen=True)
class RolloutConfig:
    """Rollout settings for local and cloud comparisons."""

    steps: int = 1000
    batch_size: int = 32
    max_backlog: float = 1e6
    show_progress: bool = True


def build_baseline_suite(N: int, beta: float = 1.0) -> dict[str, torch.nn.Module]:
    """Return required analytic baseline policies."""
    return {
        "random": RandomPolicy(N=N, beta=beta),
        "jswq": JoinShortestWeightedQueue(N=N, beta=beta),
        "backbone": AnalyticBackbonePolicy(N=N, beta=beta),
        "sed": ShortestExpectedDelay(N=N, beta=beta),
        "qmd": QuadraticMinDrift(N=N, beta=beta),
        "soft_sed": SoftSED(N=N, tau=1.0, beta=beta),
        "soft_qmd": SoftQuadraticMinDrift(N=N, tau=1.0, beta=beta),
    }


def evaluate_policy(
    *,
    name: str,
    model: torch.nn.Module,
    env_name: str,
    seed: int,
    N: int,
    lam: float,
    mu: torch.Tensor,
    rollout: RolloutConfig,
    adapter: DispatchAdapter | None = None,
) -> ExperimentMetrics:
    """Run one CTMC rollout and return dual metrics."""
    torch.manual_seed(seed)
    if hasattr(model, "reset_dispatch_state"):
        model.reset_dispatch_state()
    adapter = adapter if adapter is not None else QueueingAdapter(assumptions_satisfied=True)
    env = CTMCEnvironment(N=N, lam=lam, mu=mu, B=rollout.batch_size)
    queue_trace: list[torch.Tensor] = []
    cost_trace: list[torch.Tensor] = []
    dt_trace: list[torch.Tensor] = []
    diagnostics = []
    diverged = False
    iterator = progress(
        range(rollout.steps),
        total=rollout.steps,
        desc=f"rollout:{name}",
        disable=not rollout.show_progress,
    )
    for _ in iterator:
        mu_b = mu.unsqueeze(0).expand(env.B, -1)
        Q_obs, mu_obs, xi_obs = adapter.make_observation(env.Q, mu_b)
        with torch.no_grad():
            pi, diag = model(Q_obs, mu_obs, xi_obs, training_mode=False)
        out = env.step(pi)
        queue_trace.append(out["Q"].detach().cpu())
        cost_trace.append(out["cost"].detach().cpu())
        dt_trace.append(out["dt"].detach().cpu())
        diagnostics.append(diag)
        if out["Q"].sum(dim=-1).max().item() > rollout.max_backlog:
            diverged = True
            break
    return aggregate_metrics(
        model_name=name,
        env_name=env_name,
        seed=seed,
        lam=lam,
        queue_trace=torch.cat(queue_trace, dim=0),
        cost_trace=torch.cat(cost_trace, dim=0),
        dt_trace=torch.cat(dt_trace, dim=0),
        diagnostics=diagnostics,
        diverged=diverged,
        arrivals=rollout.steps * rollout.batch_size,
    )


def run_baseline_comparison(
    *,
    env_name: str,
    N: int,
    lam: float,
    mu: torch.Tensor,
    seed: int,
    output_dir: Path,
    rollout: RolloutConfig,
    extra_models: dict[str, torch.nn.Module] | None = None,
    adapter: DispatchAdapter | None = None,
) -> list[ExperimentMetrics]:
    """Evaluate baseline suite plus optional learned models and persist results."""
    models = build_baseline_suite(N=N)
    if extra_models:
        models.update(extra_models)
    metrics = [
        evaluate_policy(
            name=name,
            model=model,
            env_name=env_name,
            seed=seed,
            N=N,
            lam=lam,
            mu=mu,
            rollout=rollout,
            adapter=adapter,
        )
        for name, model in models.items()
    ]
    save_metrics(metrics, output_dir)
    return metrics
