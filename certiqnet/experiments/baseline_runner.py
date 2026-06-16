"""Baseline comparison runner with persisted dual metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from certiqnet.adapters.common.base import DispatchAdapter
from certiqnet.adapters.queueing.adapter import QueueingAdapter
from certiqnet.experiments.metrics import ExperimentMetrics, aggregate_metrics, save_metrics
from certiqnet.utils.progress import progress
from certiqnet.models.baselines import (
    AnalyticBackbonePolicy,
    CMuRule,
    MaxWeight,
    MaximumPressure,
    QuadraticMinDrift,
    RandomPolicy,
    ShortestExpectedDelay,
    SoftCMuRule,
    SoftMaxWeight,
)
from certiqnet.utils.ctmc import CTMCEnvironment


def _has_learnable_params(model: torch.nn.Module) -> bool:
    """Return True if the model has at least one learnable parameter."""
    return any(p.requires_grad for p in model.parameters())


def _greedy_pi(pi: torch.Tensor) -> torch.Tensor:
    """Convert a soft probability distribution to a one-hot argmax."""
    idx = pi.argmax(dim=-1)
    hard = torch.zeros_like(pi)
    hard.scatter_(1, idx.unsqueeze(-1), 1.0)
    return hard


@dataclass(frozen=True)
class RolloutConfig:
    """Rollout settings for local and cloud comparisons."""

    steps: int = 1000
    batch_size: int = 32
    max_backlog: float = 1e6
    show_progress: bool = True
    greedy_eval: bool = True


def build_baseline_suite(N: int, beta: float = 1.0) -> dict[str, torch.nn.Module]:
    """Return analytic baseline policies."""
    return {
        "random": RandomPolicy(N=N, beta=beta),
        "backbone": AnalyticBackbonePolicy(N=N, beta=beta),
        "sed": ShortestExpectedDelay(N=N, beta=beta),
        "qmd": QuadraticMinDrift(N=N, beta=beta),
        "c_mu": CMuRule(N=N, beta=beta),
        "soft_c_mu": SoftCMuRule(N=N, tau=1.0, beta=beta),
        "max_weight": MaxWeight(N=N, beta=beta),
        "soft_max_weight": SoftMaxWeight(N=N, tau=1.0, beta=beta),
        "max_pressure": MaximumPressure(N=N, beta=beta),
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
    qgym_test_states: torch.Tensor | None = None,
) -> ExperimentMetrics:
    """Run one CTMC rollout and return dual metrics.

    When ``qgym_test_states`` is provided, the CTMC is initialised to
    those queue lengths (drawn from the QGym test set) rather than
    starting from zero.
    """
    torch.manual_seed(seed)
    if hasattr(model, "reset_dispatch_state"):
        model.reset_dispatch_state()
    adapter = adapter if adapter is not None else QueueingAdapter(assumptions_satisfied=True)
    env = CTMCEnvironment(N=N, lam=lam, mu=mu, B=rollout.batch_size)
    # Initialise from QGym test states when available
    if qgym_test_states is not None:
        n_avail = qgym_test_states.shape[0]
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n_avail, generator=gen)[:rollout.batch_size]
        env.reset(qgym_test_states[idx].float().clone())
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
        # Use argmax dispatch for learned models to match hard baselines
        if rollout.greedy_eval and _has_learnable_params(model):
            pi = _greedy_pi(pi)
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
    qgym_test_path: str | Path | None = None,
) -> list[ExperimentMetrics]:
    """Evaluate baseline suite plus optional learned models and persist results.

    When ``qgym_test_path`` points to a ``test.pt`` file inside a
    ``dataset/qgym/<name>/`` directory, the rollouts start from the
    QGym test-state distribution instead of zero-initialised queues.
    """
    qgym_test_states: torch.Tensor | None = None
    if qgym_test_path is not None:
        test_path = Path(qgym_test_path)
        if test_path.is_file():
            data = torch.load(test_path, weights_only=True)
            qgym_test_states = data["Q"]
        else:
            test_file = test_path / "test" / "test.pt"
            if test_file.exists():
                data = torch.load(test_file, weights_only=True)
                qgym_test_states = data["Q"]
    models = build_baseline_suite(N=N)
    if extra_models:
        models.update(extra_models)
    import copy
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
            adapter=copy.deepcopy(adapter) if adapter is not None else None,
            qgym_test_states=qgym_test_states,
        )
        for name, model in models.items()
    ]
    save_metrics(metrics, output_dir)
    return metrics
