"""Baseline comparison runner with persisted dual metrics."""

from __future__ import annotations

import copy
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from certiqnet.adapters.common.base import DispatchAdapter
from certiqnet.adapters.queueing.adapter import QueueingAdapter
from certiqnet.dispatcher.types import DispatcherDiagnostics
from certiqnet.experiments.evaluators.metrics import (
    CertificateMetrics,
    ExperimentMetrics,
    GenericDispatchMetrics,
    PerformanceMetrics,
    save_metrics,
)
from certiqnet.utils.progress import progress
from certiqnet.utils.qgym_rollout import (
    as_batch_tensor,
    build_qgym_rollout_env,
    extract_qgym_queues,
    qgym_action_from_pi,
)
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

try:
    from rich.console import Console
    _console = Console(highlight=False)
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    _console = None


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    if _HAS_RICH:
        _console.print(f"[dim][{ts}][/dim] {msg}", markup=True)
    else:
        print(f"[{ts}] {msg}")


def _has_learnable_params(model: torch.nn.Module) -> bool:
    """Return True if the model has at least one learnable parameter."""
    return any(p.requires_grad for p in model.parameters())


def _model_device(model: torch.nn.Module) -> torch.device:
    """Infer the primary device used by a model."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _greedy_pi(pi: torch.Tensor) -> torch.Tensor:
    """Convert a soft probability distribution to a one-hot argmax."""
    idx = pi.argmax(dim=-1)
    hard = torch.zeros_like(pi)
    hard.scatter_(1, idx.unsqueeze(-1), 1.0)
    return hard


def _policy_forward(
    model: torch.nn.Module,
    Q: torch.Tensor,
    mu: torch.Tensor,
    xi: torch.Tensor | None,
    *,
    training_mode: bool,
) -> tuple[torch.Tensor, DispatcherDiagnostics]:
    """Call either ``forward_full`` or the standard forward contract."""
    if hasattr(model, "forward_full"):
        out = model.forward_full(Q, mu, xi, training_mode=training_mode)
        return out.pi, out.diagnostics
    out = model(Q, mu, xi, training_mode=training_mode)
    if hasattr(out, "pi") and hasattr(out, "diagnostics"):
        return out.pi, out.diagnostics
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    raise TypeError(
        f"Model {type(model).__name__} returned unsupported output type {type(out).__name__}."
    )


@dataclass(frozen=True)
class RolloutConfig:
    """Rollout settings for local and cloud comparisons."""

    steps: int = 2000
    trajectories: int = 100
    max_backlog: float = 1e6
    show_progress: bool = True
    greedy_eval: bool = True


def build_baseline_suite(
    N: int,
    beta: float = 1.0,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, torch.nn.Module]:
    """Return analytic baseline policies, filtered by include/exclude lists.

    Args:
        N: Number of queues.
        beta: Service rate scaling factor.
        include: Whitelist of baseline names to include.  ``["*"]`` (default)
            means all baselines.  Any other list restricts to those names.
        exclude: Blacklist of names to drop after the include filter.

    Returns:
        Ordered dict mapping baseline name → model instance.
    """
    all_baselines: dict[str, torch.nn.Module] = {
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

    # ── Include filter ───────────────────────────────────────────────────────
    if include is None or include == ["*"] or include == ["all"]:
        selected = dict(all_baselines)
    else:
        unknown = set(include) - set(all_baselines)
        if unknown:
            raise ValueError(
                f"Unknown baseline name(s) in include list: {sorted(unknown)}. "
                f"Valid names: {list(all_baselines)}"
            )
        selected = {k: v for k, v in all_baselines.items() if k in include}

    # ── Exclude filter ───────────────────────────────────────────────────────
    if exclude:
        selected = {k: v for k, v in selected.items() if k not in exclude}

    return selected


def evaluate_policy(
    *,
    name: str,
    model: torch.nn.Module,
    env_name: str,
    qgym_env_config: str | Path | dict,
    seed: int,
    N: int,
    mu: torch.Tensor,
    rollout: RolloutConfig,
    adapter: DispatchAdapter | None = None,
    qgym_test_states: torch.Tensor | None = None,
) -> ExperimentMetrics:
    """Run QGym rollouts and return benchmark metrics.

    When ``qgym_test_states`` is provided, each trajectory is reset from
    the provided QGym test-state bank instead of a fresh environment reset.
    """
    torch.manual_seed(seed)
    if hasattr(model, "reset_dispatch_state"):
        model.reset_dispatch_state()
    adapter = adapter if adapter is not None else QueueingAdapter(assumptions_satisfied=True)
    rollout_device = _model_device(model)
    n_trajectories = max(int(rollout.trajectories), 1)
    trajectory_costs: list[float] = []
    trajectory_backlogs: list[float] = []
    trajectory_p95: list[float] = []
    trajectory_p99: list[float] = []
    trajectory_diverged: list[bool] = []
    all_diagnostics: list[DispatcherDiagnostics] = []
    evaluation_start = "qgym_test" if qgym_test_states is not None else "env_reset"
    test_states = qgym_test_states.float() if qgym_test_states is not None else None

    def _select_test_batch(states: Tensor, batch_size: int) -> Tensor:
        if states.shape[0] >= batch_size:
            return states[:batch_size]
        idx = torch.arange(batch_size, device=states.device) % states.shape[0]
        return states[idx]

    iterator = progress(
        range(1),
        total=1,
        desc=f"qgym:{name}",
        disable=not rollout.show_progress,
    )
    for _ in iterator:
        env = build_qgym_rollout_env(
            qgym_env_config,
            batch=n_trajectories,
            seed=seed,
            device=rollout_device,
        )
        if test_states is not None and test_states.numel() > 0:
            env.reset(_select_test_batch(test_states.to(device=rollout_device).clone(), n_trajectories))
        else:
            env.reset()

        queue_trace: list[Tensor] = []
        cost_trace: list[Tensor] = []
        dt_trace: list[Tensor] = []
        diagnostics = []
        diverged = torch.zeros(n_trajectories, dtype=torch.bool, device=rollout_device)

        with torch.inference_mode():
            for _ in range(rollout.steps):
                Q_raw = as_batch_tensor(
                    env.obs.queues if hasattr(env, "obs") else env.env_state.queues,
                    device=rollout_device,
                )
                Q_obs = Q_raw
                mu_obs = as_batch_tensor(mu, device=rollout_device)
                xi_obs = None
                if hasattr(adapter, "make_observation"):
                    Q_obs, mu_obs, xi_obs = adapter.make_observation(Q_obs, mu_obs)
                pi, diag = _policy_forward(model, Q_obs, mu_obs, xi_obs, training_mode=False)
                if rollout.greedy_eval and _has_learnable_params(model):
                    pi = _greedy_pi(pi)
                action = qgym_action_from_pi(pi, env.network, Q_raw, sample=False)
                step_obs, _reward, _done, _truncated, info = env.step(action)
                q_t = extract_qgym_queues(step_obs, info, device=rollout_device)
                cost_t = as_batch_tensor(info.get("cost", _reward), device=rollout_device).reshape(-1)
                dt_t = as_batch_tensor(info.get("event_time", 1.0), device=rollout_device).reshape(-1)
                queue_trace.append(Q_raw.detach().cpu())
                cost_trace.append(cost_t.detach().cpu())
                dt_trace.append(dt_t.detach().cpu())
                diagnostics.append(diag)
                diverged |= q_t.sum(dim=-1) > rollout.max_backlog
                if bool(diverged.all()):
                    break

        queue_trace_t = torch.stack(queue_trace, dim=0)
        cost_trace_t = torch.stack(cost_trace, dim=0)
        dt_trace_t = torch.stack(dt_trace, dim=0)
        backlog_t = queue_trace_t.sum(dim=-1).float()
        total_time = dt_trace_t.sum(dim=0).clamp_min(1e-9)
        trajectory_costs.extend(
            (cost_trace_t.sum(dim=0) / total_time).tolist()
        )
        trajectory_backlogs.extend(
            ((backlog_t * dt_trace_t).sum(dim=0) / total_time).tolist()
        )
        trajectory_p95.extend(backlog_t.quantile(0.95, dim=0).tolist())
        trajectory_p99.extend(backlog_t.quantile(0.99, dim=0).tolist())
        trajectory_diverged.extend(diverged.detach().cpu().tolist())
        all_diagnostics.extend(diagnostics)

    def _stderr(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        tensor = torch.tensor(values, dtype=torch.float32)
        return float(tensor.std(unbiased=False).div(len(values) ** 0.5).item())

    backlog_tensor = torch.tensor(trajectory_backlogs, dtype=torch.float32)
    cost_tensor = torch.tensor(trajectory_costs, dtype=torch.float32)
    if not all_diagnostics:
        raise RuntimeError("QGym evaluation produced no diagnostics.")
    slack_all = torch.cat([d.certificate_slack.detach().flatten().cpu() for d in all_diagnostics])
    proposal_slack_all = torch.cat(
        [(d.B_Q.detach() - d.A_proposal.detach()).flatten().cpu() for d in all_diagnostics]
    )
    usage_final_all = torch.cat([d.usage_final.detach().flatten().cpu() for d in all_diagnostics])
    finite_usage = usage_final_all[torch.isfinite(usage_final_all)]
    fallback_all = torch.cat([d.fallback_flag.detach().flatten().cpu().to(torch.bool) for d in all_diagnostics])
    projection_active_all = torch.cat([d.projection_active.detach().flatten().cpu().to(torch.bool) for d in all_diagnostics])
    projection_nu_all = torch.cat([d.projection_multiplier.detach().flatten().cpu() for d in all_diagnostics])
    correction_all = torch.cat([d.correction_magnitude.detach().flatten().cpu() for d in all_diagnostics])
    pressure_mean_all = torch.cat([d.pressure_mean.detach().flatten().cpu() for d in all_diagnostics])
    pressure_max_all = torch.cat([d.pressure_max.detach().flatten().cpu() for d in all_diagnostics])
    pressure_update_norm_all = torch.cat([d.pressure_update_norm.detach().flatten().cpu() for d in all_diagnostics])

    performance = PerformanceMetrics(
        generic=GenericDispatchMetrics(
            avg_cost=float(cost_tensor.mean().item()),
            drop_rate=0.0,
        ),
        queueing={
            "avg_queue_length": float(backlog_tensor.mean().item()),
            "std_error_cost": _stderr(trajectory_costs),
            "std_error_queue": _stderr(trajectory_backlogs),
            "p95_backlog": float(torch.tensor(trajectory_p95, dtype=torch.float32).mean().item()),
            "p99_backlog": float(torch.tensor(trajectory_p99, dtype=torch.float32).mean().item()),
            "divergence_rate": float(sum(trajectory_diverged) / len(trajectory_diverged)),
            "trajectories": float(n_trajectories),
            "horizon": float(rollout.steps),
        },
    )
    certificate = CertificateMetrics(
        certificate_violation_rate=float((slack_all < -1e-5).float().mean().item()),
        min_certificate_slack=float(slack_all.min().item()),
        avg_certificate_slack=float(slack_all.mean().item()),
        projection_activation_rate=float(projection_active_all.float().mean().item()),
        projection_dual_mean=float(projection_nu_all.mean().item()),
        proposal_slack_min=float(proposal_slack_all.min().item()),
        proposal_slack_mean=float(proposal_slack_all.mean().item()),
        tail_fallback_activation_rate=float(fallback_all.float().mean().item()),
        usage_activation_rate=float((finite_usage > 0.1).float().mean().item()) if finite_usage.numel() else float("nan"),
        usage_mean_activation=float(finite_usage.mean().item()) if finite_usage.numel() else float("nan"),
        correction_magnitude=float(correction_all.max().item()),
        instability_rate=float(sum(trajectory_diverged) / len(trajectory_diverged)),
        pressure_mean=float(pressure_mean_all.mean().item()),
        pressure_max=float(pressure_max_all.max().item()),
        pressure_update_norm=float(pressure_update_norm_all.mean().item()),
    )
    return ExperimentMetrics(
        model_name=name,
        env_name=env_name,
        seed=seed,
        performance=performance,
        certificate=certificate,
        evaluation_start=evaluation_start,
        greedy_eval=rollout.greedy_eval and _has_learnable_params(model),
    )


def run_baseline_comparison(
    *,
    env_name: str,
    qgym_env_config: str | Path | dict,
    N: int,
    mu: torch.Tensor,
    seed: int,
    output_dir: Path,
    rollout: RolloutConfig,
    extra_models: dict[str, torch.nn.Module] | None = None,
    adapter: DispatchAdapter | None = None,
    qgym_test_path: str | Path | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[ExperimentMetrics]:
    """Evaluate baseline suite plus optional learned models and persist results.

    When ``qgym_test_path`` points to a ``test.pt`` file inside a
    ``dataset/qgym/<name>/`` directory, the rollouts start from the
    QGym test-state distribution instead of zero-initialised queues.

    Args:
        include: Whitelist of baseline names (e.g. ``["sed", "max_weight"]``).
            ``None`` or ``["*"]`` selects all baselines.
        exclude: Blacklist of baseline names to drop (applied after include).
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

    models = build_baseline_suite(N=N, include=include, exclude=exclude)
    if extra_models:
        models.update(extra_models)

    total = len(models)
    metrics: list[ExperimentMetrics] = []
    for idx, (name, model) in enumerate(models.items(), start=1):
        model_type = type(model).__name__
        use_greedy = rollout.greedy_eval and _has_learnable_params(model)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) if use_greedy else 0
        desc = f"type={model_type}"
        if use_greedy:
            desc += f" greedy params={n_params}"
        _log(f"── Baseline [{idx}/{total}] {name} ── ({desc})")
        try:
            result = evaluate_policy(
                name=name,
                model=model,
                env_name=env_name,
                qgym_env_config=qgym_env_config,
                seed=seed,
                N=N,
                mu=mu,
                rollout=rollout,
                adapter=copy.deepcopy(adapter) if adapter is not None else None,
                qgym_test_states=qgym_test_states,
            )
            flat = result.flat()
            cost_val = flat.get("avg_cost")
            vr_val = flat.get("certificate_violation_rate")
            parts = []
            if isinstance(cost_val, (int, float)):
                parts.append(f"cost={cost_val:.4f}")
            if isinstance(vr_val, (int, float)):
                parts.append(f"violation_rate={vr_val:.4f}")
            _log(f"── Baseline [{idx}/{total}] {name} ── {'  '.join(parts)}")
            metrics.append(result)
        except Exception:
            _log(f"-- Baseline [{idx}/{total}] {name} -- (FAILED)")
            print(f"\n[ERROR] Exception in baseline '{name}':", file=sys.stderr)
            traceback.print_exc()
            print("\nContinuing to next baseline...", file=sys.stderr)

    save_metrics(metrics, output_dir)
    return metrics
