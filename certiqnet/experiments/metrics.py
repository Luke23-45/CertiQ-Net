"""Dual performance and certificate metric aggregation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from certiqnet.dispatcher.types import DispatcherDiagnostics
from certiqnet.utils.platform import windows_safe_path


@dataclass(frozen=True)
class GenericDispatchMetrics:
    """Core metrics applying to all dispatch domains."""
    avg_cost: float
    drop_rate: float

@dataclass(frozen=True)
class PerformanceMetrics:
    """Multi-layer performance metrics."""
    generic: GenericDispatchMetrics
    queueing: dict[str, float] | None = None
    domain: dict[str, float] | None = None


@dataclass(frozen=True)
class CertificateMetrics:
    """Mandatory certificate/safety metrics."""

    certificate_violation_rate: float
    min_certificate_slack: float
    avg_certificate_slack: float
    projection_activation_rate: float
    projection_dual_mean: float
    projection_slack_min: float
    projection_slack_mean: float
    tail_fallback_activation_rate: float
    usage_activation_rate: float
    usage_mean_activation: float
    correction_magnitude: float
    instability_rate: float
    pressure_mean: float
    pressure_max: float
    pressure_update_norm: float


@dataclass(frozen=True)
class ExperimentMetrics:
    """Combined result row for one policy on one experiment."""

    model_name: str
    env_name: str
    seed: int
    performance: PerformanceMetrics
    certificate: CertificateMetrics

    def flat(self) -> dict[str, float | int | str]:
        """Return a single-level row for CSV/paper tables."""
        row: dict[str, float | int | str] = {
            "model_name": self.model_name,
            "env_name": self.env_name,
            "seed": self.seed,
        }

        def _flatten(prefix: str, d: dict[str, Any]) -> None:
            for k, v in d.items():
                if isinstance(v, dict):
                    _flatten(f"{prefix}{k}.", v)
                elif v is not None:
                    row[f"{prefix}{k}"] = v
                    if prefix.startswith("performance.") or prefix.startswith("certificate."):
                        row.setdefault(k, v)

        _flatten("performance.", asdict(self.performance))
        _flatten("certificate.", asdict(self.certificate))
        return row


def aggregate_metrics(
    *,
    model_name: str,
    env_name: str,
    seed: int,
    lam: float,
    queue_trace: Tensor,
    cost_trace: Tensor,
    dt_trace: Tensor,
    diagnostics: list[DispatcherDiagnostics],
    diverged: bool = False,
    drop_count: int = 0,
    arrivals: int = 0,
) -> ExperimentMetrics:
    """Aggregate event traces into the required dual-column metric set."""
    weights = dt_trace.clamp_min(1e-9)
    total_time = weights.sum().clamp_min(1e-9)
    backlog = queue_trace.sum(dim=-1)
    weighted_backlog = (backlog * weights).sum() / total_time
    weighted_cost = (cost_trace * weights).sum() / total_time
    slack = torch.cat([d.certificate_slack.detach().flatten().cpu() for d in diagnostics])
    usage_final = torch.cat([d.usage_final.detach().flatten().cpu() for d in diagnostics])
    fallback = torch.cat([d.fallback_active.detach().flatten().cpu() for d in diagnostics])
    projection_active = torch.cat([d.projection_active.detach().flatten().cpu() for d in diagnostics])
    projection_nu = torch.cat([d.projection_nu.detach().flatten().cpu() for d in diagnostics])
    projection_slack = torch.cat([d.projection_slack.detach().flatten().cpu() for d in diagnostics])
    correction = torch.cat([d.correction_magnitude.detach().flatten().cpu() for d in diagnostics])
    pressure_mean = torch.cat([d.pressure_mean.detach().flatten().cpu() for d in diagnostics])
    pressure_max = torch.cat([d.pressure_max.detach().flatten().cpu() for d in diagnostics])
    pressure_update_norm = torch.cat(
        [d.pressure_update_norm.detach().flatten().cpu() for d in diagnostics]
    )
    generic = GenericDispatchMetrics(
        avg_cost=float(weighted_cost.item()),
        drop_rate=float(drop_count / max(arrivals, 1)),
    )
    
    queueing = {
        "avg_queue_length": float(weighted_backlog.item()),
        "latency_proxy": float((weighted_backlog / max(lam, 1e-9)).item()),
        "p95_backlog": float(torch.quantile(backlog.detach().cpu().float(), 0.95).item()),
        "p99_backlog": float(torch.quantile(backlog.detach().cpu().float(), 0.99).item()),
        "max_backlog": float(backlog.max().item()),
    }
    
    performance = PerformanceMetrics(
        generic=generic,
        queueing=queueing,
    )
    certificate = CertificateMetrics(
        certificate_violation_rate=float((slack < -1e-5).float().mean().item()),
        min_certificate_slack=float(slack.min().item()),
        avg_certificate_slack=float(slack.mean().item()),
        projection_activation_rate=float(projection_active.float().mean().item()),
        projection_dual_mean=float(projection_nu.mean().item()),
        projection_slack_min=float(projection_slack.min().item()),
        projection_slack_mean=float(projection_slack.mean().item()),
        tail_fallback_activation_rate=float(fallback.float().mean().item()),
        usage_activation_rate=float((usage_final > 0.1).float().mean().item()),
        usage_mean_activation=float(usage_final.mean().item()),
        correction_magnitude=float(correction.max().item()),
        instability_rate=float(1.0 if diverged else 0.0),
        pressure_mean=float(pressure_mean.mean().item()),
        pressure_max=float(pressure_max.max().item()),
        pressure_update_norm=float(pressure_update_norm.mean().item()),
    )
    return ExperimentMetrics(
        model_name=model_name,
        env_name=env_name,
        seed=seed,
        performance=performance,
        certificate=certificate,
    )


def save_metrics(
    metrics: list[ExperimentMetrics],
    output_dir: Path,
    filename: str = "results",
) -> None:
    """Persist metrics as JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{filename}.json"
    csv_path = output_dir / f"{filename}.csv"
    with open(windows_safe_path(json_path), "w", encoding="utf-8") as f:
        f.write(json.dumps([asdict(metric) for metric in metrics], indent=2, sort_keys=True))
    rows = [metric.flat() for metric in metrics]
    if not rows:
        return
    with open(windows_safe_path(csv_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Keep a stable compatibility alias for downstream tools that expect
    # a canonical metrics.csv artifact in the run metrics directory.
    if filename == "results":
        alias_csv_path = output_dir / "metrics.csv"
        alias_json_path = output_dir / "metrics.json"
        with open(windows_safe_path(csv_path), encoding="utf-8") as f_in:
            csv_text = f_in.read()
        with open(windows_safe_path(json_path), encoding="utf-8") as f_in:
            json_text = f_in.read()
        with open(windows_safe_path(alias_csv_path), "w", encoding="utf-8") as f_out:
            f_out.write(csv_text)
        with open(windows_safe_path(alias_json_path), "w", encoding="utf-8") as f_out:
            f_out.write(json_text)
