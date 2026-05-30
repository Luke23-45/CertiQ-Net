"""Deterministic experiment output paths and manifests."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class RunPaths:
    """Dedicated folders for one experiment run."""

    root: Path
    checkpoints: Path
    artifacts: Path
    logs: Path
    metrics: Path
    audits: Path
    configs: Path


@dataclass(frozen=True)
class RunManifest:
    """Run identity and reproducibility metadata persisted as JSON."""

    run_id: str
    experiment_family: str
    experiment_name: str
    domain: str
    adapter: str
    backend: str
    model_category: str
    certificate_status: str
    assumptions_status: bool
    seed: int
    metric_schema: str
    created_at_utc: str
    git_commit: str | None
    git_dirty: bool
    command: list[str]
    paths: dict[str, str]


def slugify(value: str) -> str:
    """Return a filesystem-safe experiment slug."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in cleaned.split("-") if part)


def utc_timestamp() -> str:
    """Return sortable UTC timestamp for run IDs."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit(cwd: Path) -> str | None:
    """Return current git commit hash when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_dirty(cwd: Path) -> bool:
    """Return whether the worktree has uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(result.stdout.strip())


def make_run_id(experiment_name: str, seed: int) -> str:
    """Return a stable human-readable run ID."""
    return f"{utc_timestamp()}_{slugify(experiment_name)}_seed{seed}"


def create_run_paths(output_root: Path, experiment_name: str, run_id: str) -> RunPaths:
    """Create and return dedicated run artifact folders."""
    root = output_root / slugify(experiment_name) / run_id
    paths = RunPaths(
        root=root,
        checkpoints=root / "checkpoints",
        artifacts=root / "artifacts",
        logs=root / "logs",
        metrics=root / "metrics",
        audits=root / "audits",
        configs=root / "configs",
    )
    for path in asdict(paths).values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths


def save_resolved_config(cfg: DictConfig, paths: RunPaths) -> Path:
    """Persist the resolved Hydra config for reproducibility."""
    out = paths.configs / "resolved_config.yaml"
    OmegaConf.save(config=cfg, f=out, resolve=True)
    return out


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON with parent creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_manifest(
    cfg: DictConfig,
    paths: RunPaths,
    experiment_name: str,
    run_id: str,
    cwd: Path,
) -> RunManifest:
    """Persist run manifest and return it."""
    experiment_family = str(cfg.get("experiment_family", "unknown"))
    domain = str(cfg.get("domain", "queueing"))
    adapter = str(cfg.get("adapter", {}).get("_target_", "QueueingAdapter")).split(".")[-1] if cfg.get("adapter") else "QueueingAdapter"
    backend = str(cfg.get("env", {}).get("_target_", "QueueingCTMC")).split(".")[-1] if cfg.get("env") else "QueueingCTMC"
    model_category = str(cfg.get("model", {}).get("_target_", "Unknown")).split(".")[-1] if cfg.get("model") else "Unknown"
    certificate_status = str(cfg.get("certificate_status", "exact"))
    assumptions_status = bool(cfg.get("assumptions_satisfied", False))
    seed = int(cfg.get("project", {}).get("seed", 42)) if cfg.get("project") else 42
    metric_schema = str(cfg.get("metric_schema", "queueing"))

    manifest = RunManifest(
        run_id=run_id,
        experiment_family=experiment_family,
        experiment_name=experiment_name,
        domain=domain,
        adapter=adapter,
        backend=backend,
        model_category=model_category,
        certificate_status=certificate_status,
        assumptions_status=assumptions_status,
        seed=seed,
        metric_schema=metric_schema,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit(cwd),
        git_dirty=git_dirty(cwd),
        command=sys.argv,
        paths={key: str(value) for key, value in asdict(paths).items()},
    )
    save_json(paths.root / "manifest.json", asdict(manifest))
    return manifest
