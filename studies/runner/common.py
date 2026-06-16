"""Shared helpers for study runner entrypoints.

This is the authoritative location for ``StudyRunnerSpec`` and
``run_study_family``.  The old ``studies.queue.runner.common`` module
re-exports from here for backwards compatibility.

Features over the original common.py:
  - Rich console output with timestamps and boxed stage headers
  - Per-stage elapsed time reporting
  - Multi-seed execution controlled via ``cfg.studies.runner.seeds``
  - ``failure_mode`` support: "stop" (default) or "skip" on seed failure
  - ``--dry-run`` support: prints the execution plan without running anything
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from certiqnet.experiments.pipeline import (
    run_baseline_paper_comparison,
    run_state_bank_audit,
    run_training,
)

try:
    from rich.console import Console
    from rich.rule import Rule
    from rich.text import Text

    _console = Console(highlight=False)
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False
    _console = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]  # studies/runner/ → project root
CONFIG_DIR = ROOT / "configs"


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StudyRunnerSpec:
    """Describe one dedicated study runner family."""

    config_name: str
    stages: tuple[str, ...]
    default_overrides: tuple[str, ...] = field(default_factory=tuple)


# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ts() -> str:
    """Return HH:MM:SS timestamp string."""
    return datetime.now().strftime("%H:%M:%S")


def _print(msg: str, *, style: str = "") -> None:
    if _HAS_RICH:
        _console.print(f"[dim][{_ts()}][/dim] {msg}", markup=True)
    else:
        print(f"[{_ts()}] {msg}")


def _rule(title: str = "", *, style: str = "bold cyan") -> None:
    if _HAS_RICH:
        _console.print(Rule(title=title, style=style))
    else:
        width = 72
        if title:
            pad = (width - len(title) - 2) // 2
            print("═" * pad + f" {title} " + "═" * (width - pad - len(title) - 2))
        else:
            print("═" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Config composition
# ─────────────────────────────────────────────────────────────────────────────


def compose_study_config(spec: StudyRunnerSpec, cli_overrides: Sequence[str]) -> DictConfig:
    """Compose a full Hydra config for a study family plus CLI overrides."""
    overrides = [*spec.default_overrides, *cli_overrides]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name=spec.config_name, overrides=list(overrides))


# ─────────────────────────────────────────────────────────────────────────────
# Stage runner
# ─────────────────────────────────────────────────────────────────────────────


def run_stage(
    name: str,
    fn: Callable[[], None],
    *,
    index: int,
    total: int,
    seed: int,
    config_name: str,
    dry_run: bool = False,
) -> None:
    """Run one study stage with Rich console framing and elapsed time."""
    import time

    label = f"Stage [{index}/{total}]: {name.capitalize()}"
    _rule(label, style="bold cyan")
    _print(
        f"[cyan]Seed:[/cyan] {seed}  [cyan]Config:[/cyan] {config_name}",
    )

    if dry_run:
        _print(f"[yellow]  (dry-run) Would execute stage: {name}[/yellow]")
        return

    t0 = time.monotonic()
    try:
        fn()
    except Exception:
        elapsed = time.monotonic() - t0
        _print(f"[red bold]✗ Stage [{index}/{total}] failed in {_fmt_elapsed(elapsed)}[/red bold]")
        traceback.print_exc()
        raise
    elapsed = time.monotonic() - t0
    _print(f"[green bold]✓ Stage [{index}/{total}] completed in {_fmt_elapsed(elapsed)}[/green bold]")


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed study family runner
# ─────────────────────────────────────────────────────────────────────────────


def run_study_family(
    spec: StudyRunnerSpec,
    cli_overrides: Sequence[str],
    *,
    dry_run: bool = False,
) -> None:
    """Compose config and execute the declared study stages, iterating seeds.

    Seeds and other runner settings are read from
    ``cfg.studies.runner`` (composed via ``configs/studies/runner/default.yaml``).
    Any CLI overrides take precedence.
    """
    import time

    cfg = compose_study_config(spec, cli_overrides)

    # ── Runner meta-settings ─────────────────────────────────────────────────
    runner_node = cfg.get("studies", {}).get("runner", {})
    if OmegaConf.is_config(runner_node):
        runner_cfg = OmegaConf.to_container(runner_node, resolve=True)
    elif isinstance(runner_node, dict):
        runner_cfg = runner_node
    else:
        runner_cfg = {}
    if not isinstance(runner_cfg, dict):
        runner_cfg = {}

    seeds: list[int] = list(runner_cfg.get("seeds", [42]))
    failure_mode: str = str(runner_cfg.get("failure_mode", "stop"))
    stages_override: list[str] | None = runner_cfg.get("stages", None)
    verbose: bool = bool(runner_cfg.get("verbose", True))
    tag: str | None = runner_cfg.get("tag", None)

    active_stages = list(stages_override) if stages_override else list(spec.stages)

    # ── Dry-run header ───────────────────────────────────────────────────────
    if dry_run:
        _rule("DRY RUN — Execution Plan", style="bold yellow")
        _print(f"[yellow]Config:[/yellow]  {spec.config_name}")
        _print(f"[yellow]Seeds:[/yellow]   {seeds}")
        _print(f"[yellow]Stages:[/yellow]  {active_stages}")
        _print(f"[yellow]Failure:[/yellow] {failure_mode}")
        if tag:
            _print(f"[yellow]Tag:[/yellow]     {tag}")
        _print("")

    # ── Validate stages ──────────────────────────────────────────────────────
    stage_fns_factory: dict[str, Callable[[DictConfig], Callable[[], None]]] = {
        "train": lambda c: (lambda: run_training(c, cwd=ROOT)),
        "audit": lambda c: (lambda: run_state_bank_audit(c, cwd=ROOT)),
        "baselines": lambda c: (lambda: run_baseline_paper_comparison(c, cwd=ROOT)),
    }
    for stage in active_stages:
        if stage not in stage_fns_factory:
            raise ValueError(
                f"Unknown study stage '{stage}'. Valid stages: {list(stage_fns_factory)}"
            )

    # ── Seed loop ────────────────────────────────────────────────────────────
    t_total_start = time.monotonic()
    seed_results: list[tuple[int, str]] = []  # (seed, "ok" | "failed")

    for seed in seeds:
        _rule(f"Seed {seed}", style="bold magenta")
        if tag:
            _print(f"[magenta]Tag: {tag}[/magenta]")

        # Re-compose config with the concrete seed override
        seed_overrides = [*cli_overrides, f"project.seed={seed}"]
        seed_cfg = compose_study_config(spec, seed_overrides)

        stage_fns = {
            name: factory(seed_cfg) for name, factory in stage_fns_factory.items()
        }

        seed_ok = True
        for idx, stage in enumerate(active_stages, start=1):
            try:
                run_stage(
                    stage,
                    stage_fns[stage],
                    index=idx,
                    total=len(active_stages),
                    seed=seed,
                    config_name=spec.config_name,
                    dry_run=dry_run,
                )
            except Exception:
                seed_ok = False
                if failure_mode == "stop":
                    raise
                # failure_mode == "skip" — log and continue to next seed
                _print(f"[red]Seed {seed} failed on stage '{stage}' — skipping to next seed.[/red]")
                break

        seed_results.append((seed, "ok" if seed_ok else "failed"))

    # ── Final summary ────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t_total_start
    _rule("Run Summary", style="bold green")
    _print(f"Total time: [bold]{_fmt_elapsed(total_elapsed)}[/bold]")
    for seed, status in seed_results:
        icon = "[green]✓[/green]" if status == "ok" else "[red]✗[/red]"
        _print(f"  {icon} Seed {seed}: {status}")
