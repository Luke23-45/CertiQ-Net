"""Persistent checkpoint state manifest for cross-process model loading.

After training completes, ``save_checkpoint_state`` writes a small JSON
manifest to the experiment run directory plus an experiment-level
``.last_run.json`` that points to the latest trained run.  Downstream
evaluation functions use ``require_checkpoint_state`` to discover the
checkpoint path, or exit with a clear error if no trained checkpoint exists.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from certiqnet.utils.platform import windows_safe_path

_CHECKPOINT_STATE_FILE = ".checkpoint_state.json"
_LAST_RUN_FILE = ".last_run.json"


@dataclass(frozen=True)
class CheckpointState:
    experiment_name: str
    run_id: str
    checkpoint_path: str
    model_target: str
    seed: int
    max_epochs: int
    timestamp_utc: str
    status: str


def save_checkpoint_state(
    paths_root: Path,
    checkpoint_path: Path,
    *,
    experiment_name: str,
    run_id: str,
    model_target: str,
    seed: int,
    max_epochs: int,
) -> Path:
    state = CheckpointState(
        experiment_name=experiment_name,
        run_id=run_id,
        checkpoint_path=str(checkpoint_path.resolve()),
        model_target=model_target,
        seed=seed,
        max_epochs=max_epochs,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        status="completed",
    )
    state_file = paths_root / _CHECKPOINT_STATE_FILE
    with open(windows_safe_path(state_file), "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)
    return state_file


def read_checkpoint_state(paths_root: Path) -> CheckpointState | None:
    state_file = paths_root / _CHECKPOINT_STATE_FILE
    if not state_file.exists():
        return None
    try:
        with open(windows_safe_path(state_file), encoding="utf-8") as f:
            data = json.load(f)
        return CheckpointState(**data)
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        print(
            f"[error] Corrupted checkpoint state file: {state_file}\n"
            f"        {exc}",
            file=sys.stderr,
        )
        return None


def require_checkpoint_state(paths_root: Path) -> Path:
    state = read_checkpoint_state(paths_root)
    if state is None:
        print(
            f"[error] No trained checkpoint found.\n"
            f"        Expected state file: {paths_root / _CHECKPOINT_STATE_FILE}\n"
            f"        Run training first, or verify the experiment output root\n"
            f"        and run-id match the trained run.",
            file=sys.stderr,
        )
        sys.exit(1)

    ckpt = Path(state.checkpoint_path)
    if not ckpt.exists():
        print(
            f"[error] Checkpoint file referenced in state manifest does not exist:\n"
            f"        {ckpt}\n"
            f"        The file may have been moved or deleted.",
            file=sys.stderr,
        )
        sys.exit(1)

    return ckpt


def load_checkpoint_weights(model: torch.nn.Module, paths_root: Path) -> None:
    ckpt_path = require_checkpoint_state(paths_root)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)


# ── Experiment-level "last run" discovery ──────────────────────────────


def save_last_run(experiment_root: Path, *, run_id: str, experiment_name: str) -> Path:
    last_run_file = experiment_root / _LAST_RUN_FILE
    experiment_root.mkdir(parents=True, exist_ok=True)
    with open(windows_safe_path(last_run_file), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "experiment_name": experiment_name}, f, indent=2)
    return last_run_file


def read_last_run(experiment_root: Path) -> dict | None:
    last_run_file = experiment_root / _LAST_RUN_FILE
    if not last_run_file.exists():
        return None
    try:
        with open(windows_safe_path(last_run_file), encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, TypeError) as exc:
        print(
            f"[error] Corrupted last-run file: {last_run_file}\n"
            f"        {exc}",
            file=sys.stderr,
        )
        return None
