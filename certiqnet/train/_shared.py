from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

import torch

from certiqnet.experiments.logging import BufferedExperimentLogger as ExperimentLogger
from certiqnet.experiments.paths import RunPaths


def validate_exact_certificate_constant(model: torch.nn.Module, *, context: str) -> float:
    constant = float(getattr(model, "C", float("inf")))
    if not math.isfinite(constant) or constant < 0:
        raise ValueError(
            f"Exact queueing {context} require a finite model.C. "
            "Update the CertiQ Index model config to provide a finite C."
        )
    return constant


def set_model_certificate_constant(cfg, constant: float) -> None:
    model_cfg = cfg.model
    if "geometry" in model_cfg and model_cfg.geometry is not None:
        model_cfg.geometry.C = constant
        return
    if "C" in model_cfg:
        model_cfg.C = constant
        return
    raise KeyError("Model config does not expose a geometry.C or C field.")


def failure_paths(
    *,
    cwd: Path,
    paths: RunPaths | None,
    stage: str,
) -> tuple[Path, Path]:
    if paths is not None:
        failure_dir = paths.metrics
    else:
        failure_dir = (cwd / "outputs" / "_uncaught_failures").resolve()
    failure_dir.mkdir(parents=True, exist_ok=True)
    return failure_dir / f"{stage}_failure.json", failure_dir / f"{stage}_traceback.txt"


def write_failure_artifacts(
    *,
    cwd: Path,
    paths: RunPaths | None,
    stage: str,
    error: BaseException,
    tb: str,
    logger: ExperimentLogger | None = None,
) -> None:
    failure_json, failure_txt = failure_paths(cwd=cwd, paths=paths, stage=stage)
    payload = {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": tb,
    }
    try:
        failure_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass
    try:
        failure_txt.write_text(tb, encoding="utf-8")
    except Exception:
        pass
    if logger is not None:
        try:
            logger.info(
                f"{stage}_failed",
                error_type=type(error).__name__,
                message=str(error),
                failure_json=str(failure_json),
                failure_traceback=str(failure_txt),
            )
        except Exception:
            pass
