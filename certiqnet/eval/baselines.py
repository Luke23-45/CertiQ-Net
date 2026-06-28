from __future__ import annotations

import traceback
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from certiqnet.data.registry import DatasetRegistry
from certiqnet.eval._base import discover_and_prepare
from certiqnet.experiments.evaluators.baseline_runner import RolloutConfig, run_baseline_comparison
from certiqnet.experiments.persistence.checkpoint import load_checkpoint_weights
from certiqnet.experiments.persistence.checkpoint import infer_checkpoint_context_dim
from certiqnet.experiments.evaluators.factory import build_model, build_mu
from certiqnet.experiments.persistence.logging import BufferedExperimentLogger as ExperimentLogger
from certiqnet.experiments.persistence.paths import RunPaths
from certiqnet.train._shared import (
    validate_exact_certificate_constant,
    write_failure_artifacts,
)
from certiqnet.utils.platform import detect_platform
from certiqnet.utils.progress import configure_progress


def run_baseline_paper_comparison(cfg: DictConfig, *, cwd: Path) -> None:
    paths: RunPaths | None = None
    run_logger: ExperimentLogger | None = None
    baseline_error: BaseException | None = None
    baseline_traceback: str | None = None
    platform_info = detect_platform()

    if "progress" in cfg:
        configure_progress(OmegaConf.to_container(cfg.progress, resolve=True))

    try:
        paths, run_logger = discover_and_prepare(cfg, cwd=cwd)
        run_logger.info(
            "baseline_platform_info",
            os=platform_info.os_name,
            torch=platform_info.torch_version,
            gpu=platform_info.gpu_count,
        )

        adapter = instantiate(cfg.adapter) if "adapter" in cfg else None
        d_xi = int(getattr(adapter, "context_dim", 0))

        datatype = str(cfg.get("datatype", "qgym"))
        if datatype != "qgym":
            raise ValueError(f"Baseline stage requires datatype='qgym', got '{datatype}'.")
        profile = cfg.get(datatype)

        if datatype == "qgym":
            data_init_args = dict(profile.data.init_args) if profile is not None else {}
            dataset_name = data_init_args.get("dataset_name")
            if not dataset_name:
                raise ValueError("QGym profile must include data.init_args.dataset_name")
            spec = DatasetRegistry().get(dataset_name)
            N_bl = int(spec.env_N)
            mu = torch.tensor(spec.env_mu_fixed, dtype=torch.float32) if spec.env_mu_fixed is not None else build_mu(cfg)[0]
            lam = float(spec.env_lam) if spec.env_lam is not None else float(build_mu(cfg)[1])
        checkpoint_d_xi = infer_checkpoint_context_dim(paths.root)
        if checkpoint_d_xi > 0:
            d_xi = checkpoint_d_xi
        model = build_model(cfg, N=N_bl, d_xi=d_xi)
        if str(cfg.get("certificate_status", "exact")) == "exact":
            validate_exact_certificate_constant(model, context="baseline comparisons")
        load_checkpoint_weights(model, paths.root)
        rollout = RolloutConfig(
            steps=int(cfg.runner.rollout_steps),
            batch_size=int(cfg.runner.rollout_batch_size),
            max_backlog=float(cfg.runner.max_backlog),
            show_progress=bool(cfg.runner.show_progress),
        )

        baseline_include: list[str] | None = None
        baseline_exclude: list[str] | None = None
        try:
            bl_cfg = cfg.studies.runner.baselines
            raw_include = OmegaConf.to_container(bl_cfg.include, resolve=True)
            raw_exclude = OmegaConf.to_container(bl_cfg.exclude, resolve=True)
            if isinstance(raw_include, list):
                baseline_include = [str(x) for x in raw_include]
            if isinstance(raw_exclude, list) and raw_exclude:
                baseline_exclude = [str(x) for x in raw_exclude]
        except Exception:
            run_logger.info(
                "studies.runner config not found — using all baselines (default)",
            )

        run_logger.info(
            "baseline_filter",
            include=str(baseline_include),
            exclude=str(baseline_exclude),
        )

        bl_env_name = str(data_init_args.get("dataset_name", "")) if datatype == "qgym" else str(cfg.env.mu_mode)
        metrics = run_baseline_comparison(
            env_name=bl_env_name,
            N=N_bl,
            lam=lam,
            mu=mu,
            seed=int(cfg.project.seed),
            output_dir=paths.metrics,
            rollout=rollout,
            extra_models={"configured_model": model},
            adapter=adapter,
            qgym_test_path=cfg.runner.get("qgym_test_path"),
            include=baseline_include,
            exclude=baseline_exclude,
        )
        for row in metrics:
            run_logger.metric(row.flat())
        run_logger.flush()
        run_logger.table("Baseline Comparison", [row.flat() for row in metrics])

    except BaseException as exc:
        baseline_error = exc
        baseline_traceback = traceback.format_exc()
        if run_logger is not None:
            try:
                run_logger.info(
                    "baseline_failed",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            except Exception:
                pass
        print(f"\n[ERROR] Stage 'baselines' failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        if baseline_error is not None and baseline_traceback is not None:
            try:
                write_failure_artifacts(
                    cwd=cwd,
                    paths=paths,
                    stage="baselines",
                    error=baseline_error,
                    tb=baseline_traceback,
                    logger=run_logger,
                )
            except Exception:
                pass
        if run_logger is not None:
            try:
                run_logger.flush()
            except Exception:
                pass
