"""Shared experiment execution pipelines for training and evaluation."""

from __future__ import annotations

import json
import random
import math
import shutil
import traceback
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from certiqnet.diagnostics.state_bank import generate_state_bank
from certiqnet.experiments.baseline_runner import RolloutConfig, run_baseline_comparison
from certiqnet.experiments.checkpoint_state import (
    load_checkpoint_weights,
    read_checkpoint_state,
    read_last_run,
    save_checkpoint_state,
    save_last_run,
)
from certiqnet.experiments.factory import build_model, build_mu
from certiqnet.experiments.logging import BufferedExperimentLogger as ExperimentLogger
from certiqnet.experiments.metrics import aggregate_metrics, save_metrics
from certiqnet.experiments.paths import RunPaths, slugify
from certiqnet.experiments.runner import experiment_name_from_cfg, prepare_run
from certiqnet.training.datamodule import CertiQNetDataModule
from certiqnet.training.lightning_module import CertiQNetLightningModule
from certiqnet.utils.platform import detect_platform, resolve_trainer_config
from certiqnet.utils.progress import configure_progress

try:
    import pytorch_lightning as pl
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("pytorch-lightning is required. Run `pip install -e .`.") from exc


def _validate_exact_certificate_constant(model: torch.nn.Module, *, context: str) -> float:
    """Return the finite certificate constant required for exact queueing runs."""
    constant = float(getattr(model, "C", float("inf")))
    if not math.isfinite(constant) or constant < 0:
        raise ValueError(
            f"Exact queueing {context} require a finite model.C. "
            "Update the CertiQ Dispatcher model config to provide a finite geometry.C."
        )
    return constant


def _set_model_certificate_constant(cfg: DictConfig, constant: float) -> None:
    """Write the exact-certificate constant back into the resolved config."""
    model_cfg = cfg.model
    if "geometry" in model_cfg and model_cfg.geometry is not None:
        model_cfg.geometry.C = constant
        return
    if "C" in model_cfg:
        model_cfg.C = constant
        return
    raise KeyError("Model config does not expose a geometry.C or C field.")


def _failure_paths(
    *,
    cwd: Path,
    paths: RunPaths | None,
    stage: str,
) -> tuple[Path, Path]:
    """Return stable paths for persisting failure artifacts."""
    if paths is not None:
        failure_dir = paths.metrics
    else:
        failure_dir = (cwd / "outputs" / "_uncaught_failures").resolve()
    failure_dir.mkdir(parents=True, exist_ok=True)
    return failure_dir / f"{stage}_failure.json", failure_dir / f"{stage}_traceback.txt"


def _write_failure_artifacts(
    *,
    cwd: Path,
    paths: RunPaths | None,
    stage: str,
    error: BaseException,
    tb: str,
    logger: ExperimentLogger | None = None,
) -> None:
    """Persist failure details without letting secondary errors hide the crash."""
    failure_json, failure_txt = _failure_paths(cwd=cwd, paths=paths, stage=stage)
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


def run_training(cfg: DictConfig, *, cwd: Path) -> None:
    """Run the full Lightning training pipeline from a resolved config."""
    paths: RunPaths | None = None
    run_logger: ExperimentLogger | None = None
    training_error: BaseException | None = None
    training_traceback: str | None = None
    seed = int(cfg.project.seed)
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        platform_info = detect_platform()

        if "progress" in cfg:
            configure_progress(OmegaConf.to_container(cfg.progress, resolve=True))

        if bool(cfg.get("_print_platform", True)):
            py_ver = platform_info.python_version.split()[0]
            gpu_str = f"{platform_info.gpu_count}x{platform_info.gpu_names}"
            cpu_str = f"{platform_info.cpu_count_logical}log/{platform_info.cpu_count_physical}phys"
            print(
                f"[platform] os={platform_info.os_name} python={py_ver} "
                f"torch={platform_info.torch_version} gpu={gpu_str} "
                f"cpu={cpu_str} precision={platform_info.recommended_precision}"
            )

        trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
        if not isinstance(trainer_cfg, dict):
            raise TypeError("cfg.trainer must resolve to a mapping")
        resolved_trainer = resolve_trainer_config(trainer_cfg, platform_info)

        if resolved_trainer["precision"] != str(cfg.trainer.precision):
            print(
                f"[platform] [warn] precision '{cfg.trainer.precision}' not supported, "
                f"falling back to '{resolved_trainer['precision']}'"
            )

        paths, run_logger = prepare_run(cfg, cwd=cwd)
        run_logger.info(
            "platform_info",
            os=platform_info.os_name,
            python=platform_info.python_version.split()[0],
            torch=platform_info.torch_version,
            gpu_count=platform_info.gpu_count,
            gpu_names=",".join(platform_info.gpu_names),
            cpu_logical=platform_info.cpu_count_logical,
            accelerator=resolved_trainer["accelerator"],
            precision=resolved_trainer["precision"],
        )

        adapter = instantiate(cfg.adapter) if "adapter" in cfg else None
        adapter_name = adapter.__class__.__name__ if adapter is not None else "QueueingAdapter"
        d_xi = int(getattr(adapter, "context_dim", 0))
        cert_status = str(getattr(adapter, "CERTIFICATE_STATUS", "exact"))
        assumptions_satisfied = bool(getattr(adapter, "assumptions_satisfied", False))
        cfg_cert_status = str(cfg.get("certificate_status", "exact"))

        if cfg_cert_status == "exact" and cert_status != "exact":
            raise ValueError(
                f"Constraint Violation: Config specifies 'exact' certification, but adapter "
                f"'{adapter_name}' provides '{cert_status}'. Update config to "
                f"'certificate_status=approximate'."
            )

        env_target = str(cfg.get("env", {}).get("_target_", "QueueingCTMC"))
        if cfg_cert_status == "exact" and "QueueingCTMC" not in env_target:
            raise ValueError(
                "Constraint Violation: Exact certification requires the QueueingCTMC backend. "
                f"Found backend target: {env_target}."
            )

        run_logger.info(
            "adapter_info",
            adapter=adapter_name,
            context_dim=d_xi,
            certificate_status=cert_status,
            assumptions_satisfied=assumptions_satisfied,
        )

        mu, _ = build_mu(cfg)
        model = build_model(cfg, N=int(cfg.env.N), d_xi=d_xi)
        if str(cfg.get("certificate_status", "exact")) == "exact":
            model_constant = _validate_exact_certificate_constant(model, context="training runs")
            _set_model_certificate_constant(cfg, model_constant)
            OmegaConf.save(config=cfg, f=paths.configs / "resolved_config.yaml", resolve=True)
        torch.save(model.state_dict(), paths.artifacts / "initial_model_state.pt")

        actual_batch_size = int(cfg.get("_batch_size", 64))
        num_workers_raw: object = resolved_trainer.get("_num_workers")
        num_workers: int | None = num_workers_raw if isinstance(num_workers_raw, int) else None
        max_queue = int(cfg.runner.get("max_queue", 15))
        resample_epoch = bool(cfg.runner.get("resample_every_epoch", True))
        dm = CertiQNetDataModule(
            N=int(cfg.env.N),
            mu=mu,
            batch_size=actual_batch_size,
            n_samples=512,
            num_workers=num_workers,
            adapter=adapter,
            seed=seed,
            max_queue=max_queue,
            resample_every_epoch=resample_epoch,
            policy_buffer_max=int(cfg.trainer.policy_buffer_max),
            synthetic_mix_fraction=float(cfg.trainer.synthetic_mix_fraction),
            teacher_mix_fraction=float(cfg.trainer.teacher_mix_fraction),
            policy_mix_fraction=float(cfg.trainer.policy_mix_fraction),
        )
        run_logger.info(
            "datamodule_info",
            batch_size=dm.batch_size,
            n_samples=dm.n_samples,
            num_workers=dm._num_workers,
            max_queue=dm.max_queue,
        )

        lightning = CertiQNetLightningModule(model, cfg)
        logger = instantiate(cfg.logger, save_dir=str(paths.logs)) if "logger" in cfg else False
        callbacks = [instantiate(cb) for cb in cfg.get("callbacks", {}).values()]

        if bool(cfg.project.get("save_checkpoints", True)):
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=str(paths.checkpoints),
                    filename="{epoch:04d}",
                    monitor="val/selection_score",
                    mode="min",
                    save_last=True,
                    save_top_k=3,
                    every_n_epochs=1,
                )
            )
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))

        if bool(cfg.runner.get("show_progress", True)):
            callbacks.append(
                pl.callbacks.TQDMProgressBar(refresh_rate=int(cfg.runner.get("refresh_rate", 10)))
            )

        trainer = pl.Trainer(
            max_epochs=int(cfg.trainer.max_epochs),
            accelerator=resolved_trainer["accelerator"],
            devices=resolved_trainer["devices"],
            precision=resolved_trainer["precision"],
            gradient_clip_val=float(cfg.trainer.gradient_clip_val),
            val_check_interval=float(cfg.trainer.val_check_interval),
            log_every_n_steps=int(cfg.trainer.log_every_n_steps),
            logger=logger,
            callbacks=callbacks,
            default_root_dir=str(paths.root),
        )

        resume = cfg.project.get("resume_from_checkpoint")
        run_logger.info("starting_training", checkpoint=str(resume) if resume else None)
        try:
            trainer.fit(lightning, datamodule=dm, ckpt_path=str(resume) if resume else None)
        finally:
            try:
                if logger is not False and hasattr(logger, "flush"):
                    try:
                        logger.flush()
                    except Exception:
                        pass
                if logger is not False and hasattr(logger, "log_dir"):
                    lightning_metrics = Path(str(getattr(logger, "log_dir")))
                    source_metrics = lightning_metrics / "metrics.csv"
                    if source_metrics.exists():
                        shutil.copyfile(source_metrics, paths.metrics / "training_metrics.csv")
            except Exception:
                pass

    except BaseException as exc:  # pragma: no cover - failure path
        training_error = exc
        training_traceback = traceback.format_exc()
        if run_logger is not None:
            try:
                run_logger.info(
                    "training_failed",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            except Exception:
                pass
        raise
    finally:
        if training_error is not None and training_traceback is not None:
            try:
                _write_failure_artifacts(
                    cwd=cwd,
                    paths=paths,
                    stage="training",
                    error=training_error,
                    tb=training_traceback,
                    logger=run_logger,
                )
            except Exception:
                pass

    ckpt_path = paths.artifacts / "final_model_state.pt"
    torch.save(model.state_dict(), ckpt_path)
    manifest_path = paths.root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["effective_certificate_constants"] = {"C": float(getattr(model, "C", float("inf")))}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    # The actual run_id and experiment_name are auto-generated inside
    # prepare_run and baked into paths.root (…/experiment_name/run_id).
    actual_run_id = paths.root.name
    actual_experiment_name = experiment_name_from_cfg(cfg)
    save_checkpoint_state(
        paths.root,
        ckpt_path,
        experiment_name=actual_experiment_name,
        run_id=actual_run_id,
        model_target=str(cfg.model._target_),
        seed=int(cfg.project.seed),
        max_epochs=int(cfg.trainer.max_epochs),
    )
    # Persist run identity at experiment level so evaluation scripts can
    # discover the correct output directory even when ``run_id`` is
    # auto-generated from a timestamp.
    save_last_run(
        paths.root.parent,
        run_id=actual_run_id,
        experiment_name=actual_experiment_name,
    )
    run_logger.info("finished_training", root=str(paths.root))


def _discover_and_prepare(
    cfg: DictConfig, *, cwd: Path
) -> tuple[RunPaths, ExperimentLogger]:
    """Prepare experiment directories, discovering the trained run if needed.

    Calls ``prepare_run`` to auto-generate paths from *cfg*.  If the
    checkpoint state does not exist at those paths (e.g. because training
    ran in a separate call with a different auto-generated ``run_id``),
    falls back to reading the experiment-level ``.last_run.json`` file.
    """
    paths, run_logger = prepare_run(cfg, cwd=cwd)
    if read_checkpoint_state(paths.root) is not None:
        return paths, run_logger

    # ---- discovery fallback ----
    output_root = Path(str(cfg.project.get("output_root", "outputs")))
    if not output_root.is_absolute():
        output_root = (cwd / output_root).resolve()
    experiment_name = experiment_name_from_cfg(cfg)
    experiment_root = output_root / slugify(experiment_name)

    last = read_last_run(experiment_root)
    if last is None:
        return paths, run_logger  # will fail later with a clear error

    # Re-run prepare_run with the discovered run_id so that the logger,
    # manifest, and config all point to the correct trained-run directory.
    if "project" not in cfg:
        cfg.project = OmegaConf.create()
    cfg.project.run_id = last["run_id"]
    cfg.project.experiment_name = experiment_name
    paths, run_logger = prepare_run(cfg, cwd=cwd)
    return paths, run_logger


def run_state_bank_audit(cfg: DictConfig, *, cwd: Path) -> None:
    """Run the state-bank audit and persist audit metrics."""
    platform_info = detect_platform()

    if "progress" in cfg:
        configure_progress(OmegaConf.to_container(cfg.progress, resolve=True))

    paths, run_logger = _discover_and_prepare(cfg, cwd=cwd)
    run_logger.info(
        "audit_platform_info",
        os=platform_info.os_name,
        torch=platform_info.torch_version,
        gpu=platform_info.gpu_count,
    )

    adapter = instantiate(cfg.adapter) if "adapter" in cfg else None
    d_xi = int(getattr(adapter, "context_dim", 0))
    mu, lam = build_mu(cfg)
    model = build_model(cfg, N=int(cfg.env.N), d_xi=d_xi)
    if str(cfg.get("certificate_status", "exact")) == "exact":
        _validate_exact_certificate_constant(model, context="audits")
    load_checkpoint_weights(model, paths.root)
    model.eval()

    run_logger.info("generating_state_bank")
    Q_bank = generate_state_bank(
        N=int(cfg.env.N),
        mu=mu,
        beta=float(getattr(model, "beta", 1.0)),
        R_cert=float(cfg.model.certificate.get("fallback_radius", float("inf"))),
        n_random=512,
        n_grid=128,
        n_boundary=128,
    )
    run_logger.info("state_bank_generated", states=Q_bank.shape[0])

    mu_bank = mu.unsqueeze(0).expand(Q_bank.shape[0], -1)
    if adapter is not None:
        Q_bank, mu_bank, xi_bank = adapter.make_observation(Q_bank, mu_bank)
    else:
        xi_bank = None
    with torch.no_grad():
        if hasattr(model, "reset_dispatch_state"):
            model.reset_dispatch_state()
        _, diag = model(Q_bank, mu_bank, xi_bank, certify=True, training_mode=False)

    violation = (diag.A_final - diag.B_Q).clamp(min=0.0)
    audit_metrics = aggregate_metrics(
        model_name=str(cfg.model._target_).split(".")[-1],
        env_name=str(cfg.env.mu_mode),
        seed=int(cfg.project.seed),
        lam=lam,
        queue_trace=Q_bank,
        cost_trace=Q_bank.sum(dim=-1),
        dt_trace=torch.ones(Q_bank.shape[0]),
        diagnostics=[diag],
    )
    save_metrics([audit_metrics], paths.audits, filename="state_bank_audit")
    run_logger.metric(audit_metrics.flat())
    run_logger.flush()

    print(f"states={Q_bank.shape[0]}")
    print(f"max_violation={violation.max().item():.6e}")
    print(f"violation_rate={(violation > 0).float().mean().item():.6e}")
    print(f"fallback_rate={diag.fallback_active.float().mean().item():.6e}")
    print(f"usage_open_rate={(diag.usage_final > 0.1).float().mean().item():.6e}")
    print(f"usage_mean={diag.usage_final.nanmean().item():.6e}")
    print(f"min_certificate_slack={diag.certificate_slack.min().item():.6e}")


def run_baseline_paper_comparison(cfg: DictConfig, *, cwd: Path) -> None:
    """Run the baseline comparison suite and print a summary table."""
    platform_info = detect_platform()

    if "progress" in cfg:
        configure_progress(OmegaConf.to_container(cfg.progress, resolve=True))

    paths, run_logger = _discover_and_prepare(cfg, cwd=cwd)
    run_logger.info(
        "baseline_platform_info",
        os=platform_info.os_name,
        torch=platform_info.torch_version,
        gpu=platform_info.gpu_count,
    )

    adapter = instantiate(cfg.adapter) if "adapter" in cfg else None
    d_xi = int(getattr(adapter, "context_dim", 0))
    mu, lam = build_mu(cfg)
    model = build_model(cfg, N=int(cfg.env.N), d_xi=d_xi)
    if str(cfg.get("certificate_status", "exact")) == "exact":
        _validate_exact_certificate_constant(model, context="baseline comparisons")
    load_checkpoint_weights(model, paths.root)
    rollout = RolloutConfig(
        steps=int(cfg.runner.rollout_steps),
        batch_size=int(cfg.runner.rollout_batch_size),
        max_backlog=float(cfg.runner.max_backlog),
        show_progress=bool(cfg.runner.show_progress),
    )
    metrics = run_baseline_comparison(
        env_name=str(cfg.env.mu_mode),
        N=int(cfg.env.N),
        lam=lam,
        mu=mu,
        seed=int(cfg.project.seed),
        output_dir=paths.metrics,
        rollout=rollout,
        extra_models={"configured_model": model},
        adapter=adapter,
    )
    for row in metrics:
        run_logger.metric(row.flat())
    run_logger.flush()
    run_logger.table("Baseline Comparison", [row.flat() for row in metrics])
