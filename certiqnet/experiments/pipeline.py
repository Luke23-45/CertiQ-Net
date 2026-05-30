"""Shared experiment execution pipelines for training and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from certiqnet.diagnostics.state_bank import generate_state_bank
from certiqnet.experiments.baseline_runner import RolloutConfig, run_baseline_comparison
from certiqnet.experiments.factory import build_model, build_mu
from certiqnet.experiments.metrics import aggregate_metrics, save_metrics
from certiqnet.experiments.runner import prepare_run
from certiqnet.training.datamodule import CertiQNetDataModule
from certiqnet.training.lightning_module import CertiQNetLightningModule
from certiqnet.utils.platform import detect_platform, resolve_trainer_config
from certiqnet.utils.progress import configure_progress

try:
    import pytorch_lightning as pl
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("pytorch-lightning is required. Run `pip install -e .`.") from exc


def run_training(cfg: DictConfig, *, cwd: Path) -> None:
    """Run the full Lightning training pipeline from a resolved config."""
    seed = int(cfg.project.seed)
    random.seed(seed)
    torch.manual_seed(seed)

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
        num_workers=resolved_trainer.get("_num_workers", 0),
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
    torch.save(model.state_dict(), paths.artifacts / "initial_model_state.pt")

    actual_batch_size = int(cfg.get("_batch_size", 64))
    num_workers_raw: object = resolved_trainer.get("_num_workers")
    num_workers: int | None = num_workers_raw if isinstance(num_workers_raw, int) else None
    dm = CertiQNetDataModule(
        N=int(cfg.env.N),
        mu=mu,
        batch_size=actual_batch_size,
        n_samples=512,
        num_workers=num_workers,
        adapter=adapter,
        seed=seed,
    )

    lightning = CertiQNetLightningModule(model, cfg)
    logger = instantiate(cfg.logger, save_dir=str(paths.logs)) if "logger" in cfg else False
    callbacks = [instantiate(cb) for cb in cfg.get("callbacks", {}).values()]

    if bool(cfg.project.get("save_checkpoints", True)):
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(paths.checkpoints),
                filename="{epoch:04d}",
                monitor="val/CERTIFICATE_VIOLATION",
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
    trainer.fit(lightning, datamodule=dm, ckpt_path=str(resume) if resume else None)
    torch.save(model.state_dict(), paths.artifacts / "final_model_state.pt")
    run_logger.info("finished_training", root=str(paths.root))


def run_state_bank_audit(cfg: DictConfig, *, cwd: Path) -> None:
    """Run the state-bank audit and persist audit metrics."""
    platform_info = detect_platform()

    if "progress" in cfg:
        configure_progress(OmegaConf.to_container(cfg.progress, resolve=True))

    paths, run_logger = prepare_run(cfg, cwd=cwd)
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
    model.eval()

    run_logger.info("generating_state_bank")
    Q_bank = generate_state_bank(
        N=int(cfg.env.N),
        mu=mu,
        beta=float(cfg.model.beta),
        R_cert=float(cfg.model.get("R_cert", float("inf"))),
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
        _, diag = model(Q_bank, mu_bank, xi_bank, training_mode=False)

    violation = (diag.A_pi - diag.B_Q).clamp(min=0.0)
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
    print(f"gate_rate={(diag.eta_final > 0).float().mean().item():.6e}")
    print(f"min_drift_slack={diag.drift_slack.min().item():.6e}")


def run_baseline_paper_comparison(cfg: DictConfig, *, cwd: Path) -> None:
    """Run the baseline comparison suite and print a summary table."""
    platform_info = detect_platform()

    if "progress" in cfg:
        configure_progress(OmegaConf.to_container(cfg.progress, resolve=True))

    paths, run_logger = prepare_run(cfg, cwd=cwd)
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
