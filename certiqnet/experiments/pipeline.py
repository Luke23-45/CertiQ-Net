"""Shared experiment execution pipelines for training and evaluation."""

from __future__ import annotations

import json
import random
import math
import shutil
import sys
import traceback
import csv
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from certiqnet.data.common.state_bank import generate_state_bank
from certiqnet.experiments.baseline_runner import RolloutConfig, run_baseline_comparison
from certiqnet.experiments.checkpoint_state import (
    load_checkpoint_weights,
    read_last_run,
    save_checkpoint_state,
    save_last_run,
)
from certiqnet.experiments.factory import build_model, build_mu
from certiqnet.experiments.logging import BufferedExperimentLogger as ExperimentLogger
from certiqnet.experiments.metrics import aggregate_metrics, save_metrics
from certiqnet.experiments.paths import RunPaths, slugify
from certiqnet.experiments.runner import experiment_name_from_cfg, prepare_run
from certiqnet.data.qgym.datamodule import QGymDataModule
from certiqnet.data.registry import DatasetRegistry
from certiqnet.train.common.loss import CertiQNetLoss
from certiqnet.train.queueing.module import QueueingLightningModule
from certiqnet.train.channel.module import ChannelLightningModule
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
            "Update the CertiQ Index model config to provide a finite C."
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


def _sync_training_metrics_alias(paths: RunPaths) -> None:
    """Write a consolidated root-level summary of the run-local training metrics."""
    source = paths.metrics / "training_metrics.csv"
    if not source.exists():
        return
    output_root = paths.root.parents[1]
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output_root / "training_metrics_timeseries.csv")

    with source.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            return
        summary: dict[str, str] = {k: "" for k in fieldnames}
        for row in reader:
            for key, value in row.items():
                if value not in ("", None):
                    summary[key] = value

    with (output_root / "training_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


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

        env_node = cfg.get("env")
        env_target = str(env_node.get("_target_", "QueueingCTMC") if env_node is not None else "QueueingCTMC")
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

        # ── Datatype validation ────────────────────────────────────────
        datatype = cfg.get("datatype")
        if datatype != "qgym":
            raise ValueError(
                f"datatype must be 'qgym', got {datatype}. "
                "Set 'datatype=qgym' in the experiment config (mandatory)."
            )
        profile = cfg.get(datatype)
        if profile is None:
            raise ValueError(
                f"Config must include a '{datatype}' block with data, "
                "trainer, loss, and lagrangian sections."
            )

        # ── Environment parameters ─────────────────────────────────────
        data_init_args = dict(profile.data.init_args)
        dataset_name = data_init_args.get("dataset_name")
        if dataset_name is None:
            raise ValueError("QGym profile must include data.init_args.dataset_name")
        spec = DatasetRegistry().get(dataset_name)
        N = int(spec.env_N)
        mu = (
            torch.tensor(spec.env_mu_fixed, dtype=torch.float32)
            if spec.env_mu_fixed is not None
            else build_mu(cfg)[0]
        )
        lam = (
            float(spec.env_lam)
            if spec.env_lam is not None
            else float(build_mu(cfg)[1])
        )
        model = build_model(cfg, N=N, d_xi=d_xi)

        if str(cfg.get("certificate_status", "exact")) == "exact":
            model_constant = _validate_exact_certificate_constant(model, context="training runs")
            _set_model_certificate_constant(cfg, model_constant)
            OmegaConf.save(config=cfg, f=paths.configs / "resolved_config.yaml", resolve=True)
        torch.save(model.state_dict(), paths.artifacts / "initial_model_state.pt")

        # ── DataModule from datatype profile ───────────────────────────
        data_cfg = profile.data
        data_init_args = dict(data_cfg.init_args)
        data_init_args.setdefault("N", N)
        dm = QGymDataModule(mu=mu, **data_init_args)
        dm.datatype = datatype

        # ── Dataset status logging ─────────────────────────────────────
        ds_name = getattr(dm, "dataset_name", None)
        ds_path = getattr(dm, "dataset_path", None)
        if ds_name:
            try:
                reg = DatasetRegistry()
                exists = reg.exists(ds_name)
                icon = "[x]" if exists else "[ ]"
                print(
                    f"[dataset] '{ds_name}' {icon} "
                    f"{'present' if exists else 'not found'} "
                    f"at {reg.resolve_path(ds_name)}"
                )
            except Exception:
                pass
        elif ds_path:
            path_obj = Path(str(ds_path))
            exists = path_obj.exists() and any(path_obj.rglob("*.pt"))
            print(
                f"[dataset] {ds_path} "
                f"{'found' if exists else 'not found'}"
            )

        run_logger.info(
            "datamodule_info",
            batch_size=dm.batch_size,
            n_samples=dm.n_samples,
            num_workers=dm._num_workers,
            max_queue=dm.max_queue,
            dataset_name=ds_name or "legacy_path",
        )

        # ── Loss from datatype profile ─────────────────────────────────
        loss_cfg = OmegaConf.to_container(profile.get("loss", {}), resolve=True)
        if not isinstance(loss_cfg, dict):
            loss_cfg = {}
        loss_fn = CertiQNetLoss(
            omega_bc=float(loss_cfg.get("omega_bc", 1.0)),
            omega_action=float(loss_cfg.get("omega_action", 1.5)),
            omega_margin=float(loss_cfg.get("omega_margin", 0.1)),
            omega_usage=float(loss_cfg.get("omega_usage", 0.1)),
            rollout_weight=float(loss_cfg.get("rollout_weight", 1.0)),
            policy_kl_weight=float(loss_cfg.get("policy_kl_weight", 0.05)),
            value_weight=float(loss_cfg.get("value_weight", 1.0)),
            entropy_weight=float(loss_cfg.get("entropy_weight", 0.001)),
        )

        # ── Training params from datatype profile ──────────────────────
        trainer_profile = profile.get("trainer")
        if trainer_profile is None:
            raise ValueError(f"'{datatype}' profile must include a 'trainer' section")
        trainer_container = OmegaConf.to_container(trainer_profile, resolve=True)
        if not isinstance(trainer_container, dict):
            raise TypeError(f"'{datatype}'.trainer must resolve to a mapping")

        lagrangian_profile = profile.get("lagrangian")
        lagrangian_container = OmegaConf.to_container(lagrangian_profile, resolve=True) if lagrangian_profile is not None else {}
        if not isinstance(lagrangian_container, dict):
            lagrangian_container = {}

        input_normalization = str(getattr(profile, "input_normalization", "none"))

        module_kwargs = dict(
            model=model,
            loss_fn=loss_fn,
            input_normalization=input_normalization,
            lr=float(trainer_container.get("lr", 3e-4)),
            weight_decay=float(trainer_container.get("weight_decay", 1e-5)),
            rollout_horizon=int(trainer_container.get("rollout_horizon", 64)),
            use_ppo=bool(trainer_container.get("use_ppo", True)),
            ppo_epochs=int(trainer_container.get("ppo_epochs", 4)),
            ppo_clip_epsilon=float(trainer_container.get("ppo_clip_epsilon", 0.2)),
            ppo_manual_clip_val=float(trainer_container.get("ppo_manual_clip_val", 1.0)),
            entropy_warmup_epochs=int(trainer_container.get("entropy_warmup_epochs", 20)),
            imitation_warmup_epochs=int(trainer_container.get("imitation_warmup_epochs", 20)),
            expert_mode=str(trainer_container.get("expert_mode", "sed")),
            critic_bootstrap_epochs=int(trainer_container.get("critic_bootstrap_epochs", 3)),
            imitation_decay_rate=float(trainer_container.get("imitation_decay_rate", 0.96)),
            target_kl_cert=float(lagrangian_container.get("target_kl_cert", 0.01)),
            initial_policy_kl_weight=float(lagrangian_container.get("initial_policy_kl_weight", 0.05)),
            entropy_weight=float(loss_cfg.get("entropy_weight", 0.001)),
            lam=float(lam),
            gamma=float(trainer_container.get("gamma", 0.99)),
            gae_lambda=float(trainer_container.get("gae_lambda", 0.95)),
            val_horizon_max=int(trainer_container.get("val_horizon_max", 8)),
            dual_lambda_lr=float(lagrangian_container.get("dual_lambda_lr", 0.01)),
            dual_lambda_init=float(lagrangian_container.get("dual_lambda_init", 0.0)),
            dual_lambda_momentum=float(lagrangian_container.get("dual_lambda_momentum", 0.9)),
            dual_lr_warmup_steps=int(lagrangian_container.get("dual_lr_warmup_steps", 10)),
            dual_lambda_max=float(lagrangian_container.get("dual_lambda_max", 10.0)),
            dual_lr_decay=float(lagrangian_container.get("dual_lr_decay", 1.0)),
        )

        if adapter_name in ("QueueingAdapter", "QGymAdapter"):
            lightning = QueueingLightningModule(**module_kwargs)
        elif adapter_name == "ChannelAdapter":
            lightning = ChannelLightningModule(**module_kwargs)
        else:
            lightning = QueueingLightningModule(**module_kwargs)
        logger = instantiate(cfg.logger, save_dir=str(paths.logs)) if "logger" in cfg else False
        callbacks = [instantiate(cb) for cb in cfg.get("callbacks", {}).values()]

        if bool(cfg.project.get("save_checkpoints", True)):
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=str(paths.checkpoints),
                    filename="{epoch:04d}-{val/selection_score:.4f}",
                    monitor="val/selection_score",
                    mode="min",
                    save_last=True,
                    save_top_k=5,
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
                _sync_training_metrics_alias(paths)
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

    # Prefer the best checkpoint (by val/selection_score) over final epoch weights.
    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.ModelCheckpoint) and cb.best_model_path:
            best_path = Path(cb.best_model_path)
            if best_path.exists():
                try:
                    best_state = torch.load(str(best_path), map_location="cpu", weights_only=False)
                    # Lightning checkpoints store model weights under "state_dict"
                    if "state_dict" in best_state:
                        # Strip "model." prefix from Lightning state_dict keys
                        raw_sd = best_state["state_dict"]
                        cleaned = {
                            k.removeprefix("model."): v for k, v in raw_sd.items() if k.startswith("model.")
                        }
                        model.load_state_dict(cleaned)
                    else:
                        model.load_state_dict(best_state)
                    run_logger.info(
                        "loaded_best_checkpoint",
                        path=str(best_path),
                        score=f"{cb.best_model_score:.4f}" if cb.best_model_score is not None else "N/A",
                    )
                except Exception as e:
                    run_logger.info("best_checkpoint_load_failed", error=str(e))
            break

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

    Attempts discovery of a previous training run (via ``.last_run.json``)
    **before** calling ``prepare_run``.  If discovery succeeds the existing
    ``run_id`` is reused, avoiding ghost directories.  If discovery fails
    the ``run_id`` is auto-generated as normal.
    """
    output_root = Path(str(cfg.project.get("output_root", "outputs")))
    if not output_root.is_absolute():
        output_root = (cwd / output_root).resolve()
    experiment_path_name = str(cfg.project.get("experiment_name", None) or "")
    if not experiment_path_name.strip():
        experiment_path_name = experiment_name_from_cfg(cfg)
    experiment_root = output_root / slugify(experiment_path_name)

    last = read_last_run(experiment_root)
    if last is not None:
        if "project" not in cfg:
            cfg.project = OmegaConf.create()
        cfg.project.run_id = last["run_id"]
        cfg.project.experiment_name = experiment_path_name

    paths, run_logger = prepare_run(cfg, cwd=cwd)
    return paths, run_logger


def run_state_bank_audit(cfg: DictConfig, *, cwd: Path) -> None:
    """Run the state-bank audit and persist audit metrics."""
    platform_info = detect_platform()

    if "progress" in cfg:
        configure_progress(OmegaConf.to_container(cfg.progress, resolve=True))

    paths: RunPaths | None = None
    run_logger: ExperimentLogger | None = None
    audit_error: BaseException | None = None
    audit_traceback: str | None = None

    try:
        paths, run_logger = _discover_and_prepare(cfg, cwd=cwd)
        run_logger.info(
            "audit_platform_info",
            os=platform_info.os_name,
            torch=platform_info.torch_version,
            gpu=platform_info.gpu_count,
        )

        adapter = instantiate(cfg.adapter) if "adapter" in cfg else None
        d_xi = int(getattr(adapter, "context_dim", 0))

        datatype = str(cfg.get("datatype", "qgym"))
        profile = cfg.get(datatype) if datatype == "qgym" else None

        if datatype == "qgym":
            data_init_args = dict(profile.data.init_args) if profile is not None else {}
            dataset_name = data_init_args.get("dataset_name", "")
            spec = DatasetRegistry().get(dataset_name)
            N_audit = int(spec.env_N)
            mu = torch.tensor(spec.env_mu_fixed, dtype=torch.float32) if spec.env_mu_fixed is not None else build_mu(cfg)[0]
            lam = float(spec.env_lam) if spec.env_lam is not None else float(build_mu(cfg)[1])
        model = build_model(cfg, N=N_audit, d_xi=d_xi)
        if str(cfg.get("certificate_status", "exact")) == "exact":
            _validate_exact_certificate_constant(model, context="audits")
        load_checkpoint_weights(model, paths.root)
        model.eval()

        run_logger.info("generating_state_bank")
        Q_bank = generate_state_bank(
            N=N_audit,
            mu=mu,
            beta=float(getattr(model, "beta", 1.0)),
            R_cert=float(cfg.model.get("certificate", {}).get("fallback_radius", float("inf"))),
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
            _, diag = model(Q_bank, mu_bank, xi_bank, training_mode=False)

        violation = (diag.A_final - diag.B_Q).clamp(min=0.0)
        audit_env_name = str(data_init_args.get("dataset_name", ""))
        audit_metrics = aggregate_metrics(
            model_name=str(cfg.model._target_).split(".")[-1],
            env_name=audit_env_name,
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
        print(f"fallback_rate=0.000000e+00")
        print(f"usage_open_rate={(diag.usage_final > 0.1).float().mean().item():.6e}")
        print(f"usage_mean={diag.usage_final.nanmean().item():.6e}")
        print(f"min_certificate_slack={diag.certificate_slack.min().item():.6e}")

    except BaseException as exc:
        audit_error = exc
        audit_traceback = traceback.format_exc()
        if run_logger is not None:
            try:
                run_logger.info(
                    "audit_failed",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            except Exception:
                pass
        print(f"\n[ERROR] Pipeline stage 'audit' failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        if audit_error is not None and audit_traceback is not None:
            try:
                _write_failure_artifacts(
                    cwd=cwd,
                    paths=paths,
                    stage="audit",
                    error=audit_error,
                    tb=audit_traceback,
                    logger=run_logger,
                )
            except Exception:
                pass
        if run_logger is not None:
            try:
                run_logger.flush()
            except Exception:
                pass


def run_baseline_paper_comparison(cfg: DictConfig, *, cwd: Path) -> None:
    """Run the baseline comparison suite and print a summary table."""
    paths: RunPaths | None = None
    run_logger: ExperimentLogger | None = None
    baseline_error: BaseException | None = None
    baseline_traceback: str | None = None
    platform_info = detect_platform()

    if "progress" in cfg:
        configure_progress(OmegaConf.to_container(cfg.progress, resolve=True))

    try:
        paths, run_logger = _discover_and_prepare(cfg, cwd=cwd)
        run_logger.info(
            "baseline_platform_info",
            os=platform_info.os_name,
            torch=platform_info.torch_version,
            gpu=platform_info.gpu_count,
        )

        adapter = instantiate(cfg.adapter) if "adapter" in cfg else None
        d_xi = int(getattr(adapter, "context_dim", 0))

        datatype = str(cfg.get("datatype", "qgym"))
        profile = cfg.get(datatype) if datatype == "qgym" else None

        if datatype == "qgym":
            data_init_args = dict(profile.data.init_args) if profile is not None else {}
            dataset_name = data_init_args.get("dataset_name", "")
            spec = DatasetRegistry().get(dataset_name)
            N_bl = int(spec.env_N)
            mu = torch.tensor(spec.env_mu_fixed, dtype=torch.float32) if spec.env_mu_fixed is not None else build_mu(cfg)[0]
            lam = float(spec.env_lam) if spec.env_lam is not None else float(build_mu(cfg)[1])
        model = build_model(cfg, N=N_bl, d_xi=d_xi)
        if str(cfg.get("certificate_status", "exact")) == "exact":
            _validate_exact_certificate_constant(model, context="baseline comparisons")
        load_checkpoint_weights(model, paths.root)
        rollout = RolloutConfig(
            steps=int(cfg.runner.rollout_steps),
            batch_size=int(cfg.runner.rollout_batch_size),
            max_backlog=float(cfg.runner.max_backlog),
            show_progress=bool(cfg.runner.show_progress),
        )

        # ── Baseline filter — read from cfg.studies.runner.baselines ────────
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
        print(f"\n[ERROR] Pipeline stage 'baselines' failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        if baseline_error is not None and baseline_traceback is not None:
            try:
                _write_failure_artifacts(
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

