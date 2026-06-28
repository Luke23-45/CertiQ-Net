from __future__ import annotations

import csv
import json
import random
import shutil
import traceback
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from certiqnet.data.qgym.datamodule import QGymDataModule
from certiqnet.data.registry import DatasetRegistry
from certiqnet.experiments.evaluators.factory import build_model, build_mu
from certiqnet.experiments.persistence.checkpoint import (
    save_checkpoint_state,
    save_last_run,
)
from certiqnet.experiments.persistence.logging import BufferedExperimentLogger as ExperimentLogger
from certiqnet.experiments.persistence.paths import RunPaths
from certiqnet.experiments.runner import experiment_name_from_cfg, prepare_run
from certiqnet.train._shared import (
    set_model_certificate_constant,
    validate_exact_certificate_constant,
    write_failure_artifacts,
)
from certiqnet.train.channel.module import ChannelLightningModule
from certiqnet.train.common.loss import CertiQNetLoss
from certiqnet.train.queueing.module import QueueingLightningModule
from certiqnet.utils.platform import detect_platform, resolve_trainer_config
from certiqnet.utils.progress import configure_progress

try:
    import pytorch_lightning as pl
except ModuleNotFoundError as exc:
    raise SystemExit("pytorch-lightning is required. Run `pip install -e .`.") from exc


def _sync_training_metrics_alias(paths: RunPaths) -> None:
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
                "trainer and loss sections."
            )
        model_profile = profile.get("model", {})
        model_profile_container = OmegaConf.to_container(model_profile, resolve=True) if model_profile is not None else {}
        if not isinstance(model_profile_container, dict):
            model_profile_container = {}
        if model_profile_container:
            OmegaConf.set_struct(cfg, False)
            cfg.model = OmegaConf.merge(cfg.model, OmegaConf.create(model_profile_container))
            OmegaConf.set_struct(cfg, True)

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
        context_dim = int(model_profile_container.get("context_dim", 0))
        d_xi = max(d_xi, context_dim)
        model = build_model(cfg, N=N, d_xi=d_xi)

        if str(cfg.get("certificate_status", "exact")) == "exact":
            model_constant = validate_exact_certificate_constant(model, context="training runs")
            set_model_certificate_constant(cfg, model_constant)
            OmegaConf.save(config=cfg, f=paths.configs / "resolved_config.yaml", resolve=True)
        torch.save(model.state_dict(), paths.artifacts / "initial_model_state.pt")

        data_init_args.setdefault("N", N)
        dm = QGymDataModule(mu=mu, **data_init_args)
        dm.datatype = datatype
        dm.context_dim = max(int(getattr(dm, "context_dim", 0)), d_xi)

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

        loss_cfg = OmegaConf.to_container(profile.get("loss", {}), resolve=True)
        if not isinstance(loss_cfg, dict):
            loss_cfg = {}
        loss_fn = CertiQNetLoss(
            omega_roll=float(loss_cfg.get("omega_roll", 1.0)),
            omega_ent=float(loss_cfg.get("omega_ent", 0.001)),
            omega_kl=float(loss_cfg.get("omega_kl", 0.05)),
        )

        trainer_profile = profile.get("trainer")
        if trainer_profile is None:
            raise ValueError(f"'{datatype}' profile must include a 'trainer' section")
        trainer_container = OmegaConf.to_container(trainer_profile, resolve=True)
        if not isinstance(trainer_container, dict):
            raise TypeError(f"'{datatype}'.trainer must resolve to a mapping")

        input_normalization = str(getattr(profile, "input_normalization", "none"))

        module_kwargs = dict(
            model=model,
            loss_fn=loss_fn,
            input_normalization=input_normalization,
            lr=float(trainer_container.get("lr", 3e-4)),
            weight_decay=float(trainer_container.get("weight_decay", 1e-5)),
            rollout_horizon=int(trainer_container.get("rollout_horizon", 64)),
            context_dim=d_xi,
            lam=float(lam),
            gamma=float(trainer_container.get("gamma", 0.99)),
            val_horizon_max=int(trainer_container.get("val_horizon_max", 64)),
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

    except BaseException as exc:
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
                write_failure_artifacts(
                    cwd=cwd,
                    paths=paths,
                    stage="training",
                    error=training_error,
                    tb=training_traceback,
                    logger=run_logger,
                )
            except Exception:
                pass

    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.ModelCheckpoint) and cb.best_model_path:
            best_path = Path(cb.best_model_path)
            if best_path.exists():
                try:
                    best_state = torch.load(str(best_path), map_location="cpu", weights_only=False)
                    if "state_dict" in best_state:
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
    save_last_run(
        paths.root.parent,
        run_id=actual_run_id,
        experiment_name=actual_experiment_name,
    )
    run_logger.info("finished_training", root=str(paths.root))
