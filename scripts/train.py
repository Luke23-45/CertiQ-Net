"""Hydra training entry point with platform-aware config resolution."""

from __future__ import annotations

import random
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from certiqnet.experiments.runner import prepare_run
from certiqnet.models.baselines import (
    AnalyticBackbonePolicy,
    CertiQNetSAblation_NoGate,
    JoinShortestWeightedQueue,
    RandomPolicy,
)
from certiqnet.models.certiqnet_p import CertiQNetP
from certiqnet.models.certiqnet_s import CertiQNetS
from certiqnet.training.datamodule import CertiQNetDataModule
from certiqnet.training.lightning_module import CertiQNetLightningModule
from certiqnet.utils.config_schemas import CertiQNetPConfig, CertiQNetSConfig, RootConfig
from certiqnet.utils.platform import detect_platform, resolve_trainer_config
from certiqnet.utils.progress import configure_progress

try:
    import pytorch_lightning as pl
except ModuleNotFoundError as exc:
    raise SystemExit("pytorch-lightning is required. Run `pip install -e .`.") from exc

cs = ConfigStore.instance()
cs.store(name="root_config", node=RootConfig)

T = TypeVar("T")


def _dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value = data[f.name]
        target = f.type
        if is_dataclass(target) and isinstance(value, dict):
            kwargs[f.name] = _dataclass_from_dict(target, value)  # type: ignore[arg-type]
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def build_mu(cfg: DictConfig) -> tuple[torch.Tensor, float]:
    env = cfg.env
    if env.mu_mode == "fixed":
        assert env.mu_fixed is not None
        mu = torch.tensor(list(env.mu_fixed), dtype=torch.float32)
    elif env.mu_mode == "lognormal":
        mu = torch.distributions.LogNormal(0.0, float(env.mu_lognormal_sigma)).sample((int(env.N),))
    else:
        raise ValueError(f"Unsupported mu_mode: {env.mu_mode}")
    lam = float(env.lam)
    if env.rho_target is not None:
        lam = float(env.rho_target) * mu.sum().item()
    assert lam < mu.sum().item(), "lambda must remain subcritical."
    return mu, lam


def build_model(cfg: DictConfig, N: int, d_xi: int = 0) -> torch.nn.Module:
    model_data = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_data, dict)
    target = str(model_data.get("_target_", ""))
    if target.endswith("CertiQNetP"):
        return CertiQNetP(_dataclass_from_dict(CertiQNetPConfig, model_data), N=N, d_xi=d_xi)
    if target.endswith("AnalyticBackbonePolicy"):
        return AnalyticBackbonePolicy(N=N, beta=float(model_data.get("beta", 1.0)))
    if target.endswith("RandomPolicy"):
        return RandomPolicy(N=N, beta=float(model_data.get("beta", 1.0)))
    if target.endswith("JoinShortestWeightedQueue"):
        return JoinShortestWeightedQueue(N=N, beta=float(model_data.get("beta", 1.0)))
    if target.endswith("CertiQNetSAblation_NoGate"):
        return CertiQNetSAblation_NoGate(
            _dataclass_from_dict(CertiQNetSConfig, model_data), N=N, d_xi=d_xi
        )
    return CertiQNetS(_dataclass_from_dict(CertiQNetSConfig, model_data), N=N, d_xi=d_xi)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")  # type: ignore[untyped-decorator]
def main(cfg: DictConfig) -> None:
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
        print(f"[platform] os={platform_info.os_name} python={py_ver} "
              f"torch={platform_info.torch_version} gpu={gpu_str} "
              f"cpu={cpu_str} precision={platform_info.recommended_precision}")

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_cfg, dict)
    resolved_trainer = resolve_trainer_config(trainer_cfg, platform_info)

    if resolved_trainer["precision"] != str(cfg.trainer.precision):
        fallback_msg = (
            f"[warn] precision '{cfg.trainer.precision}' not supported, "
            f"falling back to '{resolved_trainer['precision']}'"
        )
        print(f"[platform] {fallback_msg}")

    paths, run_logger = prepare_run(cfg, cwd=ROOT)
    run_logger.info("platform_info",
                    os=platform_info.os_name,
                    python=platform_info.python_version.split()[0],
                    torch=platform_info.torch_version,
                    gpu_count=platform_info.gpu_count,
                    gpu_names=",".join(platform_info.gpu_names),
                    cpu_logical=platform_info.cpu_count_logical,
                    accelerator=resolved_trainer["accelerator"],
                    precision=resolved_trainer["precision"],
                    num_workers=resolved_trainer.get("_num_workers", 0))

    adapter = instantiate(cfg.adapter) if "adapter" in cfg else None
    adapter_name = adapter.__class__.__name__ if adapter is not None else "QueueingAdapter"
    d_xi = int(getattr(adapter, "context_dim", 0))
    cert_status = str(getattr(adapter, "CERTIFICATE_STATUS", "exact"))
    assumptions_satisfied = bool(getattr(adapter, "assumptions_satisfied", False))
    cfg_cert_status = str(cfg.get("certificate_status", "exact"))

    if cfg_cert_status == "exact" and cert_status != "exact":
        raise ValueError(
            f"Constraint Violation: Config specifies 'exact' certification, but adapter "
            f"'{adapter_name}' provides '{cert_status}'. Update config to 'certificate_status=approximate'."
        )
        
    env_target = str(cfg.get("env", {}).get("_target_", "QueueingCTMC"))
    if cfg_cert_status == "exact" and "QueueingCTMC" not in env_target:
        raise ValueError(
            f"Constraint Violation: Exact certification requires the QueueingCTMC backend. "
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
        callbacks.append(pl.callbacks.TQDMProgressBar(
            refresh_rate=int(cfg.runner.get("refresh_rate", 10)),
        ))

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


if __name__ == "__main__":
    main()
