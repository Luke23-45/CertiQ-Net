"""Hydra training entry point."""

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

from certiqnet.models.baselines import AnalyticBackbonePolicy
from certiqnet.models.certiqnet_p import CertiQNetP
from certiqnet.models.certiqnet_s import CertiQNetS
from certiqnet.training.datamodule import CertiQNetDataModule
from certiqnet.training.lightning_module import CertiQNetLightningModule
from certiqnet.utils.config_schemas import CertiQNetPConfig, CertiQNetSConfig, RootConfig

try:
    import pytorch_lightning as pl
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("pytorch-lightning is required. Run `pip install -e .`.") from exc

cs = ConfigStore.instance()
cs.store(name="root_config", node=RootConfig)

T = TypeVar("T")


def _dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        target = f.type
        if is_dataclass(target) and isinstance(value, dict):
            kwargs[f.name] = _dataclass_from_dict(target, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def build_mu(cfg: DictConfig) -> tuple[torch.Tensor, float]:
    """Build service rates and lambda from env config."""
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


def build_model(cfg: DictConfig, N: int) -> torch.nn.Module:
    """Construct model while preserving config dataclass semantics."""
    model_data = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_data, dict)
    target = str(model_data.get("_target_", ""))
    if target.endswith("CertiQNetP"):
        return CertiQNetP(_dataclass_from_dict(CertiQNetPConfig, model_data), N=N)
    if target.endswith("AnalyticBackbonePolicy"):
        return AnalyticBackbonePolicy(N=N, beta=float(model_data.get("beta", 1.0)))
    return CertiQNetS(_dataclass_from_dict(CertiQNetSConfig, model_data), N=N)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train CertiQ-Net with exact-gate validation."""
    seed = int(cfg.project.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    mu, _ = build_mu(cfg)
    model = build_model(cfg, N=int(cfg.env.N))
    dm = CertiQNetDataModule(N=int(cfg.env.N), mu=mu, batch_size=64, n_samples=512)
    lightning = CertiQNetLightningModule(model, cfg)
    logger = instantiate(cfg.logger) if "logger" in cfg else False
    callbacks = [instantiate(cb) for cb in cfg.get("callbacks", {}).values()]
    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=int(cfg.trainer.devices),
        precision=cfg.trainer.precision,
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        val_check_interval=float(cfg.trainer.val_check_interval),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(lightning, datamodule=dm)


if __name__ == "__main__":
    main()
