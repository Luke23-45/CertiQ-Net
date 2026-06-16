"""Training entrypoints.

Usage
-----
New (LightningCLI):
    python certiqnet/scripts/train.py --cli --config configs/experiments/queueing/certiq_index.yaml

Legacy (Hydra):
    python certiqnet/scripts/train.py model=certiq_index env=family_a adapter=queueing trainer=default
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── CLI mode (LightningCLI, BGSL-style) ─────────────────────────────


_CLI_TRAINER_KEYS = {
    "accelerator", "strategy", "devices", "num_nodes", "precision",
    "logger", "callbacks", "fast_dev_run", "max_epochs", "min_epochs",
    "max_steps", "min_steps", "max_time", "limit_train_batches",
    "limit_val_batches", "limit_test_batches", "limit_predict_batches",
    "overfit_batches", "val_check_interval", "check_val_every_n_epoch",
    "num_sanity_val_steps", "log_every_n_steps", "enable_checkpointing",
    "enable_progress_bar", "enable_model_summary", "accumulate_grad_batches",
    "gradient_clip_val", "gradient_clip_algorithm", "deterministic",
    "benchmark", "inference_mode", "use_distributed_sampler", "profiler",
    "detect_anomaly", "barebones", "plugins", "sync_batchnorm",
    "reload_dataloaders_every_n_epochs", "default_root_dir",
}

def _run_cli(config_path: str, unknown_args: list[str]) -> None:
    import sys
    import tempfile
    import yaml
    from omegaconf import OmegaConf

    cfg = OmegaConf.create()
    exp_cfg = OmegaConf.load(config_path)

    def _merge_defaults(source_cfg, source_path: str, seen: set[str]) -> None:
        nonlocal cfg
        defaults = source_cfg.get("defaults", [])
        for d in defaults:
            if isinstance(d, str):
                p = Path(source_path).resolve().parent / f"{d}.yaml"
                resolved = str(p.resolve())
                if p.exists() and resolved not in seen:
                    seen.add(resolved)
                    sub = OmegaConf.load(p)
                    _merge_defaults(sub, resolved, seen)
                    if "defaults" in sub:
                        del sub["defaults"]
                    cfg = OmegaConf.merge(cfg, sub)
            else:
                for key, val in d.items():
                    if key == "override /dataset":
                        key = "data"
                    safe_key = key.lstrip("/")
                    p = ROOT / "configs" / "cli" / safe_key / f"{val}.yaml"
                    if not p.exists():
                        p = ROOT / "configs" / "cli" / f"{val}.yaml"
                    if p.exists():
                        resolved = str(p.resolve())
                        if resolved not in seen:
                            seen.add(resolved)
                            sub = OmegaConf.load(p)
                            _merge_defaults(sub, resolved, seen)
                            if "defaults" in sub:
                                del sub["defaults"]
                            cfg = OmegaConf.merge(cfg, sub)

    _merge_defaults(exp_cfg, config_path, set())

    if "defaults" in exp_cfg:
        del exp_cfg["defaults"]
    cfg = OmegaConf.merge(cfg, exp_cfg)

    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(OmegaConf.to_container(cfg, resolve=True), tf)
    tf.close()

    sys.argv = ["certiqnet-train", "fit", "--config", tf.name] + unknown_args
    from certiqnet.cli import main as cli_main
    cli_main()


# ── Legacy mode (Hydra, backward-compat) ────────────────────────────


def _run_legacy(config_path: str | None = None, overrides: list[str] | None = None) -> None:
    from omegaconf import OmegaConf

    from certiqnet.experiments.pipeline import run_training

    # Load base config
    base_cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cfg = OmegaConf.create()
    defaults = base_cfg.get("defaults", [])
    for d in defaults:
        if isinstance(d, str):
            continue  # skip special keys like _self_
        for key, val in d.items():
            if key == "_self_":
                continue
            p = ROOT / "configs" / key / f"{val}.yaml"
            if p.exists():
                sub = OmegaConf.load(p)
                cfg = OmegaConf.merge(cfg, sub)
    cfg = OmegaConf.merge(cfg, base_cfg)

    if config_path:
        exp_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, exp_cfg)

    if overrides:
        overrides_merged = OmegaConf.from_cli(overrides)
        cfg = OmegaConf.merge(cfg, overrides_merged)

    run_training(cfg, cwd=ROOT)


# ── Dispatch ────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CertiQ-Net training")
    parser.add_argument("--cli", action="store_true", help="Use LightningCLI mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args, unknown = parser.parse_known_args()

    if args.cli:
        if args.config is None:
            print("ERROR: --cli requires --config <path>")
            sys.exit(1)
        _run_cli(args.config, unknown)
    elif args.config:
        _run_legacy(config_path=args.config)
    else:
        sys.argv = [sys.argv[0]] + unknown
        _run_legacy()
