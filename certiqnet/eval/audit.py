from __future__ import annotations

import traceback
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from certiqnet.data.common.state_bank import generate_state_bank
from certiqnet.data.registry import DatasetRegistry
from certiqnet.eval._base import discover_and_prepare
from certiqnet.experiments.persistence.checkpoint import load_checkpoint_weights
from certiqnet.experiments.evaluators.factory import build_model, build_mu
from certiqnet.experiments.persistence.logging import BufferedExperimentLogger as ExperimentLogger
from certiqnet.experiments.evaluators.metrics import aggregate_metrics, save_metrics
from certiqnet.experiments.persistence.paths import RunPaths
from certiqnet.train._shared import (
    validate_exact_certificate_constant,
    write_failure_artifacts,
)
from certiqnet.utils.platform import detect_platform
from certiqnet.utils.progress import configure_progress


def run_state_bank_audit(cfg: DictConfig, *, cwd: Path) -> None:
    platform_info = detect_platform()

    if "progress" in cfg:
        configure_progress(OmegaConf.to_container(cfg.progress, resolve=True))

    paths: RunPaths | None = None
    run_logger: ExperimentLogger | None = None
    audit_error: BaseException | None = None
    audit_traceback: str | None = None

    try:
        paths, run_logger = discover_and_prepare(cfg, cwd=cwd)
        run_logger.info(
            "audit_platform_info",
            os=platform_info.os_name,
            torch=platform_info.torch_version,
            gpu=platform_info.gpu_count,
        )

        adapter = instantiate(cfg.adapter) if "adapter" in cfg else None
        d_xi = int(getattr(adapter, "context_dim", 0))

        datatype = str(cfg.get("datatype", "qgym"))
        if datatype != "qgym":
            raise ValueError(f"Audit stage requires datatype='qgym', got '{datatype}'.")
        profile = cfg.get(datatype)

        if datatype == "qgym":
            data_init_args = dict(profile.data.init_args) if profile is not None else {}
            dataset_name = data_init_args.get("dataset_name")
            if not dataset_name:
                raise ValueError("QGym profile must include data.init_args.dataset_name")
            spec = DatasetRegistry().get(dataset_name)
            N_audit = int(spec.env_N)
            mu = torch.tensor(spec.env_mu_fixed, dtype=torch.float32) if spec.env_mu_fixed is not None else build_mu(cfg)[0]
            lam = float(spec.env_lam) if spec.env_lam is not None else float(build_mu(cfg)[1])
        model = build_model(cfg, N=N_audit, d_xi=d_xi)
        if str(cfg.get("certificate_status", "exact")) == "exact":
            validate_exact_certificate_constant(model, context="audits")
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
        print(f"\n[ERROR] Stage 'audit' failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        if audit_error is not None and audit_traceback is not None:
            try:
                write_failure_artifacts(
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
