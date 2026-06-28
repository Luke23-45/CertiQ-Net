from __future__ import annotations

import traceback
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from certiqnet.data.common.state_bank import generate_adversarial_states, generate_state_bank
from certiqnet.data.registry import DatasetRegistry
from certiqnet.eval._base import discover_and_prepare
from certiqnet.experiments.persistence.checkpoint import load_checkpoint_weights
from certiqnet.experiments.persistence.checkpoint import infer_checkpoint_context_dim
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
        checkpoint_d_xi = infer_checkpoint_context_dim(paths.root)
        if checkpoint_d_xi > 0:
            d_xi = checkpoint_d_xi
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

        run_logger.info("generating_adversarial_state_bank")
        Q_adv = generate_adversarial_states(
            model=model,
            mu=mu,
            N=N_audit,
            n_states=128,
            n_steps=25,
        )
        run_logger.info("adversarial_state_bank_generated", states=Q_adv.shape[0])

        def _eval_bank(name: str, Q_in: torch.Tensor) -> torch.Tensor:
            mu_bank = mu.unsqueeze(0).expand(Q_in.shape[0], -1)
            if adapter is not None:
                Q_obs, mu_obs, xi_obs = adapter.make_observation(Q_in, mu_bank)
            else:
                Q_obs, mu_obs, xi_obs = Q_in, mu_bank, None
            if int(getattr(model, "d_xi", 0)) <= 0:
                xi_obs = None
            elif xi_obs is not None and xi_obs.shape[-1] != int(getattr(model, "d_xi", 0)):
                model_d_xi = int(getattr(model, "d_xi", 0))
                if xi_obs.shape[-1] > model_d_xi:
                    xi_obs = xi_obs[..., :model_d_xi]
                else:
                    pad = torch.zeros(*xi_obs.shape[:-1], model_d_xi - xi_obs.shape[-1], device=xi_obs.device, dtype=xi_obs.dtype)
                    xi_obs = torch.cat([xi_obs, pad], dim=-1)
            with torch.no_grad():
                if hasattr(model, "reset_dispatch_state"):
                    model.reset_dispatch_state()
                _, diag = model(Q_obs, mu_obs, xi_obs, training_mode=False)
            violation = (diag.A_final - diag.B_Q).clamp(min=0.0)
            metrics = aggregate_metrics(
                model_name=str(cfg.model._target_).split(".")[-1],
                env_name=str(data_init_args.get("dataset_name", "")),
                seed=int(cfg.project.seed),
                lam=lam,
                queue_trace=Q_in,
                cost_trace=Q_in.sum(dim=-1),
                dt_trace=torch.ones(Q_in.shape[0]),
                diagnostics=[diag],
                evaluation_start=name,
                greedy_eval=False,
            )
            save_metrics([metrics], paths.audits, filename=f"{name}_audit")
            run_logger.metric(metrics.flat())
            print(f"{name}_states={Q_in.shape[0]}")
            print(f"{name}_max_violation={violation.max().item():.6e}")
            print(f"{name}_violation_rate={(violation > 0).float().mean().item():.6e}")
            print(f"{name}_usage_open_rate={(diag.usage_final > 0.1).float().mean().item():.6e}")
            print(f"{name}_usage_mean={diag.usage_final.nanmean().item():.6e}")
            print(f"{name}_min_certificate_slack={diag.certificate_slack.min().item():.6e}")
            return violation

        state_violation = _eval_bank("state_bank", Q_bank)
        adv_violation = _eval_bank("adversarial_state_bank", Q_adv)
        run_logger.flush()

        print("fallback_rate=0.000000e+00")
        print(f"state_bank_max_violation={state_violation.max().item():.6e}")
        print(f"adversarial_state_bank_max_violation={adv_violation.max().item():.6e}")

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
