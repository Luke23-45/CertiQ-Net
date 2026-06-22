"""High-level experiment runner utilities and study orchestration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig

from certiqnet.data.registry import DatasetRegistry
from certiqnet.experiments.catalog.runtime import (
    RunRequest,
    Stage,
    filter_overrides_for_method,
    select_seeds,
    select_variants,
)
from certiqnet.experiments.catalog.specs import OverrideSpec, StudySpec
from certiqnet.experiments.catalog.registry import (
    get_registered_studies,
    get_study_entry,
    print_study_listing,
)
from certiqnet.experiments.logging import BufferedExperimentLogger as ExperimentLogger
from certiqnet.experiments.paths import (
    RunPaths,
    create_run_paths,
    make_run_id,
    save_manifest,
    save_resolved_config,
    slugify,
)
from certiqnet.experiments.stages import (
    build_audit_stage,
    build_baselines_stage,
    build_training_stage,
)
from certiqnet.experiments.stages._base import OUTPUT_ROOT, ROOT


# ── Existing utility functions (kept as-is) ──────────────────────────


def _optional_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "none" else text


def experiment_name_from_cfg(cfg: DictConfig) -> str:
    model_name = str(cfg.model._target_).split(".")[-1]
    family = str(cfg.get("experiment_family", "default"))

    if cfg.datatype == "qgym":
        profile = cfg.get("qgym")
        if profile is not None:
            ds_name = str(profile.data.init_args.get("dataset_name", "qgym"))
        else:
            ds_name = str(cfg.get("qgym_data", "qgym"))
        ds_name_clean = ds_name.replace("/", "_").replace("\\", "_")
        spec = DatasetRegistry().get(ds_name_clean) if ds_name_clean != "qgym" else None
        if spec is not None:
            n = int(spec.env_N)
            lam_str = str(spec.env_lam) if spec.env_lam is not None else "auto"
        else:
            n = int(cfg.env.N) if "env" in cfg else 6
            lam_str = "auto"
        return f"{family}_{model_name}_N{n}_{ds_name_clean}_lam{lam_str}"
    else:
        env_name = str(cfg.env.mu_mode)
        n = int(cfg.env.N)
        rho = cfg.env.get("rho_target")
        rho_part = f"rho{rho}" if rho is not None else f"lam{cfg.env.lam}"
        return f"{family}_{model_name}_N{n}_{env_name}_{rho_part}"


def prepare_run(
    cfg: DictConfig,
    *,
    cwd: Path,
    output_root: Path | None = None,
) -> tuple[RunPaths, ExperimentLogger]:
    experiment_name = _optional_text(cfg.project.get("experiment_name", None))
    if not experiment_name:
        experiment_name = experiment_name_from_cfg(cfg)
    run_id = _optional_text(cfg.project.get("run_id", None))
    if not run_id:
        run_id = make_run_id(experiment_name, int(cfg.project.seed))
    output_root_value = _optional_text(cfg.project.get("output_root", "outputs")) or "outputs"
    raw_root = output_root or Path(output_root_value)
    root = raw_root if raw_root.is_absolute() else (cwd / raw_root).resolve()
    paths = create_run_paths(root, experiment_name, run_id)
    save_resolved_config(cfg, paths)
    save_manifest(cfg, paths, experiment_name, run_id, cwd)
    logger = ExperimentLogger(paths.logs)
    logger.info(
        "prepared_run",
        run_id=run_id,
        experiment_name=experiment_name,
        root=str(paths.root),
    )
    return paths, logger


# ── New orchestration functions ──────────────────────────────────────


def parse_overrides(raw_overrides: list[str]) -> list[OverrideSpec]:
    parsed: list[OverrideSpec] = []
    for raw in raw_overrides:
        try:
            parsed.append(OverrideSpec.parse(raw))
        except ValueError as exc:
            _safe_print(f"Invalid override: {exc}")
            sys.exit(1)
    return parsed


def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        safe = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe)


def run_study(
    spec: StudySpec,
    request: RunRequest,
    *,
    dry_run: bool = False,
    keep_going: bool = False,
) -> int:
    selected_variants = select_variants(spec, request.variant_labels)

    for variant in selected_variants:
        seeds = select_seeds(variant.seeds, request.seeds)
        method_overrides = filter_overrides_for_method(
            request.overrides, getattr(variant, "method", "")
        )

        for seed in seeds:
            experiment_name = f"{spec.name}_{variant.label}"
            run_id = make_run_id(experiment_name, seed)

            stages: list[Stage] = []

            if "train" in spec.stages:
                stages.append(
                    build_training_stage(spec, variant, seed, tuple(method_overrides), run_id)
                )

            if "audit" in spec.stages or "baselines" in spec.stages:
                run_dir = str(OUTPUT_ROOT / slugify(experiment_name) / run_id)
                if "audit" in spec.stages:
                    stages.append(
                        build_audit_stage(spec, variant, seed, tuple(method_overrides), run_dir)
                    )
                if "baselines" in spec.stages:
                    stages.append(
                        build_baselines_stage(spec, variant, seed, tuple(method_overrides), run_dir)
                    )

            for idx, stage in enumerate(stages, start=1):
                _safe_print(f"[{idx}/{len(stages)}] {stage.label}")
                if dry_run:
                    _safe_print(f"  (dry-run) cwd={stage.cwd}")
                    _safe_print(f"  (dry-run) {' '.join(stage.command)}")
                    continue
                result = subprocess.run(stage.command, cwd=stage.cwd).returncode
                if result != 0:
                    _safe_print(f"Stage failed with exit code {result}: {stage.label}")
                    if not keep_going:
                        return result

    return 0


# ── CLI ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CertiQ-Net study runner — plan and execute experiment stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true", help="List registered studies and exit.")
    p.add_argument("--study", type=str, help="Study name to execute.")
    p.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant label to run. Repeat to select multiple variants.",
    )
    p.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help="Seed to run. Repeat to select multiple seeds.",
    )
    p.add_argument(
        "--set",
        dest="raw_overrides",
        action="append",
        default=[],
        help="Runtime override in key=value or method:key=value form. Repeatable.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print execution plan without running.")
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue remaining stages after failures.",
    )
    return p


def parse_and_run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.list:
        print("Registered studies:")
        print_study_listing()
        return 0

    if not args.study:
        parser.print_help()
        return 1

    try:
        entry = get_study_entry(args.study)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    overrides = parse_overrides(args.raw_overrides)

    request = RunRequest(
        variant_labels=tuple(args.variant),
        seeds=tuple(args.seed),
        overrides=tuple(overrides),
    )

    return run_study(
        entry.spec,
        request,
        dry_run=args.dry_run,
        keep_going=args.keep_going,
    )


def main() -> int:
    raise SystemExit(parse_and_run())


if __name__ == "__main__":
    main()
