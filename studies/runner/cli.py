"""Argparse-based CLI wrapper for study runner families.

This module translates argparse flags into Hydra override strings, then
delegates to ``run_study_family``.

Usage (via a family entrypoint):
    python -m studies.runner.families.main_queueing \\
        --stages train,baselines \\
        --seeds 42,43,44 \\
        --baselines-include sed,max_weight \\
        --tag experiment-v3 \\
        --dry-run

    # Or mix with raw Hydra overrides:
    python -m studies.runner.families.main_queueing \\
        studies.runner.verbose=false \\
        project.seed=99
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from certiqnet.data.registry import DatasetRegistry
from studies.runner.common import StudyRunnerSpec, run_study_family


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """Return an ArgumentParser for the study runner CLI."""
    p = argparse.ArgumentParser(
        prog=prog,
        description="CertiQ-Net study runner — translates flags to Hydra overrides.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Any unrecognised arguments are forwarded directly as Hydra overrides.\n"
            "Example: studies.runner.verbose=false  project.seed=99"
        ),
    )
    p.add_argument(
        "--stages",
        metavar="STAGE[,STAGE...]",
        default=None,
        help="Comma-separated list of stages to run (e.g. train,baselines).",
    )
    p.add_argument(
        "--seeds",
        metavar="SEED[,SEED...]",
        default=None,
        help="Comma-separated list of integer seeds (e.g. 42,43,44).",
    )
    p.add_argument(
        "--baselines-include",
        metavar="NAME[,NAME...]",
        default=None,
        dest="baselines_include",
        help=(
            "Whitelist of baseline names to run (e.g. sed,max_weight). "
            "Defaults to all baselines (*)."
        ),
    )
    p.add_argument(
        "--baselines-exclude",
        metavar="NAME[,NAME...]",
        default=None,
        dest="baselines_exclude",
        help="Blacklist of baseline names to skip (e.g. random).",
    )
    p.add_argument(
        "--failure-mode",
        choices=["stop", "skip"],
        default=None,
        dest="failure_mode",
        help="What to do when a seed fails: stop (default) or skip to next seed.",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Optional label attached to the run for identification.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Enable verbose Rich logging.",
    )
    p.add_argument(
        "--no-verbose",
        action="store_false",
        dest="verbose",
        help="Disable verbose output.",
    )
    p.add_argument(
        "--dataset",
        dest="dataset_name",
        default=None,
        help="Dataset type codename (e.g. reentrant_2, reentrant_3_hyper). "
        "Sets datatype=qgym and data.init_args.dataset_name.",
    )
    p.add_argument(
        "--family",
        dest="family_name",
        default=None,
        help="Synthetic env family (e.g. synthetic/family_a, synthetic/family_b). "
        "Sets datatype=synthetic and env=<family>.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print the execution plan without actually running any stages.",
    )
    return p


def parse_and_run(
    spec: StudyRunnerSpec,
    argv: Sequence[str] | None = None,
) -> None:
    """Parse CLI arguments for *spec* and delegate to ``run_study_family``.

    Unrecognised args are forwarded as Hydra overrides verbatim.
    """
    parser = build_parser()
    args, hydra_overrides = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    # ── Mode resolution ───────────────────────────────────────────────────────
    if args.dataset_name is not None and args.family_name is not None:
        print(
            "Cannot specify both --dataset and --family. Choose one mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dataset_name is not None:
        # QGym mode: override datatype and the dataset_name in the qgym profile
        try:
            ds_reg = DatasetRegistry()
            ds_spec = ds_reg.get(args.dataset_name)
            hydra_overrides.append("datatype=qgym")
            hydra_overrides.append(
                f"qgym.data.init_args.dataset_name={args.dataset_name}"
            )
        except KeyError:
            print(
                f"Unknown dataset '{args.dataset_name}'. "
                f"Available: {', '.join(DatasetRegistry().list_datasets())}",
                file=sys.stderr,
            )
            sys.exit(1)

    elif args.family_name is not None:
        # Synthetic mode: override datatype and env
        hydra_overrides.append("datatype=synthetic")
        hydra_overrides.append(f"env={args.family_name}")

    # Translate parsed flags → Hydra overrides
    if args.stages is not None:
        stages_list = ",".join(s.strip() for s in args.stages.split(","))
        hydra_overrides.append(f"studies.runner.stages=[{stages_list}]")

    if args.seeds is not None:
        seeds_list = ",".join(s.strip() for s in args.seeds.split(","))
        hydra_overrides.append(f"studies.runner.seeds=[{seeds_list}]")

    if args.baselines_include is not None:
        names = ",".join(s.strip() for s in args.baselines_include.split(","))
        hydra_overrides.append(f"studies.runner.baselines.include=[{names}]")

    if args.baselines_exclude is not None:
        names = ",".join(s.strip() for s in args.baselines_exclude.split(","))
        hydra_overrides.append(f"studies.runner.baselines.exclude=[{names}]")

    if args.failure_mode is not None:
        hydra_overrides.append(f"studies.runner.failure_mode={args.failure_mode}")

    if args.tag is not None:
        hydra_overrides.append(f"studies.runner.tag={args.tag}")

    if args.verbose is not None:
        hydra_overrides.append(f"studies.runner.verbose={str(args.verbose).lower()}")

    run_study_family(spec, hydra_overrides, dry_run=args.dry_run)
