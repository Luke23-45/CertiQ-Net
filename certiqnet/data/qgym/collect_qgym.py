"""Collect a QGym dataset to disk as ``.pt`` shards.

Usage
-----
    # Collect a registered dataset by name (primary workflow)
    python -m certiqnet.data.qgym.collect_qgym collect reentrant_2
    python -m certiqnet.data.qgym.collect_qgym collect reentrant_2 --force

    # List all known datasets
    python -m certiqnet.data.qgym.collect_qgym list

    # Show collection status
    python -m certiqnet.data.qgym.collect_qgym status
    python -m certiqnet.data.qgym.collect_qgym status reentrant_2 --verbose

    # Verify an existing dataset
    python -m certiqnet.data.qgym.collect_qgym verify reentrant_2

    # Print resolved config for a dataset
    python -m certiqnet.data.qgym.collect_qgym config reentrant_2

    # Legacy: collect from a collection YAML directly
    python -m certiqnet.data.qgym.collect_qgym --config path/to/collection.yaml

The primary workflow uses the **centralized dataset registry**
(``configs/dataset/qgym/*.yaml``) as the single source of truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certiqnet.adapters.qgym.config import QGymCollectionConfig
from certiqnet.data.collection_manager import DatasetCollectionManager
from certiqnet.data.registry import DatasetRegistry


# ── Helpers ──────────────────────────────────────────────────────────


def _print_status(status_data: dict[str, dict], verbose: bool = False) -> None:
    """Pretty-print dataset status."""
    for name, info in sorted(status_data.items()):
        exists = info.get("exists", False)
        path = info.get("path", "?")
        icon = "[x]" if exists else "[ ]"
        print(f"  {icon} {name}")
        print(f"       path: {path}")
        if "error" in info:
            print(f"       error: {info['error']}")
        elif verbose and exists and "metadata" in info:
            meta = info["metadata"]
            print(f"       N={meta.get('N', '?')}, "
                  f"train={meta.get('n_train', '?'):,}, "
                  f"valid={meta.get('n_valid', '?'):,}, "
                  f"test={meta.get('n_test', '?'):,}")
            print(f"       policy={meta.get('policy', '?')}, "
                  f"seed={meta.get('seed', '?')}")
            ct = meta.get("collection_time", "?")
            print(f"       collected: {ct}")
        print()


# ── Subcommands ──────────────────────────────────────────────────────


def cmd_collect(args: argparse.Namespace) -> None:
    """Collect a dataset by registry name."""
    registry = DatasetRegistry()
    manager = DatasetCollectionManager(registry)

    name = args.dataset_name
    if name is None:
        try:
            name = registry.get_default()
            print(f"[collect] No dataset specified — using registry default '{name}'.")
        except KeyError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    if name not in registry:
        print(f"Error: unknown dataset '{name}'.")
        print(f"Available: {', '.join(registry.list_datasets())}")
        sys.exit(1)

    if args.trust_override and args.n_steps is None:
        print(
            "[collect] warning: --trust-override has no effect without --n-steps; "
            "the collected dataset will still follow the registry spec."
        )

    spec = registry.get(name)
    if args.n_steps is not None:
        from certiqnet.data.registry import CollectionConfig, DatasetSpec

        spec = DatasetSpec(
            name=spec.name,
            env=spec.env,
            output_dir=spec.output_dir,
            collection=CollectionConfig(
                n_steps=args.n_steps,
                n_valid=spec.collection.n_valid,
                n_test=spec.collection.n_test,
                shard_size=spec.collection.shard_size,
                batch_size_env=spec.collection.batch_size_env,
                seed=spec.collection.seed,
                policy=spec.collection.policy,
                policy_weights=spec.collection.policy_weights,
            ),
            training_defaults=spec.training_defaults,
        )
        if args.trust_override:
            print(
                "[collect] warning: trust-override requested; the collected dataset "
                "will be annotated as an override/mismatch and must not be treated "
                "as registry-default data."
            )
    output = manager.collect(spec, force=args.force, skip_verify=args.no_verify)
    if args.n_steps is not None:
        meta_path = Path(output) / "metadata.yaml"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = yaml.safe_load(f) or {}
            meta["collection_override"] = {"n_steps": args.n_steps}
            meta["trust_override_requested"] = bool(args.trust_override)
            with open(meta_path, "w") as f:
                yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
            print(f"[collect] metadata annotated with collection override")
        spec_path = Path(output) / "dataset_spec.yaml"
        if spec_path.exists():
            with open(spec_path) as f:
                spec_d = yaml.safe_load(f) or {}
            spec_d["collection_override"] = {"n_steps": args.n_steps}
            spec_d["trust_override_requested"] = bool(args.trust_override)
            with open(spec_path, "w") as f:
                yaml.dump(spec_d, f, default_flow_style=False, sort_keys=False)
            print("[collect] dataset_spec annotated with collection override")
    print(f"\n[collect] done -> {output}")


def cmd_list(args: argparse.Namespace) -> None:  # noqa: ARG001
    """List all known datasets in the registry."""
    registry = DatasetRegistry()
    names = registry.list_datasets()
    if not names:
        print("No datasets found in registry.")
        return
    print(f"Known datasets ({len(names)}):")
    for name in names:
        spec = registry.get(name)
        exists = registry.exists(name)
        icon = "[x]" if exists else "[ ]"
        path = registry.resolve_path(name)
        print(f"  {icon} {name}  -> {path}")
        print(f"       env: {spec.env}")
        print(f"       collection: {spec.collection.n_steps:,} train + "
              f"{spec.collection.n_valid:,} valid + {spec.collection.n_test:,} test"
              f"  ({spec.collection.policy} policy)")


def cmd_status(args: argparse.Namespace) -> None:
    """Show collection status for all or one dataset."""
    registry = DatasetRegistry()
    data = registry.status(name=args.dataset_name)
    _print_status(data, verbose=args.verbose)


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify integrity of an existing dataset."""
    registry = DatasetRegistry()
    name = args.dataset_name
    if not registry.exists(name):
        print(f"Dataset '{name}' does not exist or is incomplete.")
        sys.exit(1)
    manager = DatasetCollectionManager(registry)
    path = registry.resolve_path(name)
    try:
        n = manager.verify(path)
        print(f"\n[verify] passed: {n} split(s) verified.")
    except Exception as exc:
        print(f"[verify] FAILED: {exc}")
        sys.exit(1)


def cmd_config(args: argparse.Namespace) -> None:
    """Print the resolved dataset spec for a dataset."""
    registry = DatasetRegistry()
    name = args.dataset_name
    if name not in registry:
        print(f"Error: unknown dataset '{name}'.")
        sys.exit(1)
    spec = registry.get(name)
    as_dict = registry.spec_to_dict(spec)
    yaml.dump(as_dict, sys.stdout, default_flow_style=False, sort_keys=False)


# ── Legacy compat ────────────────────────────────────────────────────


def cmd_legacy_collect(args: argparse.Namespace) -> None:
    """Legacy collection from a ``QGymCollectionConfig`` YAML file."""
    cfg = QGymCollectionConfig.from_yaml(args.config)
    if args.force:
        cfg.force_collect = True

    # Build a temporary DatasetSpec from the legacy config
    from certiqnet.data.registry import CollectionConfig, DatasetSpec

    spec = DatasetSpec(
        name=cfg.dataset_name,
        env=str(cfg.env_config_path),
        output_dir=str(cfg.output_dir),
        collection=CollectionConfig(
            n_steps=cfg.n_steps,
            n_valid=cfg.n_valid,
            n_test=cfg.n_test,
            shard_size=cfg.shard_size,
            batch_size_env=cfg.batch_size_env,
            seed=cfg.seed,
            policy=cfg.policy,
            policy_weights=cfg.policy_weights,
        ),
    )
    manager = DatasetCollectionManager()
    output = manager.collect(spec, force=args.force)
    print(f"\n[collect] done -> {output}")


# ── Main CLI ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and manage QGym datasets.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # ── collect ───────────────────────────────────────────────────────
    p_collect = subparsers.add_parser(
        "collect", help="Collect a dataset by registry name"
    )
    p_collect.add_argument("dataset_name", type=str, nargs="?", default=None,
                           help="Registered dataset name (default: registry default)")
    p_collect.add_argument("--force", action="store_true", help="Overwrite existing")
    p_collect.add_argument(
        "--no-verify", action="store_true", help="Skip post-collection verification"
    )
    p_collect.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override number of training states from registry default (e.g. --n-steps 10000)",
    )
    p_collect.add_argument(
        "--trust-override",
        action="store_true",
        help="Deprecated compatibility flag. Collects the override but never "
             "rewrites registry hashes; metadata is marked as a mismatch.",
    )

    # ── list ──────────────────────────────────────────────────────────
    subparsers.add_parser("list", help="List all known datasets")

    # ── status ────────────────────────────────────────────────────────
    p_status = subparsers.add_parser(
        "status", help="Show collection status for all or one dataset"
    )
    p_status.add_argument(
        "dataset_name", type=str, nargs="?", default=None, help="Dataset name (optional)"
    )
    p_status.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed stats"
    )

    # ── verify ────────────────────────────────────────────────────────
    p_verify = subparsers.add_parser(
        "verify", help="Verify integrity of an existing dataset"
    )
    p_verify.add_argument("dataset_name", type=str, help="Dataset name")

    # ── config ────────────────────────────────────────────────────────
    p_config = subparsers.add_parser(
        "config", help="Print resolved spec for a dataset"
    )
    p_config.add_argument("dataset_name", type=str, help="Dataset name")

    # ── Legacy: --config ─────────────────────────────────────────────
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "[Legacy] Path to a QGymCollectionConfig YAML. "
            "Replaced by ``collect <name>``."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="[Legacy] Overwrite existing dataset (used with --config).",
    )

    args = parser.parse_args()

    # Dispatch
    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.config:
        # Legacy path
        cmd_legacy_collect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
