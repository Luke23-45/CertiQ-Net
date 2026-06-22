from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from certiqnet.experiments.catalog.specs import (
    OverrideSpec,
    StudySpec,
    TrainingVariant,
)


@dataclass(frozen=True)
class Stage:
    label: str
    command: list[str]
    cwd: Path


@dataclass(frozen=True)
class RunRequest:
    variant_labels: tuple[str, ...] = field(default_factory=tuple)
    seeds: tuple[int, ...] = field(default_factory=tuple)
    overrides: tuple[OverrideSpec, ...] = field(default_factory=tuple)
    skip_prep: bool = False


def sort_studies(studies: Iterable[StudySpec]) -> list[StudySpec]:
    def _sort_key(study: StudySpec) -> tuple[int, str]:
        prefix = study.name.split("_", 1)[0]
        try:
            return (int(prefix), study.name)
        except ValueError:
            return (9999, study.name)

    return sorted(studies, key=_sort_key)


def filter_overrides_for_method(
    overrides: Sequence[OverrideSpec], method: str
) -> list[OverrideSpec]:
    return [o for o in overrides if o.applies_to_method(method)]


def select_variants(
    study: StudySpec, variant_labels: Sequence[str]
) -> tuple[TrainingVariant, ...]:
    variants = tuple(study.variants)
    if not variant_labels:
        return variants

    by_label = {variant.label: variant for variant in variants}
    missing = [label for label in variant_labels if label not in by_label]
    if missing:
        available = ", ".join(sorted(by_label))
        joined = ", ".join(missing)
        raise KeyError(
            f"Unknown variant(s) for study '{study.name}': {joined}. "
            f"Available: {available}"
        )
    return tuple(by_label[label] for label in variant_labels)


def select_seeds(available: Sequence[int], requested: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in available)
    if not requested:
        return seeds

    requested_set = {int(seed) for seed in requested}
    missing = sorted(seed for seed in requested_set if seed not in seeds)
    if missing:
        available_text = ", ".join(str(seed) for seed in seeds)
        missing_text = ", ".join(str(seed) for seed in missing)
        raise KeyError(
            f"Unknown seed(s): {missing_text}. Available: {available_text}"
        )
    return tuple(seed for seed in seeds if seed in requested_set)
