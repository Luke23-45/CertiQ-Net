from __future__ import annotations

from certiqnet.experiments.catalog.runtime import sort_studies
from certiqnet.experiments.catalog.specs import StudySpec


class StudyEntry:
    __slots__ = ("spec",)

    def __init__(self, spec: StudySpec) -> None:
        self.spec = spec


def get_registered_studies() -> list[StudySpec]:
    from certiqnet.experiments.catalog.studies import _collect_specs

    return sort_studies(_collect_specs())


def get_study_entry(name: str) -> StudyEntry:
    for spec in get_registered_studies():
        if spec.name == name:
            return StudyEntry(spec=spec)
    raise KeyError(f"Unknown study: {name}")


def print_study_listing() -> None:
    studies = get_registered_studies()
    for study in studies:
        variants = ", ".join(v.label for v in study.variants)
        stages = ", ".join(study.stages)
        print(f"  {study.name}")
        print(f"    {study.description}")
        print(f"    config: {study.config_name}")
        print(f"    stages: {stages}")
        print(f"    variants: {variants}")
        print()
