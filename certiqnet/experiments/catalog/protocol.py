from __future__ import annotations

from typing import Protocol, runtime_checkable

from certiqnet.experiments.catalog.runtime import RunRequest, Stage
from certiqnet.experiments.catalog.specs import StudySpec


@runtime_checkable
class StudyRunner(Protocol):
    def load_studies(self) -> list[StudySpec]:
        ...

    def plan(self, study: StudySpec, request: RunRequest) -> list[Stage]:
        ...
