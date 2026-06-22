from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class OverrideSpec:
    raw: str
    key: str
    value: str
    scope_method: str | None = None

    @classmethod
    def parse(cls, raw: str) -> OverrideSpec:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}': expected key=value syntax.")
        lhs, value = raw.split("=", 1)
        if ":" in lhs:
            scope_method, key = lhs.split(":", 1)
            if not scope_method or not key:
                raise ValueError(
                    f"Invalid override '{raw}': both scope and key must be "
                    f"non-empty in 'scope:key=value' form."
                )
            return cls(raw=raw, key=key, value=value, scope_method=scope_method)
        if not lhs:
            raise ValueError(f"Invalid override '{raw}': key must be non-empty.")
        return cls(raw=raw, key=lhs, value=value, scope_method=None)

    def applies_to_method(self, method: str) -> bool:
        if self.scope_method is None:
            return True
        return self.scope_method == method

    def to_hydra_arg(self) -> str:
        return f"{self.key}={self.value}"


@dataclass(frozen=True)
class TrainingVariant:
    label: str
    dataset: str
    model: str
    adapter: str
    seeds: Sequence[int]
    extra_overrides: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class StudySpec:
    name: str
    description: str
    config_name: str
    stages: tuple[str, ...]
    variants: Sequence[TrainingVariant]
    default_overrides: tuple[str, ...] = field(default_factory=tuple)
