"""Robust, configurable progress bar for all experiment types."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

from tqdm import tqdm  # type: ignore[import-untyped]

T = TypeVar("T")


@dataclass
class ProgressConfig:
    new_line_after_iteration: bool = True
    mininterval: float = 0.1
    maxinterval: float = 1.0
    miniters: int | None = None
    smoothing: float = 0.3
    dynamic_ncols: bool = True
    leave: bool = False
    position: int = 0
    unit: str = "it"
    bar_format: str | None = None
    ascii: bool = False
    ncols: int | None = None


DEFAULT_BAR_FORMAT = "{l_bar}{bar:20}{r_bar}"


class RobustProgressBar:
    """tqdm wrapper with configurable new-line behavior and robust formatting."""

    def __init__(self, config: ProgressConfig | None = None) -> None:
        self.cfg = config or ProgressConfig()

    def __call__(
        self,
        iterable: Iterable[T],
        *,
        total: int | None = None,
        desc: str = "",
        unit: str | None = None,
        disable: bool = False,
        **kwargs: Any,
    ) -> Iterator[T]:
        return self.iterate(iterable, total=total, desc=desc, unit=unit, disable=disable, **kwargs)

    def iterate(
        self,
        iterable: Iterable[T],
        *,
        total: int | None = None,
        desc: str = "",
        unit: str | None = None,
        disable: bool = False,
        **kwargs: Any,
    ) -> Iterator[T]:
        bar = tqdm(
            iterable,
            total=total,
            desc=desc,
            disable=disable,
            mininterval=kwargs.pop("mininterval", self.cfg.mininterval),
            maxinterval=kwargs.pop("maxinterval", self.cfg.maxinterval),
            miniters=kwargs.pop("miniters", self.cfg.miniters),
            smoothing=kwargs.pop("smoothing", self.cfg.smoothing),
            dynamic_ncols=kwargs.pop("dynamic_ncols", self.cfg.dynamic_ncols),
            leave=kwargs.pop("leave", self.cfg.leave),
            position=kwargs.pop("position", self.cfg.position),
            unit=kwargs.pop("unit", unit or self.cfg.unit),
            bar_format=kwargs.pop("bar_format", self.cfg.bar_format or DEFAULT_BAR_FORMAT),
            ascii=kwargs.pop("ascii", self.cfg.ascii),
            ncols=kwargs.pop("ncols", self.cfg.ncols),
            file=kwargs.pop("file", sys.stdout),
            **kwargs,
        )
        try:
            yield from bar
        finally:
            bar.close()
            if not disable and self.cfg.new_line_after_iteration:
                sys.stdout.write("\n")
                sys.stdout.flush()


_PROGRESS_BAR_INSTANCE: RobustProgressBar | None = None


def configure_progress(cfg: ProgressConfig | dict[str, Any] | None = None) -> RobustProgressBar:
    global _PROGRESS_BAR_INSTANCE
    if isinstance(cfg, dict):
        cfg = ProgressConfig(**cfg)
    _PROGRESS_BAR_INSTANCE = RobustProgressBar(cfg or ProgressConfig())
    return _PROGRESS_BAR_INSTANCE


def get_progress() -> RobustProgressBar:
    global _PROGRESS_BAR_INSTANCE
    if _PROGRESS_BAR_INSTANCE is None:
        _PROGRESS_BAR_INSTANCE = RobustProgressBar()
    return _PROGRESS_BAR_INSTANCE


def progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "",
    disable: bool = False,
    **kwargs: Any,
) -> Iterator[T]:
    return get_progress().iterate(iterable, total=total, desc=desc, disable=disable, **kwargs)
