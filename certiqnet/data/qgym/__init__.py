"""QGym data sources for CertiQ-Net.

Keep imports lazy so ``certiqnet.data.qgym.context`` can be imported
without triggering the full datamodule dependency graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "QGymDataModule",
    "QGymDataset",
]

if TYPE_CHECKING:  # pragma: no cover
    from certiqnet.data.qgym.datamodule import QGymDataModule
    from certiqnet.data.qgym.dataset import QGymDataset


def __getattr__(name: str):
    if name == "QGymDataModule":
        from certiqnet.data.qgym.datamodule import QGymDataModule

        return QGymDataModule
    if name == "QGymDataset":
        from certiqnet.data.qgym.dataset import QGymDataset

        return QGymDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
