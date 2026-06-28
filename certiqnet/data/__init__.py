"""Data modules and samplers for CertiQ-Net.

Keep package imports lazy to avoid circular import chains during
submodule initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "DatasetRegistry",
    "DatasetSpec",
    "DatasetCollectionManager",
]

if TYPE_CHECKING:  # pragma: no cover
    from certiqnet.data.collection_manager import DatasetCollectionManager
    from certiqnet.data.registry import DatasetRegistry, DatasetSpec


def __getattr__(name: str):
    if name == "DatasetRegistry":
        from certiqnet.data.registry import DatasetRegistry

        return DatasetRegistry
    if name == "DatasetSpec":
        from certiqnet.data.registry import DatasetSpec

        return DatasetSpec
    if name == "DatasetCollectionManager":
        from certiqnet.data.collection_manager import DatasetCollectionManager

        return DatasetCollectionManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
