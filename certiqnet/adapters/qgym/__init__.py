"""QGym adapter interface for CertiQ-Net.

Lazy imports prevent circular initialization when the adapter imports
QGym context helpers via ``certiqnet.data.qgym.context``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "QGymAdapter",
    "QGymCollectionConfig",
    "QGymEnvConfig",
    "SUPPORTED_POLICIES",
    "compute_effective_mu",
    "compute_queue_holding_cost",
    "load_qgym_env",
    "resolve_env_config_path",
]

if TYPE_CHECKING:  # pragma: no cover
    from certiqnet.adapters.qgym.adapter import QGymAdapter
    from certiqnet.adapters.qgym.config import (
        QGymCollectionConfig,
        QGymEnvConfig,
        SUPPORTED_POLICIES,
        resolve_env_config_path,
    )
    from certiqnet.adapters.qgym.env_loader import (
        compute_effective_mu,
        compute_queue_holding_cost,
        load_qgym_env,
    )


def __getattr__(name: str):
    if name == "QGymAdapter":
        from certiqnet.adapters.qgym.adapter import QGymAdapter

        return QGymAdapter
    if name in {"QGymCollectionConfig", "QGymEnvConfig", "SUPPORTED_POLICIES", "resolve_env_config_path"}:
        from certiqnet.adapters.qgym.config import (
            QGymCollectionConfig,
            QGymEnvConfig,
            SUPPORTED_POLICIES,
            resolve_env_config_path,
        )

        return {
            "QGymCollectionConfig": QGymCollectionConfig,
            "QGymEnvConfig": QGymEnvConfig,
            "SUPPORTED_POLICIES": SUPPORTED_POLICIES,
            "resolve_env_config_path": resolve_env_config_path,
        }[name]
    if name in {"compute_effective_mu", "compute_queue_holding_cost", "load_qgym_env"}:
        from certiqnet.adapters.qgym.env_loader import (
            compute_effective_mu,
            compute_queue_holding_cost,
            load_qgym_env,
        )

        return {
            "compute_effective_mu": compute_effective_mu,
            "compute_queue_holding_cost": compute_queue_holding_cost,
            "load_qgym_env": load_qgym_env,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
