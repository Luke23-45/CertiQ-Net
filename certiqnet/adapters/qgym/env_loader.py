from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from certiqnet.adapters.qgym.config import QGymEnvConfig, resolve_env_config_path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  QGym submodule discovery
# ---------------------------------------------------------------------------


def _find_qgym_root() -> Path:
    """Locate the QGym submodule from the project root.

    Navigates from this file (``certiqnet/adapters/qgym/env_loader.py``)
    up three parents to reach the project root, then checks
    ``extern/QGym``.
    """
    here = Path(__file__).resolve().parent          # .../certiqnet/adapters/qgym
    project_root = here.parents[2]                  # .../CertiQ-Net
    qgym_root = project_root / "extern" / "QGym"
    if qgym_root.exists():
        return qgym_root
    raise FileNotFoundError(
        f"QGym submodule not found at expected path: {qgym_root}.  "
        "Run `git submodule update --init` from the project root."
    )


def _ensure_qgym_importable() -> None:
    """Ensure the QGym module is on ``sys.path``."""
    qgym_root = _find_qgym_root()
    rl_root = qgym_root / "RL"
    for p in [str(qgym_root), str(rl_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Apply runtime patches to the QGym submodule before any import.
    from certiqnet.utils.qgym_patch import apply_qgym_patches

    apply_qgym_patches()


# ---------------------------------------------------------------------------
#  Environment instantiation
# ---------------------------------------------------------------------------


def _load_npy_or_raise(path: Path, description: str) -> np.ndarray:
    """Load a ``.npy`` file with a clear error on failure."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required QGym data file missing: {path}  ({description})"
        )
    return np.load(str(path))


def load_qgym_env(
    env_config: str | Path | dict | QGymEnvConfig,
    batch: int = 1,
    seed: int = 42,
    policy_name: str = "WC",
    device: torch.device | str = "cpu",
):
    """Instantiate a QGym ``DiffDiscreteEventSystem`` from a config source.

    Parameters
    ----------
    env_config : str | Path | dict | QGymEnvConfig
        YAML path (relative or absolute), raw ``dict``, or a ``QGymEnvConfig``.
    batch : int
        Number of parallel environments (default 1).
    seed : int
        Random seed for reproducibility.
    policy_name : str
        QGym policy name (``"WC"``, ``"vanilla"``, ``"discrete"``).
    device : torch.device | str
        Torch device.

    Returns
    -------
    dq : RL_Wrapper_P_DiffDiscreteEventSystem
        Ready-to-use QGym environment (SB3-compatible wrapper).
    """
    _ensure_qgym_importable()
    from RL.utils.rl_env import load_rl_p_env

    # ── Normalise to dict ────────────────────────────────────────────
    if isinstance(env_config, (str, Path)):
        resolved = resolve_env_config_path(str(env_config))
        cfg = QGymEnvConfig.from_yaml(resolved)
        raw_dict = dataclasses.asdict(cfg)
    elif isinstance(env_config, QGymEnvConfig):
        raw_dict = dataclasses.asdict(env_config)
    elif isinstance(env_config, dict):
        raw_dict = env_config.copy()
    else:
        raise TypeError(f"Unsupported env_config type: {type(env_config)}")

    # ── Fill in missing arrays from .npy data files ──────────────────
    if raw_dict.get("network") is None or raw_dict.get("mu") is None:
        env_type = raw_dict.get("env_type") or raw_dict.get("name")
        qgym_root = _find_qgym_root()
        data_dir = qgym_root / "configs" / "env_data" / env_type

        if raw_dict.get("network") is None:
            arr = _load_npy_or_raise(
                data_dir / f"{env_type}_network.npy", "network matrix"
            )
            raw_dict["network"] = arr.tolist()

        if raw_dict.get("mu") is None:
            arr = _load_npy_or_raise(
                data_dir / f"{env_type}_mu.npy", "service-rate matrix"
            )
            raw_dict["mu"] = arr.tolist()

        if raw_dict.get("queue_event_options") == "custom":
            arr = _load_npy_or_raise(
                data_dir / f"{env_type}_delta.npy", "arrival delta"
            )
            raw_dict["queue_event_options"] = arr.tolist()

        if raw_dict.get("lam_params", {}).get("val") is None:
            lam_path = data_dir / f"{env_type}_lam.npy"
            if lam_path.exists():
                raw_dict.setdefault("lam_params", {})["val"] = (
                    np.load(str(lam_path)).tolist()
                )

    # ── Remove keys not expected by load_rl_p_env ────────────────────
    for key in ("env_type", "num_pool", "server_pool_size"):
        raw_dict.pop(key, None)

    device = torch.device(device) if isinstance(device, str) else device
    env = load_rl_p_env(
        env_config=raw_dict,
        temp=1.0,
        batch=batch,
        seed=seed,
        policy_name=policy_name,
        device=device,
    )
    return env


# ---------------------------------------------------------------------------
#  Effective-rate & holding-cost helpers
# ---------------------------------------------------------------------------


def compute_effective_mu(
    network: torch.Tensor,
    mu_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute per-queue effective service rate from ``(s, q)`` topology.

    ``effective_mu[q] = sum_s network[s, q] * mu_matrix[s, q]``

    This is the total service capacity directed at each queue.
    """
    return (network * mu_matrix).sum(dim=0)


def compute_queue_holding_cost(
    Q: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Compute holding cost ``cost_i = Q_i @ h``.

    Parameters
    ----------
    Q : Tensor
        Queue-length matrix of shape ``(B, N)``.
    h : Tensor
        Per-queue holding-cost vector of shape ``(N,)``.

    Returns
    -------
    Tensor of shape ``(B,)``.
    """
    if h.dim() == 1:
        h = h.unsqueeze(0)
    return (Q * h).sum(dim=-1)
