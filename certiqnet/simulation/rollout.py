"""Rollout utilities."""

import torch
from torch import Tensor

from certiqnet.simulation.ctmc import CTMCEnvironment


def run_rollout(model: torch.nn.Module, env: CTMCEnvironment, steps: int) -> dict[str, Tensor]:
    """Run a model-controlled CTMC rollout for a fixed number of events."""
    costs: list[Tensor] = []
    dts: list[Tensor] = []
    for _ in range(steps):
        mu = env.mu.unsqueeze(0).expand(env.B, -1)
        with torch.no_grad():
            pi, _ = model(env.Q, mu, training_mode=False)
        out = env.step(pi)
        costs.append(out["cost"])
        dts.append(out["dt"])
    return {"cost": torch.stack(costs), "dt": torch.stack(dts), "Q": env.Q.clone()}


def collect_transitions(
    model: torch.nn.Module, env: CTMCEnvironment, steps: int
) -> dict[str, Tensor]:
    """Collect rollout states and actions for training buffers."""
    states: list[Tensor] = []
    probs: list[Tensor] = []
    for _ in range(steps):
        mu = env.mu.unsqueeze(0).expand(env.B, -1)
        with torch.no_grad():
            pi, _ = model(env.Q, mu, training_mode=False)
        states.append(env.Q.clone())
        probs.append(pi.clone())
        env.step(pi)
    return {"Q": torch.stack(states), "pi": torch.stack(probs)}
