"""Shared helpers used across all baseline policies."""

import torch
import torch.nn as nn
from torch import Tensor




def expand_mu(Q: Tensor, mu: Tensor) -> Tensor:
    if mu.dim() == 1:
        return mu.unsqueeze(0).expand(Q.shape[0], -1)
    return mu


def baseline_device(module: nn.Module, reference: Tensor) -> torch.device:
    param = next(module.parameters(), None)
    return reference.device if param is None else param.device



