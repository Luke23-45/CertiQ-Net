"""Constrained parameterisation for analytic backbone parameters."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConstrainedBackboneParams(nn.Module):
    """Learnable backbone parameters with hard constraint enforcement."""

    def __init__(
        self,
        alpha_min: float = 1e-3,
        beta_min: float = 1e-3,
        beta_max: float = 10.0,
        gamma_max: float = 2.0,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        gamma_init: float = 0.0,
        c_init: float = 0.0,
    ) -> None:
        super().__init__()
        assert alpha_min > 0 and beta_min > 0, "Parameter lower bounds must be positive."
        assert beta_max >= beta_min, "beta_max must be >= beta_min."
        assert gamma_max > 0, "gamma_max must be positive."
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma_max = gamma_max
        self.raw_alpha = nn.Parameter(torch.tensor(self._inv_softplus(alpha_init - alpha_min)))
        self.raw_beta = nn.Parameter(torch.tensor(self._inv_softplus(beta_init - beta_min)))
        self.raw_gamma = nn.Parameter(torch.tensor(gamma_init))
        self.raw_c = nn.Parameter(torch.tensor(self._inv_softplus(max(c_init, 0.0) + 1e-6)))

    @staticmethod
    def _inv_softplus(x: float) -> float:
        if x <= 0:
            raise ValueError("Softplus inverse input must be strictly positive.")
        return math.log(math.expm1(x))

    @property
    def alpha(self) -> Tensor:
        return self.alpha_min + F.softplus(self.raw_alpha)

    @property
    def beta(self) -> Tensor:
        beta = self.beta_min + F.softplus(self.raw_beta)
        if self.beta_max < float("inf"):
            beta = beta.clamp(max=self.beta_max)
        return beta

    @property
    def gamma(self) -> Tensor:
        return self.gamma_max * torch.tanh(self.raw_gamma)

    @property
    def c(self) -> Tensor:
        return F.softplus(self.raw_c)

    def envelope_constant(self, mu: Tensor, N: int | None = None) -> Tensor:
        """Return exact backbone envelope constant ``C_B`` from the proof file."""
        assert (mu > 0).all(), "Service rates must be strictly positive."
        n = int(N or mu.shape[-1])
        kappa = self.c / mu.pow(self.beta) - (self.gamma / self.alpha) * torch.log(mu)
        return torch.log(torch.tensor(float(n), device=mu.device)) / self.alpha + (
            kappa.max(dim=-1).values - kappa.min(dim=-1).values
        )
