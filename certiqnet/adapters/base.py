"""Abstract domain adapter interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class DispatchTuple:
    """Abstract dispatch tuple ``D = (N, lambda, mu, xi, Q, c)``."""

    N: int
    lam: float
    mu: Tensor
    xi: Tensor | None
    Q: Tensor
    cost: float


class DispatchAdapter(ABC):
    """Map a real system into the abstract dispatch model."""

    CERTIFICATE_STATUS: str

    @abstractmethod
    def to_dispatch_tuple(self, system_state: dict[str, object]) -> DispatchTuple:
        """Convert a domain state to the abstract dispatch tuple."""

    @abstractmethod
    def apply_action(self, action: int, system_state: dict[str, object]) -> dict[str, object]:
        """Apply a resource assignment action to a domain state."""

    @abstractmethod
    def certificate_assumptions(self) -> list[str]:
        """Return assumptions required for exact certificate inheritance."""
