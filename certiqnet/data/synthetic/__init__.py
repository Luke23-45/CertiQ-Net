"""Synthetic data module for CertiQ‑Net."""

from certiqnet.data.synthetic.datamodule import CertiQNetDataModule
from certiqnet.data.synthetic.state_bank import generate_adversarial_states, generate_state_bank

__all__ = [
    "CertiQNetDataModule",
    "generate_adversarial_states",
    "generate_state_bank",
]
