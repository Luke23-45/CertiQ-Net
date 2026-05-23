"""Mixture-of-experts adapter with empirical certificate status by default."""

from certiqnet.adapters.queueing import QueueingAdapter


class MoEAdapter(QueueingAdapter):
    """Approximate adapter for MoE routing; CTMC theorem is not claimed exactly."""

    CERTIFICATE_STATUS = "approximate"

    def certificate_assumptions(self) -> list[str]:
        return [
            "Expert service approximated as exponential",
            "Batched execution effects ignored",
            "Empirical diagnostics required; exact CTMC theorem not claimed",
        ]
