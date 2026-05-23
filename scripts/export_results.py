"""Export dual performance/certificate result tables."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultColumns:
    """Mandatory result table columns."""

    performance: tuple[str, ...] = (
        "avg_queue_length",
        "avg_cost",
        "latency_proxy",
        "p95_backlog",
        "max_backlog",
        "drop_rate",
    )
    certificate: tuple[str, ...] = (
        "drift_violation_rate",
        "min_drift_slack",
        "avg_drift_slack",
        "projection_activation_rate",
        "tail_fallback_activation_rate",
        "gate_activation_rate",
        "residual_logit_magnitude",
        "instability_rate",
    )


def main() -> None:
    """Print mandatory result schema."""
    cols = ResultColumns()
    print("performance," + ",".join(cols.performance))
    print("certificate," + ",".join(cols.certificate))


if __name__ == "__main__":
    main()
