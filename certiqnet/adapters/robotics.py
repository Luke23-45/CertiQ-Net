"""Multi-robot task-allocation adapter."""

from certiqnet.adapters.queueing import QueueingAdapter


class RoboticsAdapter(QueueingAdapter):
    """Robotics adapter; exact only under Poisson arrivals and exponential service."""

    CERTIFICATE_STATUS = "empirical"

    def certificate_assumptions(self) -> list[str]:
        return [
            "Poisson task arrivals",
            "Exponential task completion per robot",
            "Constant travel-time abstraction or separately certified generator",
        ]
