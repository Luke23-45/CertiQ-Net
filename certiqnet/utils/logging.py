"""Rich-based logger factory."""

import logging

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Return a configured Rich logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
        force=False,
    )
    return logging.getLogger(name)
