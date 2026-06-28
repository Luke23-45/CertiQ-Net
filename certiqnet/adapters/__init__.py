"""Domain adapters for CertiQ-Net.

Lazy imports avoid circular initialization when subpackages import each
other indirectly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "AdapterBatch",
    "ChannelAdapter",
    "DispatchAdapter",
    "DispatchTuple",
    "QGymAdapter",
    "QueueingAdapter",
]

if TYPE_CHECKING:  # pragma: no cover
    from certiqnet.adapters.channel.adapter import ChannelAdapter
    from certiqnet.adapters.common.base import AdapterBatch, DispatchAdapter, DispatchTuple
    from certiqnet.adapters.qgym.adapter import QGymAdapter
    from certiqnet.adapters.queueing.adapter import QueueingAdapter


def __getattr__(name: str):
    if name in {"AdapterBatch", "DispatchAdapter", "DispatchTuple"}:
        from certiqnet.adapters.common.base import AdapterBatch, DispatchAdapter, DispatchTuple

        return {
            "AdapterBatch": AdapterBatch,
            "DispatchAdapter": DispatchAdapter,
            "DispatchTuple": DispatchTuple,
        }[name]
    if name == "ChannelAdapter":
        from certiqnet.adapters.channel.adapter import ChannelAdapter

        return ChannelAdapter
    if name == "QGymAdapter":
        from certiqnet.adapters.qgym.adapter import QGymAdapter

        return QGymAdapter
    if name == "QueueingAdapter":
        from certiqnet.adapters.queueing.adapter import QueueingAdapter

        return QueueingAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
