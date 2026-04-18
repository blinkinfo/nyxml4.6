"""Strategy orchestrator — delegates to the active strategy via registry.

The old monolithic check_signal() (with ADX, threshold logic) has been
replaced by a pluggable strategy system.  Each strategy implements
check_signal() and returns a standard sentinel dict.

Registry is loaded from core.strategies package.  The active strategy name
is stored in config (STRATEGY_NAME, default "pattern").
"""

from __future__ import annotations

import logging
from typing import Any

import config as cfg

log = logging.getLogger(__name__)

# Lazy-loaded strategy instance — set on first call.
_strategy: Any | None = None


def _get_strategy():
    """Return the active strategy instance, loading it lazily.

    Returns None if the strategy cannot be instantiated (constructor crashed
    or unknown name). The caller must check for None before calling
    check_signal() — returning None propagates the hard-error sentinel up
    to the scheduler which logs and skips the slot.
    """
    global _strategy
    if _strategy is None:
        strategy_name = getattr(cfg, "STRATEGY_NAME", "ml")
        try:
            from core.strategies import get_strategy
            _strategy = get_strategy(strategy_name)
            log.info("Strategy engine: loaded '%s' strategy", strategy_name)
        except Exception:
            log.exception(
                "Strategy engine: failed to instantiate strategy '%s' — "
                "check_signal() will return None (hard error) until the next "
                "successful instantiation attempt",
                strategy_name,
            )
            # Do NOT cache a broken/None value — allow retry on the next slot
            # so a transient initialization failure does not permanently disable
            # signals for the life of the process.
            return None
    return _strategy


async def check_signal() -> dict[str, Any] | None:
    """Delegate to the active strategy's check_signal().

    Returns the strategy's signal/skip dict, or None on hard failure.
    The orchestrator adds no extra logic — strategies return all fields
    needed by the scheduler.
    """
    strategy = _get_strategy()
    if strategy is None:
        # _get_strategy already logged the error; return hard-error sentinel
        return None
    return await strategy.check_signal()
