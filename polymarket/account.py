"""Account helpers — balance, positions, connection status.

Uses the correct py-clob-client API:
  - get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
    returns {"balance": "<wei_str>", "allowance": "<wei_str>"}
    USDC has 6 decimals -> balance_usdc = int(balance_wei) / 1_000_000

  - Open positions are fetched via:
    1. ClobClient.get_positions() if available (future-proof)
    2. Fallback: authenticated REST GET /positions?user=<funder_address>
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

import config as cfg

log = logging.getLogger(__name__)

# USDC uses 6 decimal places on Polygon
_USDC_DECIMALS = 1_000_000


async def get_balance(poly_client) -> float | None:
    """Return USDC balance (float, rounded to 2 dp) for the wallet.

    Uses ClobClient.get_balance_allowance() with AssetType.COLLATERAL --
    the correct py-clob-client method for on-chain USDC balance.
    The returned 'balance' field is a wei string (6 decimals for USDC).

    Returns None on any error so the caller can display 'N/A'.
    """
    try:
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams  # type: ignore

        result: dict[str, Any] = await asyncio.to_thread(
            poly_client.client.get_balance_allowance,
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
        )

        if not isinstance(result, dict):
            log.error(
                "get_balance_allowance returned unexpected type %s: %r",
                type(result).__name__,
                result,
            )
            return None

        raw = result.get("balance")
        if raw is None:
            log.error("get_balance_allowance response missing 'balance' key: %r", result)
            return None

        balance_usdc = int(raw) / _USDC_DECIMALS
        return round(balance_usdc, 2)

    except Exception:
        log.exception("Failed to fetch Polymarket USDC balance")
        return None


async def get_open_positions(poly_client) -> list[dict[str, Any]]:
    """Return a list of open conditional-token positions for the wallet.

    Strategy (in order of preference):
      1. ClobClient.get_positions() -- present in newer py-clob-client builds.
      2. Authenticated REST GET /positions?user=<funder_address> -- stable
         fallback available on the public CLOB API.

    Returns an empty list on any error.
    """
    # Attempt 1 -- native client method
    native = getattr(poly_client.client, "get_positions", None)
    if callable(native):
        try:
            positions = await asyncio.to_thread(native)
            if isinstance(positions, list):
                log.debug("get_positions() returned %d position(s)", len(positions))
                return positions
            log.warning("get_positions() returned non-list %r -- falling back", type(positions))
        except Exception:
            log.warning("Native get_positions() failed -- falling back to REST", exc_info=True)

    # Attempt 2 -- authenticated REST endpoint
    funder = getattr(poly_client.config, "POLYMARKET_FUNDER_ADDRESS", None)
    if not funder:
        log.warning("POLYMARKET_FUNDER_ADDRESS not set -- cannot fetch positions")
        return []

    url = f"{cfg.CLOB_HOST}/positions"
    params = {"user": funder}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        if isinstance(data, list):
            log.debug("REST /positions returned %d position(s)", len(data))
            return data

        # Some API versions wrap the list in a dict
        if isinstance(data, dict):
            for key in ("positions", "data", "results"):
                if isinstance(data.get(key), list):
                    return data[key]

        log.warning("Unexpected /positions response shape: %r", type(data))
        return []

    except httpx.HTTPStatusError as exc:
        log.error(
            "REST /positions HTTP error %d for user=%s",
            exc.response.status_code,
            funder,
        )
        return []
    except Exception:
        log.exception("Failed to fetch open positions via REST")
        return []


async def get_connection_status(poly_client) -> bool:
    """Quick connectivity check -- call get_server_time on the CLOB.

    Returns True if the server responds, False on any error.
    """
    try:
        info = await asyncio.to_thread(poly_client.client.get_server_time)
        return info is not None
    except Exception:
        log.exception("Polymarket connection check failed")
        return False
