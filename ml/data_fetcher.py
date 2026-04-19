"""Data fetchers for BTC/USDT market inputs used by training and live inference.

Active design:
- OHLCV comes from MEXC spot/futures.
- CVD comes from Gate.io futures contract_stats only.
- Training and live both normalize Gate CVD through the same internal schema
  before feature engineering.

There is intentionally no active MEXC CVD path in this module.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta

import httpx
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
    """Convert ccxt OHLCV list to a clean DataFrame."""
    df = pd.DataFrame(ohlcv, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


def _paginate_ohlcv(exchange, symbol: str, timeframe: str, start_ms: int, end_ms: int, batch: int = 500) -> pd.DataFrame:
    """Paginate ccxt fetch_ohlcv calls from start_ms to end_ms.

    MEXC spot caps at 500 candles per request; futures may allow more.
    We probe the actual page size from the first response and stop when
    returned count < that size (meaning we hit the end of history).
    """
    all_rows = []
    since = start_ms
    actual_page_size = None  # determined from first successful response

    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch)
        except Exception as e:
            log.warning("fetch_ohlcv error (%s %s since=%d): %s", symbol, timeframe, since, e)
            break
        if not ohlcv:
            break
        all_rows.extend(ohlcv)

        # Determine effective page size from first batch
        if actual_page_size is None:
            actual_page_size = len(ohlcv)

        last_ts = ohlcv[-1][0]
        # Stop if we reached end of requested range or got a partial page
        if last_ts >= end_ms or len(ohlcv) < actual_page_size:
            break
        since = last_ts + 1
        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = _ohlcv_to_df(all_rows)
    # Deduplicate on timestamp, sort ascending
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Filter to [start_ms, end_ms)
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_dt = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Section 3.1 — 5m candles (spot)
# ---------------------------------------------------------------------------

def fetch_5m(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch BTC/USDT 5m spot candles from MEXC."""
    import ccxt
    exchange = ccxt.mexc()
    exchange.load_markets()
    return _paginate_ohlcv(exchange, "BTC/USDT", "5m", start_ms, end_ms)


# ---------------------------------------------------------------------------
# Section 3.2 — 15m candles (swap/futures)
# ---------------------------------------------------------------------------

def fetch_15m(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch BTC/USDT:USDT 15m futures candles from MEXC."""
    import ccxt
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    exchange.load_markets()
    return _paginate_ohlcv(exchange, "BTC/USDT:USDT", "15m", start_ms, end_ms)


# ---------------------------------------------------------------------------
# Section 3.3 — 1h candles (swap/futures)
# ---------------------------------------------------------------------------

def fetch_1h(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch BTC/USDT:USDT 1h futures candles from MEXC."""
    import ccxt
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    exchange.load_markets()
    return _paginate_ohlcv(exchange, "BTC/USDT:USDT", "1h", start_ms, end_ms)


# ---------------------------------------------------------------------------
# Section 3.4 — Gate.io CVD (taker buy/sell volume)
# ---------------------------------------------------------------------------

GATE_CONTRACT_STATS_URL = "https://api.gateio.ws/api/v4/futures/usdt/contract_stats"
_GATE_CONTRACT = "BTC_USDT"
_GATE_MAX_LIMIT = 2000  # Gate allows up to 2000 rows per request


def fetch_gate_cvd(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch BTC/USDT 5m taker buy/sell volume from Gate.io contract_stats.

    Gate.io /futures/usdt/contract_stats returns per-candle aggregate stats
    including long_taker_size (aggressive buy volume), short_taker_size
    (aggressive sell volume), and open_interest — real exchange-reported taker
    flow and position data, no estimation.

    Gate timestamps are in SECONDS; we convert to ms for alignment with MEXC.
    Paginates forward from start_ms to end_ms in chunks of _GATE_MAX_LIMIT candles.

    Args:
        start_ms: Start of range in milliseconds UTC (inclusive).
        end_ms:   End of range in milliseconds UTC (exclusive).

    Returns:
        DataFrame with columns:
            timestamp         (datetime64[ms, UTC]) — 5m bucket open time
            long_taker_size   (float) — aggressive buy volume (contracts)
            short_taker_size  (float) — aggressive sell volume (contracts)
            open_interest     (float) — open interest in contracts
        Sorted ascending by timestamp, deduplicated, filtered to [start_ms, end_ms).
        Returns empty DataFrame with correct columns on any hard failure.
    """

    # Gate uses seconds; convert ms → s for params
    start_sec = start_ms // 1000
    end_sec = end_ms // 1000
    step_sec = _GATE_MAX_LIMIT * 300  # 2000 bars * 5 min = 10000 min = 600 000 s

    records: list[dict] = []
    cursor = start_sec

    log.info(
        "fetch_gate_cvd: fetching BTC_USDT 5m taker volume from Gate.io "
        "(start=%d, end=%d)", start_ms, end_ms,
    )

    with httpx.Client(timeout=30) as client:
        while cursor < end_sec:
            batch_end = min(cursor + step_sec, end_sec)
            params = {
                "contract": _GATE_CONTRACT,
                "interval": "5m",
                "from": cursor,
                "to": batch_end,
                "limit": _GATE_MAX_LIMIT,
            }
            try:
                resp = client.get(GATE_CONTRACT_STATS_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                log.warning(
                    "fetch_gate_cvd: request error at cursor=%d: %s", cursor, exc
                )
                break

            if not isinstance(data, list) or not data:
                log.info(
                    "fetch_gate_cvd: empty response at cursor=%d — stopping", cursor
                )
                break

            for row in data:
                try:
                    ts_sec = int(row["time"])
                    lts = float(row.get("long_taker_size", 0) or 0)
                    sts = float(row.get("short_taker_size", 0) or 0)
                    oi  = float(row.get("open_interest", 0) or 0)
                    records.append({
                        # Convert Gate seconds → ms for parity with all other DFs
                        "timestamp": pd.Timestamp(ts_sec * 1000, unit="ms", tz="UTC"),
                        "long_taker_size": lts,
                        "short_taker_size": sts,
                        "open_interest": oi,
                    })
                except (KeyError, TypeError, ValueError) as exc:
                    log.debug("fetch_gate_cvd: skipping malformed row: %s — %s", row, exc)
                    continue

            # Advance cursor past the last returned timestamp
            last_ts_sec = int(data[-1]["time"])
            if last_ts_sec >= end_sec or len(data) < _GATE_MAX_LIMIT:
                break  # reached end or partial page
            cursor = last_ts_sec + 300  # step one 5m bar forward
            time.sleep(0.1)

    log.info("fetch_gate_cvd: fetched %d raw candles total", len(records))

    if not records:
        log.warning("fetch_gate_cvd: no data returned for window [%d, %d)", start_ms, end_ms)
        return empty_cvd_frame()

    df = normalize_gate_cvd(pd.DataFrame(records))

    # Filter to requested range
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_dt = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].reset_index(drop=True)

    log.info(
        "fetch_gate_cvd: returning %d candles after range filter "
        "(%s → %s)",
        len(df),
        df["timestamp"].iloc[0].isoformat() if len(df) > 0 else "N/A",
        df["timestamp"].iloc[-1].isoformat() if len(df) > 0 else "N/A",
    )
    return df


def fetch_live_gate_cvd(limit: int = 400) -> pd.DataFrame:
    """Fetch the most recent `limit` 5m Gate.io CVD candles for live inference.

    Uses the `limit` parameter directly (no from/to pagination needed for
    recent data). Gate returns results in ascending timestamp order.

    Args:
        limit: Number of 5m candles to fetch (default 400, max 2000).

    Returns:
        DataFrame with columns: timestamp, long_taker_size, short_taker_size, open_interest
        Sorted ascending by timestamp. Returns empty DataFrame on failure.
    """

    params = {
        "contract": _GATE_CONTRACT,
        "interval": "5m",
        "limit": min(limit, _GATE_MAX_LIMIT),
    }
    log.info("fetch_live_gate_cvd: fetching last %d 5m candles from Gate.io", limit)

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(GATE_CONTRACT_STATS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        log.warning("fetch_live_gate_cvd: request error: %s", exc)
        return empty_cvd_frame()

    if not isinstance(data, list) or not data:
        log.warning("fetch_live_gate_cvd: empty response from Gate.io")
        return empty_cvd_frame()

    records = []
    for row in data:
        try:
            ts_sec = int(row["time"])
            lts = float(row.get("long_taker_size", 0) or 0)
            sts = float(row.get("short_taker_size", 0) or 0)
            oi  = float(row.get("open_interest", 0) or 0)
            records.append({
                "timestamp": pd.Timestamp(ts_sec * 1000, unit="ms", tz="UTC"),
                "long_taker_size": lts,
                "short_taker_size": sts,
                "open_interest": oi,
            })
        except (KeyError, TypeError, ValueError) as exc:
            log.debug("fetch_live_gate_cvd: skipping malformed row: %s — %s", row, exc)
            continue

    if not records:
        log.warning("fetch_live_gate_cvd: no valid records parsed from response")
        return empty_cvd_frame()

    df = normalize_gate_cvd(pd.DataFrame(records))
    log.info("fetch_live_gate_cvd: returning %d candles", len(df))
    return df


# ---------------------------------------------------------------------------
# Training fetch bundle — fetch last N months of active training inputs
# ---------------------------------------------------------------------------

def fetch_all(months: int = 5) -> dict:
    """Fetch the active training inputs for the last `months` months.

    Returns dict with keys: df5, df15, df1h, cvd
      cvd — Gate.io 5m taker buy/sell volume (long_taker_size, short_taker_size, open_interest)
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=months * 30)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    log.info("fetch_all: start=%s end=%s", start.isoformat(), now.isoformat())

    print(f"  Fetching 5m candles ({months} months)...")
    df5 = fetch_5m(start_ms, end_ms)
    print(f"  -> {len(df5)} 5m candles")

    print("  Fetching 15m candles...")
    df15 = fetch_15m(start_ms, end_ms)
    print(f"  -> {len(df15)} 15m candles")

    print("  Fetching 1h candles...")
    df1h = fetch_1h(start_ms, end_ms)
    print(f"  -> {len(df1h)} 1h candles")


    print("  Fetching Gate.io CVD (taker buy/sell volume)...")
    cvd = fetch_gate_cvd(start_ms, end_ms)
    print(f"  -> {len(cvd)} CVD candles")

    return {"df5": df5, "df15": df15, "df1h": df1h, "cvd": cvd}


# ---------------------------------------------------------------------------
# Live fetchers (for MLStrategy real-time inference)
# ---------------------------------------------------------------------------

def fetch_live_5m(limit: int = 400) -> pd.DataFrame:
    """Fetch last `limit` 5m candles from MEXC spot."""
    import ccxt
    exchange = ccxt.mexc()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="5m", limit=limit)
    df = _ohlcv_to_df(ohlcv)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_live_15m(limit: int = 100) -> pd.DataFrame:
    """Fetch last `limit` 15m candles from MEXC futures."""
    import ccxt
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe="15m", limit=limit)
    df = _ohlcv_to_df(ohlcv)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_live_1h(limit: int = 60) -> pd.DataFrame:
    """Fetch last `limit` 1h candles from MEXC futures."""
    import ccxt
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe="1h", limit=limit)
    df = _ohlcv_to_df(ohlcv)
    return df.sort_values("timestamp").reset_index(drop=True)
