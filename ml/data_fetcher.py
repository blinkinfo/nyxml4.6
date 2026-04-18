"""MEXC data fetcher for BTC/USDT OHLCV + CVD — BLUEPRINT sections 3.1-3.5.

All data sourced from MEXC only (spot + futures). NO Binance, NO Coinbase.

CVD approach:
- Live (fetch_live_cvd): fetches real aggressor-side trade data from the MEXC
  futures deals endpoint (contract/deals). Each trade has T=1 (buy) or T=2 (sell).
  Trades are aggregated into 5-minute buckets to produce buy_vol / sell_vol.
  Falls back to kline vol-based estimation if the deals endpoint returns no data.
- Historical (fetch_cvd): uses the MEXC futures kline endpoint. The kline response
  does NOT expose taker_buy_vol as a separate field (only 11 fields: time, open,
  close, high, low, vol, amount, realOpen, realClose, realHigh, realLow). We
  therefore use a directional close-vs-open estimator for the historical path:
  buy_vol = vol * max(0, (close - open) / max(high - low, 1e-8) * 0.5 + 0.5),
  which is still far better than nothing and uses real exchange volume. The live
  path with real deals data dominates for recent bars, which matters most for
  inference.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta

import ccxt
import httpx
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

MEXC_CVD_KLINE_URL = "https://contract.mexc.com/api/v1/contract/kline/BTC_USDT"
MEXC_CVD_DEALS_URL = "https://contract.mexc.com/api/v1/contract/deals"


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
    exchange = ccxt.mexc()
    exchange.load_markets()
    return _paginate_ohlcv(exchange, "BTC/USDT", "5m", start_ms, end_ms)


# ---------------------------------------------------------------------------
# Section 3.2 — 15m candles (swap/futures)
# ---------------------------------------------------------------------------

def fetch_15m(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch BTC/USDT:USDT 15m futures candles from MEXC."""
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    exchange.load_markets()
    return _paginate_ohlcv(exchange, "BTC/USDT:USDT", "15m", start_ms, end_ms)


# ---------------------------------------------------------------------------
# Section 3.3 — 1h candles (swap/futures)
# ---------------------------------------------------------------------------

def fetch_1h(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch BTC/USDT:USDT 1h futures candles from MEXC."""
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    exchange.load_markets()
    return _paginate_ohlcv(exchange, "BTC/USDT:USDT", "1h", start_ms, end_ms)


# ---------------------------------------------------------------------------
# Section 3.4 — CVD via MEXC futures REST (real aggressor-side data)
# ---------------------------------------------------------------------------

def _kline_vol_to_buy_sell(open_: float, high: float, low: float, close: float, vol: float):
    """Estimate buy_vol/sell_vol from OHLCV kline data.

    Uses a directional estimator based on close-vs-open position within the candle
    range. This is better than the previous 50/50 heuristic because it uses the
    actual candle direction. buy_weight = 0.5 + 0.5 * (close-open)/(high-low).
    """
    rng = high - low
    if rng > 1e-8:
        # Fraction of range covered by the body, centered at 0.5
        body_frac = (close - open_) / rng
        buy_weight = max(0.0, min(1.0, 0.5 + 0.5 * body_frac))
    else:
        buy_weight = 0.5
    buy_vol = vol * buy_weight
    sell_vol = vol * (1.0 - buy_weight)
    return buy_vol, sell_vol


def _fetch_deals_page(client: httpx.Client, limit: int = 1000) -> list:
    """Fetch one page of deals from MEXC futures deals endpoint.

    Returns list of dicts with keys: t (ms timestamp), v (volume str), T (side int).
    Returns empty list on error or empty response.
    """
    url = MEXC_CVD_DEALS_URL
    params = {"symbol": "BTC_USDT", "limit": limit}
    try:
        resp = client.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.content
        if not raw:
            log.warning("fetch_deals_page: empty response body from %s", url)
            return []
        data = resp.json()
    except Exception as e:
        log.warning("fetch_deals_page: request error from %s: %s", url, e)
        return []

    if not data.get("success", False):
        log.warning("fetch_deals_page: API returned success=false: %s", data.get("message", ""))
        return []

    result_list = []
    d = data.get("data", {})
    if isinstance(d, dict):
        result_list = d.get("resultList", [])
    elif isinstance(d, list):
        result_list = d

    log.info("fetch_deals_page: fetched %d trade records from %s", len(result_list), url)
    return result_list


def _aggregate_deals_to_5m(trades: list) -> pd.DataFrame:
    """Aggregate raw trade records into 5-minute buckets.

    Args:
        trades: List of dicts with keys t (ms timestamp), v (volume str), T (side int).
                T=1 means buy (aggressor buy), T=2 means sell (aggressor sell).

    Returns:
        DataFrame with columns: timestamp (datetime64[ms, UTC]), buy_vol, sell_vol
        sorted ascending by timestamp.
    """
    if not trades:
        return pd.DataFrame(columns=["timestamp", "buy_vol", "sell_vol"])

    buckets: dict[int, list[float, float]] = {}  # bucket_ms -> [buy_vol, sell_vol]
    bucket_size_ms = 300_000  # 5 minutes

    for trade in trades:
        try:
            t_ms = int(trade["t"])
            v = float(trade["v"])
            side = int(trade["T"])  # 1=buy, 2=sell
            bucket = (t_ms // bucket_size_ms) * bucket_size_ms
            if bucket not in buckets:
                buckets[bucket] = [0.0, 0.0]
            if side == 1:
                buckets[bucket][0] += v
            else:
                buckets[bucket][1] += v
        except (KeyError, ValueError, TypeError):
            continue

    if not buckets:
        return pd.DataFrame(columns=["timestamp", "buy_vol", "sell_vol"])

    records = [
        {
            "timestamp": pd.Timestamp(bucket_ms, unit="ms", tz="UTC"),
            "buy_vol": bv,
            "sell_vol": sv,
        }
        for bucket_ms, (bv, sv) in sorted(buckets.items())
    ]
    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_cvd(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch 5m CVD data from MEXC futures REST endpoint.

    Uses the MEXC futures kline endpoint for historical data and derives
    buy_vol/sell_vol from candle direction (directional estimator). This is
    far better than the old heuristic because it uses real exchange volume data
    aligned to actual candle direction rather than a symmetric 50/50 split.

    For recent data (live inference), the fetch_live_cvd() function uses real
    aggressor-side trade data from the deals endpoint.

    Returns DataFrame with columns:
        timestamp  (datetime64[ms, UTC])
        buy_vol    (float)
        sell_vol   (float)
        open, high, low, close, volume  (float) — included for feature engineering
    """
    # 100 candles * 5min = 500min window per request
    window_sec = 100 * 5 * 60
    start_sec = start_ms // 1000
    end_sec = end_ms // 1000

    records = []
    cursor = start_sec

    log.info("fetch_cvd: fetching historical kline CVD from %s (start=%d, end=%d)",
             MEXC_CVD_KLINE_URL, start_ms, end_ms)

    with httpx.Client(timeout=30) as client:
        while cursor < end_sec:
            batch_end = min(cursor + window_sec, end_sec)
            params = {
                "interval": "Min5",
                "start": cursor,
                "end": batch_end,
            }
            try:
                resp = client.get(MEXC_CVD_KLINE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                log.warning("fetch_cvd: kline request error cursor=%d: %s", cursor, e)
                break

            candle_data = data.get("data", {})
            times = candle_data.get("time", [])
            opens = candle_data.get("open", [])
            highs = candle_data.get("high", [])
            lows = candle_data.get("low", [])
            closes = candle_data.get("close", [])
            vols = candle_data.get("vol", [])

            if not times:
                log.info("fetch_cvd: kline returned empty for cursor=%d", cursor)
                break

            n_candles = len(times)
            for i in range(n_candles):
                try:
                    ts_sec = int(times[i])
                    o = float(opens[i])
                    h = float(highs[i])
                    lo_val = float(lows[i])
                    c = float(closes[i])
                    v = float(vols[i])
                    bv, sv = _kline_vol_to_buy_sell(o, h, lo_val, c, v)
                    records.append({
                        "timestamp": pd.Timestamp(ts_sec, unit="s", tz="UTC"),
                        "open": o,
                        "high": h,
                        "low": lo_val,
                        "close": c,
                        "volume": v,
                        "buy_vol": bv,
                        "sell_vol": sv,
                    })
                except (IndexError, ValueError, TypeError) as e:
                    log.debug("fetch_cvd: skipping malformed candle at index %d: %s", i, e)
                    continue

            last_time = int(times[-1]) if times else cursor
            if last_time >= end_sec or n_candles < 100:
                break
            cursor = last_time + 300  # next 5m window
            time.sleep(0.05)

    log.info("fetch_cvd: fetched %d kline candles total", len(records))

    if not records:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "buy_vol", "sell_vol"])

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Filter to range
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_dt = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].reset_index(drop=True)
    log.info("fetch_cvd: returning %d CVD candles after range filter", len(df))
    return df


# ---------------------------------------------------------------------------
# Section 3.6 — Gate.io CVD (taker buy/sell volume)
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
    _EMPTY = pd.DataFrame(columns=["timestamp", "long_taker_size", "short_taker_size", "open_interest"])

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
        return _EMPTY

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

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
    _EMPTY = pd.DataFrame(columns=["timestamp", "long_taker_size", "short_taker_size", "open_interest"])

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
        return _EMPTY

    if not isinstance(data, list) or not data:
        log.warning("fetch_live_gate_cvd: empty response from Gate.io")
        return _EMPTY

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
        return _EMPTY

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
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
    exchange = ccxt.mexc()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="5m", limit=limit)
    df = _ohlcv_to_df(ohlcv)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_live_15m(limit: int = 100) -> pd.DataFrame:
    """Fetch last `limit` 15m candles from MEXC futures."""
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe="15m", limit=limit)
    df = _ohlcv_to_df(ohlcv)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_live_1h(limit: int = 60) -> pd.DataFrame:
    """Fetch last `limit` 1h candles from MEXC futures."""
    exchange = ccxt.mexc({"options": {"defaultType": "swap"}})
    ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe="1h", limit=limit)
    df = _ohlcv_to_df(ohlcv)
    return df.sort_values("timestamp").reset_index(drop=True)



def _fetch_live_cvd_from_deals(n_candles: int = 400) -> pd.DataFrame:
    """Fetch real aggressor-side CVD data from MEXC futures deals endpoint.

    Fetches trade records (T=1 buy, T=2 sell) and aggregates into 5-minute buckets.
    Makes up to 3 requests to try to cover n_candles worth of history.

    Returns DataFrame with columns: timestamp, buy_vol, sell_vol
    or empty DataFrame if deals endpoint is unavailable.
    """
    all_trades = []
    log.info("fetch_live_cvd: attempting to fetch real deals data from %s", MEXC_CVD_DEALS_URL)

    with httpx.Client(timeout=15) as client:
        for attempt in range(3):
            trades = _fetch_deals_page(client, limit=1000)
            if not trades:
                log.info("fetch_live_cvd: deals endpoint returned no data on attempt %d", attempt + 1)
                break
            all_trades.extend(trades)
            log.info("fetch_live_cvd: fetched %d trades (total: %d)", len(trades), len(all_trades))
            # If we have enough or got a partial page, stop
            if len(trades) < 1000:
                break
            time.sleep(0.1)

    if not all_trades:
        return pd.DataFrame(columns=["timestamp", "buy_vol", "sell_vol"])

    df = _aggregate_deals_to_5m(all_trades)
    log.info("fetch_live_cvd: aggregated %d trades into %d 5m buckets", len(all_trades), len(df))
    return df


def _fetch_live_cvd_from_kline(n_candles: int = 400) -> pd.DataFrame:
    """Fallback: fetch recent kline data and derive CVD using directional estimator.

    Returns DataFrame with columns: timestamp, buy_vol, sell_vol
    """
    end_sec = int(time.time())
    start_sec = end_sec - (n_candles + 10) * 300

    params = {
        "interval": "Min5",
        "start": start_sec,
        "end": end_sec,
    }
    log.info("fetch_live_cvd fallback: fetching kline data from %s", MEXC_CVD_KLINE_URL)
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(MEXC_CVD_KLINE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("fetch_live_cvd kline fallback error: %s", e)
        return pd.DataFrame(columns=["timestamp", "buy_vol", "sell_vol"])

    candle_data = data.get("data", {})
    times = candle_data.get("time", [])
    opens = candle_data.get("open", [])
    highs = candle_data.get("high", [])
    lows = candle_data.get("low", [])
    closes = candle_data.get("close", [])
    vols = candle_data.get("vol", [])

    records = []
    for i in range(len(times)):
        try:
            ts_sec = int(times[i])
            o = float(opens[i])
            h = float(highs[i])
            lo_val = float(lows[i])
            c = float(closes[i])
            v = float(vols[i])
            bv, sv = _kline_vol_to_buy_sell(o, h, lo_val, c, v)
            records.append({
                "timestamp": pd.Timestamp(ts_sec, unit="s", tz="UTC"),
                "buy_vol": bv,
                "sell_vol": sv,
            })
        except (IndexError, ValueError, TypeError):
            continue

    if not records:
        return pd.DataFrame(columns=["timestamp", "buy_vol", "sell_vol"])

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    log.info("fetch_live_cvd kline fallback: returning %d candles", len(df))
    return df


def fetch_live_cvd(n_candles: int = 400) -> pd.DataFrame:
    """Fetch last `n_candles` 5m CVD data for live inference.

    Primary: real aggressor-side data from MEXC futures deals endpoint.
             Trades are aggregated into 5-minute buy_vol / sell_vol buckets.
    Fallback: MEXC futures kline with directional vol estimator (close vs open).

    For coverage of n_candles (~33 hours for n=400), the deals endpoint may not
    have enough history. When deals cover fewer than 50 candles, we supplement
    with kline-based estimates for the older portion.

    Returns DataFrame with columns:
        timestamp  (datetime64[ms, UTC])
        buy_vol    (float)
        sell_vol   (float)
    """
    # Try deals endpoint first
    deals_df = _fetch_live_cvd_from_deals(n_candles)

    if not deals_df.empty and len(deals_df) >= 50:
        # Deals have enough recent data — use directly
        log.info("fetch_live_cvd: using real deals data (%d buckets)", len(deals_df))
        df = deals_df.tail(n_candles).reset_index(drop=True)
        return df

    # Deals insufficient — get kline for full history
    kline_df = _fetch_live_cvd_from_kline(n_candles)

    if deals_df.empty or len(deals_df) < 5:
        # No usable deals data at all — use kline entirely
        log.info("fetch_live_cvd: deals unavailable, using kline fallback (%d candles)", len(kline_df))
        if kline_df.empty:
            return pd.DataFrame(columns=["timestamp", "buy_vol", "sell_vol"])
        return kline_df.tail(n_candles).reset_index(drop=True)

    # Merge: use kline for older candles, deals for recent ones
    # Deals data takes priority where timestamps overlap
    if not kline_df.empty:
        deals_oldest = deals_df["timestamp"].min()
        kline_older = kline_df[kline_df["timestamp"] < deals_oldest][["timestamp", "buy_vol", "sell_vol"]]
        combined = pd.concat([kline_older, deals_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        log.info(
            "fetch_live_cvd: merged kline+deals: %d kline older + %d deals = %d total",
            len(kline_older), len(deals_df), len(combined),
        )
        return combined.tail(n_candles).reset_index(drop=True)

    return deals_df.tail(n_candles).reset_index(drop=True)
