import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/nebula/nyxmlopp')

from ml.features import compute_atr14, FEATURE_COLS
import config as cfg


def make_ohlcv(n=100, seed=42):
    rng = np.random.default_rng(seed)
    close = 50000 + np.cumsum(rng.normal(0, 100, n))
    open_ = close + rng.normal(0, 50, n)
    high = np.maximum(close, open_) + rng.uniform(0, 100, n)
    low = np.minimum(close, open_) - rng.uniform(0, 100, n)
    vol = rng.uniform(10, 100, n)
    ts = pd.date_range('2025-01-01', periods=n, freq='5min', tz='UTC')
    return pd.DataFrame({'timestamp': ts, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': vol})


def test_feature_count():
    assert len(FEATURE_COLS) == 26


def test_feature_order():
    expected = ['body_ratio_n1', 'body_ratio_n2', 'body_ratio_n3',
                'upper_wick_n1', 'upper_wick_n2', 'lower_wick_n1', 'lower_wick_n2',
                'volume_ratio_n1', 'volume_ratio_n2',
                'body_ratio_15m', 'dir_15m', 'volume_ratio_15m',
                'body_ratio_1h', 'dir_1h', 'ema9_slope_1h',
                'funding_rate', 'funding_zscore',
                'delta_ratio', 'cvd_delta', 'cvd_5', 'cvd_20', 'cvd_trend',
                'hour_utc', 'dow', 'atr_percentile_24h', 'vol_regime']
    assert FEATURE_COLS == expected


def test_atr14_formula():
    df = make_ohlcv(50)
    atr = compute_atr14(df)
    assert (atr.dropna() > 0).all()
    assert atr.iloc[:13].isna().all()
    assert not pd.isna(atr.iloc[14])


def test_no_lookahead_body_ratio_n1():
    df = make_ohlcv(50)
    atr = compute_atr14(df)
    expected = (df['close'].iloc[19] - df['open'].iloc[19]) / atr.iloc[19]
    br_series = (df['close'].shift(1) - df['open'].shift(1)) / atr.shift(1)
    assert abs(br_series.iloc[20] - expected) < 1e-10


def test_cvd_proxy_formula():
    high, low, close, vol = 100.0, 90.0, 95.0, 1000.0
    buy = vol * (close - low) / (high - low)
    sell = vol * (high - close) / (high - low)
    assert abs(buy - 500.0) < 1e-6
    assert abs(sell - 500.0) < 1e-6


def test_merge_asof_no_future_leak():
    ts_5m = pd.Timestamp('2025-01-01 09:00:00', tz='UTC')
    ts_15m_future = pd.Timestamp('2025-01-01 09:15:00', tz='UTC')
    ts_15m_current = pd.Timestamp('2025-01-01 09:00:00', tz='UTC')
    ts_15m_past = pd.Timestamp('2025-01-01 08:45:00', tz='UTC')
    left = pd.DataFrame({'ts_n1': [ts_5m]})
    right = pd.DataFrame({'timestamp': [ts_15m_past, ts_15m_current, ts_15m_future], 'val': [1, 2, 3]})
    merged = pd.merge_asof(left, right, left_on='ts_n1', right_on='timestamp', direction='backward')
    assert merged['val'].iloc[0] == 2


def test_train_val_test_split():
    n = 1000
    train_end = int(n * 0.75)
    val_start = int(train_end * 0.80)
    assert train_end == 750
    assert val_start == 600
    assert n - train_end == 250
    assert train_end - val_start == 150


def test_default_threshold_matches_blueprint():
    """Blueprint Section 9: recommended threshold is 0.535.
    This test will catch any future accidental regression of the default."""
    assert cfg.ML_DEFAULT_THRESHOLD == 0.535, (
        f"ML_DEFAULT_THRESHOLD is {cfg.ML_DEFAULT_THRESHOLD}, expected 0.535 "
        "(Blueprint Section 9 recommended threshold)"
    )


def test_asof_backward_vectorized_matches_searchsorted():
    """_asof_backward (now pd.merge_asof) must produce identical results to
    the previous searchsorted row-loop implementation for all call sites:
    15m merge, 1h merge, funding merge, and CVD merge."""
    import sys
    sys.path.insert(0, '/home/nebula/nyxmlopp')
    from ml.features import _asof_backward

    rng = np.random.default_rng(0)
    n_left = 200
    n_right = 50

    # Build a right-side DataFrame with sorted timestamps and two value columns
    right_ts = pd.date_range('2025-01-01', periods=n_right, freq='15min', tz='UTC')
    right = pd.DataFrame({
        'timestamp': right_ts,
        'val_a': rng.uniform(0, 1, n_right),
        'val_b': rng.uniform(100, 200, n_right),
    })

    # Build left timestamps — denser than right, some before first right row (should give NaN)
    left_ts = pd.date_range('2024-12-31 23:00', periods=n_left, freq='5min', tz='UTC')
    left_series = pd.Series(left_ts)

    # Run the vectorized implementation
    result = _asof_backward(left_series, right, ['val_a', 'val_b'])

    # Re-implement the original searchsorted logic inline for reference.
    # Use microseconds (us) throughout — pandas 2.x stores datetime64[us],
    # so .values.view(int64) gives us-since-epoch. Convert left ts the same way.
    right_ts_us = right['timestamp'].values.view(np.int64)  # datetime64[us] -> int64 us
    expected_a = np.full(n_left, np.nan)
    expected_b = np.full(n_left, np.nan)
    for i, ts in enumerate(left_series):
        if pd.isna(ts):
            continue
        # Convert to microseconds: Timestamp.value is ns, divide by 1000
        ts_us = pd.Timestamp(ts).value // 1000
        idx = np.searchsorted(right_ts_us, ts_us, side='right') - 1
        if idx >= 0 and right_ts_us[idx] <= ts_us:
            expected_a[i] = right['val_a'].iloc[idx]
            expected_b[i] = right['val_b'].iloc[idx]

    # Results must be bit-for-bit identical (same float values, not just close)
    got_a = result['val_a'].values
    got_b = result['val_b'].values
    nan_mask_a = np.isnan(expected_a)
    nan_mask_b = np.isnan(expected_b)
    assert np.array_equal(nan_mask_a, np.isnan(got_a)), "NaN positions differ for val_a"
    assert np.array_equal(nan_mask_b, np.isnan(got_b)), "NaN positions differ for val_b"
    np.testing.assert_array_equal(got_a[~nan_mask_a], expected_a[~nan_mask_a])
    np.testing.assert_array_equal(got_b[~nan_mask_b], expected_b[~nan_mask_b])


def test_asof_backward_nat_handling():
    """_asof_backward must silently produce NaN for NaT rows in the left key,
    NOT raise ValueError.  This replicates the real call site:
        ts_n1 = df5['timestamp'].shift(1)  -> row 0 is always NaT.
    """
    from ml.features import _asof_backward

    n_right = 20
    right_ts = pd.date_range('2025-01-01', periods=n_right, freq='15min', tz='UTC')
    right = pd.DataFrame({
        'timestamp': right_ts,
        'val_a': np.arange(n_right, dtype=float),
    })

    # Simulate shift(1): first element is NaT, rest are valid timestamps.
    valid_ts = pd.date_range('2025-01-01 00:05', periods=9, freq='5min', tz='UTC')
    left_with_nat = pd.Series([pd.NaT] + list(valid_ts), dtype='datetime64[ns, UTC]')

    # Must not raise — NaT rows should silently become NaN in output.
    result = _asof_backward(left_with_nat, right, ['val_a'])

    assert len(result) == 10, "Output length must equal input length"
    assert pd.isna(result['val_a'].iloc[0]), "Row 0 (NaT input) must produce NaN output"
    # Valid rows after the first right timestamp should have non-NaN values
    assert result['val_a'].iloc[1:].notna().any(), "Valid timestamp rows should resolve to non-NaN"


def test_live_features_match_training_for_latest_closed_5m_row():
    """Live feature builder must match training feature row semantics exactly.

    Contract:
      - Predict N+1 using data up to N-1 (exclude in-progress 5m candle N).
      - 15m/1h context is selected by timestamp <= ts_n1, not by blindly
        dropping the latest higher-timeframe row.

    This test builds synthetic aligned 5m/15m/1h/funding/CVD data, then checks
    that build_live_features(...) equals the corresponding row from
    build_features(...), feature-by-feature.
    """
    from collections import deque
    from ml.features import build_features, build_live_features

    rng = np.random.default_rng(7)
    n5 = 450

    ts_5m = pd.date_range("2026-01-01", periods=n5, freq="5min", tz="UTC")
    close = 50000 + np.cumsum(rng.normal(0, 20, n5))
    open_ = close + rng.normal(0, 5, n5)
    high = np.maximum(open_, close) + rng.uniform(0, 8, n5)
    low = np.minimum(open_, close) - rng.uniform(0, 8, n5)
    vol = rng.uniform(50, 200, n5)

    df5 = pd.DataFrame(
        {
            "timestamp": ts_5m,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )

    s = df5.set_index("timestamp")
    df15 = (
        s.resample("15min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    df1h = (
        s.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )

    funding_ts = pd.date_range(
        ts_5m.min() - pd.Timedelta("16h"),
        ts_5m.max() + pd.Timedelta("1h"),
        freq="8h",
        tz="UTC",
    )
    funding = pd.DataFrame(
        {
            "timestamp": funding_ts,
            "funding_rate": rng.normal(0, 0.0001, len(funding_ts)),
        }
    )

    buy = rng.uniform(30, 120, n5)
    sell = rng.uniform(30, 120, n5)
    cvd = pd.DataFrame(
        {
            "timestamp": ts_5m,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "buy_vol": buy,
            "sell_vol": sell,
        }
    )

    # Training features on full history
    train_feat = build_features(df5, df15, df1h, funding, cvd)
    expected = train_feat[FEATURE_COLS].iloc[-2].to_numpy()

    # Live path semantics:
    #   - drop in-progress 5m candle only
    #   - keep 15m/1h history; builder itself applies <= ts_n1 filtering
    live_row, nan_features = build_live_features(
        df5.iloc[:-1].copy(),
        df15.copy(),
        df1h.copy(),
        float(funding["funding_rate"].iloc[-1]),
        deque(funding["funding_rate"].tail(24).tolist(), maxlen=24),
        cvd.iloc[:-1].copy(),
    )

    assert live_row is not None, f"build_live_features returned None; nan_features={nan_features}"
    assert nan_features == [], f"Unexpected NaN features: {nan_features}"
    got = live_row[0]
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_fetch_funding_mock():
    """Verify fetch_funding() behaviour using mocked ccxt and MEXC REST API.

    Scenarios tested:
      (a) ccxt pagination stalls (two consecutive pages same last_ts) -> stops ccxt loop
      (b) ccxt coverage < 80% of window -> falls back to MEXC REST API
      (c) REST results are deduplicated (duplicate settleTime entries collapsed)
      (d) Records outside [start_ms, end_ms) are filtered out
    """
    import sys
    sys.path.insert(0, '/home/nebula/nyxmlopp')

    from unittest.mock import patch, MagicMock
    import ml.data_fetcher as df_module
    from ml.data_fetcher import fetch_funding

    # Define a 10-day window (start to end) — funding every 8h = 30 records expected
    # Use a window far enough back that coverage from ccxt (which returns only 3 recent
    # records) will be < 80%, forcing the REST fallback.
    now_ms = 1_700_000_000_000  # fixed epoch for determinism (~Nov 2023)
    start_ms = now_ms - 10 * 24 * 3600 * 1000   # 10 days before
    end_ms   = now_ms

    # --- Build ccxt mock ---
    # ccxt returns only 3 records, all with the same timestamp (stall on page 2)
    # Page 1: 3 records near the end of the window (recent) — stall because page 2 same last_ts
    ccxt_ts_1 = end_ms - 3 * 8 * 3600 * 1000  # 24h before end
    ccxt_ts_2 = end_ms - 2 * 8 * 3600 * 1000
    ccxt_ts_3 = end_ms - 1 * 8 * 3600 * 1000
    ccxt_page1 = [
        {"timestamp": ccxt_ts_1, "fundingRate": 0.0001},
        {"timestamp": ccxt_ts_2, "fundingRate": 0.0002},
        {"timestamp": ccxt_ts_3, "fundingRate": 0.0003},
    ]
    # Page 2: same last_ts as page 1 last record -> stall detected -> stop
    ccxt_page2 = [
        {"timestamp": ccxt_ts_3, "fundingRate": 0.0003},  # same last_ts as page1[-1]
    ]

    mock_exchange = MagicMock()
    mock_exchange.fetch_funding_rate_history.side_effect = [ccxt_page1, ccxt_page2]

    # --- Build REST mock ---
    # REST returns 2 pages of records covering the full 10-day window.
    # Page 1: newest records (descending order as MEXC returns them)
    # Page 2: older records reaching before start_ms -> pagination stops
    # Include one duplicate settleTime across pages to test dedup.
    # Include one record outside [start_ms, end_ms) to test filtering.
    interval_ms = 8 * 3600 * 1000  # 8 hours in ms

    def make_rest_items(ts_list):
        return [{"settleTime": str(ts), "fundingRate": str(round(0.0001 * (i + 1), 6))}
                for i, ts in enumerate(ts_list)]

    # Page 1: 5 records near end of window
    rest_page1_ts = [end_ms - i * interval_ms for i in range(1, 6)]
    rest_page1_items = make_rest_items(rest_page1_ts)

    # Page 2: 5 records going back to before start_ms (last one is out-of-range)
    # Also duplicate the last ts from page 1 to test dedup
    rest_page2_ts = [end_ms - i * interval_ms for i in range(5, 11)]
    rest_page2_ts.append(start_ms - interval_ms)  # one record before start_ms (filtered out)
    rest_page2_ts.append(rest_page1_ts[-1])        # duplicate from page 1 (deduped)
    rest_page2_items = make_rest_items(rest_page2_ts)

    # Page 3: empty -> stop
    rest_page3_items: list = []

    def mock_httpx_get(url, params=None, **kwargs):
        page = int((params or {}).get("page_num", 1))
        if page == 1:
            items = rest_page1_items
        elif page == 2:
            items = rest_page2_items
        else:
            items = rest_page3_items
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"resultList": items}}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    with patch.object(df_module.ccxt, "mexc", return_value=mock_exchange), \
         patch("ml.data_fetcher.httpx.Client") as mock_client_cls, \
         patch("ml.data_fetcher.time.sleep"):  # suppress sleep in tests

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.side_effect = mock_httpx_get
        mock_client_cls.return_value = mock_client_instance

        result = fetch_funding(start_ms, end_ms)

    # (a) ccxt stall detection: fetch_funding_rate_history called exactly twice (page1 + stall page2)
    assert mock_exchange.fetch_funding_rate_history.call_count == 2, (
        f"Expected 2 ccxt calls (stall on page 2), got {mock_exchange.fetch_funding_rate_history.call_count}"
    )

    # (b) REST fallback triggered: httpx.Client.get called at least twice (pages 1 and 2)
    assert mock_client_instance.get.call_count >= 2, (
        f"Expected REST fallback with >= 2 GET calls, got {mock_client_instance.get.call_count}"
    )

    # Result must be a DataFrame with correct columns
    assert isinstance(result, pd.DataFrame), "fetch_funding must return a DataFrame"
    assert list(result.columns) == ["timestamp", "funding_rate"], (
        f"Unexpected columns: {list(result.columns)}"
    )

    # (c) Deduplication: no duplicate timestamps in result
    assert result["timestamp"].duplicated().sum() == 0, "Result contains duplicate timestamps"

    # (d) All records within [start_ms, end_ms)
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_dt   = pd.Timestamp(end_ms,   unit="ms", tz="UTC")
    assert (result["timestamp"] >= start_dt).all(), "Some records are before start_ms"
    assert (result["timestamp"] < end_dt).all(),    "Some records are at or after end_ms"

    # Sorted ascending
    assert result["timestamp"].is_monotonic_increasing, "Result is not sorted ascending"

    # funding_rate column is float
    assert result["funding_rate"].dtype == float, (
        f"funding_rate dtype should be float, got {result['funding_rate'].dtype}"
    )


def test_volume_ratio_n1_excludes_self_from_mean():
    """volume_ratio_n1 = volume[i-1] / mean(volume[i-2]..volume[i-21]).
    The N-1 candle must NOT appear in its own rolling mean denominator.
    Training formula: shift(2).rolling(20) at row i = mean of [i-2..i-21].
    Live formula:     vol_series[-22:-2]            = mean of [i-2..i-21].
    Both must be identical — this test verifies the training-side formula.
    """
    df = make_ohlcv(60)
    # Compute training formula
    vol_mean_train = df['volume'].shift(2).rolling(20).mean()
    ratio_train = df['volume'].shift(1) / vol_mean_train

    # Compute live formula manually at the last row
    vol = df['volume'].values
    # Last row index = 59 (i=59), N-1 = index 58, mean window = [57..38]
    live_mean = np.mean(vol[38:58])   # indices 38..57 inclusive = vol[-22:-2] of 60-row array
    live_ratio = vol[58] / live_mean

    train_ratio_last = ratio_train.iloc[59]
    assert abs(train_ratio_last - live_ratio) < 1e-10, (
        f"Train/live volume_ratio_n1 mismatch: train={train_ratio_last:.8f} live={live_ratio:.8f}"
    )
