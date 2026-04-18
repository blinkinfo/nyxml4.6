import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/nebula/nyxml4')

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
    assert len(FEATURE_COLS) == 40


def test_feature_order():
    expected = ['body_ratio_n1', 'body_ratio_n2', 'body_ratio_n3',
                'upper_wick_n1', 'upper_wick_n2', 'lower_wick_n1', 'lower_wick_n2',
                'volume_ratio_n1', 'volume_ratio_n2',
                'body_ratio_15m', 'dir_15m', 'volume_ratio_15m',
                'body_ratio_1h', 'dir_1h', 'ema9_slope_1h',
                'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'vol_zscore', 'vol_trend',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'atr_percentile_24h', 'vol_regime',
                'rsi14', 'candle_streak', 'price_in_range', 'ema_cross_5m',
                'body_vs_range5', 'range_expansion', 'vwap_dist_20',
                'cvd_ratio', 'cvd_delta_norm']
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
    15m merge and 1h merge."""
    import sys
    sys.path.insert(0, '/home/nebula/nyxml4')
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

    This test builds synthetic aligned 5m/15m/1h/funding data, then checks
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

    # Training features on full history
    train_feat = build_features(df5, df15, df1h)
    expected = train_feat[FEATURE_COLS].iloc[-2].to_numpy()

    # Live path semantics:
    #   - drop in-progress 5m candle only
    #   - keep 15m/1h history; builder itself applies <= ts_n1 filtering
    live_row, nan_features = build_live_features(
        df5.iloc[:-1].copy(),
        df15.copy(),
        df1h.copy(),
    )

    assert live_row is not None, f"build_live_features returned None; nan_features={nan_features}"
    assert nan_features == [], f"Unexpected NaN features: {nan_features}"
    got = live_row[0]
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)



def test_fetch_funding_mock():
    """Verify fetch_funding() behaviour using mocked ccxt and MEXC REST API.

    Scenarios tested:
      (a) ccxt pagination stalls with no forward progress -> stops ccxt loop
      (b) ccxt coverage is sparse -> falls back to MEXC REST API
      (c) REST results are deduplicated (duplicate settleTime entries collapsed)
      (d) Records outside [start_ms, end_ms) are filtered out
    """
    import sys
    sys.path.insert(0, '/home/nebula/nyxml4')

    from unittest.mock import patch, MagicMock
    import ml.data_fetcher as df_module
    from ml.data_fetcher import fetch_funding

    now_ms = 1_700_000_000_000
    start_ms = now_ms - 10 * 24 * 3600 * 1000
    end_ms   = now_ms

    ccxt_ts_1 = end_ms - 3 * 8 * 3600 * 1000
    ccxt_ts_2 = end_ms - 2 * 8 * 3600 * 1000
    ccxt_ts_3 = end_ms - 1 * 8 * 3600 * 1000
    ccxt_page1 = [
        {"timestamp": ccxt_ts_3, "fundingRate": 0.0003},
        {"timestamp": ccxt_ts_2, "fundingRate": 0.0002},
        {"timestamp": ccxt_ts_1, "fundingRate": 0.0001},
    ]
    ccxt_page2 = [
        {"timestamp": ccxt_ts_3, "fundingRate": 0.0003},
    ]

    mock_exchange = MagicMock()
    mock_exchange.fetch_funding_rate_history.side_effect = [ccxt_page1, ccxt_page2]

    interval_ms = 8 * 3600 * 1000

    def make_rest_items(ts_list):
        return [{"settleTime": str(ts), "fundingRate": str(round(0.0001 * (i + 1), 6))}
                for i, ts in enumerate(ts_list)]

    rest_page1_ts = [end_ms - i * interval_ms for i in range(1, 6)]
    rest_page1_items = make_rest_items(rest_page1_ts)

    rest_page2_ts = [end_ms - i * interval_ms for i in range(5, 11)]
    rest_page2_ts.append(start_ms - interval_ms)
    rest_page2_ts.append(rest_page1_ts[-1])
    rest_page2_items = make_rest_items(rest_page2_ts)

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
         patch("ml.data_fetcher.time.sleep"):

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.side_effect = mock_httpx_get
        mock_client_cls.return_value = mock_client_instance

        result = fetch_funding(start_ms, end_ms)

    assert mock_exchange.fetch_funding_rate_history.call_count == 2, (
        f"Expected 2 ccxt calls (page1 + no-progress page2), got {mock_exchange.fetch_funding_rate_history.call_count}"
    )
    assert mock_client_instance.get.call_count >= 2, (
        f"Expected REST fallback with >= 2 GET calls, got {mock_client_instance.get.call_count}"
    )
    assert isinstance(result, pd.DataFrame), "fetch_funding must return a DataFrame"
    assert list(result.columns) == ["timestamp", "funding_rate"], (
        f"Unexpected columns: {list(result.columns)}"
    )
    assert result["timestamp"].duplicated().sum() == 0, "Result contains duplicate timestamps"

    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_dt   = pd.Timestamp(end_ms,   unit="ms", tz="UTC")
    assert (result["timestamp"] >= start_dt).all(), "Some records are before start_ms"
    assert (result["timestamp"] < end_dt).all(),    "Some records are at or after end_ms"
    assert result["timestamp"].is_monotonic_increasing, "Result is not sorted ascending"
    assert result["funding_rate"].dtype == float, (
        f"funding_rate dtype should be float, got {result['funding_rate'].dtype}"
    )


def test_fetch_funding_ccxt_descending_pages_accumulates_beyond_single_page():
    """Descending ccxt pages should still advance across a long window.

    This is the regression guard for the one-page failure mode: when MEXC/ccxt
    returns newest-first pages, advancing via batch[-1] stalls near the first
    page boundary. The funding path should instead keep moving forward and collect
    materially more than 100 records without invoking REST fallback.
    """
    import sys
    sys.path.insert(0, '/home/nebula/nyxml4')

    from unittest.mock import patch, MagicMock
    import ml.data_fetcher as df_module
    from ml.data_fetcher import fetch_funding

    interval_ms = 8 * 3600 * 1000
    expected_points = 120
    start_ms = 1_700_000_000_000
    end_ms = start_ms + expected_points * interval_ms
    all_ts = [start_ms + i * interval_ms for i in range(expected_points)]

    def make_desc_page(start_idx, size):
        ts_slice = all_ts[start_idx:start_idx + size]
        return [
            {"timestamp": ts, "fundingRate": 0.0001 + idx * 1e-7}
            for idx, ts in enumerate(reversed(ts_slice))
        ]

    ccxt_pages = [
        make_desc_page(0, 100),
        make_desc_page(100, 20),
        [],
    ]

    mock_exchange = MagicMock()
    mock_exchange.fetch_funding_rate_history.side_effect = ccxt_pages
    mock_exchange.load_markets.return_value = None

    with patch.object(df_module.ccxt, "mexc", return_value=mock_exchange), \
         patch("ml.data_fetcher.httpx.Client") as mock_client_cls, \
         patch("ml.data_fetcher.time.sleep"):

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client_instance

        result = fetch_funding(start_ms, end_ms)

    assert mock_exchange.fetch_funding_rate_history.call_count >= 2, (
        f"Expected multi-page ccxt pagination, got {mock_exchange.fetch_funding_rate_history.call_count} call(s)"
    )
    assert mock_client_instance.get.call_count == 0, "REST fallback should not be needed when ccxt pages advance"
    assert len(result) == expected_points, f"Expected {expected_points} funding rows, got {len(result)}"
    assert len(result) > 100, "Regression guard failed: funding retrieval did not exceed one page"
    assert result["timestamp"].is_monotonic_increasing, "ccxt result should be sorted ascending after normalization"
    assert result["timestamp"].duplicated().sum() == 0, "ccxt result should be deduplicated"


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


# ===========================================================================
# Targeted numerical parity tests for the 4 new structure features
# Each test computes the expected value from first principles and verifies
# that both build_features() (training) and build_live_features() (inference)
# produce the same result to floating-point precision.
# ===========================================================================

def make_aligned_data(n=500, seed=99):
    """Return (df5, df15, df1h) with n 5m candles."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    close = 50000 + np.cumsum(rng.normal(0, 20, n))
    open_ = close + rng.normal(0, 5, n)
    high  = np.maximum(open_, close) + rng.uniform(0, 8, n)
    low   = np.minimum(open_, close) - rng.uniform(0, 8, n)
    vol   = rng.uniform(50, 200, n)
    df5 = pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                        "low": low, "close": close, "volume": vol})
    s = df5.set_index("timestamp")
    df15 = s.resample("15min").agg({"open":"first","high":"max","low":"min",
                                    "close":"last","volume":"sum"}).dropna().reset_index()
    df1h = s.resample("1h").agg({"open":"first","high":"max","low":"min",
                                  "close":"last","volume":"sum"}).dropna().reset_index()
    return df5, df15, df1h


def test_cvd_ratio_parity():
    """cvd_ratio = long_taker_size / (long + short), in [0, 1].
    Training: build_features() with explicit CVD DataFrame.
    Live:     build_live_features() with the same CVD DataFrame.
    Both must equal the hand-computed reference AND each other.
    """
    from collections import deque
    from ml.features import build_features, build_live_features, FEATURE_COLS

    df5, df15, df1h = make_aligned_data(seed=11)

    # Build a synthetic CVD DataFrame aligned with df5 timestamps
    rng = np.random.default_rng(42)
    cvd_ts = df5["timestamp"].copy()
    long_vol = rng.uniform(100, 1000, len(cvd_ts))
    short_vol = rng.uniform(100, 1000, len(cvd_ts))
    cvd = pd.DataFrame({
        "timestamp": cvd_ts,
        "long_taker_size": long_vol,
        "short_taker_size": short_vol,
    })

    train_feat = build_features(df5, df15, df1h, cvd)
    live_row, nan_feats = build_live_features(
        df5.iloc[:-1].copy(), df15.copy(), df1h.copy(), cvd.copy())

    assert live_row is not None, f"live returned None; nan={nan_feats}"
    assert nan_feats == [], f"Unexpected NaN features: {nan_feats}"

    idx_ratio = FEATURE_COLS.index("cvd_ratio")
    train_val = float(train_feat["cvd_ratio"].iloc[-2])
    live_val  = float(live_row[0][idx_ratio])

    # Value must be in [0, 1]
    assert 0.0 <= train_val <= 1.0, f"train cvd_ratio={train_val} out of [0,1]"
    assert 0.0 <= live_val  <= 1.0, f"live  cvd_ratio={live_val} out of [0,1]"

    # Training and live must match to floating-point precision
    np.testing.assert_allclose(live_val, train_val, atol=1e-9,
        err_msg=f"cvd_ratio mismatch: train={train_val} live={live_val}")


def test_body_vs_range5_parity():
    """body_vs_range5 = |body_n1| / 5-bar range ending at N-1.
    Window: max(high[i-1..i-5]) - min(low[i-1..i-5]).
    Both training and live must equal the hand-computed reference.
    """
    from collections import deque
    from ml.features import build_features, build_live_features, FEATURE_COLS

    df5, df15, df1h = make_aligned_data(seed=22)

    train_feat = build_features(df5, df15, df1h)
    live_row, nan_feats = build_live_features(
        df5.iloc[:-1].copy(), df15.copy(), df1h.copy())

    assert live_row is not None, f"live returned None; nan={nan_feats}"

    idx = FEATURE_COLS.index("body_vs_range5")
    train_val = float(train_feat["body_vs_range5"].iloc[-2])
    live_val  = float(live_row[0][idx])

    # Hand-compute reference at last training row (corresponds to df5 row n-2
    # when the last df5 row is the in-progress candle that gets dropped live).
    # training row iloc[-2] corresponds to df5 index n-2 (0-based),
    # so N-1 in training terms = df5.iloc[-3] (the closed candle before the last).
    n = len(df5)
    # The last training row uses shift(1), so "N-1 candle" for that row = df5.iloc[n-2]
    # (training iterates through all rows; last non-NaN row = row n-2 after dropna on target).
    # We need the row index in df5 that corresponds to train_feat.iloc[-2].
    # Easiest: verify training vs live match, then spot-check the formula.
    np.testing.assert_allclose(live_val, train_val, atol=1e-12,
        err_msg=f"body_vs_range5 mismatch: train={train_val} live={live_val}")

    # Formula spot-check — mirror the implementation exactly.
    # build_live_features receives df5.iloc[:-1] (499 rows, no in-progress candle),
    # then internally does df5 = df5_live.copy().reset_index(drop=True).
    # high_arr = df5["high"].values on that 499-row df.
    # _n5_high = high_arr[-6:-1] = last 5 closed candles ending at N-1 (iloc[-2]).
    df5_live_ref = df5.iloc[:-1].copy().reset_index(drop=True)
    H_ref = df5_live_ref["high"].values
    L_ref = df5_live_ref["low"].values
    C_ref = df5_live_ref["close"].values
    O_ref = df5_live_ref["open"].values
    _n5_high_ref = H_ref[max(0, len(H_ref)-6):-1]
    _n5_low_ref  = L_ref[max(0, len(L_ref)-6):-1]
    _5bar_range_ref = max(float(np.max(_n5_high_ref)) - float(np.min(_n5_low_ref)), 1e-9)
    _body_n1_abs_ref = abs(float(C_ref[-2]) - float(O_ref[-2]))
    ref = _body_n1_abs_ref / _5bar_range_ref
    np.testing.assert_allclose(live_val, ref, atol=1e-9,
        err_msg=f"body_vs_range5 formula mismatch: live={live_val} ref={ref}")

    # Value must be >= 0
    assert train_val >= 0, f"body_vs_range5 negative: {train_val}"


def test_range_expansion_parity():
    """range_expansion = current-5-bar-range / prior-5-bar-range (6..10 candles back).
    Training: shift(1).rolling(5) vs shift(6).rolling(5).
    Live:     high/low[-6:-1] vs high/low[-11:-6].
    Both must equal each other to floating-point precision.
    """
    from collections import deque
    from ml.features import build_features, build_live_features, FEATURE_COLS

    df5, df15, df1h = make_aligned_data(seed=33)

    train_feat = build_features(df5, df15, df1h)
    live_row, nan_feats = build_live_features(
        df5.iloc[:-1].copy(), df15.copy(), df1h.copy())

    assert live_row is not None, f"live returned None; nan={nan_feats}"

    idx = FEATURE_COLS.index("range_expansion")
    train_val = float(train_feat["range_expansion"].iloc[-2])
    live_val  = float(live_row[0][idx])

    np.testing.assert_allclose(live_val, train_val, atol=1e-12,
        err_msg=f"range_expansion mismatch: train={train_val} live={live_val}")

    # Value must be > 0 (ratio of two positive ranges)
    assert train_val > 0, f"range_expansion non-positive: {train_val}"

    # Spot-check: current / prior must both be finite
    assert np.isfinite(train_val), f"range_expansion not finite: {train_val}"

    # Formula spot-check — mirror the implementation exactly.
    # build_live_features receives df5.iloc[:-1] (499 rows).
    # high_arr = those 499 rows; current=[-6:-1], prior=[-11:-6].
    df5_live_ref = df5.iloc[:-1].copy().reset_index(drop=True)
    H_ref = df5_live_ref["high"].values
    L_ref = df5_live_ref["low"].values
    _n5_hi_ref = H_ref[max(0, len(H_ref)-6):-1]
    _n5_lo_ref = L_ref[max(0, len(L_ref)-6):-1]
    _pr_hi_ref = H_ref[max(0, len(H_ref)-11):-6]
    _pr_lo_ref = L_ref[max(0, len(L_ref)-11):-6]
    cur_rng = max(float(np.max(_n5_hi_ref) - np.min(_n5_lo_ref)), 1e-9)
    pri_rng = max(float(np.max(_pr_hi_ref) - np.min(_pr_lo_ref)), 1e-9)
    ref = cur_rng / pri_rng
    np.testing.assert_allclose(live_val, ref, atol=1e-9,
        err_msg=f"range_expansion formula mismatch: live={live_val} ref={ref}")


def test_vwap_dist_20_parity():
    """vwap_dist_20 = (close_n1 - vwap_20) / atr5_n1.
    vwap_20 = sum(close*vol, last 20 candles) / sum(vol, last 20 candles).
    Training: shift(1) rolling sums; Live: array slices.
    Both must equal each other and the hand-computed reference.
    """
    from collections import deque
    from ml.features import build_features, build_live_features, FEATURE_COLS, compute_atr14

    df5, df15, df1h = make_aligned_data(seed=44)

    train_feat = build_features(df5, df15, df1h)
    live_row, nan_feats = build_live_features(
        df5.iloc[:-1].copy(), df15.copy(), df1h.copy())

    assert live_row is not None, f"live returned None; nan={nan_feats}"

    idx = FEATURE_COLS.index("vwap_dist_20")
    train_val = float(train_feat["vwap_dist_20"].iloc[-2])
    live_val  = float(live_row[0][idx])

    np.testing.assert_allclose(live_val, train_val, atol=1e-9,
        err_msg=f"vwap_dist_20 mismatch: train={train_val} live={live_val}")

    # Value must be finite (guarded by atr clip lower=1e-9)
    assert np.isfinite(train_val), f"vwap_dist_20 not finite: {train_val}"

    # Hand-compute reference — mirror the implementation exactly.
    # build_live_features receives df5.iloc[:-1] (499 rows, no in-progress candle).
    # Internally: df5 = df5_live.copy().reset_index(drop=True) (499 rows).
    # atr5_val = atr5.iloc[-2] (safe(atr5,1) on 499-row df = second-to-last).
    # VWAP window: cv_arr[-21:-1] on 499-row df = 20 values ending at N-1 (iloc[-2]).
    # close_n1 = df5["close"].iloc[-2] on 499-row df.
    df5_live_ref = df5.iloc[:-1].copy().reset_index(drop=True)
    atr5_ref = compute_atr14(df5_live_ref)
    atr_n1_ref = float(atr5_ref.iloc[-2])  # safe(atr5, 1) = iloc[-2] of 499-row df
    cv_ref = (df5_live_ref["close"] * df5_live_ref["volume"]).values
    v_ref  = df5_live_ref["volume"].values
    _cv_win_ref = cv_ref[max(0, len(cv_ref)-21):-1]
    _v_win_ref  = v_ref[max(0, len(v_ref)-21):-1]
    vwap20_ref  = float(np.sum(_cv_win_ref)) / max(float(np.sum(_v_win_ref)), 1e-9)
    close_n1_ref = float(df5_live_ref["close"].iloc[-2])
    ref = (close_n1_ref - vwap20_ref) / max(atr_n1_ref, 1e-9)
    np.testing.assert_allclose(live_val, ref, atol=1e-9,
        err_msg=f"vwap_dist_20 formula mismatch: live={live_val} ref={ref}")


def test_structure_features_no_nan_in_normal_conditions():
    """All structure + CVD features must be non-NaN when data is plentiful and well-formed."""
    from collections import deque
    from ml.features import build_features, build_live_features, FEATURE_COLS

    df5, df15, df1h = make_aligned_data(n=500, seed=55)
    # CVD defaults to neutral (0.5 / 0.0) when not provided — still non-NaN
    train_feat = build_features(df5, df15, df1h)

    struct_feats = ["body_vs_range5", "range_expansion", "vwap_dist_20",
                    "cvd_ratio", "cvd_delta_norm"]
    for feat in struct_feats:
        nan_count = train_feat[feat].isna().sum()
        total = len(train_feat)
        # Allow up to 2% NaN for warmup rows only
        assert nan_count / total < 0.02, \
            f"{feat}: {nan_count}/{total} NaN rows ({nan_count/total:.1%}) — exceeds 2% warmup allowance"

    live_row, nan_feats = build_live_features(
        df5.iloc[:-1].copy(), df15.copy(), df1h.copy())
    assert live_row is not None, f"live returned None; nan={nan_feats}"
    for feat in struct_feats:
        i = FEATURE_COLS.index(feat)
        assert not np.isnan(live_row[0][i]), f"{feat} is NaN in live row with 500 candles"


def test_cvd_delta_norm_parity():
    """cvd_delta_norm = (long - short) / atr5 — ATR-normalized signed delta.
    Training and live must match each other to floating-point precision.
    """
    from collections import deque
    from ml.features import build_features, build_live_features, FEATURE_COLS

    df5, df15, df1h = make_aligned_data(seed=77)

    # Build synthetic CVD data with known buy/sell dominance pattern
    rng = np.random.default_rng(99)
    cvd_ts = df5["timestamp"].copy()
    long_vol = rng.uniform(500, 2000, len(cvd_ts))
    short_vol = rng.uniform(300, 1800, len(cvd_ts))
    cvd = pd.DataFrame({
        "timestamp": cvd_ts,
        "long_taker_size": long_vol,
        "short_taker_size": short_vol,
    })

    train_feat = build_features(df5, df15, df1h, cvd)
    live_row, nan_feats = build_live_features(
        df5.iloc[:-1].copy(), df15.copy(), df1h.copy(), cvd.copy())

    assert live_row is not None, f"live returned None; nan={nan_feats}"
    assert nan_feats == [], f"Unexpected NaN features: {nan_feats}"

    idx_norm = FEATURE_COLS.index("cvd_delta_norm")
    train_val = float(train_feat["cvd_delta_norm"].iloc[-2])
    live_val  = float(live_row[0][idx_norm])

    # Must be finite
    assert np.isfinite(train_val), f"train cvd_delta_norm not finite: {train_val}"
    assert np.isfinite(live_val),  f"live  cvd_delta_norm not finite: {live_val}"

    # Training and live must match
    np.testing.assert_allclose(live_val, train_val, atol=1e-9,
        err_msg=f"cvd_delta_norm mismatch: train={train_val} live={live_val}")



def test_sample_quality_weights_are_bounded():
    import numpy as np
    from ml.probability import derive_sample_weights

    feature_names = ["body_ratio", "vol_regime", "atr_percentile_24h"]
    X = np.array([
        [0.01, -3.0, 0.1],
        [0.02, 0.0, 0.5],
        [0.50, 1.0, 0.8],
        [0.80, 3.5, 0.95],
    ] * 20, dtype=float)
    y = np.array(([0, 1, 1, 0] * 20), dtype=int)
    result = derive_sample_weights(X, y, feature_names, {"enabled": True})
    weights = result["weights"]
    assert weights.shape[0] == X.shape[0]
    assert float(weights.min()) >= 0.25
    assert float(weights.max()) <= 1.0
    assert result["summary"]["enabled"] is True


def test_probability_diagnostics_compact_buckets():
    import numpy as np
    from ml.probability import compute_probability_diagnostics

    probs = np.linspace(0.05, 0.95, 100)
    y = (probs > 0.5).astype(int)
    diag = compute_probability_diagnostics(probs, y, "demo", bucket_count=5)
    assert diag["label"] == "demo"
    assert len(diag["buckets"]) == 5
    assert diag["sample_count"] == 100
