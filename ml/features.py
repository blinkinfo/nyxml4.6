"""Feature engineering for LightGBM ML strategy — BLUEPRINT sections 4, 5, 6.

ZERO lookahead bias: all features use shift(k>=1). Target uses shift(-1) (future,
only for training labels — never used as a feature).

Target semantics: 1 if the NEXT candle closes at or above its own open
(close[i+1] >= open[i+1]), matching Polymarket's settlement logic
(resolver.py: winner = "Up" if close_price >= open_price else "Down").

40 features total: candle shape (7), volume (2), 15m context (3), 1h context (3),
OHLCV pressure (5), time-of-day cyclical (4), volatility regime (2),
momentum (4: rsi14, candle_streak, price_in_range, ema_cross_5m),
structure (3: body_vs_range5, range_expansion, vwap_dist_20),
Gate.io CVD taker flow (7: cvd_ratio, cvd_delta_norm, cvd_cumulative_5,
cvd_cumulative_20, cvd_trend_slope, cvd_divergence, oi_change_5bar).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column order — MUST match exactly (40 features)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "body_ratio_n1", "body_ratio_n2", "body_ratio_n3",
    "upper_wick_n1", "upper_wick_n2",
    "lower_wick_n1", "lower_wick_n2",
    "volume_ratio_n1", "volume_ratio_n2",
    "body_ratio_15m", "dir_15m", "volume_ratio_15m",
    "body_ratio_1h", "dir_1h", "ema9_slope_1h",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "vol_zscore", "vol_trend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",  # cyclical time (replaces hour_utc, dow)
    "atr_percentile_24h", "vol_regime",
    "rsi14", "candle_streak", "price_in_range", "ema_cross_5m",  # momentum features
    # structure features
    "body_vs_range5", "range_expansion", "vwap_dist_20",
    # Gate.io CVD taker flow features (indices 35-41)
    "cvd_ratio", "cvd_delta_norm",
    # CVD accumulation + open interest features (indices 37-41)
    "cvd_cumulative_5", "cvd_cumulative_20", "cvd_trend_slope",
    "cvd_divergence", "oi_change_5bar",
]


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    """ATR14 using EWM (BLUEPRINT spec)."""
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()


def _asof_backward(left_ts: pd.Series, right: pd.DataFrame, right_cols: list[str]) -> pd.DataFrame:
    """
    Backward-fill lookup: for each timestamp in left_ts, find the last row in
    right where right['timestamp'] <= left_ts.  Uses pd.merge_asof (vectorized,
    C-level) instead of a Python row loop — identical semantics, ~100x faster.

    left_ts  : Series of tz-aware timestamps (may contain NaT), any name.
    right    : sorted DataFrame with a 'timestamp' column + right_cols.
    Returns  : DataFrame indexed 0..len(left_ts)-1 with right_cols,
               NaN where no prior right row exists or left_ts is NaT.
    """
    n = len(left_ts)

    # Build a left frame with a positional index column so we can reindex after
    # the merge.  Give the key a unique name to avoid collisions with `right`.
    left_df = pd.DataFrame({"_left_ts": left_ts.values, "_pos": np.arange(n)})

    # Ensure both key columns share the exact same dtype before merge_asof.
    # Localize if tz-naive, then cast to datetime64[ms, UTC].
    col = left_df["_left_ts"]
    if col.dt.tz is None:
        col = col.dt.tz_localize("UTC")
    left_df["_left_ts"] = col.astype("datetime64[ms, UTC]")
    right = right.copy()
    ts_col = right["timestamp"]
    if ts_col.dt.tz is None:
        ts_col = ts_col.dt.tz_localize("UTC")
    right["timestamp"] = ts_col.astype("datetime64[ms, UTC]")

    # pd.merge_asof refuses NaT in the left key (raises ValueError).
    # ts_n1 = df5["timestamp"].shift(1) always produces NaT at row 0.
    # Solution: filter those rows out, merge the valid subset, then reindex
    # back to the full 0..n-1 range — NaT positions stay NaN in output.
    valid_mask = left_df["_left_ts"].notna()
    left_valid = left_df[valid_mask].reset_index(drop=True)

    if left_valid.empty:
        # All rows were NaT — return all-NaN frame of correct shape.
        return pd.DataFrame(np.nan, index=np.arange(n), columns=right_cols)

    merged = pd.merge_asof(
        left_valid,
        right[["timestamp"] + right_cols],
        left_on="_left_ts",
        right_on="timestamp",
        direction="backward",
    )

    # Restore original positions: set _pos as index, reindex to 0..n-1.
    # Rows that were NaT (excluded above) will have NaN filled automatically.
    result = (
        merged[["_pos"] + right_cols]
        .set_index("_pos")
        .reindex(np.arange(n))
    )
    return result.reset_index(drop=True)


def build_features(
    df5: pd.DataFrame,
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    cvd: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build 40 features per BLUEPRINT sections 4-6. Returns df with FEATURE_COLS + 'target'.

    Args:
        df5:     5m OHLCV candles from MEXC spot.
        df15:    15m OHLCV candles from MEXC futures.
        df1h:    1h OHLCV candles from MEXC futures.
        cvd:     Gate.io 5m taker volume DataFrame with columns:
                 timestamp, long_taker_size, short_taker_size, open_interest.
                 Used to compute 7 CVD/OI features:
                   - cvd_ratio:         long_taker_size / (long + short), clamped [0, 1]
                   - cvd_delta_norm:    (long - short) / ATR5, ATR-normalised delta
                   - cvd_cumulative_5:  rolling 5-bar sum of delta / ATR5, shift(1)
                   - cvd_cumulative_20: rolling 20-bar sum of delta / ATR5, shift(1)
                   - cvd_trend_slope:   OLS slope of delta over rolling 10-bar window / ATR5, shift(1)
                   - cvd_divergence:    +1 price/CVD disagree, -1 agree, 0 flat (5-bar window, shift(1))
                   - oi_change_5bar:    (oi[t-1] - oi[t-6]) / |oi[t-6]|, pct change over 5 bars
                 If None or empty, cvd_ratio defaults to 0.5, cvd_delta_norm to 0.0,
                 and all 5 accumulation/OI features default to 0.0
                 (neutral — no directional information available).
    """

    # Work on copies with clean RangeIndex
    df5 = df5.copy().reset_index(drop=True)
    df15 = df15.copy().reset_index(drop=True)
    df1h = df1h.copy().reset_index(drop=True)

    # Sort ascending (should already be sorted, but be safe)
    df5 = df5.sort_values("timestamp").reset_index(drop=True)
    df15 = df15.sort_values("timestamp").reset_index(drop=True)
    df1h = df1h.sort_values("timestamp").reset_index(drop=True)

    # Normalize all timestamps to ms UTC for consistent merging
    for df in [df5, df15, df1h]:
        df["timestamp"] = df["timestamp"].astype("datetime64[ms, UTC]")

    # -----------------------------------------------------------------------
    # 5m features — all use shift(k>=1), NEVER shift(0)
    # -----------------------------------------------------------------------
    atr5 = compute_atr14(df5)

    df5["body_ratio_n1"] = (df5["close"].shift(1) - df5["open"].shift(1)) / atr5.shift(1)
    df5["body_ratio_n2"] = (df5["close"].shift(2) - df5["open"].shift(2)) / atr5.shift(2)
    df5["body_ratio_n3"] = (df5["close"].shift(3) - df5["open"].shift(3)) / atr5.shift(3)

    df5["upper_wick_n1"] = (
        df5["high"].shift(1) - df5[["open", "close"]].shift(1).max(axis=1)
    ) / atr5.shift(1)
    df5["upper_wick_n2"] = (
        df5["high"].shift(2) - df5[["open", "close"]].shift(2).max(axis=1)
    ) / atr5.shift(2)

    df5["lower_wick_n1"] = (
        df5[["open", "close"]].shift(1).min(axis=1) - df5["low"].shift(1)
    ) / atr5.shift(1)
    df5["lower_wick_n2"] = (
        df5[["open", "close"]].shift(2).min(axis=1) - df5["low"].shift(2)
    ) / atr5.shift(2)

    # volume_ratio_n1: N-1 volume divided by rolling mean of the 20 candles
    # ending at N-2 (i.e. vol[i-2]..vol[i-21]).
    # shift(2).rolling(20) at row i = mean of vol[i-2]..vol[i-21] — N-1 candle
    # is deliberately excluded from its own mean, matching the live formula
    # vol_series[-22:-2] and the blueprint Section 5 English spec.
    vol_mean_n1 = df5["volume"].shift(2).rolling(20).mean()
    df5["volume_ratio_n1"] = df5["volume"].shift(1) / vol_mean_n1
    # volume_ratio_n2: N-2 volume divided by rolling mean of vol[i-3]..vol[i-22]
    vol_mean_n2 = df5["volume"].shift(3).rolling(20).mean()
    df5["volume_ratio_n2"] = df5["volume"].shift(2) / vol_mean_n2

    # ts_n1 = N-1 timestamp (shift by 1 for all multi-tf merges)
    ts_n1 = df5["timestamp"].shift(1)

    # -----------------------------------------------------------------------
    # 15m features — merge_asof backward on ts_n1
    # -----------------------------------------------------------------------
    atr15 = compute_atr14(df15)
    df15["body_ratio_15m"] = (df15["close"] - df15["open"]) / atr15
    df15["dir_15m"] = np.sign(df15["close"] - df15["open"])
    df15["volume_ratio_15m"] = df15["volume"] / df15["volume"].rolling(20, min_periods=2).mean()

    r15 = _asof_backward(ts_n1, df15, ["body_ratio_15m", "dir_15m", "volume_ratio_15m"])
    df5["body_ratio_15m"] = r15["body_ratio_15m"].values
    df5["dir_15m"] = r15["dir_15m"].values
    df5["volume_ratio_15m"] = r15["volume_ratio_15m"].values

    # -----------------------------------------------------------------------
    # 1h features — same merge_asof pattern
    # -----------------------------------------------------------------------
    atr1h = compute_atr14(df1h)
    df1h["body_ratio_1h"] = (df1h["close"] - df1h["open"]) / atr1h
    df1h["dir_1h"] = np.sign(df1h["close"] - df1h["open"])
    ema9 = df1h["close"].ewm(span=9, adjust=False).mean()
    df1h["ema9_slope_1h"] = (ema9 - ema9.shift(1)) / atr1h

    r1h = _asof_backward(ts_n1, df1h, ["body_ratio_1h", "dir_1h", "ema9_slope_1h"])
    df5["body_ratio_1h"] = r1h["body_ratio_1h"].values
    df5["dir_1h"] = r1h["dir_1h"].values
    df5["ema9_slope_1h"] = r1h["ema9_slope_1h"].values

    # -----------------------------------------------------------------------
    # OHLCV-native pressure features — computed purely from df5, zero parity gap
    # -----------------------------------------------------------------------
    hl_range = (df5["high"] - df5["low"]).clip(lower=1e-9)
    body      = df5["close"] - df5["open"]

    # body_ratio: candle body direction and strength, [-1, 1]
    df5["body_ratio"] = (body / hl_range).clip(-1.0, 1.0).shift(1)

    # upper_wick_ratio: selling rejection at highs, [0, 1]
    upper_wick = df5["high"] - df5[["open", "close"]].max(axis=1)
    df5["upper_wick_ratio"] = (upper_wick / hl_range).clip(0.0, 1.0).shift(1)

    # lower_wick_ratio: buying rejection at lows, [0, 1]
    lower_wick = df5[["open", "close"]].min(axis=1) - df5["low"]
    df5["lower_wick_ratio"] = (lower_wick / hl_range).clip(0.0, 1.0).shift(1)

    # vol_zscore: volume surge detection vs 20-bar rolling mean/std
    vol_mean20 = df5["volume"].rolling(20).mean()
    vol_std20  = df5["volume"].rolling(20).std(ddof=1).clip(lower=1e-8)
    df5["vol_zscore"] = ((df5["volume"] - vol_mean20) / vol_std20).shift(1)

    # vol_trend: short vs long volume momentum (5-bar / 20-bar rolling mean)
    vol_ma5  = df5["volume"].rolling(5).mean()
    vol_ma20 = df5["volume"].rolling(20).mean().clip(lower=1e-8)
    df5["vol_trend"] = (vol_ma5 / vol_ma20).shift(1)

    # -----------------------------------------------------------------------
    # Time-of-day cyclical features — derived from N-1 candle timestamp
    # Replaces raw hour_utc and dow with sine/cosine encoding so the model
    # can learn periodic patterns without discontinuities at midnight / week-end.
    # -----------------------------------------------------------------------
    ts_n1_series = df5["timestamp"].shift(1)
    hour_raw = ts_n1_series.dt.hour
    dow_raw = ts_n1_series.dt.dayofweek
    df5["hour_sin"] = np.sin(2 * np.pi * hour_raw / 24)
    df5["hour_cos"] = np.cos(2 * np.pi * hour_raw / 24)
    df5["dow_sin"]  = np.sin(2 * np.pi * dow_raw / 7)
    df5["dow_cos"]  = np.cos(2 * np.pi * dow_raw / 7)

    # Volatility regime features — derived from ATR of the N-1 candle
    # atr_percentile_24h: percentile rank (0.0–1.0) of atr5[i-1] within a 288-candle rolling window
    # vol_regime: zscore of atr5[i-1] within same 288-candle rolling window (std-normalized)
    # 288 = 24 hours * 12 five-minute candles per hour
    _ATR_WINDOW = 288
    atr_shifted = atr5.shift(1)
    def _rolling_percentile(s: pd.Series, w: int) -> pd.Series:
        # min_periods=14 allows partial windows during ATR warmup rows;
        # NaNs within the window are stripped before ranking so warmup NaNs
        # don't silently corrupt the percentile (NaN < value == False in numpy).
        def _pct(x: np.ndarray) -> float:
            x = x[~np.isnan(x)]
            if len(x) < 2:
                return np.nan
            return float(np.sum(x[:-1] < x[-1])) / max(len(x) - 1, 1)
        return s.rolling(w, min_periods=14).apply(_pct, raw=True)
    df5["atr_percentile_24h"] = _rolling_percentile(atr_shifted, _ATR_WINDOW)
    roll = atr_shifted.rolling(_ATR_WINDOW, min_periods=14)
    atr_roll_mean = roll.mean()
    atr_roll_std  = roll.std()
    df5["vol_regime"] = (atr_shifted - atr_roll_mean) / atr_roll_std.clip(lower=1e-10)

    # -----------------------------------------------------------------------
    # Momentum features (new) — all use shift(k>=1) for zero lookahead
    # -----------------------------------------------------------------------

    # rsi14: Wilder's RSI(14) on 5m closes, N-1 value
    _delta = df5["close"].diff()
    _gain = _delta.clip(lower=0)
    _loss = (-_delta).clip(lower=0)
    _avg_gain = _gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    _avg_loss = _loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    _rs = _avg_gain / _avg_loss.clip(lower=1e-10)
    _rsi = 100.0 - (100.0 / (1.0 + _rs))
    df5["rsi14"] = _rsi.shift(1)  # N-1 value, zero lookahead

    # candle_streak: consecutive same-direction candles ending at N-1
    # Vectorized approach: group consecutive same-direction runs
    _direction = np.sign(df5["close"] - df5["open"])
    _same_as_prev = (_direction == _direction.shift(1)) & (_direction != 0)
    _streak = _same_as_prev.groupby((~_same_as_prev).cumsum()).cumsum()
    _streak = _streak.where(_direction != 0, 0).astype(float)
    df5["candle_streak"] = _streak.shift(1)  # N-1 value

    # price_in_range: where N-1 close sits within 20-candle range ending at N-1
    # rolling(20).max/min on shift(1) gives range of [i-20..i-1] — zero lookahead
    _rolling_high = df5["high"].shift(1).rolling(20, min_periods=5).max()
    _rolling_low  = df5["low"].shift(1).rolling(20, min_periods=5).min()
    _rng = (_rolling_high - _rolling_low).clip(lower=1e-10)
    df5["price_in_range"] = (df5["close"].shift(1) - _rolling_low) / _rng

    # ema_cross_5m: sign of EMA9 vs EMA21 at N-1 candle (-1, 0, +1)
    _ema9_5m  = df5["close"].ewm(span=9,  adjust=False).mean()
    _ema21_5m = df5["close"].ewm(span=21, adjust=False).mean()
    df5["ema_cross_5m"] = np.sign(_ema9_5m - _ema21_5m).shift(1)  # N-1 cross state

    # -----------------------------------------------------------------------
    # Structure features — all use shift(k>=1) for zero lookahead
    # -----------------------------------------------------------------------

    # body_vs_range5: |body_n1| normalised by the 5-candle range ending at N-1.
    # 5-bar range = max(high[i-1..i-5]) - min(low[i-1..i-5]) — shift(1).rolling(5)
    # gives exactly that window at each row i with zero lookahead.
    _5bar_high   = df5["high"].shift(1).rolling(5, min_periods=2).max()
    _5bar_low    = df5["low"].shift(1).rolling(5, min_periods=2).min()
    _5bar_range  = (_5bar_high - _5bar_low).clip(lower=1e-9)
    df5["body_vs_range5"] = (df5["close"].shift(1) - df5["open"].shift(1)).abs() / _5bar_range

    # range_expansion: current 5-bar range vs prior 5-bar range (6..10 candles back).
    # shift(6).rolling(5) at row i = max/min of [i-6..i-10] — no overlap with current.
    _prior_high  = df5["high"].shift(6).rolling(5, min_periods=2).max()
    _prior_low   = df5["low"].shift(6).rolling(5, min_periods=2).min()
    _prior_range = (_prior_high - _prior_low).clip(lower=1e-9)
    df5["range_expansion"] = _5bar_range / _prior_range

    # vwap_dist_20: (close_n1 - vwap_20) / atr5_n1.
    # VWAP over the 20 candles ending at N-1: shift(1) before rolling ensures
    # the window [i-1..i-20] — zero lookahead. Divided by ATR for scale invariance.
    _cv_20  = (df5["close"].shift(1) * df5["volume"].shift(1)).rolling(20, min_periods=5).sum()
    _v_20   = df5["volume"].shift(1).rolling(20, min_periods=5).sum().clip(lower=1e-9)
    _vwap20 = _cv_20 / _v_20
    df5["vwap_dist_20"] = (df5["close"].shift(1) - _vwap20) / atr5.shift(1).clip(lower=1e-9)

    # -----------------------------------------------------------------------
    # Gate.io CVD taker flow features — merge_asof backward on ts_n1
    #
    # cvd_ratio     : long_taker_size / total_taker_size — [0, 1]
    #                 > 0.5 = net buy pressure, < 0.5 = net sell pressure.
    #                 Neutral default 0.5 when no CVD data is available.
    #
    # cvd_delta_norm: (long_taker_size - short_taker_size) / atr5 — signed,
    #                 ATR-normalized bar delta. Positive = buy dominance.
    #                 Neutral default 0.0 when no CVD data is available.
    #
    # Both use ts_n1 (N-1 candle timestamp) in the merge — zero lookahead.
    # -----------------------------------------------------------------------
    cvd_available = (
        cvd is not None
        and not cvd.empty
        and "long_taker_size" in cvd.columns
        and "short_taker_size" in cvd.columns
    )

    if cvd_available:
        cvd_clean = cvd.copy().reset_index(drop=True)
        cvd_clean["timestamp"] = cvd_clean["timestamp"].astype("datetime64[ms, UTC]")
        cvd_clean = cvd_clean.sort_values("timestamp").reset_index(drop=True)

        # Derive ratio and delta directly on the CVD frame before merging
        _total = (cvd_clean["long_taker_size"] + cvd_clean["short_taker_size"]).clip(lower=1e-9)
        cvd_clean["cvd_ratio"] = (cvd_clean["long_taker_size"] / _total).clip(0.0, 1.0)
        cvd_clean["cvd_delta"] = cvd_clean["long_taker_size"] - cvd_clean["short_taker_size"]

        # -----------------------------------------------------------------------
        # CVD accumulation + OI features — computed on the CVD frame (per-bar),
        # then merged onto df5 via _asof_backward exactly like cvd_ratio/delta.
        #
        # ALL use shift(k>=1) relative to the CVD frame so that when they are
        # merged onto df5 at ts_n1 (N-1 bar timestamp) the values they carry
        # correspond to history BEFORE N-1 — zero lookahead bias guaranteed.
        #
        # cvd_cumulative_5:  sum of cvd_delta over bars [t-5 .. t-1] (rolling 5,
        #                    then shift 1), ATR-normalized.  Captures short-term
        #                    directional buy/sell pressure accumulation.
        #
        # cvd_cumulative_20: same over bars [t-20 .. t-1] (rolling 20, shift 1).
        #                    Medium-term flow regime.
        #
        # cvd_trend_slope:   Linear regression slope of cvd_delta over the 10-bar
        #                    window [t-10 .. t-1] (rolling 10, shift 1), normalized
        #                    by ATR.  Positive = buy pressure accelerating.
        #                    Uses least-squares via polyfit on positions [0..9].
        #
        # cvd_divergence:    +1 if price direction (last 5 bars ending at t-1) and
        #                    CVD direction (last 5 bars ending at t-1) disagree,
        #                    -1 if they agree, 0 if either is flat.
        #                    Classical divergence: price up + CVD falling = reversal.
        #
        # oi_change_5bar:    (oi[t-1] - oi[t-6]) / |oi[t-6]|  — percentage change
        #                    in open interest over the prior 5 bars.  Positive =
        #                    new positions opening (trend continuation bias).
        #                    oi[t-6] is shift(6) so oi[t-1] = shift(1); diff is
        #                    (shift(1) - shift(6)) / abs(shift(6)).
        # -----------------------------------------------------------------------

        # ATR proxy for CVD normalization: we need ATR at the CVD bar's own
        # timestamp. We merge the ATR series from df5 onto cvd_clean via
        # _asof_backward so each CVD bar gets the ATR of the corresponding
        # 5m candle. Then we use that for normalization — not a fixed scalar.
        _df5_atr = pd.DataFrame({
            "timestamp": df5["timestamp"].values,
            "atr5_for_cvd": atr5.values,
        })
        _atr_merged = _asof_backward(
            cvd_clean["timestamp"], _df5_atr, ["atr5_for_cvd"]
        )
        _cvd_atr = _atr_merged["atr5_for_cvd"].clip(lower=1e-9)
        # _atr_s1: ATR at the bar BEFORE each CVD bar (shift(1) with proper index).
        # Using the Series directly (not wrapping in pd.Series()) preserves the
        # RangeIndex alignment with cvd_clean and eliminates any positional mismatch.
        _atr_s1 = _cvd_atr.shift(1)

        # cvd_cumulative_5: rolling sum of delta over 5 bars, then shift(1)
        # shift(1) moves the window one bar back so the value at CVD bar t represents
        # the sum of bars [t-5 .. t-1] — matching the live path which uses bars ending
        # at N-2 (one bar before the N-1 bar used as the merge anchor).
        cvd_clean["cvd_cumulative_5"] = (
            cvd_clean["cvd_delta"].rolling(5, min_periods=2).sum().shift(1)
            / _atr_s1.clip(lower=1e-9)
        )

        # cvd_cumulative_20: rolling sum of delta over 20 bars, then shift(1)
        cvd_clean["cvd_cumulative_20"] = (
            cvd_clean["cvd_delta"].rolling(20, min_periods=5).sum().shift(1)
            / _atr_s1.clip(lower=1e-9)
        )

        # cvd_trend_slope: OLS slope of cvd_delta over rolling 10-bar window, shift(1)
        # polyfit on positions [0..9] — normalized by ATR
        _x_slope = np.arange(10, dtype=np.float64)

        def _slope(vals: np.ndarray) -> float:
            """Return OLS slope of vals over positions 0..len-1."""
            v = vals[~np.isnan(vals)]
            if len(v) < 3:
                return np.nan
            x = np.arange(len(v), dtype=np.float64)
            try:
                return float(np.polyfit(x, v, 1)[0])
            except Exception:
                return np.nan

        cvd_clean["cvd_trend_slope"] = (
            cvd_clean["cvd_delta"]
            .rolling(10, min_periods=3)
            .apply(_slope, raw=True)
            .shift(1)
            / _atr_s1.clip(lower=1e-9)
        )

        # cvd_divergence: sign disagreement between price direction and CVD direction
        # price_dir_5: sign of sum of (close-open) over last 5 bars ending at t-1.
        # We pre-shift body5_sum in df5 coordinates BEFORE merging so that after
        # _asof_backward the merged value at each CVD timestamp already reflects
        # the price-body sum ending one 5m candle before that CVD bar.
        # This is equivalent to (and consistent with) the live path which reads
        # df5[-6:-1] (5 closed candles ending at N-1, not including forming bar N).
        _df5_body = pd.DataFrame({
            "timestamp": df5["timestamp"].values,
            "body5_sum": (df5["close"] - df5["open"]).rolling(5, min_periods=2).sum().shift(1).values,
        })
        _body_merged = _asof_backward(cvd_clean["timestamp"], _df5_body, ["body5_sum"])
        _price_dir_5 = np.sign(_body_merged["body5_sum"].values)
        _cvd_dir_5 = np.sign(
            cvd_clean["cvd_delta"].rolling(5, min_periods=2).sum().shift(1).values
        )
        # +1 = diverging (disagree), -1 = aligning (agree), 0 = either flat
        _div_raw = np.where(
            (_price_dir_5 == 0) | (_cvd_dir_5 == 0),
            0.0,
            np.where(_price_dir_5 != _cvd_dir_5, 1.0, -1.0),
        )
        cvd_clean["cvd_divergence"] = _div_raw

        # oi_change_5bar: (oi_t-1 - oi_t-6) / |oi_t-6|
        # open_interest column present when cvd_available (fetched from Gate.io)
        _oi_col = "open_interest"
        if _oi_col in cvd_clean.columns:
            _oi_s1 = cvd_clean[_oi_col].shift(1)
            _oi_s6 = cvd_clean[_oi_col].shift(6)
            cvd_clean["oi_change_5bar"] = (
                (_oi_s1 - _oi_s6) / _oi_s6.abs().clip(lower=1e-9)
            )
        else:
            cvd_clean["oi_change_5bar"] = np.nan

        _cvd_cols = [
            "cvd_ratio", "cvd_delta",
            "cvd_cumulative_5", "cvd_cumulative_20", "cvd_trend_slope",
            "cvd_divergence", "oi_change_5bar",
        ]
        rcvd = _asof_backward(ts_n1, cvd_clean, _cvd_cols)
        df5["cvd_ratio"] = rcvd["cvd_ratio"].values

        # Normalize delta by ATR — use atr5 (already computed on df5)
        # Clip denominator away from zero to prevent inf/nan
        df5["cvd_delta_norm"] = rcvd["cvd_delta"].values / atr5.shift(1).clip(lower=1e-9).values

        # Accumulation + OI features are already ATR-normalized on the CVD frame
        df5["cvd_cumulative_5"]  = rcvd["cvd_cumulative_5"].values
        df5["cvd_cumulative_20"] = rcvd["cvd_cumulative_20"].values
        df5["cvd_trend_slope"]   = rcvd["cvd_trend_slope"].values
        df5["cvd_divergence"]    = rcvd["cvd_divergence"].values
        df5["oi_change_5bar"]    = rcvd["oi_change_5bar"].values
    else:
        log.warning(
            "build_features: CVD data not provided or empty — "
            "using neutral defaults (cvd_ratio=0.5, cvd_delta_norm=0.0, "
            "cvd_cumulative_5=0.0, cvd_cumulative_20=0.0, cvd_trend_slope=0.0, "
            "cvd_divergence=0.0, oi_change_5bar=0.0)"
        )
        df5["cvd_ratio"] = 0.5
        df5["cvd_delta_norm"] = 0.0
        df5["cvd_cumulative_5"] = 0.0
        df5["cvd_cumulative_20"] = 0.0
        df5["cvd_trend_slope"] = 0.0
        df5["cvd_divergence"] = 0.0
        df5["oi_change_5bar"] = 0.0

    # -----------------------------------------------------------------------
    # Target: 1 if next candle closes >= its own open (future label, NOT a feature)
    # Matches Polymarket settlement: close >= open within candle i+1
    # (resolver.py: winner = "Up" if close_price >= open_price else "Down")
    # -----------------------------------------------------------------------
    df5["target"] = (df5["close"].shift(-1) >= df5["open"].shift(-1)).astype(int)

    # -----------------------------------------------------------------------
    # Drop rows with any NaN in features or target, return feature cols + target
    # -----------------------------------------------------------------------
    all_cols = FEATURE_COLS + ["target"]
    df_out = df5[["timestamp"] + all_cols].dropna(subset=all_cols)
    log.info("build_features: %d rows after dropna (started with %d)", len(df_out), len(df5))
    return df_out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live feature computation
# ---------------------------------------------------------------------------

def build_live_features(
    df5_live: pd.DataFrame,
    df15_live: pd.DataFrame,
    df1h_live: pd.DataFrame,
    cvd_live: pd.DataFrame | None = None,
) -> "tuple[np.ndarray, list[str]] | tuple[None, list[str]]":
    """
    Build a single feature row (shape 1×40) for live inference.

    Returns a 2-tuple (feature_row, nan_features):
      - feature_row : np.ndarray shape (1, 40), or None on hard failure.
      - nan_features: list of feature names that were NaN (empty on success).
                      Populated even when feature_row is None so callers can
                      log exactly which features caused the skip.

    Returns (None, []) if ATR warmup is not satisfied (fewer than 14 candles).
    Returns (None, [<name>, ...]) when one or more features are NaN.
    Returns (row, []) on full success.

    Args:
        df5_live:           5m OHLCV candles (live window).
        df15_live:          15m OHLCV candles (live window).
        df1h_live:          1h OHLCV candles (live window).
        cvd_live:           Gate.io 5m taker volume DataFrame
                            (columns: timestamp, long_taker_size, short_taker_size,
                            open_interest).
                            If None or empty, cvd_ratio defaults to 0.5,
                            cvd_delta_norm to 0.0, and all accumulation/OI
                            features to 0.0 (neutral — no bias applied).
    """
    # Validate ATR warmup
    if len(df5_live) < 14:
        log.debug(
            "build_live_features: insufficient candles for ATR warmup "
            "(have %d, need 14) — skipping inference",
            len(df5_live),
        )
        return None, []

    df5 = df5_live.copy().reset_index(drop=True)
    df15 = df15_live.copy().reset_index(drop=True)
    df1h = df1h_live.copy().reset_index(drop=True)

    # Normalize timestamps
    for df in [df5, df15, df1h]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ms, UTC]")

    atr5 = compute_atr14(df5)
    if atr5.iloc[-1] is None or pd.isna(atr5.iloc[-1]):
        log.warning(
            "build_live_features: ATR is NaN at current candle "
            "(insufficient warmup data, have %d candles) — skipping inference",
            len(df5),
        )
        return None, []

    # df5_live[-1] is the still-forming candle N (present, never read as a feature).
    # safe(series, k=1) returns the N-1 value (last fully-closed candle).
    # safe(series, k=2) returns N-2, etc.
    # This matches the training convention where all features use shift(k>=1).
    def safe(series, k=1):
        idx = len(series) - 1 - k
        return series.iloc[idx] if idx >= 0 else np.nan

    atr5_val = safe(atr5, 1)
    if pd.isna(atr5_val) or atr5_val <= 0:
        log.warning(
            "build_live_features: atr5_val is NaN or zero (%.6g) at N-1 — skipping inference",
            atr5_val if not pd.isna(atr5_val) else float("nan"),
        )
        return None, []

    body_ratio_n1 = (safe(df5["close"], 1) - safe(df5["open"], 1)) / atr5_val
    body_ratio_n2 = (safe(df5["close"], 2) - safe(df5["open"], 2)) / safe(atr5, 2)
    body_ratio_n3 = (safe(df5["close"], 3) - safe(df5["open"], 3)) / safe(atr5, 3)

    upper_wick_n1 = (safe(df5["high"], 1) - max(safe(df5["open"], 1), safe(df5["close"], 1))) / atr5_val
    upper_wick_n2 = (safe(df5["high"], 2) - max(safe(df5["open"], 2), safe(df5["close"], 2))) / safe(atr5, 2)
    lower_wick_n1 = (min(safe(df5["open"], 1), safe(df5["close"], 1)) - safe(df5["low"], 1)) / atr5_val
    lower_wick_n2 = (min(safe(df5["open"], 2), safe(df5["close"], 2)) - safe(df5["low"], 2)) / safe(atr5, 2)

    vol_series = df5["volume"].values
    # df5[-1] is forming candle N; vol_series[-2] = N-1, vol_series[-3] = N-2.
    #
    # volume_ratio_n1: N-1 volume / mean of the 20 candles ending at N-2.
    # Matches training: df['volume'].shift(2).rolling(20).mean() at row i
    # = mean(vol[i-2]..vol[i-21]) — N-1 is excluded from its own normaliser.
    # Live equivalent: vol_series[-2] / mean(vol_series[-22:-2]).
    #
    # volume_ratio_n2: N-2 volume / mean of the 20 candles ending at N-3.
    # Matches training: df['volume'].shift(3).rolling(20).mean() at row i
    # = mean(vol[i-3]..vol[i-22]).
    # Live equivalent: vol_series[-3] / mean(vol_series[-23:-3]).
    if len(vol_series) >= 22:
        vol_ratio_n1 = vol_series[-2] / np.mean(vol_series[-22:-2])
    elif len(vol_series) >= 4:
        # Fewer than 20 prior candles available — use what we have (graceful degradation)
        vol_ratio_n1 = vol_series[-2] / np.mean(vol_series[:-2]) if len(vol_series) > 2 else np.nan
    else:
        vol_ratio_n1 = np.nan

    if len(vol_series) >= 23:
        vol_ratio_n2 = vol_series[-3] / np.mean(vol_series[-23:-3])
    elif len(vol_series) >= 5:
        vol_ratio_n2 = vol_series[-3] / np.mean(vol_series[:-3]) if len(vol_series) > 3 else np.nan
    else:
        vol_ratio_n2 = np.nan

    # 15m features
    if len(df15) >= 14:
        atr15 = compute_atr14(df15)
        # ts_n1: timestamp of the N-1 (last fully-closed) 5m candle.
        # df5[-1] is forming candle N, so df5[-2] is N-1.
        ts_n1 = df5["timestamp"].iloc[-2] if len(df5) >= 2 else None
        if ts_n1 is not None and not pd.isna(ts_n1):
            # Find the last 15m candle whose open is at or before ts_n1
            mask15 = df15["timestamp"] <= ts_n1
            if mask15.any():
                idx15 = df15[mask15].index[-1]
                atr15_val = atr15.iloc[idx15]
                if pd.notna(atr15_val) and atr15_val > 0:
                    body_ratio_15m = (df15["close"].iloc[idx15] - df15["open"].iloc[idx15]) / atr15_val
                    dir_15m = np.sign(df15["close"].iloc[idx15] - df15["open"].iloc[idx15])
                    # volume_ratio_15m: matches training — rolling(20, min_periods=2).mean()
                    vol15_rolling_mean = df15["volume"].rolling(20, min_periods=2).mean()
                    vol15_mean_val = vol15_rolling_mean.iloc[idx15]
                    vol_ratio_15m = df15["volume"].iloc[idx15] / vol15_mean_val if pd.notna(vol15_mean_val) and vol15_mean_val > 0 else np.nan
                else:
                    body_ratio_15m = dir_15m = vol_ratio_15m = np.nan
            else:
                body_ratio_15m = dir_15m = vol_ratio_15m = np.nan
        else:
            body_ratio_15m = dir_15m = vol_ratio_15m = np.nan
    else:
        body_ratio_15m = dir_15m = vol_ratio_15m = np.nan

    # 1h features
    if len(df1h) >= 14:
        atr1h = compute_atr14(df1h)
        # ts_n1: timestamp of the N-1 (last fully-closed) 5m candle.
        # df5[-1] is forming candle N, so df5[-2] is N-1.
        ts_n1 = df5["timestamp"].iloc[-2] if len(df5) >= 2 else None
        if ts_n1 is not None and not pd.isna(ts_n1):
            mask1h = df1h["timestamp"] <= ts_n1
            if mask1h.any():
                idx1h = df1h[mask1h].index[-1]
                # Use ATR at idx1h only — do NOT scan forward into future candles.
                # Scanning forward would pull OHLC/EMA data from candles that haven't
                # closed yet at the time of the N-1 5m bar, introducing lookahead bias.
                # If ATR is NaN here (warmup), fall back to NaN for all 1h features.
                atr1h_val = atr1h.iloc[idx1h]
                if pd.notna(atr1h_val) and atr1h_val > 0:
                    body_ratio_1h = (df1h["close"].iloc[idx1h] - df1h["open"].iloc[idx1h]) / atr1h_val
                    dir_1h = np.sign(df1h["close"].iloc[idx1h] - df1h["open"].iloc[idx1h])
                    ema9 = df1h["close"].ewm(span=9, adjust=False).mean()
                    ema9_slope_1h = (ema9.iloc[idx1h] - ema9.iloc[idx1h - 1]) / atr1h_val if idx1h > 0 else np.nan
                else:
                    body_ratio_1h = dir_1h = ema9_slope_1h = np.nan
            else:
                body_ratio_1h = dir_1h = ema9_slope_1h = np.nan
        else:
            body_ratio_1h = dir_1h = ema9_slope_1h = np.nan
    else:
        body_ratio_1h = dir_1h = ema9_slope_1h = np.nan

    # OHLCV-native pressure features (live) — identical formulas to build_features.
    # df5[-1] is the forming candle N (present, never read as a feature).
    # All of these use index -2, which is the N-1 (last fully-closed) candle.
    hl_range_live = (df5["high"] - df5["low"]).clip(lower=1e-9)
    body_live      = df5["close"] - df5["open"]

    # body_ratio: [-1, 1]
    body_ratio = float(np.clip(body_live.iloc[-2] / hl_range_live.iloc[-2], -1.0, 1.0))

    # upper_wick_ratio: [0, 1]
    upper_wick_live = df5["high"] - df5[["open", "close"]].max(axis=1)
    upper_wick_ratio = float(np.clip(upper_wick_live.iloc[-2] / hl_range_live.iloc[-2], 0.0, 1.0))

    # lower_wick_ratio: [0, 1]
    lower_wick_live = df5[["open", "close"]].min(axis=1) - df5["low"]
    lower_wick_ratio = float(np.clip(lower_wick_live.iloc[-2] / hl_range_live.iloc[-2], 0.0, 1.0))

    # vol_zscore: (vol_n1 - mean20) / std20.
    # df5[-2] = N-1 volume; window [-21:-1] = 20 bars ending at N-1 (df5[-1] excluded).
    if len(df5) >= 21:
        vol_window = df5["volume"].iloc[-21:-1]  # 20 bars ending at N-1 (index -2)
        v_mean = float(vol_window.mean())
        v_std  = float(vol_window.std(ddof=1))
        vol_zscore = (float(df5["volume"].iloc[-2]) - v_mean) / max(v_std, 1e-8)
    else:
        vol_zscore = np.nan

    # vol_trend: ma5 / ma20 at N-1.
    # Slices [-6:-1] (5 bars) and [-21:-1] (20 bars) both end at N-1 (index -2).
    if len(df5) >= 21:
        vol_ma5_live  = float(df5["volume"].iloc[-6:-1].mean())   # 5 bars ending at N-1
        vol_ma20_live = float(df5["volume"].iloc[-21:-1].mean())  # 20 bars ending at N-1
        vol_trend = vol_ma5_live / max(vol_ma20_live, 1e-8)
    else:
        vol_trend = np.nan

    # -----------------------------------------------------------------------
    # Time-of-day cyclical features — derived from the N-1 candle timestamp.
    # df5[-1] is forming candle N, so df5[-2] is N-1 (last fully-closed bar).
    # Replaces raw hour_utc and dow with sin/cos encoding (no discontinuities).
    # -----------------------------------------------------------------------
    ts_n1_live = df5["timestamp"].iloc[-2] if len(df5) >= 2 else None
    if ts_n1_live is not None and not pd.isna(ts_n1_live):
        ts = pd.Timestamp(ts_n1_live)
        hour_raw_live = float(ts.hour)
        dow_raw_live  = float(ts.dayofweek)
        hour_sin = float(np.sin(2 * np.pi * hour_raw_live / 24))
        hour_cos = float(np.cos(2 * np.pi * hour_raw_live / 24))
        dow_sin  = float(np.sin(2 * np.pi * dow_raw_live / 7))
        dow_cos  = float(np.cos(2 * np.pi * dow_raw_live / 7))
    else:
        hour_sin = hour_cos = dow_sin = dow_cos = np.nan

    # Volatility regime features — rolling window on the ATR series.
    # atr5_arr[-1] corresponds to forming candle N (present, never used as a feature).
    # atr5_arr[-2] is the N-1 ATR value, matching atr5.shift(1) at the current row
    # in training. The rolling window also ends at N-1: slice [max(0,L-289):-1]
    # gives up to 288 values with atr_n1 = atr5_arr[-2] as the last (most recent) entry.
    _ATR_WINDOW = 288
    if len(atr5) >= 14:
        atr5_arr = atr5.values  # full series including forming candle N at index -1
        atr_n1 = atr5_arr[-2] if len(atr5_arr) >= 2 else np.nan  # N-1 ATR value
        if pd.notna(atr_n1):
            # Window: up to 288 values ending at atr_n1 (index -2), excluding forming candle.
            # atr5_arr[max(0, L-289):-1] gives indices [L-289 .. L-2] in 0-based terms.
            window_vals = atr5_arr[max(0, len(atr5_arr)-_ATR_WINDOW-1):-1]
            window_vals = window_vals[~np.isnan(window_vals)]  # strip ATR warmup NaNs
            if len(window_vals) >= 14:
                # Rank atr_n1 (last element) against all prior values in the window,
                # matching the training _rolling_percentile logic: x[-1] vs x[:-1].
                atr_percentile_24h = float(np.sum(window_vals[:-1] < atr_n1)) / max(len(window_vals) - 1, 1)
            else:
                atr_percentile_24h = np.nan
            if len(window_vals) >= 2:
                w_mean = float(np.mean(window_vals))
                w_std  = float(np.std(window_vals, ddof=1))
                vol_regime = (atr_n1 - w_mean) / max(w_std, 1e-10)
            else:
                vol_regime = np.nan
        else:
            atr_percentile_24h = np.nan
            vol_regime = np.nan
    else:
        atr_percentile_24h = np.nan
        vol_regime = np.nan

    # -----------------------------------------------------------------------
    # Momentum features (live) — zero lookahead, all use N-1 values
    # -----------------------------------------------------------------------

    # rsi14 (live): Wilder's RSI(14) on 5m closes at N-1
    if len(df5) >= 15:
        delta_live = df5["close"].diff()
        gain_live  = delta_live.clip(lower=0)
        loss_live  = (-delta_live).clip(lower=0)
        avg_gain_live = gain_live.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss_live = loss_live.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs_live  = avg_gain_live / avg_loss_live.clip(lower=1e-10)
        rsi_live = 100.0 - (100.0 / (1.0 + rs_live))
        rsi14 = float(rsi_live.iloc[-2])  # N-1 value
        if np.isnan(rsi14):
            rsi14 = np.nan
    else:
        rsi14 = np.nan

    # candle_streak (live): how many consecutive same-direction closed candles
    # precede the N-1 bar (i.e. streak accumulated by N-2 and earlier).
    #
    # Training formula: _streak.shift(1) at row i gives the streak count of
    # candles [i-2, i-3, ...] that share the same direction as candle i-1.
    # Candle i-1 (N-1) itself sets the reference direction but is NOT counted.
    #
    # Live mapping (df5[-1] = forming candle N, df5[-2] = N-1, df5[-3] = N-2):
    #   dir_live[-1] = forming candle N direction (ignored)
    #   dir_live[-2] = N-1 direction  → reference direction
    #   dir_live[-3] = N-2 direction  → first candle checked in the streak walk
    if len(df5) >= 2:
        dir_live = np.sign(df5["close"].values - df5["open"].values)
        ref_dir = dir_live[-2]  # N-1 direction: reference, not counted in streak
        streak_val = 0.0
        if ref_dir != 0:
            # Walk backwards from N-2 (index -3 in dir_live) counting matching candles
            for k in range(3, len(dir_live) + 1):
                if dir_live[-k] == ref_dir:
                    streak_val += 1.0
                else:
                    break
        candle_streak = streak_val
    else:
        candle_streak = np.nan

    # price_in_range (live): where N-1 close sits within 20-candle range ending at N-1
    # high_arr/low_arr/close_arr defined unconditionally so structure features can reuse them
    high_arr  = df5["high"].values
    low_arr   = df5["low"].values
    close_arr = df5["close"].values
    if len(df5) >= 6:
        # N-1 close: index -2
        # 20-candle range ending at N-1 (inclusive): high/low of [-21:-1] or available
        window_hi = high_arr[max(0, len(high_arr)-21):-1]
        window_lo = low_arr[max(0, len(low_arr)-21):-1]
        if len(window_hi) >= 5:
            rng_hi = float(np.max(window_hi))
            rng_lo = float(np.min(window_lo))
            rng = max(rng_hi - rng_lo, 1e-10)
            price_in_range = (close_arr[-2] - rng_lo) / rng
        else:
            price_in_range = np.nan
    else:
        price_in_range = np.nan

    # ema_cross_5m (live): sign of EMA9 vs EMA21 at N-1
    if len(df5) >= 22:
        ema9_live  = df5["close"].ewm(span=9,  adjust=False).mean()
        ema21_live = df5["close"].ewm(span=21, adjust=False).mean()
        ema_cross_5m = float(np.sign(ema9_live.iloc[-2] - ema21_live.iloc[-2]))  # N-1
    else:
        ema_cross_5m = np.nan

    # -----------------------------------------------------------------------
    # Structure features (live) — mirrors build_features() formulas exactly
    # -----------------------------------------------------------------------

    # body_vs_range5 (live): |body_n1| / 5-bar range ending at N-1
    # Window: high/low of last 5 closed candles = indices [-6:-1]
    _n5_high = high_arr[max(0, len(high_arr)-6):-1]  # up to 5 values ending at N-1
    _n5_low  = low_arr[max(0, len(low_arr)-6):-1]
    if len(_n5_high) >= 2:
        _5bar_range_live = float(np.max(_n5_high) - np.min(_n5_low))
        _5bar_range_live = max(_5bar_range_live, 1e-9)
        _body_n1_abs = abs(float(df5["close"].iloc[-2]) - float(df5["open"].iloc[-2]))
        body_vs_range5 = _body_n1_abs / _5bar_range_live
    else:
        body_vs_range5 = np.nan

    # range_expansion (live): 5-bar range N-1 / 5-bar range 6..10 candles back
    # Prior window: high/low of indices [-11:-6] (candles i-6 through i-10)
    _pr_high = high_arr[max(0, len(high_arr)-11):-6] if len(high_arr) >= 7 else np.array([])
    _pr_low  = low_arr[max(0, len(low_arr)-11):-6]   if len(low_arr) >= 7 else np.array([])
    if len(_pr_high) >= 2 and len(_n5_high) >= 2:
        _prior_range_live = float(np.max(_pr_high) - np.min(_pr_low))
        _prior_range_live = max(_prior_range_live, 1e-9)
        range_expansion = _5bar_range_live / _prior_range_live
    else:
        range_expansion = np.nan

    # vwap_dist_20 (live): (close_n1 - vwap_20) / atr5_n1
    # VWAP over last 20 closed candles ending at N-1 = indices [-21:-1]
    if len(df5) >= 6:
        _cv_arr  = (df5["close"] * df5["volume"]).values
        _v_arr   = df5["volume"].values
        _cv_win  = _cv_arr[max(0, len(_cv_arr)-21):-1]   # up to 20 values ending at N-1
        _v_win   = _v_arr[max(0, len(_v_arr)-21):-1]
        if len(_cv_win) >= 5 and float(np.sum(_v_win)) > 1e-9:
            _vwap20_live = float(np.sum(_cv_win)) / float(np.sum(_v_win))
            vwap_dist_20 = (float(df5["close"].iloc[-2]) - _vwap20_live) / max(atr5_val, 1e-9)
        else:
            vwap_dist_20 = np.nan
    else:
        vwap_dist_20 = np.nan

    # -----------------------------------------------------------------------
    # Gate.io CVD taker flow features (live)
    #
    # Mirrors build_features() exactly:
    #   cvd_ratio          = long_taker_size / (long + short), clamped [0, 1]
    #   cvd_delta_norm     = (long - short) / atr5_val  (ATR-normalized)
    #   cvd_cumulative_5   = sum(delta[-6:-2]) / atr5_val  (5-bar rolling sum ending at N-2)
    #   cvd_cumulative_20  = sum(delta[-21:-2]) / atr5_val (20-bar rolling sum ending at N-2)
    #   cvd_trend_slope    = OLS slope of delta[-11:-2] / atr5_val (10-bar window ending at N-2)
    #   cvd_divergence     = +1 disagree / -1 agree / 0 flat (price vs CVD dir, 5-bar window ending at N-2)
    #   oi_change_5bar     = (oi[-2] - oi[-7]) / |oi[-7]|  (N-2 vs N-7, matches training shift(1)/shift(6))
    #
    # We look up the last CVD candle whose timestamp <= ts_n1_live (N-1 bar),
    # giving a slice where index -1 = N-1 bar. We then work on _delta_hist =
    # _delta_arr[:-1] (dropping N-1) so all rolling windows end at N-2.
    #
    # Parity guarantee: training uses rolling(...).shift(1) which at the N-1
    # merge point gives a window ending at N-2. Live mirrors this exactly by
    # excluding the last bar from all rolling computations. Neither path reads
    # the forming candle (N).
    # -----------------------------------------------------------------------
    cvd_live_available = (
        cvd_live is not None
        and not cvd_live.empty
        and "long_taker_size" in cvd_live.columns
        and "short_taker_size" in cvd_live.columns
    )

    # neutral defaults — used on any failure path
    cvd_ratio_live         = 0.5
    cvd_delta_norm_live    = 0.0
    cvd_cumulative_5_live  = 0.0
    cvd_cumulative_20_live = 0.0
    cvd_trend_slope_live   = 0.0
    cvd_divergence_live    = 0.0
    oi_change_5bar_live    = 0.0

    if cvd_live_available:
        # ts_n1_for_cvd: N-1 timestamp. df5[-1] is forming candle N, df5[-2] is N-1.
        ts_n1_for_cvd = df5["timestamp"].iloc[-2] if len(df5) >= 2 else None
        if ts_n1_for_cvd is not None and not pd.isna(ts_n1_for_cvd):
            _cvd_ts = pd.to_datetime(cvd_live["timestamp"], utc=True).astype("datetime64[ms, UTC]")
            _mask_cvd = _cvd_ts <= pd.Timestamp(ts_n1_for_cvd)
            if _mask_cvd.any():
                # Full slice of CVD bars up to and including N-1
                _cvd_slice = cvd_live[_mask_cvd].reset_index(drop=True)
                _cvd_row = _cvd_slice.iloc[-1]

                # --- cvd_ratio and cvd_delta_norm (single bar N-1) ---
                _lts = float(_cvd_row.get("long_taker_size", 0) or 0)
                _sts = float(_cvd_row.get("short_taker_size", 0) or 0)
                _total = max(_lts + _sts, 1e-9)
                cvd_ratio_live = float(np.clip(_lts / _total, 0.0, 1.0))
                _delta_n1 = _lts - _sts
                cvd_delta_norm_live = float(_delta_n1 / max(atr5_val, 1e-9))

                # --- rolling delta series (all bars up to N-1 inclusive) ---
                _lts_arr = _cvd_slice["long_taker_size"].astype(float).values
                _sts_arr = _cvd_slice["short_taker_size"].astype(float).values
                _delta_arr = _lts_arr - _sts_arr  # shape (K,), last element = N-1 bar

                # Training uses rolling(N).sum().shift(1) on the CVD frame, then
                # merges at ts_n1 (N-1 bar). The shift(1) means the value stored at
                # CVD bar index t is the rolling sum of bars [t-N .. t-1] — it does
                # NOT include bar t itself. So when merged at N-1, the window ends at
                # N-2 (one bar before N-1).
                #
                # Live fix: exclude the last element (_delta_arr[-1] = N-1) from all
                # rolling computations by working on _delta_hist = _delta_arr[:-1].
                # This gives a window ending at N-2, exactly matching training.
                _delta_hist = _delta_arr[:-1]  # bars up to N-2 (training-equivalent)

                # cvd_cumulative_5: sum of last 5 bars ending at N-2
                if len(_delta_hist) >= 2:
                    _slice5  = _delta_hist[max(0, len(_delta_hist) - 5):]
                    _slice20 = _delta_hist[max(0, len(_delta_hist) - 20):]
                    cvd_cumulative_5_live  = float(np.sum(_slice5)  / max(atr5_val, 1e-9))
                    cvd_cumulative_20_live = float(np.sum(_slice20) / max(atr5_val, 1e-9))
                else:
                    cvd_cumulative_5_live  = 0.0
                    cvd_cumulative_20_live = 0.0

                # cvd_trend_slope: OLS slope over last 10 bars ending at N-2
                if len(_delta_hist) >= 3:
                    _slope_vals = _delta_hist[max(0, len(_delta_hist) - 10):]
                    _v = _slope_vals[~np.isnan(_slope_vals)]
                    if len(_v) >= 3:
                        _x = np.arange(len(_v), dtype=np.float64)
                        try:
                            _slope = float(np.polyfit(_x, _v, 1)[0])
                        except Exception:
                            _slope = 0.0
                        cvd_trend_slope_live = float(_slope / max(atr5_val, 1e-9))
                    else:
                        cvd_trend_slope_live = 0.0
                else:
                    cvd_trend_slope_live = 0.0

                # cvd_divergence: sign disagreement between price dir and CVD dir.
                # Both directions use the 5-bar window ending at N-2 to match
                # training's shift(1) semantics.
                #
                # Price direction: training pre-shifts body5_sum before the asof
                # merge, so the merged value at CVD bar t reflects the price-body
                # sum ending one 5m candle BEFORE t. When t = N-1, that window ends
                # at N-2. Live equivalent: df5[-7:-2] (5 closed candles ending at N-2;
                # df5[-1] = forming N, df5[-2] = N-1, df5[-7:-2] = N-2..N-6).
                if len(df5) >= 7:
                    _price_bodies = (df5["close"].iloc[-7:-2] - df5["open"].iloc[-7:-2]).values
                    _price_dir = np.sign(float(np.sum(_price_bodies)))
                else:
                    _price_dir = 0.0
                # CVD direction: last 5 bars of _delta_hist (ending at N-2)
                _cvd_dir_5 = np.sign(float(np.sum(_delta_hist[max(0, len(_delta_hist) - 5):])))
                if _price_dir == 0.0 or _cvd_dir_5 == 0.0:
                    cvd_divergence_live = 0.0
                elif _price_dir != _cvd_dir_5:
                    cvd_divergence_live = 1.0   # diverging
                else:
                    cvd_divergence_live = -1.0  # aligning

                # oi_change_5bar: (oi[N-2] - oi[N-7]) / |oi[N-7]|
                # Training: shift(1) → oi[N-2], shift(6) → oi[N-7] when merged at N-1.
                # Live fix: use index -2 (N-2) and -7 (N-7) in the OI array.
                if "open_interest" in _cvd_slice.columns and len(_cvd_slice) >= 7:
                    _oi_arr = _cvd_slice["open_interest"].astype(float).values
                    _oi_n2  = _oi_arr[-2]   # N-2 (matches training shift(1))
                    _oi_n7  = _oi_arr[-7]   # N-7 (matches training shift(6))
                    _oi_denom = max(abs(_oi_n7), 1e-9)
                    oi_change_5bar_live = float((_oi_n2 - _oi_n7) / _oi_denom)
                else:
                    oi_change_5bar_live = 0.0

            else:
                log.debug(
                    "build_live_features: no CVD candle at or before ts_n1=%s — using neutrals",
                    ts_n1_for_cvd,
                )
        # else: ts_n1_for_cvd is None/NaT — neutrals already set above
    else:
        log.debug(
            "build_live_features: cvd_live not provided or empty — "
            "using neutral defaults for all CVD/OI features"
        )

    row = np.array([[
        body_ratio_n1, body_ratio_n2, body_ratio_n3,
        upper_wick_n1, upper_wick_n2,
        lower_wick_n1, lower_wick_n2,
        vol_ratio_n1, vol_ratio_n2,
        body_ratio_15m, dir_15m, vol_ratio_15m,
        body_ratio_1h, dir_1h, ema9_slope_1h,
        body_ratio, upper_wick_ratio, lower_wick_ratio, vol_zscore, vol_trend,
        hour_sin, hour_cos, dow_sin, dow_cos,
        atr_percentile_24h, vol_regime,
        rsi14, candle_streak, price_in_range, ema_cross_5m,
        body_vs_range5, range_expansion, vwap_dist_20,
        cvd_ratio_live, cvd_delta_norm_live,
        cvd_cumulative_5_live, cvd_cumulative_20_live, cvd_trend_slope_live,
        cvd_divergence_live, oi_change_5bar_live,
    ]], dtype=np.float64)

    nan_features = [FEATURE_COLS[i] for i in range(len(FEATURE_COLS)) if np.isnan(row[0][i])]
    if nan_features:
        log.warning("build_live_features: NaN in features, skipping inference. NaN features: %s", nan_features)
        return None, nan_features

    return row, []
