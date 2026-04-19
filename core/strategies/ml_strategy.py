"""ML strategy using a trained LightGBM model for BTC/USDT 5-min binary prediction.

Returns the IDENTICAL signal dict schema as PatternStrategy.
Uses get_next_slot_info() + get_slot_prices() exactly as PatternStrategy does.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any


from core.strategies.base import BaseStrategy
from ml import data_fetcher
from ml import features as feat_eng
from ml import model_store
from ml import inference_logger
from ml.probability import apply_probability_calibration, build_live_trust_report
from db import queries
from polymarket.markets import get_next_slot_info, get_slot_prices
import config as cfg

log = logging.getLogger(__name__)

FEATURE_COLS = feat_eng.FEATURE_COLS  # 40 features in exact order


def _trust_reason_summary(reasons: list[str] | None) -> str:
    if not reasons:
        return "none"
    return ";".join(reasons[:3])


def _calibration_metadata_status(metadata: dict[str, Any] | None) -> tuple[bool, str | None]:
    if not isinstance(metadata, dict):
        return False, None
    calibration = metadata.get("probability_calibration")
    if not isinstance(calibration, dict):
        return False, None
    method = calibration.get("method")
    return True, method if isinstance(method, str) else None

# Module-level reload flag so cmd_promote_model can signal a reload
_RELOAD_REQUESTED = False

# Module-level preloaded model — injected at startup via set_model()
_PRELOADED_MODEL = None


def set_model(model) -> None:
    """Inject a pre-loaded model instance at startup (or after retrain/promote).

    Called from main.py post_init after loading the model from DB or disk.
    The injected model is consumed by _load_model() on the next MLStrategy
    instantiation or reload cycle.
    """
    global _PRELOADED_MODEL
    _PRELOADED_MODEL = model


def request_model_reload() -> None:
    """Signal that the model should be reloaded on the next check_signal call."""
    global _RELOAD_REQUESTED
    _RELOAD_REQUESTED = True


class MLStrategy(BaseStrategy):
    """LightGBM-based signal strategy. Replaces PatternStrategy as the default."""

    def __init__(self):
        self._model = None
        self._model_slot = "current"
        try:
            self._load_model()
        except Exception:
            log.exception(
                "MLStrategy.__init__: _load_model failed — model will be None; "
                "signals will be skipped until a model is loaded via set_model() or /retrain"
            )


    def _load_model(self) -> None:
        """Load the current model — use preloaded model if available, else load from disk."""
        global _RELOAD_REQUESTED, _PRELOADED_MODEL
        if _PRELOADED_MODEL is not None:
            self._model = _PRELOADED_MODEL
            _PRELOADED_MODEL = None
            _RELOAD_REQUESTED = False
            log.info("MLStrategy: model set from preloaded instance")
        else:
            self._model = model_store.load_model("current")
            _RELOAD_REQUESTED = False
            if self._model is None:
                log.warning("MLStrategy: no trained model found at models/model_current.lgb")
            else:
                log.info("MLStrategy: model loaded successfully")

    async def _get_threshold(self) -> float:
        """Read UP threshold from ml_config table, fall back to cfg default."""
        try:
            val = await queries.get_ml_threshold()
            return val
        except Exception:
            pass
        # Legacy fallback: check settings table
        try:
            val = await queries.get_setting("ml_threshold")
            if val is not None:
                return float(val)
        except Exception:
            pass
        return cfg.ML_DEFAULT_THRESHOLD

    async def _get_down_threshold(self, up_threshold: float) -> float:
        """Read DOWN threshold from ml_config table.

        Falls back to the symmetric complement (1 - up_threshold) so the
        system is fully backwards-compatible with models trained before
        down_threshold was stored explicitly.
        """
        try:
            val = await queries.get_ml_down_threshold()
            if val is not None:
                return val
        except Exception:
            pass
        # Symmetric fallback: mirror of the UP threshold around 0.5
        return round(1.0 - up_threshold, 4)

    def _get_down_enabled(self) -> bool:
        """Read down_enabled flag from current model metadata.

        Returns False if metadata is missing or down_enabled is not set,
        ensuring backwards-compatibility with models trained before Option B.
        """
        try:
            meta = model_store.load_metadata(self._model_slot)
            if meta is not None:
                if meta.get("down_override", False):
                    return True
                return bool(meta.get("down_enabled", False))
        except Exception:
            pass
        return False

    async def check_signal(self) -> dict[str, Any] | None:
        """Generate an ML-based signal for slot N+1.

        Called at T-85s before the current slot ends.

        Returns the same signal dict schema as PatternStrategy:
          - skipped=True dict when no trade (below threshold or data issues)
          - skipped=False dict with full trade fields when model fires
          - None on hard failure
        """
        global _RELOAD_REQUESTED

        # Reload model if requested (e.g., after promote_model)
        if _RELOAD_REQUESTED:
            self._load_model()

        # Get next slot info — identical pattern to PatternStrategy
        slot_n1 = get_next_slot_info()
        slug = slot_n1["slug"]
        slot_ts = slot_n1["slot_start_ts"]
        slot_start_str = slot_n1["slot_start_str"]
        slot_end_str = slot_n1["slot_end_str"]

        # Standard base fields used in all return dicts (matches PatternStrategy exactly)
        base_fields: dict[str, Any] = {
            "skipped": True,
            "pattern": None,
            "candles_used": 400,
            "slot_n1_start_full": slot_n1["slot_start_full"],
            "slot_n1_end_full":   slot_n1["slot_end_full"],
            "slot_n1_start_str":  slot_start_str,
            "slot_n1_end_str":    slot_end_str,
            "slot_n1_ts":         slot_ts,
            "slot_n1_slug":       slug,
        }

        if self._model is None:
            self._load_model()
            if self._model is None:
                log.error("MLStrategy: no model loaded, skipping slot %s", slug)
                inference_logger.log_skipped_data(
                    slot_slug=slug,
                    slot_ts=slot_ts,
                    slot_start_str=slot_start_str,
                    slot_end_str=slot_end_str,
                    skip_reason="No model loaded",
                )
                return {**base_fields, "reason": "No model loaded"}

        try:
            # Fetch live data in parallel using executor (blocking ccxt calls)
            loop = asyncio.get_running_loop()
            df5, df15, df1h, cvd_live = await asyncio.gather(
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_5m(400)),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_15m(100)),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_1h(60)),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_gate_cvd(400)),
            )

            # --- Data quality snapshot (before dropping the forming candle) ---
            df5_rows_raw  = len(df5)      if df5      is not None else 0
            df15_rows     = len(df15)     if df15     is not None else 0
            df1h_rows     = len(df1h)     if df1h     is not None else 0
            cvd_rows_raw  = len(cvd_live) if cvd_live is not None and not cvd_live.empty else 0

            # df5 is passed to build_live_features WITH the still-forming candle N
            # intact as the last row (index -1).  build_live_features always uses
            # safe(series, k=1) / iloc[-2] to reference the N-1 (last fully-closed)
            # candle — the forming candle is never read as a feature value.
            #
            # Keeping the forming candle present is required for parity with training:
            # build_features() operates on a full df5 that includes row i (the "current"
            # row whose target we are predicting), and all feature shifts are k>=1.
            # Trimming df5 before calling build_live_features shifts every index by one,
            # making safe(s,1) return N-2 instead of N-1 — a systematic one-candle lag
            # that corrupts the feature row and inverts the model's predictions.
            #
            # 15m/1h/CVD are also NOT trimmed: build_live_features selects the most
            # recent candle with timestamp <= ts_n1 (the N-1 5m bar's timestamp) via
            # backward merge, so no trimming is needed or correct there either.

            # Row counts (what the model actually sees)
            df5_rows  = len(df5)
            cvd_rows  = len(cvd_live) if cvd_live is not None and not cvd_live.empty else 0

            # N-1 candle metadata for the log
            candle_n1_ts    = None
            candle_n1_close = None
            candle_n1_vol   = None
            if df5_rows >= 2:
                try:
                    import pandas as _pd
                    # df5[-1] is the still-forming candle N; df5[-2] is the last
                    # fully-closed candle N-1 — that is what we log as "candle_n1".
                    n1 = df5.iloc[-2]
                    ts_raw = n1["timestamp"]
                    if isinstance(ts_raw, _pd.Timestamp):
                        candle_n1_ts = str(ts_raw.tz_localize("UTC").isoformat() if ts_raw.tzinfo is None else ts_raw.isoformat())
                    elif ts_raw is not None:
                        candle_n1_ts = str(_pd.Timestamp(int(ts_raw), unit="ms", tz="UTC").isoformat())
                    else:
                        candle_n1_ts = None
                    candle_n1_close = float(n1["close"])
                    candle_n1_vol   = float(n1["volume"])
                except Exception as _e:
                    log.debug("inference_logger: candle_n1 extraction failed: %s", _e)


            # Build feature row — returns (row, nan_features) 2-tuple

            try:
                feature_row, nan_features = feat_eng.build_live_features(
                    df5, df15, df1h, cvd_live
                )
            except feat_eng.MissingLiveCVDDataError as exc:
                log.warning("MLStrategy: %s", exc)
                inference_logger.log_inference(
                    slot_slug=slug,
                    slot_ts=slot_ts,
                    slot_start_str=slot_start_str,
                    slot_end_str=slot_end_str,
                    df5_rows=df5_rows,
                    df15_rows=df15_rows,
                    df1h_rows=df1h_rows,
                    cvd_rows=cvd_rows,
                    candle_n1_ts=candle_n1_ts,
                    candle_n1_close=candle_n1_close,
                    candle_n1_vol=candle_n1_vol,
                    feature_names=FEATURE_COLS,
                    feature_row=None,
                    nan_features=[],
                    p_up=None,
                    p_down=None,
                    up_threshold=None,
                    down_threshold=None,
                    down_enabled=False,
                    fired=False,
                    side=None,
                    skip_reason=f"Missing required Gate CVD: {exc}",
                )
                return {**base_fields, "reason": f"Missing required Gate CVD: {exc}"}

            metadata = model_store.load_metadata(self._model_slot) or {}
            calibration_metadata_present, calibration_metadata_method = _calibration_metadata_status(metadata)

            # Model inference — P(UP) as a single float in [0, 1]
            raw_prob = float(self._model.predict(feature_row)[0])
            calibrated_probs, calibration_applied, calibration_method = apply_probability_calibration([raw_prob], metadata)
            prob = float(calibrated_probs[0])
            up_threshold   = await self._get_threshold()
            down_threshold = await self._get_down_threshold(up_threshold)

            # P(DOWN) = 1 - P(UP)
            prob_down = round(1.0 - prob, 6)

            trust_report = build_live_trust_report(feature_row, FEATURE_COLS, metadata)
            trust_ok = bool(trust_report.get("ok", True))
            trust_reasons = trust_report.get("reasons", [])
            trust_failed_open = bool((trust_report.get("details") or {}).get("failed_open"))
            if trust_failed_open:
                log.warning(
                    "MLStrategy: live trust gate failed open slot=%s reason=%s",
                    slug, (trust_report.get("details") or {}).get("failed_open"),
                )
            log.info(
                "MLStrategy: probability_path slot=%s raw=%.4f calibrated=%.4f calibration_applied=%s calibration_metadata_present=%s metadata_method=%s method_used=%s trust_ok=%s trust_reason=%s",
                slug, raw_prob, prob, calibration_applied, calibration_metadata_present,
                calibration_metadata_method, calibration_method, trust_ok, _trust_reason_summary(trust_reasons),
            )

            up_qualifies = prob >= up_threshold

            # DOWN gate: only fire if the model's DOWN side was independently
            # validated (down_enabled=True in metadata). If the model was trained
            # before Option B or failed the DOWN sweep, down_enabled=False and
            # no DOWN trade ever fires regardless of prob_down.
            down_enabled = self._get_down_enabled()

            # ------------------------------------------------------------------
            # Regime gate -- covariate shift guard (Blueprint Option 1).
            #
            # At training time, trainer.py records the 5th and 95th percentile
            # of vol_regime across the full training dataset and stores them as
            # "regime_vol_p5" / "regime_vol_p95" in the model metadata JSON.
            #
            # Here we compare the live vol_regime value against those bounds.
            # If the live regime falls OUTSIDE [p5, p95], the model is operating
            # in a volatility environment it rarely saw during training -- its
            # probability estimates are less calibrated and the signal is suppressed.
            #
            # Design decisions:
            #   - Gate fires AFTER model.predict() so the log always contains the
            #     full p_up/p_down values. This lets you audit "what the model
            #     wanted to do" vs "what the gate blocked" -- invaluable for tuning.
            #   - If metadata is missing or the keys are absent (e.g. older model
            #     trained before this feature), the gate is silently skipped.
            #     This guarantees full backwards compatibility with no config change.
            #   - If either bound is None (degenerate training set < 10 samples),
            #     the gate is skipped -- not the live bot's fault, don't punish it.
            #   - The gate itself is wrapped in try/except so a metadata read error
            #     never crashes inference -- the model fires normally if gate errors.
            # ------------------------------------------------------------------
            try:
                _meta = metadata
                if _meta is not None:
                    _regime_p5  = _meta.get("regime_vol_p5")
                    _regime_p95 = _meta.get("regime_vol_p95")
                    if _regime_p5 is not None and _regime_p95 is not None:
                        _vol_regime_idx = FEATURE_COLS.index("vol_regime")
                        _live_regime = float(feature_row[0, _vol_regime_idx])
                        if not (_regime_p5 <= _live_regime <= _regime_p95):
                            _regime_skip_reason = (
                                f"Regime gate: vol_regime={_live_regime:.4f} outside training "
                                f"distribution [{_regime_p5:.4f}, {_regime_p95:.4f}] -- "
                                f"signal suppressed (covariate shift guard)"
                            )
                            log.warning("MLStrategy: %s", _regime_skip_reason)
                            inference_logger.log_inference(
                                slot_slug=slug,
                                slot_ts=slot_ts,
                                slot_start_str=slot_start_str,
                                slot_end_str=slot_end_str,
                                df5_rows=df5_rows,
                                df15_rows=df15_rows,
                                df1h_rows=df1h_rows,
                                cvd_rows=cvd_rows,
                                candle_n1_ts=candle_n1_ts,
                                candle_n1_close=candle_n1_close,
                                candle_n1_vol=candle_n1_vol,
                                feature_names=FEATURE_COLS,
                                feature_row=feature_row,
                                nan_features=[],
                                p_up=prob,
                                p_down=prob_down,
                                up_threshold=up_threshold,
                                down_threshold=down_threshold,
                                down_enabled=down_enabled,
                                fired=False,
                                side=None,
                                skip_reason=_regime_skip_reason,
                            )
                            return {
                                **base_fields,
                                "pattern": f"p={prob:.4f} [regime_gate]",
                                "reason": _regime_skip_reason,
                                "ml_p_up": prob,
                                "ml_p_up_raw": raw_prob,
                                "ml_probability_calibrated": calibration_applied,
                                "ml_probability_calibration_method": calibration_method,
                                "ml_p_down": prob_down,
                                "ml_up_threshold": up_threshold,
                                "ml_down_threshold": down_threshold,
                                "ml_down_enabled": down_enabled,
                            }
            except Exception as _rge:
                # Never let the regime gate itself crash inference.
                # Log and continue -- the model fires normally if the gate errors.
                log.warning(
                    "MLStrategy: regime gate check failed open slot=%s reason=%s", slug, _rge
                )

            if not trust_report.get("ok", True):
                trust_reason = "Trust gate: " + "; ".join(trust_report.get("reasons", []))
                log.warning(
                    "MLStrategy: trust gate blocked slot=%s raw=%.4f calibrated=%.4f method=%s reason=%s",
                    slug, raw_prob, prob, calibration_method, _trust_reason_summary(trust_report.get("reasons", [])),
                )
                inference_logger.log_inference(
                    slot_slug=slug,
                    slot_ts=slot_ts,
                    slot_start_str=slot_start_str,
                    slot_end_str=slot_end_str,
                    df5_rows=df5_rows,
                    df15_rows=df15_rows,
                    df1h_rows=df1h_rows,
                    cvd_rows=cvd_rows,
                    candle_n1_ts=candle_n1_ts,
                    candle_n1_close=candle_n1_close,
                    candle_n1_vol=candle_n1_vol,
                    feature_names=FEATURE_COLS,
                    feature_row=feature_row,
                    nan_features=[],
                    p_up=prob,
                    p_down=prob_down,
                    up_threshold=up_threshold,
                    down_threshold=down_threshold,
                    down_enabled=down_enabled,
                    fired=False,
                    side=None,
                    skip_reason=trust_reason,
                )
                return {
                    **base_fields,
                    "pattern": f"p={prob:.4f} [trust_gate]",
                    "reason": trust_reason,
                    "ml_p_up": prob,
                    "ml_p_up_raw": raw_prob,
                    "ml_probability_calibrated": calibration_applied,
                    "ml_probability_calibration_method": calibration_method,
                    "ml_p_down": prob_down,
                    "ml_up_threshold": up_threshold,
                    "ml_down_threshold": down_threshold,
                    "ml_down_enabled": down_enabled,
                }

            down_qualifies = down_enabled and (prob_down >= down_threshold)

            # Determine direction:
            #   - Both qualify  → pick the one with the larger margin over its threshold.
            #     With independently validated thresholds this is extremely rare
            #     (would require p_up >= up_thr AND 1-p_up >= down_thr simultaneously)
            #     but we handle it cleanly rather than crashing.
            #   - Only one      → pick that one
            #   - Neither       → skip
            if up_qualifies and down_qualifies:
                up_margin   = prob      - up_threshold
                down_margin = prob_down - down_threshold
                side = "Up" if up_margin >= down_margin else "Down"
                log.info(
                    "MLStrategy: BOTH qualify — up_margin=%.4f down_margin=%.4f → side=%s",
                    up_margin, down_margin, side,
                )
            elif up_qualifies:
                side = "Up"
            elif down_qualifies:
                side = "Down"
            else:
                # Build skip reason — include DOWN gate status so logs are clear
                if not down_enabled:
                    down_reason = "DOWN disabled (not validated)"
                else:
                    down_reason = f"p_down={prob_down:.4f}<{down_threshold:.3f}"
                skip_reason = (
                    f"Below threshold (p_up={prob:.4f}<{up_threshold:.3f}, {down_reason})"
                )
                inference_logger.log_inference(
                    slot_slug=slug,
                    slot_ts=slot_ts,
                    slot_start_str=slot_start_str,
                    slot_end_str=slot_end_str,
                    df5_rows=df5_rows,
                    df15_rows=df15_rows,
                    df1h_rows=df1h_rows,
                    cvd_rows=cvd_rows,
                    candle_n1_ts=candle_n1_ts,
                    candle_n1_close=candle_n1_close,
                    candle_n1_vol=candle_n1_vol,
                    feature_names=FEATURE_COLS,
                    feature_row=feature_row,
                    nan_features=[],
                    p_up=prob,
                    p_down=prob_down,
                    up_threshold=up_threshold,
                    down_threshold=down_threshold,
                    down_enabled=down_enabled,
                    fired=False,
                    side=None,
                    skip_reason=skip_reason,
                )
                return {
                    **base_fields,
                    "pattern": f"p={prob:.4f}<{up_threshold:.3f}",
                    "reason": skip_reason,
                    # Structured ML fields for rich Telegram formatting
                    "ml_p_up": prob,
                    "ml_p_up_raw": raw_prob,
                    "ml_probability_calibrated": calibration_applied,
                    "ml_probability_calibration_method": calibration_method,
                    "ml_p_down": prob_down,
                    "ml_up_threshold": up_threshold,
                    "ml_down_threshold": down_threshold,
                    "ml_down_enabled": down_enabled,
                }

            log.info(
                "MLStrategy: inference_decision slot=%s side=%s raw=%.4f calibrated=%.4f p_down=%.4f up_thr=%.3f down_thr=%.3f down_enabled=%s calibration_applied=%s method=%s trust_ok=%s",
                slug, side, raw_prob, prob, prob_down, up_threshold, down_threshold,
                down_enabled, calibration_applied, calibration_method, trust_ok,
            )

            # Fetch Polymarket prices — identical to PatternStrategy
            prices = await get_slot_prices(slug)
            if prices is None:
                log.warning(
                    "MLStrategy: no Polymarket prices for slug=%s, skipping", slug
                )
                inference_logger.log_inference(
                    slot_slug=slug,
                    slot_ts=slot_ts,
                    slot_start_str=slot_start_str,
                    slot_end_str=slot_end_str,
                    df5_rows=df5_rows,
                    df15_rows=df15_rows,
                    df1h_rows=df1h_rows,
                    cvd_rows=cvd_rows,
                    candle_n1_ts=candle_n1_ts,
                    candle_n1_close=candle_n1_close,
                    candle_n1_vol=candle_n1_vol,
                    feature_names=FEATURE_COLS,
                    feature_row=feature_row,
                    nan_features=[],
                    p_up=prob,
                    p_down=prob_down,
                    up_threshold=up_threshold,
                    down_threshold=down_threshold,
                    down_enabled=down_enabled,
                    fired=False,
                    side=side,
                    skip_reason="Market data unavailable (no Polymarket prices)",
                )
                return {
                    **base_fields,
                    "pattern": f"p={prob:.4f}",
                    "reason": "Market data unavailable",
                    # ML inference already completed — include structured fields so
                    # the scheduler can render the rich ML skip card instead of
                    # falling back to the generic format_skip() card.
                    "ml_p_up":           prob,
                    "ml_p_up_raw":       raw_prob,
                    "ml_probability_calibrated": calibration_applied,
                    "ml_probability_calibration_method": calibration_method,
                    "ml_p_down":         prob_down,
                    "ml_up_threshold":   up_threshold,
                    "ml_down_threshold": down_threshold,
                    "ml_down_enabled":   down_enabled,
                }

            entry_price    = prices["up_price"]    if side == "Up" else prices["down_price"]
            opposite_price = prices["down_price"]  if side == "Up" else prices["up_price"]
            token_id          = prices["up_token_id"] if side == "Up" else prices["down_token_id"]
            opposite_token_id = prices["down_token_id"] if side == "Up" else prices["up_token_id"]

            # Log the fired inference record
            inference_logger.log_inference(
                slot_slug=slug,
                slot_ts=slot_ts,
                slot_start_str=slot_start_str,
                slot_end_str=slot_end_str,
                df5_rows=df5_rows,
                df15_rows=df15_rows,
                df1h_rows=df1h_rows,
                cvd_rows=cvd_rows,
                candle_n1_ts=candle_n1_ts,
                candle_n1_close=candle_n1_close,
                candle_n1_vol=candle_n1_vol,
                feature_names=FEATURE_COLS,
                feature_row=feature_row,
                nan_features=[],
                p_up=prob,
                p_down=prob_down,
                up_threshold=up_threshold,
                down_threshold=down_threshold,
                down_enabled=down_enabled,
                fired=True,
                side=side,
                skip_reason=None,
            )

            return {
                **base_fields,
                "skipped":           False,
                "side":              side,
                "entry_price":       entry_price,
                "opposite_price":    opposite_price,
                "token_id":          token_id,
                "opposite_token_id": opposite_token_id,
                "pattern":        f"p_up={prob:.4f},p_down={prob_down:.4f}" + (f",raw={raw_prob:.4f}" if calibration_applied else ""),
                # Structured ML fields for rich Telegram formatting
                "ml_p_up":          prob,
                "ml_p_up_raw":      raw_prob,
                "ml_probability_calibrated": calibration_applied,
                "ml_probability_calibration_method": calibration_method,
                "ml_p_down":        prob_down,
                "ml_up_threshold":  up_threshold,
                "ml_down_threshold": down_threshold,
                "ml_down_enabled":  down_enabled,
            }

        except Exception as exc:
            log.exception("MLStrategy.check_signal failed: %s", exc)
            return None
