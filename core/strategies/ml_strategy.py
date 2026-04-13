"""ML strategy using a trained LightGBM model for BTC/USDT 5-min binary prediction.

Returns the IDENTICAL signal dict schema as PatternStrategy.
Uses get_next_slot_info() + get_slot_prices() exactly as PatternStrategy does.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from core.strategies.base import BaseStrategy
from ml import data_fetcher
from ml import features as feat_eng
from ml import model_store
from db import queries
from polymarket.markets import get_next_slot_info, get_slot_prices
import config as cfg

log = logging.getLogger(__name__)

FEATURE_COLS = feat_eng.FEATURE_COLS  # 26 features in exact order

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
        self._funding_buffer: deque = deque(maxlen=24)
        self._model_slot = "current"
        # Track the last funding settlement timestamp that was appended to the
        # buffer.  MEXC settles funding every 8h (00:00, 08:00, 16:00 UTC).
        # We only append to the buffer when a new settlement period has started,
        # matching the training data semantics where each buffer entry represents
        # one distinct 8h settlement — not a repeated 5m snapshot of the same rate.
        self._last_funding_settlement: datetime | None = None
        # Each step is individually guarded so a failure in one never prevents
        # the other from running, and a constructor crash can never propagate
        # up to _get_strategy() / the scheduler.
        try:
            self._load_model()
        except Exception:
            log.exception(
                "MLStrategy.__init__: _load_model failed — model will be None; "
                "signals will be skipped until a model is loaded via set_model() or /retrain"
            )
        try:
            self._seed_funding_buffer()
        except Exception:
            log.exception(
                "MLStrategy.__init__: _seed_funding_buffer failed — "
                "funding zscore will be undefined for the first live periods; "
                "inference will continue with an empty buffer"
            )

    def _seed_funding_buffer(self) -> None:
        """Seed the funding buffer with historical data on startup.

        Without seeding, the buffer starts empty and zscore is undefined for the
        first 8 days of operation (24 periods * 8h each). This pre-fills the buffer
        from MEXC historical funding so zscore is valid from the very first inference.
        """
        try:
            history = data_fetcher.fetch_live_funding_history(n_periods=24)
            if history:
                for rate in history:
                    self._funding_buffer.append(rate)
                log.info(
                    "MLStrategy: seeded funding_buffer with %d historical records",
                    len(self._funding_buffer),
                )
            else:
                log.warning("MLStrategy: could not seed funding_buffer — no historical data returned")
        except Exception as exc:
            log.warning("MLStrategy: funding_buffer seed failed: %s", exc)

    @staticmethod
    def _current_funding_settlement() -> datetime:
        """Return the most recent MEXC funding settlement timestamp (UTC).

        MEXC settles funding at 00:00, 08:00, and 16:00 UTC every day.
        This returns the floor of utcnow() to the nearest 8h boundary,
        giving a stable, deterministic key for deduplication.
        """
        now = datetime.now(timezone.utc)
        # Hours since midnight, floored to 8h block: 0, 8, or 16
        settlement_hour = (now.hour // 8) * 8
        return now.replace(hour=settlement_hour, minute=0, second=0, microsecond=0)

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

        # Standard base fields used in all return dicts (matches PatternStrategy exactly)
        base_fields: dict[str, Any] = {
            "skipped": True,
            "pattern": None,
            "candles_used": 400,
            "slot_n1_start_full": slot_n1["slot_start_full"],
            "slot_n1_end_full":   slot_n1["slot_end_full"],
            "slot_n1_start_str":  slot_n1["slot_start_str"],
            "slot_n1_end_str":    slot_n1["slot_end_str"],
            "slot_n1_ts":         slot_n1["slot_start_ts"],
            "slot_n1_slug":       slot_n1["slug"],
        }

        if self._model is None:
            self._load_model()
            if self._model is None:
                log.error("MLStrategy: no model loaded, skipping slot %s", slot_n1["slug"])
                return {**base_fields, "reason": "No model loaded"}

        try:
            # Fetch live data in parallel using executor (blocking ccxt calls)
            loop = asyncio.get_running_loop()
            df5, df15, df1h, funding_rate, cvd_live = await asyncio.gather(
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_5m(400)),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_15m(100)),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_1h(60)),
                loop.run_in_executor(None, data_fetcher.fetch_live_funding),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_cvd(400)),
            )

            # Drop the still-forming 5m candle so inference uses data only up to N-1.
            #
            # IMPORTANT PARITY NOTE:
            # We intentionally do NOT drop the last 15m/1h rows here.
            # build_live_features() already selects the most recent 15m/1h candle
            # with timestamp <= ts_n1 (where ts_n1 is the closed 5m N-1 candle).
            # Dropping 15m/1h pre-emptively can shift those context features one
            # bucket stale vs training and create train/live feature drift.
            df5 = df5.iloc[:-1].reset_index(drop=True)
            cvd_live = cvd_live.iloc[:-1].reset_index(drop=True) if cvd_live is not None and len(cvd_live) > 0 else cvd_live

            # Update funding rolling buffer — only append when a new 8h settlement
            # has occurred, matching training data semantics (one entry per settlement
            # period, not one entry per 5m check_signal call).
            if funding_rate is not None:
                current_settlement = self._current_funding_settlement()
                if self._last_funding_settlement != current_settlement:
                    self._funding_buffer.append(funding_rate)
                    self._last_funding_settlement = current_settlement
                    log.debug(
                        "MLStrategy: funding_buffer updated for settlement=%s rate=%.6f buffer_len=%d",
                        current_settlement.isoformat(), funding_rate, len(self._funding_buffer),
                    )

            # Build feature row
            feature_row = feat_eng.build_live_features(
                df5, df15, df1h, funding_rate, self._funding_buffer, cvd_live
            )
            if feature_row is None:
                log.warning("MLStrategy: insufficient data for features, skipping")
                return {**base_fields, "reason": "Insufficient data for features"}

            # Model inference — P(UP) as a single float in [0, 1]
            prob = float(self._model.predict(feature_row)[0])
            up_threshold   = await self._get_threshold()
            down_threshold = await self._get_down_threshold(up_threshold)

            # P(DOWN) = 1 - P(UP)
            prob_down = round(1.0 - prob, 6)

            up_qualifies = prob >= up_threshold

            # DOWN gate: only fire if the model's DOWN side was independently
            # validated (down_enabled=True in metadata). If the model was trained
            # before Option B or failed the DOWN sweep, down_enabled=False and
            # no DOWN trade ever fires regardless of prob_down.
            down_enabled = self._get_down_enabled()
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
                return {
                    **base_fields,
                    "pattern": f"p={prob:.4f}<{up_threshold:.3f}",
                    "reason": (
                        f"Below threshold (p_up={prob:.4f}<{up_threshold:.3f}, "
                        f"{down_reason})"
                    ),
                    # Structured ML fields for rich Telegram formatting
                    "ml_p_up": prob,
                    "ml_p_down": prob_down,
                    "ml_up_threshold": up_threshold,
                    "ml_down_threshold": down_threshold,
                    "ml_down_enabled": down_enabled,
                }

            log.info(
                "MLStrategy: side=%s p_up=%.4f p_down=%.4f up_thr=%.3f down_thr=%.3f "
                "down_enabled=%s slot=%s",
                side, prob, prob_down, up_threshold, down_threshold,
                down_enabled, slot_n1["slug"],
            )

            # Fetch Polymarket prices — identical to PatternStrategy
            prices = await get_slot_prices(slot_n1["slug"])
            if prices is None:
                log.warning(
                    "MLStrategy: no Polymarket prices for slug=%s, skipping", slot_n1["slug"]
                )
                return {
                    **base_fields,
                    "pattern": f"p={prob:.4f}",
                    "reason": "Market data unavailable",
                    # ML inference already completed — include structured fields so
                    # the scheduler can render the rich ML skip card instead of
                    # falling back to the generic format_skip() card.
                    "ml_p_up":           prob,
                    "ml_p_down":         prob_down,
                    "ml_up_threshold":   up_threshold,
                    "ml_down_threshold": down_threshold,
                    "ml_down_enabled":   down_enabled,
                }

            entry_price    = prices["up_price"]    if side == "Up" else prices["down_price"]
            opposite_price = prices["down_price"]  if side == "Up" else prices["up_price"]
            token_id          = prices["up_token_id"] if side == "Up" else prices["down_token_id"]
            opposite_token_id = prices["down_token_id"] if side == "Up" else prices["up_token_id"]

            return {
                **base_fields,
                "skipped":           False,
                "side":              side,
                "entry_price":       entry_price,
                "opposite_price":    opposite_price,
                "token_id":          token_id,
                "opposite_token_id": opposite_token_id,
                "pattern":        f"p_up={prob:.4f},p_down={prob_down:.4f}",
                # Structured ML fields for rich Telegram formatting
                "ml_p_up":          prob,
                "ml_p_down":        prob_down,
                "ml_up_threshold":  up_threshold,
                "ml_down_threshold": down_threshold,
                "ml_down_enabled":  down_enabled,
            }

        except Exception as exc:
            log.exception("MLStrategy.check_signal failed: %s", exc)
            return None
