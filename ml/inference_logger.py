"""Structured per-inference debug logger for live ML strategy.

Each call to check_signal() writes one JSONL record containing:
  - slot metadata (slug, timestamps)
  - raw data quality indicators (row counts and candle timestamps)
  - every feature value with its name
  - NaN breakdown when inference is skipped due to missing features
  - model probability and threshold values
  - fired/skipped decision and side
  - outcome (win/loss) back-filled by log_outcome() after slot resolution

Log format: newline-delimited JSON (JSONL) — one record per line.
Each record is self-contained and can be parsed independently.

Usage:
    from ml import inference_logger

    # On every check_signal() call:
    log_id = inference_logger.log_inference(...)

    # After slot resolves (win/loss known):
    inference_logger.log_outcome(slot_slug, outcome="Up", is_win=True)

Thread safety: all writes are serialised through a threading.Lock so the
file is safe even if multiple threads somehow call the logger concurrently.
The lock is module-level and persists for the lifetime of the process.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import config as cfg

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_write_lock = threading.Lock()

# In-memory index: slot_slug -> file byte offset of the record.
# Used by log_outcome() to back-fill the outcome field without rewriting
# the whole log.  Populated on every log_inference() call.
# This index is lost on restart, so log_outcome() gracefully falls back to
# an append-mode "outcome patch" record if the offset is not found.
_slug_to_offset: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_path() -> str:
    """Return the configured log path, or empty string if logging is disabled."""
    path = getattr(cfg, "INFERENCE_LOG_PATH", "inference_log.jsonl")
    if not path or path.lower() in ("", "none", "disabled"):
        return ""
    return path


def _ensure_dir(path: str) -> None:
    """Create parent directories if they don't exist."""
    parent = Path(path).parent
    if str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def _safe_float(val: Any) -> Any:
    """Convert numpy/float values to JSON-serialisable Python types.

    NaN and Inf are converted to None so the JSON remains valid.
    """
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 8)
    except (TypeError, ValueError):
        return None


def _serialise_features(feature_names: list[str], feature_row: "np.ndarray | None") -> dict[str, Any]:
    """Return a dict mapping feature name -> value (None for NaN/missing)."""
    if feature_row is None:
        return {name: None for name in feature_names}
    flat = feature_row.flatten()
    return {
        name: _safe_float(flat[i]) if i < len(flat) else None
        for i, name in enumerate(feature_names)
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_inference(
    *,
    # --- Slot context ---
    slot_slug: str,
    slot_ts: int,
    slot_start_str: str,
    slot_end_str: str,
    infer_time_utc: str | None = None,
    # --- Data quality ---
    df5_rows: int,
    df15_rows: int,
    df1h_rows: int,
    cvd_rows: int,
    candle_n1_ts: str | None,       # ISO timestamp of the N-1 5m candle
    candle_n1_close: float | None,  # close price of N-1 candle
    candle_n1_vol: float | None,    # volume of N-1 candle
    # --- Feature engineering result ---
    feature_names: list[str],
    feature_row: "np.ndarray | None",   # None if build_live_features returned None
    nan_features: list[str],            # names of features that were NaN (if any)
    # --- Model inference ---
    p_up: float | None,
    p_down: float | None,
    up_threshold: float | None,
    down_threshold: float | None,
    down_enabled: bool,
    # --- Decision ---
    fired: bool,
    side: str | None,       # "Up" / "Down" / None
    skip_reason: str | None,
) -> str:
    """Write one inference record to the JSONL log.

    Returns the slot_slug so callers can use it as an opaque log ID.
    Never raises — all errors are logged as warnings so inference is never blocked.
    """
    path = _log_path()
    if not path:
        return slot_slug

    try:
        now = infer_time_utc or datetime.now(timezone.utc).isoformat()

        record: dict[str, Any] = {
            # Schema version for forward compatibility
            "schema_version": 1,

            # Slot metadata
            "slot_slug":       slot_slug,
            "slot_ts":         slot_ts,
            "slot_start_str":  slot_start_str,
            "slot_end_str":    slot_end_str,
            "infer_time_utc":  now,

            # Data quality — lets you spot silent data shortages
            "data": {
                "df5_rows":        df5_rows,
                "df15_rows":       df15_rows,
                "df1h_rows":       df1h_rows,
                "cvd_rows":        cvd_rows,
                "candle_n1_ts":    candle_n1_ts,
                "candle_n1_close": _safe_float(candle_n1_close),
                "candle_n1_vol":   _safe_float(candle_n1_vol),
            },

            # NaN breakdown — which features were missing and why inference skipped
            "nan_features": nan_features,

            # All feature values (None where NaN/missing)
            "features": _serialise_features(feature_names, feature_row),

            # Model output
            "model": {
                "p_up":           _safe_float(p_up),
                "p_down":         _safe_float(p_down),
                "up_threshold":   _safe_float(up_threshold),
                "down_threshold": _safe_float(down_threshold),
                "down_enabled":   down_enabled,
            },

            # Trade decision
            "decision": {
                "fired":       fired,
                "side":        side,
                "skip_reason": skip_reason,
            },

            # Outcome — back-filled by log_outcome() after resolution
            "outcome": {
                "winner":       None,
                "is_win":       None,
                "resolved_utc": None,
            },
        }

        line = json.dumps(record, separators=(",", ":")) + "\n"
        encoded = line.encode("utf-8")

        _ensure_dir(path)

        with _write_lock:
            with open(path, "ab") as f:
                offset = f.tell()
                f.write(encoded)
            _slug_to_offset[slot_slug] = offset

        # Emit to stdout so the record appears in Railway download logs.
        # Prefix INFER_LOG: makes it easy to grep/filter from other bot output.
        print("INFER_LOG:", line, end="", flush=True)

        log.debug(
            "inference_logger: wrote record for slot=%s fired=%s side=%s p_up=%s nan=%s",
            slot_slug, fired, side,
            f"{p_up:.4f}" if p_up is not None else "N/A",
            nan_features or "none",
        )

    except Exception as exc:
        log.warning("inference_logger.log_inference failed (non-fatal): %s", exc)

    return slot_slug


def log_outcome(
    slot_slug: str,
    winner: str,
    is_win: bool,
) -> None:
    """Back-fill the outcome into the inference log record for slot_slug.

    Strategy:
    1. If the offset for this slug is in the in-memory index (same process
       lifetime), seek to that position and patch the "outcome" block in-place.
       This keeps the record self-contained and easy to parse.
    2. If the offset is unknown (e.g. after a restart), append a compact
       "outcome_patch" record that references the slug.  Analysis tools
       can join on slug to reconstruct the full picture.

    Never raises.
    """
    path = _log_path()
    if not path:
        return

    resolved_utc = datetime.now(timezone.utc).isoformat()

    try:
        with _write_lock:
            offset = _slug_to_offset.get(slot_slug)

            if offset is not None and os.path.exists(path):
                # --- Strategy 1: in-place patch ---
                # Read the original record bytes
                with open(path, "r+b") as f:
                    f.seek(offset)
                    line_bytes = f.readline()

                try:
                    record = json.loads(line_bytes.decode("utf-8"))
                    record["outcome"] = {
                        "winner":       winner,
                        "is_win":       is_win,
                        "resolved_utc": resolved_utc,
                    }
                    new_line = json.dumps(record, separators=(",", ":")) + "\n"
                    new_bytes = new_line.encode("utf-8")

                    if len(new_bytes) == len(line_bytes):
                        # Exact same size — safe to overwrite in-place
                        with open(path, "r+b") as f:
                            f.seek(offset)
                            f.write(new_bytes)
                        # Emit updated record to stdout for Railway logs
                        print("INFER_LOG:", new_line, end="", flush=True)
                        log.debug(
                            "inference_logger: outcome patched in-place for slot=%s is_win=%s",
                            slot_slug, is_win,
                        )
                        return
                    # Size changed (shouldn't happen with fixed schema but be safe)
                    # Fall through to append strategy
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # Corrupted record — fall through to append

            # --- Strategy 2: append outcome patch record ---
            patch: dict[str, Any] = {
                "schema_version": 1,
                "record_type":    "outcome_patch",
                "slot_slug":      slot_slug,
                "outcome": {
                    "winner":       winner,
                    "is_win":       is_win,
                    "resolved_utc": resolved_utc,
                },
            }
            patch_line = json.dumps(patch, separators=(",", ":")) + "\n"
            _ensure_dir(path)
            with open(path, "ab") as f:
                f.write(patch_line.encode("utf-8"))

            # Emit patch record to stdout for Railway logs
            print("INFER_LOG:", patch_line, end="", flush=True)

            log.debug(
                "inference_logger: outcome appended as patch for slot=%s is_win=%s",
                slot_slug, is_win,
            )

    except Exception as exc:
        log.warning("inference_logger.log_outcome failed (non-fatal): %s", exc)


def get_log_path() -> str:
    """Return the current log file path (empty string if disabled)."""
    return _log_path()


def log_skipped_data(
    *,
    slot_slug: str,
    slot_ts: int,
    slot_start_str: str,
    slot_end_str: str,
    skip_reason: str,
    df5_rows: int = 0,
    df15_rows: int = 0,
    df1h_rows: int = 0,
    cvd_rows: int = 0,
    nan_features: list[str] | None = None,
) -> None:
    """Convenience wrapper for slots skipped before model inference
    (e.g. no model loaded, data fetch failed entirely).

    Writes a minimal record so skipped slots are visible in the log and
    can be counted during analysis.
    """
    from ml.features import FEATURE_COLS
    log_inference(
        slot_slug=slot_slug,
        slot_ts=slot_ts,
        slot_start_str=slot_start_str,
        slot_end_str=slot_end_str,
        df5_rows=df5_rows,
        df15_rows=df15_rows,
        df1h_rows=df1h_rows,
        cvd_rows=cvd_rows,
        candle_n1_ts=None,
        candle_n1_close=None,
        candle_n1_vol=None,
        feature_names=FEATURE_COLS,
        feature_row=None,
        nan_features=nan_features or [],
        p_up=None,
        p_down=None,
        up_threshold=None,
        down_threshold=None,
        down_enabled=False,
        fired=False,
        side=None,
        skip_reason=skip_reason,
    )
