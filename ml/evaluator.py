"""Model evaluator — full hold-out test evaluation with metrics table."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from config import ML_PAYOUT_RATIO
from ml.probability import apply_probability_calibration, compute_probability_diagnostics

log = logging.getLogger(__name__)


def _probability_quality_outcome(raw_diag: dict, used_diag: dict) -> tuple[str, float, float]:
    delta_brier = float(used_diag.get("brier_score", 0.0)) - float(raw_diag.get("brier_score", 0.0))
    delta_ece = float(used_diag.get("ece", 0.0)) - float(raw_diag.get("ece", 0.0))
    if delta_brier < -1e-6 or delta_ece < -1e-6:
        outcome = "improved"
    elif abs(delta_brier) <= 1e-6 and abs(delta_ece) <= 1e-6:
        outcome = "neutral"
    else:
        outcome = "mixed"
    return outcome, delta_brier, delta_ece


def _bucket_preview(diagnostics: dict | None) -> str:
    if not isinstance(diagnostics, dict):
        return "na"
    buckets = diagnostics.get("buckets") or []
    if not buckets:
        return "none"
    parts = []
    for idx in sorted({0, len(buckets) // 2, len(buckets) - 1}):
        bucket = buckets[idx]
        parts.append(f"b{bucket.get('bucket')}:{bucket.get('prob_mean', 0.0):.3f}->{bucket.get('win_rate', 0.0):.3f}")
    return ",".join(parts)


def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    test_period_days: float = 37,
    payout: float = ML_PAYOUT_RATIO,
    probability_metadata: dict | None = None,
) -> dict:
    """Full evaluation of a trained LightGBM model on a hold-out test set.

    Prints a clear summary table and returns a metrics dict.

    Args:
        model: lgb.Booster instance
        X_test: Feature matrix (n_samples, 22)
        y_test: True binary labels
        threshold: Decision threshold from val-set sweep
        test_period_days: How many days the test set covers (for trades/day)
        payout: Profit per $1 wagered on a winning trade (default:
                ML_PAYOUT_RATIO from config, overridable via ML_PAYOUT_RATIO
                env var). Used to compute payout-adjusted EV/day.

    Returns:
        dict with: wr, precision, recall, f1, trades, trades_per_day,
                   ev_per_trade, ev_per_day, brier_score, calibration_mean,
                   confusion_matrix, threshold, payout
    """
    raw_probs = model.predict(X_test)
    probs, calibration_applied, calibration_method = apply_probability_calibration(raw_probs, probability_metadata)
    raw_diag = compute_probability_diagnostics(raw_probs, y_test, "test_raw")
    used_diag = compute_probability_diagnostics(probs, y_test, "test_used")
    quality_outcome, delta_brier, delta_ece = _probability_quality_outcome(raw_diag, used_diag)
    log.info(
        "evaluate: probability_quality calibration_applied=%s method=%s outcome=%s delta_brier=%+.4f delta_ece=%+.4f raw[brier=%.4f ece=%.4f buckets=%s] used[brier=%.4f ece=%.4f buckets=%s]",
        calibration_applied,
        calibration_method,
        quality_outcome,
        delta_brier,
        delta_ece,
        raw_diag.get("brier_score", 0.0),
        raw_diag.get("ece", 0.0),
        _bucket_preview(raw_diag),
        used_diag.get("brier_score", 0.0),
        used_diag.get("ece", 0.0),
        _bucket_preview(used_diag),
    )
    mask = probs >= threshold
    trades = int(mask.sum())

    if trades == 0:
        result = {
            "wr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "trades": 0,
            "trades_per_day": 0.0,
            "ev_per_trade": 0.0,
            "ev_per_day": 0.0,
            "brier_score": float(np.mean((probs - y_test) ** 2)),
            "calibration_mean": float(np.mean(probs)),
            "confusion_matrix": [[0, 0], [0, 0]],
            "threshold": threshold,
            "payout": payout,
            "raw_brier_score": float(np.mean((raw_probs - y_test) ** 2)),
            "calibration_applied": calibration_applied,
            "calibration_method": calibration_method,
            "probability_diagnostics": {
                "raw": raw_diag,
                "used": used_diag,
            },
        }
        _print_table(result)
        return result

    y_pred = mask.astype(int)
    y_sel = y_test[mask]

    wr = float(y_sel.mean())
    trades_per_day = trades / test_period_days if test_period_days > 0 else 0.0

    # Payout-adjusted EV: accounts for asymmetric win/loss payouts.
    # EV/trade = (WR * payout) - ((1 - WR) * 1.0) = WR * (1 + payout) - 1.0
    ev_per_trade = float(wr * (1.0 + payout) - 1.0)
    ev_per_day = ev_per_trade * trades_per_day

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Brier score (calibration quality)
    brier = float(np.mean((probs - y_test) ** 2))
    calib_mean = float(np.mean(probs[mask]))

    cm = confusion_matrix(y_test, y_pred).tolist()

    result = {
        "wr": wr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trades": trades,
        "trades_per_day": trades_per_day,
        "ev_per_trade": ev_per_trade,
        "ev_per_day": ev_per_day,
        "brier_score": brier,
        "calibration_mean": calib_mean,
        "confusion_matrix": cm,
        "threshold": threshold,
        "payout": payout,
        "raw_brier_score": float(np.mean((raw_probs - y_test) ** 2)),
        "calibration_applied": calibration_applied,
        "calibration_method": calibration_method,
        "probability_diagnostics": {
            "raw": raw_diag,
            "used": used_diag,
        },
    }

    _print_table(result)
    return result


def compute_risk_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    payout: float,
) -> dict:
    """Compute risk/drawdown metrics for a model at a given threshold.

    Simulates a $1 flat-bet equity curve on the ordered sequence of trades
    selected by the model (probs >= threshold), preserving time order.

    On a $1 flat-bet:
        WIN  -> +payout   (e.g. +$0.85)
        LOSS -> -1.00

    Metrics returned
    ----------------
    max_dd_dollar   : float  — worst peak-to-trough drawdown in dollars
    max_dd_pct      : float  — worst drawdown as % of peak equity
    max_loss_streak : int    — longest consecutive losing trades
    max_win_streak  : int    — longest consecutive winning trades
    profit_factor   : float  — gross wins / gross losses (inf if no losses)
    sharpe          : float  — annualised Sharpe ratio (252 trading days,
                               assuming trades_per_day from this sample)
    trades          : int    — number of trades at this threshold

    All values are 0.0 / 0 when there are no trades at the threshold.
    """
    mask = probs >= threshold
    trades = int(mask.sum())

    _zero: dict = {
        "max_dd_dollar": 0.0,
        "max_dd_pct": 0.0,
        "max_loss_streak": 0,
        "max_win_streak": 0,
        "profit_factor": 0.0,
        "sharpe": 0.0,
        "trades": 0,
    }
    if trades == 0:
        return _zero

    outcomes = y_true[mask].astype(int)
    pnl = np.where(outcomes == 1, payout, -1.0)

    equity = np.concatenate([[0.0], np.cumsum(pnl)])
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd_dollar = float(np.min(drawdown))

    with np.errstate(invalid="ignore", divide="ignore"):
        dd_pct = np.where(peak > 0, drawdown / peak * 100.0, 0.0)
    max_dd_pct = float(np.min(dd_pct))

    max_win_streak = 0
    max_loss_streak = 0
    cur_win = 0
    cur_loss = 0
    for o in outcomes:
        if o == 1:
            cur_win += 1
            cur_loss = 0
            if cur_win > max_win_streak:
                max_win_streak = cur_win
        else:
            cur_loss += 1
            cur_win = 0
            if cur_loss > max_loss_streak:
                max_loss_streak = cur_loss

    gross_wins = float(np.sum(pnl[pnl > 0]))
    gross_losses = float(np.abs(np.sum(pnl[pnl < 0])))
    if gross_losses == 0.0:
        profit_factor = float("inf") if gross_wins > 0 else 0.0
    else:
        profit_factor = round(gross_wins / gross_losses, 4)

    sharpe = 0.0
    if trades >= 2:
        mean_r = float(np.mean(pnl))
        std_r = float(np.std(pnl, ddof=1))
        if std_r > 0:
            tpd = trades / max(len(probs) * 5 / 1440, 1e-9)
            sharpe = round((mean_r / std_r) * (252 * tpd) ** 0.5, 4)

    return {
        "max_dd_dollar": round(max_dd_dollar, 4),
        "max_dd_pct": round(max_dd_pct, 4),
        "max_loss_streak": int(max_loss_streak),
        "max_win_streak": int(max_win_streak),
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "trades": trades,
    }


def _print_table(m: dict) -> None:
    """Print a readable evaluation summary."""
    print("\n" + "=" * 52)
    print("  MODEL EVALUATION (HOLD-OUT TEST SET)")
    print("=" * 52)
    print(f"  Threshold          : {m['threshold']:.3f}")
    print(f"  Payout ratio       : {m.get('payout', ML_PAYOUT_RATIO):.2f}")
    print(f"  Win Rate (WR)      : {m['wr']:.4f}  ({m['wr']*100:.2f}%)")
    print(f"  Precision          : {m['precision']:.4f}")
    print(f"  Recall             : {m['recall']:.4f}")
    print(f"  F1                 : {m['f1']:.4f}")
    print(f"  Trades total       : {m['trades']}")
    print(f"  Trades / day       : {m['trades_per_day']:.2f}")
    print(f"  EV / trade ($1)    : {m.get('ev_per_trade', 0.0):+.4f}")
    print(f"  EV / day ($1 flat) : {m.get('ev_per_day', 0.0):+.4f}")
    print(f"  Brier score        : {m['brier_score']:.4f}")
    print(f"  Mean prob (trades) : {m['calibration_mean']:.4f}")
    if m.get("confusion_matrix"):
        cm = m["confusion_matrix"]
        if len(cm) == 2 and len(cm[0]) == 2:
            print("  Confusion matrix   :")
            print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
            print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print("=" * 52 + "\n")


def compute_training_feature_stats(X: "np.ndarray", feature_names: "list[str]") -> dict:
    """Compute mean and std of each feature from the training dataset.

    Returns dict mapping feature_name -> {"mean": float, "std": float}.
    Used by check_feature_drift() to detect covariate shift at inference time.
    """
    stats = {}
    for i, fname in enumerate(feature_names):
        col = X[:, i]
        col = col[~np.isnan(col)]
        if len(col) >= 2:
            stats[fname] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col, ddof=1)),
            }
    return stats


def check_feature_drift(
    inference_log_path: str,
    training_feature_stats: dict,
    n_recent: int = 500,
    z_threshold: float = 2.0,
) -> dict:
    """Check for feature drift between recent inference data and training distribution.

    Loads the last n_recent inference log records, computes the mean of each feature,
    and flags features whose mean deviates more than z_threshold standard deviations
    from the training distribution mean.

    Args:
        inference_log_path: Path to the JSONL inference log file.
        training_feature_stats: Dict mapping feature_name -> {"mean": float, "std": float}
                                 from the training dataset. Stored in model metadata.
        n_recent: Number of recent inference records to analyze (default 500 = ~42 hours).
        z_threshold: Z-score threshold for flagging drift (default 2.0 sigma).

    Returns:
        dict with keys:
            "records_analyzed": int
            "drifted_features": list[dict]  -- each has feature, live_mean, train_mean, train_std, z_score
            "ok": bool -- True if no features drifted beyond threshold
            "checked_at": str -- ISO UTC timestamp
            "error": str | None
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from collections import defaultdict

    checked_at = datetime.now(timezone.utc).isoformat()

    if not inference_log_path or not Path(inference_log_path).exists():
        log.warning("check_feature_drift: log file not found at %s", inference_log_path)
        return {
            "records_analyzed": 0,
            "drifted_features": [],
            "ok": True,
            "checked_at": checked_at,
            "error": f"Log file not found: {inference_log_path}",
        }

    if not training_feature_stats:
        log.warning("check_feature_drift: no training_feature_stats provided -- skipping")
        return {
            "records_analyzed": 0,
            "drifted_features": [],
            "ok": True,
            "checked_at": checked_at,
            "error": "No training feature stats provided",
        }

    try:
        records = []
        with open(inference_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("record_type") == "outcome_patch":
                        continue
                    features = rec.get("features")
                    if features and any(v is not None for v in features.values()):
                        records.append(features)
                except (json.JSONDecodeError, KeyError):
                    continue

        records = records[-n_recent:]
        n = len(records)

        if n < 10:
            log.warning("check_feature_drift: only %d records available (minimum 10 required)", n)
            return {
                "records_analyzed": n,
                "drifted_features": [],
                "ok": True,
                "checked_at": checked_at,
                "error": f"Insufficient records: {n} (minimum 10 required)",
            }

        feature_values: dict = defaultdict(list)
        for feat_dict in records:
            for fname, fval in feat_dict.items():
                if fval is not None and not (isinstance(fval, float) and np.isnan(fval)):
                    feature_values[fname].append(float(fval))

        drifted = []
        for fname, stats in training_feature_stats.items():
            train_mean = stats.get("mean")
            train_std = stats.get("std")
            if train_mean is None or train_std is None or train_std <= 0:
                continue
            live_vals = feature_values.get(fname, [])
            if len(live_vals) < 10:
                continue
            live_mean = float(np.mean(live_vals))
            z = (live_mean - train_mean) / train_std
            if abs(z) >= z_threshold:
                drifted.append({
                    "feature": fname,
                    "live_mean": round(live_mean, 6),
                    "train_mean": round(float(train_mean), 6),
                    "train_std": round(float(train_std), 6),
                    "z_score": round(z, 3),
                })
                log.warning(
                    "check_feature_drift: DRIFT DETECTED feature=%s live_mean=%.4f train_mean=%.4f train_std=%.4f z=%.2f",
                    fname,
                    live_mean,
                    train_mean,
                    train_std,
                    z,
                )

        if not drifted:
            log.info(
                "check_feature_drift: no drift detected (%d records, %d features checked)",
                n,
                len(training_feature_stats),
            )
        else:
            log.warning(
                "check_feature_drift: %d feature(s) drifted beyond %.1f sigma: %s",
                len(drifted),
                z_threshold,
                [d["feature"] for d in drifted],
            )

        return {
            "records_analyzed": n,
            "drifted_features": drifted,
            "ok": len(drifted) == 0,
            "checked_at": checked_at,
            "error": None,
        }

    except Exception as exc:
        log.error("check_feature_drift: unexpected error: %s", exc, exc_info=True)
        return {
            "records_analyzed": 0,
            "drifted_features": [],
            "ok": True,
            "checked_at": checked_at,
            "error": str(exc),
        }
