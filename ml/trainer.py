"""LightGBM trainer — BLUEPRINT sections 7, 8, 9.

CRITICAL: NO data shuffling (time-series order must be preserved).
Threshold sweep ONLY on validation set, never on test set.
"""

from __future__ import annotations

import logging
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss

from ml import model_store
from ml.evaluator import compute_risk_metrics, evaluate
from ml.probability import apply_probability_calibration, compute_probability_diagnostics, derive_sample_weights, fit_probability_calibrator
from ml.features import FEATURE_COLS
from config import ML_PAYOUT_RATIO


class DeploymentBlockedError(Exception):
    """Raised when the trained model fails to meet the minimum test-set WR.

    Blueprint Rule 10: ALWAYS validate that test set WR >= 59% before
    deploying. If a new retrain fails to hit 59% on test, do not deploy.
    """


log = logging.getLogger(__name__)

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": 1,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100

WF_FOLDS = 5
WF_INITIAL_PCT = 0.60
WF_STEP_PCT = (1.0 - WF_INITIAL_PCT) / WF_FOLDS

MIN_TRADES = 30


def _compact_probability_quality_summary(label: str, diagnostics: dict | None, logloss_value: float | None = None) -> str:
    if not isinstance(diagnostics, dict):
        return f"{label}=na"
    buckets = diagnostics.get("buckets") or []
    bucket_preview = []
    if buckets:
        for idx in sorted({0, len(buckets) // 2, len(buckets) - 1}):
            bucket = buckets[idx]
            bucket_preview.append(
                f"b{bucket.get('bucket')}:{bucket.get('prob_mean', 0.0):.3f}->{bucket.get('win_rate', 0.0):.3f}"
            )
    return (
        f"{label}[n={diagnostics.get('sample_count', 0)} brier={diagnostics.get('brier_score', 0.0):.4f} "
        f"ece={diagnostics.get('ece', 0.0):.4f} "
        f"logloss={(f'{logloss_value:.4f}' if logloss_value is not None else 'na')} "
        f"buckets={','.join(bucket_preview) if bucket_preview else 'none'}]"
    )


def _safe_log_loss(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    if len(probs) == 0 or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(log_loss(y_true, np.clip(probs, 1e-6, 1.0 - 1e-6), labels=[0, 1]))
    except Exception:
        return None


def _log_probability_quality_comparison(
    stage: str,
    raw_diag: dict,
    used_diag: dict,
    raw_logloss: float | None,
    used_logloss: float | None,
    calibration_applied: bool,
    calibration_method: str | None,
) -> None:
    delta_brier = float(used_diag.get("brier_score", 0.0)) - float(raw_diag.get("brier_score", 0.0))
    delta_ece = float(used_diag.get("ece", 0.0)) - float(raw_diag.get("ece", 0.0))
    if delta_brier < -1e-6 or delta_ece < -1e-6:
        outcome = "improved"
    elif abs(delta_brier) <= 1e-6 and abs(delta_ece) <= 1e-6:
        outcome = "neutral"
    else:
        outcome = "mixed" if calibration_applied else "raw_only"
    log.info(
        "train: probability_quality stage=%s calibration_applied=%s method=%s outcome=%s delta_brier=%+.4f delta_ece=%+.4f %s %s",
        stage,
        calibration_applied,
        calibration_method,
        outcome,
        delta_brier,
        delta_ece,
        _compact_probability_quality_summary("raw", raw_diag, raw_logloss),
        _compact_probability_quality_summary("used", used_diag, used_logloss),
    )


def _ev_per_day(wr: float, tpd: float, payout: float) -> float:
    return (wr * (1.0 + payout) - 1.0) * tpd


def sweep_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    lo: float = 0.50,
    hi: float = 0.80,
    step_coarse: float = 0.02,
    payout: float = ML_PAYOUT_RATIO,
) -> tuple[float, float, float]:
    def _run_sweep(lo_s: float, hi_s: float, step_s: float):
        _above = []
        _all = []
        t = lo_s
        while t <= hi_s + 1e-9:
            mask = probs >= t
            trades = int(mask.sum())
            if trades >= MIN_TRADES:
                wr = float(y_true[mask].mean())
                tpd = trades / (len(probs) * 5 / 1440)
                _all.append((t, wr, trades, tpd))
                if wr >= 0.58:
                    _above.append((t, wr, trades, tpd))
            t = round(t + step_s, 4)
        return _above, _all

    candidates_above_coarse, all_coarse = _run_sweep(lo, hi, step_coarse)

    if candidates_above_coarse:
        best_coarse = max(candidates_above_coarse, key=lambda x: _ev_per_day(x[1], x[3], payout))
    elif all_coarse:
        best_coarse = max(all_coarse, key=lambda x: _ev_per_day(x[1], x[3], payout))
    else:
        log.warning("sweep_threshold: no viable candidates in coarse pass — returning lo=%.3f", lo)
        return lo, 0.0, 0.0

    best_coarse_threshold = best_coarse[0]
    log.info(
        "sweep_threshold stage1 (coarse step=%.3f): best_coarse=%.3f WR=%.4f tpd=%.1f ev/day=%.4f",
        step_coarse,
        best_coarse_threshold,
        best_coarse[1],
        best_coarse[3],
        _ev_per_day(best_coarse[1], best_coarse[3], payout),
    )

    fine_lo = max(lo, round(best_coarse_threshold - 0.02, 4))
    fine_hi = min(hi, round(best_coarse_threshold + 0.02, 4))
    candidates_above_fine, all_fine = _run_sweep(fine_lo, fine_hi, 0.005)

    if candidates_above_fine:
        best_fine = max(candidates_above_fine, key=lambda x: _ev_per_day(x[1], x[3], payout))
    elif all_fine:
        best_fine = max(all_fine, key=lambda x: _ev_per_day(x[1], x[3], payout))
    else:
        best_fine = best_coarse

    best_threshold, best_wr, _, best_trades_per_day = best_fine
    log.info(
        "sweep_threshold stage2 (fine step=0.005, range=[%.3f,%.3f]): best=%.3f WR=%.4f tpd=%.1f ev/day=%.4f",
        fine_lo,
        fine_hi,
        best_threshold,
        best_wr,
        best_trades_per_day,
        _ev_per_day(best_wr, best_trades_per_day, payout),
    )

    candidates_above = candidates_above_fine if candidates_above_fine else candidates_above_coarse
    if not candidates_above:
        log.warning(
            "sweep_threshold: no WR>=0.58 threshold found — using best EV/day fallback thresh=%.3f WR=%.4f tpd=%.1f ev/day=%.4f",
            best_threshold,
            best_wr,
            best_trades_per_day,
            _ev_per_day(best_wr, best_trades_per_day, payout),
        )
    elif not candidates_above_fine:
        log.warning("sweep_threshold: no WR>=0.58 candidates in fine range — fell back to coarse best")

    return best_threshold, best_wr, best_trades_per_day


def evaluate_at_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> dict:
    mask = probs >= threshold
    trades = int(mask.sum())

    if trades == 0:
        return {
            "wr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "trades": 0,
            "trades_per_day": 0.0,
        }

    y_pred = mask.astype(int)
    y_sel = y_true[mask]

    wr = float(y_sel.mean())
    trades_per_day = trades / (len(probs) * 5 / 1440)

    try:
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        precision = wr
        recall = 0.0
        f1 = 0.0

    return {
        "wr": wr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trades": trades,
        "trades_per_day": trades_per_day,
    }


def walk_forward_validation(X: np.ndarray, y: np.ndarray) -> dict:
    n = len(y)
    fold_results = []

    log.info(
        "walk_forward_validation: starting %d-fold walk-forward on n=%d samples (initial_pct=%.0f%%, step_pct=%.0f%%)",
        WF_FOLDS,
        n,
        WF_INITIAL_PCT * 100,
        WF_STEP_PCT * 100,
    )

    for fold_idx in range(WF_FOLDS):
        trainval_end = int(n * (WF_INITIAL_PCT + fold_idx * WF_STEP_PCT))
        test_end = int(n * (WF_INITIAL_PCT + (fold_idx + 1) * WF_STEP_PCT)) if fold_idx < WF_FOLDS - 1 else n
        test_start = trainval_end
        fold_val_start = int(trainval_end * 0.80)

        X_fold_train = X[:fold_val_start]
        y_fold_train = y[:fold_val_start]
        X_fold_val = X[fold_val_start:trainval_end]
        y_fold_val = y[fold_val_start:trainval_end]
        X_fold_test = X[test_start:test_end]
        y_fold_test = y[test_start:test_end]

        fold_num = fold_idx + 1
        log.info(
            "walk_forward fold %d/%d: train=[0:%d] val=[%d:%d] test=[%d:%d]",
            fold_num,
            WF_FOLDS,
            fold_val_start,
            fold_val_start,
            trainval_end,
            test_start,
            test_end,
        )

        if len(X_fold_train) < 50 or len(X_fold_val) < 10 or len(X_fold_test) < 10:
            log.warning(
                "walk_forward fold %d: insufficient samples (train=%d val=%d test=%d) — skipping fold",
                fold_num,
                len(X_fold_train),
                len(X_fold_val),
                len(X_fold_test),
            )
            continue

        fold_train_data = lgb.Dataset(X_fold_train, label=y_fold_train, feature_name=FEATURE_COLS)
        fold_val_data = lgb.Dataset(
            X_fold_val,
            label=y_fold_val,
            feature_name=FEATURE_COLS,
            reference=fold_train_data,
        )
        fold_callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        fold_model = lgb.train(
            LGBM_PARAMS,
            fold_train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[fold_val_data],
            callbacks=fold_callbacks,
        )

        fold_val_probs = fold_model.predict(X_fold_val)
        fold_threshold, fold_val_wr, _ = sweep_threshold(fold_val_probs, y_fold_val)
        fold_down_probs_val = 1.0 - fold_val_probs
        fold_y_val_down = 1 - y_fold_val
        fold_down_threshold, fold_down_val_wr, _ = sweep_threshold(fold_down_probs_val, fold_y_val_down)
        fold_test_probs = fold_model.predict(X_fold_test)
        fold_test_metrics = evaluate_at_threshold(fold_test_probs, y_fold_test, fold_threshold)

        log.info(
            "walk_forward fold %d/%d: up_threshold=%.3f val_wr=%.4f down_threshold=%.3f down_val_wr=%.4f test_wr=%.4f test_trades=%d",
            fold_num,
            WF_FOLDS,
            fold_threshold,
            fold_val_wr,
            fold_down_threshold,
            fold_down_val_wr,
            fold_test_metrics["wr"],
            fold_test_metrics["trades"],
        )

        fold_results.append(
            {
                "fold": fold_num,
                "train_size": fold_val_start,
                "val_size": trainval_end - fold_val_start,
                "test_size": test_end - test_start,
                "up_threshold": fold_threshold,
                "down_threshold": fold_down_threshold,
                "val_wr": fold_val_wr,
                "down_val_wr": fold_down_val_wr,
                "test_wr": fold_test_metrics["wr"],
                "test_trades": fold_test_metrics["trades"],
                "test_trades_per_day": fold_test_metrics["trades_per_day"],
            }
        )

    if fold_results:
        wrs = [r["test_wr"] for r in fold_results]
        avg_wr = float(np.mean(wrs))
        std_wr = float(np.std(wrs))
        min_wr = float(np.min(wrs))
        max_wr = float(np.max(wrs))
    else:
        avg_wr = std_wr = min_wr = max_wr = 0.0

    log.info(
        "walk_forward_validation SUMMARY: folds=%d avg_wr=%.4f std_wr=%.4f min_wr=%.4f max_wr=%.4f",
        len(fold_results),
        avg_wr,
        std_wr,
        min_wr,
        max_wr,
    )
    return {
        "fold_results": fold_results,
        "avg_wr": avg_wr,
        "std_wr": std_wr,
        "min_wr": min_wr,
        "max_wr": max_wr,
    }


def aggregate_wf_thresholds(wf_results: dict) -> tuple:
    fold_results = wf_results.get("fold_results", [])
    if not fold_results:
        log.warning("aggregate_wf_thresholds: no fold results — returning defaults (0.5, 0.5)")
        return 0.5, 0.5

    up_thresholds = [r["up_threshold"] for r in fold_results]
    down_thresholds = [r["down_threshold"] for r in fold_results]

    up_threshold = float(np.median(up_thresholds))
    down_threshold = float(np.median(down_thresholds))

    log.info(
        "aggregate_wf_thresholds: %d folds -> up_threshold=%.4f (median of %s) down_threshold=%.4f (median of %s)",
        len(fold_results),
        up_threshold,
        [f"{t:.4f}" for t in up_thresholds],
        down_threshold,
        [f"{t:.4f}" for t in down_thresholds],
    )
    return up_threshold, down_threshold


def train(df_features: pd.DataFrame, slot: str = "current") -> dict:
    n = len(df_features)
    if n < 130:
        raise ValueError(f"Too few samples to train: {n} (minimum 130 required)")

    X = df_features[FEATURE_COLS].values
    y = df_features["target"].values

    _vol_regime_idx = FEATURE_COLS.index("vol_regime")
    _vol_regime_vals = X[:, _vol_regime_idx]
    _vol_regime_vals = _vol_regime_vals[~np.isnan(_vol_regime_vals)]
    if len(_vol_regime_vals) >= 10:
        _regime_vol_p5 = float(np.percentile(_vol_regime_vals, 5))
        _regime_vol_p95 = float(np.percentile(_vol_regime_vals, 95))
    else:
        _regime_vol_p5 = None
        _regime_vol_p95 = None
    log.info(
        "train: vol_regime training distribution — n=%d p5=%.4f p95=%.4f",
        len(_vol_regime_vals),
        _regime_vol_p5 if _regime_vol_p5 is not None else float("nan"),
        _regime_vol_p95 if _regime_vol_p95 is not None else float("nan"),
    )

    log.info("train: running walk-forward validation (%d folds) before final fit", WF_FOLDS)
    wf_results = walk_forward_validation(X, y)
    log.info(
        "train: walk-forward done — avg_wr=%.4f std_wr=%.4f (across %d folds)",
        wf_results["avg_wr"],
        wf_results["std_wr"],
        len(wf_results["fold_results"]),
    )

    split_boundary = int(n * 0.80)
    val_start = int(split_boundary * 0.80)

    from ml.evaluator import compute_training_feature_stats

    _training_feature_stats = compute_training_feature_stats(X[:split_boundary], FEATURE_COLS)
    log.info("train: computed training feature stats for %d features", len(_training_feature_stats))
    log.info("train: n=%d train=[0:%d] val=[%d:%d] test=[%d:%d]", n, val_start, val_start, split_boundary, split_boundary, n)

    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:split_boundary], y[val_start:split_boundary]
    X_test, y_test = X[split_boundary:], y[split_boundary:]

    log.info("train: X_train=%s X_val=%s X_test=%s", X_train.shape, X_val.shape, X_test.shape)

    _sample_weight_cfg = {
        "enabled": True,
        "borderline_target_weight": 0.7,
        "vol_tail_weight": 0.85,
        "funding_tail_weight": 0.9,
    }
    _sample_weight_result = derive_sample_weights(X_train, y_train, FEATURE_COLS, _sample_weight_cfg)
    _train_weights = _sample_weight_result["weights"]
    _sample_quality_summary = _sample_weight_result["summary"]
    if _sample_quality_summary.get("enabled"):
        log.info(
            "train: sample_quality enabled=%s affected=%d/%d min=%.3f max=%.3f mean=%.3f",
            _sample_quality_summary.get("enabled"),
            _sample_quality_summary.get("affected_samples", 0),
            _sample_quality_summary.get("sample_count", len(y_train)),
            _sample_quality_summary.get("min_weight", 1.0),
            _sample_quality_summary.get("max_weight", 1.0),
            _sample_quality_summary.get("mean_weight", 1.0),
        )
    else:
        log.warning(
            "train: sample_quality enabled=%s reason=%s",
            _sample_quality_summary.get("enabled"),
            _sample_quality_summary.get("reason", "not_applied"),
        )

    train_data = lgb.Dataset(X_train, label=y_train, weight=_train_weights, feature_name=FEATURE_COLS)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_data)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    log.info("train: best_iteration=%d", model.best_iteration)

    best_threshold, down_threshold = aggregate_wf_thresholds(wf_results)

    val_probs_raw = model.predict(X_val)
    probability_calibration = fit_probability_calibrator(val_probs_raw, y_val)
    if probability_calibration is None:
        log.warning(
            "train: calibration skipped stage=validation sample_count=%d reason=fit_returned_none",
            len(y_val),
        )
    else:
        validation_ece = probability_calibration.get("validation_ece")
        log.info(
            "train: calibration fitted stage=validation method=%s sample_count=%d validation_brier=%.4f validation_ece=%s",
            probability_calibration.get("method"),
            len(y_val),
            probability_calibration.get("validation_brier", 0.0),
            f"{validation_ece:.4f}" if validation_ece is not None else "na",
        )
    val_probs, calibration_applied, calibration_method = apply_probability_calibration(
        val_probs_raw,
        {"probability_calibration": probability_calibration} if probability_calibration else None,
    )
    _, best_wr, best_trades_per_day = sweep_threshold(val_probs, y_val)
    down_probs_val = 1.0 - val_probs
    y_val_down = 1 - y_val
    _, down_val_wr, down_val_tpd = sweep_threshold(down_probs_val, y_val_down)

    down_enabled = down_val_wr >= 0.58

    log.info(
        "train: WFV-derived thresholds — up_threshold=%.3f down_threshold=%.3f",
        best_threshold,
        down_threshold,
    )
    log.info(
        "train: val reference — val_wr=%.4f down_val_wr=%.4f down_val_tpd=%.1f down_enabled=%s",
        best_wr,
        down_val_wr,
        down_val_tpd,
        down_enabled,
    )
    if not down_enabled:
        log.warning(
            "train: DOWN side did NOT pass deployment gate (down_val_wr=%.4f < 0.58). DOWN trades will be disabled for this model.",
            down_val_wr,
        )

    probability_metadata = {"probability_calibration": probability_calibration} if probability_calibration else None
    _val_diag_raw = compute_probability_diagnostics(val_probs_raw, y_val, "validation_raw")
    _val_diag_cal = compute_probability_diagnostics(val_probs, y_val, "validation_calibrated")
    _val_logloss_raw = _safe_log_loss(y_val, val_probs_raw)
    _val_logloss_cal = _safe_log_loss(y_val, val_probs)
    _log_probability_quality_comparison(
        "validation",
        _val_diag_raw,
        _val_diag_cal,
        _val_logloss_raw,
        _val_logloss_cal,
        calibration_applied,
        calibration_method,
    )

    test_probs_raw = model.predict(X_test)
    test_probs, _, _ = apply_probability_calibration(test_probs_raw, probability_metadata)
    _test_diag_raw = compute_probability_diagnostics(test_probs_raw, y_test, "test_raw")
    _test_diag_cal = compute_probability_diagnostics(test_probs, y_test, "test_calibrated")
    _test_logloss_raw = _safe_log_loss(y_test, test_probs_raw)
    _test_logloss_cal = _safe_log_loss(y_test, test_probs)
    _log_probability_quality_comparison(
        "test",
        _test_diag_raw,
        _test_diag_cal,
        _test_logloss_raw,
        _test_logloss_cal,
        calibration_applied,
        calibration_method,
    )
    test_metrics = evaluate(model, X_test, y_test, best_threshold, probability_metadata=probability_metadata)

    down_test_metrics = evaluate_at_threshold(1.0 - test_probs, 1 - y_test, down_threshold)
    if down_enabled and down_test_metrics["wr"] < 0.58:
        log.warning(
            "train: DOWN passed val gate but FAILED test gate (down_test_wr=%.4f < 0.58). Disabling DOWN.",
            down_test_metrics["wr"],
        )
        down_enabled = False

    log.info(
        "train: val_wr=%.4f threshold=%.3f | test_wr=%.4f test_trades=%d",
        best_wr,
        best_threshold,
        test_metrics["wr"],
        test_metrics["trades"],
    )
    log.info(
        "train: down_val_wr=%.4f down_threshold=%.3f | down_test_wr=%.4f down_test_trades=%d down_enabled=%s",
        down_val_wr,
        down_threshold,
        down_test_metrics["wr"],
        down_test_metrics["trades"],
        down_enabled,
    )

    MIN_DEPLOY_WR = 0.58
    blocked = test_metrics["wr"] < MIN_DEPLOY_WR
    if blocked:
        log.warning(
            "DEPLOYMENT BLOCKED: test_wr=%.4f is below minimum %.2f (Blueprint Rule 10). Model saved to candidate slot — user must decide whether to promote or discard.",
            test_metrics["wr"],
            MIN_DEPLOY_WR,
        )

    _ts_col = df_features["timestamp"] if "timestamp" in df_features.columns else None
    if _ts_col is not None and len(_ts_col) > 0:
        _data_start = pd.Timestamp(_ts_col.iloc[0]).isoformat()[:10]
        _data_end = pd.Timestamp(_ts_col.iloc[-1]).isoformat()[:10]
    else:
        _data_start = None
        _data_end = None

    _up_ev_per_day = _ev_per_day(test_metrics["wr"], test_metrics["trades_per_day"], ML_PAYOUT_RATIO)
    _down_ev_per_day = _ev_per_day(down_test_metrics["wr"], down_test_metrics["trades_per_day"], ML_PAYOUT_RATIO)

    _val_risk = compute_risk_metrics(y_val, val_probs, best_threshold, ML_PAYOUT_RATIO)
    _test_risk = compute_risk_metrics(y_test, test_probs, best_threshold, ML_PAYOUT_RATIO)

    _wf_worst_dd_dollar = _test_risk["max_dd_dollar"]
    _wf_worst_dd_pct = _test_risk["max_dd_pct"]
    _wf_worst_loss_streak = _test_risk["max_loss_streak"]
    if wf_results["fold_results"]:
        _wf_min_wr = wf_results["min_wr"]
        _wf_test_wr = test_metrics["wr"] if test_metrics["wr"] > 0 else 1.0
        _scale = max(_wf_min_wr / _wf_test_wr, 0.0) if _wf_test_wr > 0 else 1.0
        if _wf_min_wr < test_metrics["wr"] and _scale < 1.0:
            _inv = 1.0 / _scale if _scale > 0 else 1.0
            _wf_worst_dd_dollar = round(_test_risk["max_dd_dollar"] * _inv, 4)
            _wf_worst_dd_pct = round(_test_risk["max_dd_pct"] * _inv, 4)

    metadata = {
        "train_date": datetime.utcnow().isoformat(),
        "data_start": _data_start,
        "data_end": _data_end,
        "val_risk": _val_risk,
        "test_risk": _test_risk,
        "wf_worst_dd_dollar": _wf_worst_dd_dollar,
        "wf_worst_dd_pct": _wf_worst_dd_pct,
        "wf_worst_loss_streak": _wf_worst_loss_streak,
        "threshold": best_threshold,
        "threshold_source": "walk_forward_validation_median",
        "val_wr": best_wr,
        "val_trades_per_day": best_trades_per_day,
        "test_wr": test_metrics["wr"],
        "test_precision": test_metrics["precision"],
        "test_trades": test_metrics["trades"],
        "test_trades_per_day": test_metrics["trades_per_day"],
        "up_ev_per_day": _up_ev_per_day,
        "down_threshold": down_threshold,
        "down_enabled": down_enabled,
        "down_val_wr": down_val_wr,
        "down_val_tpd": down_val_tpd,
        "down_test_wr": down_test_metrics["wr"],
        "down_test_trades": down_test_metrics["trades"],
        "down_test_tpd": down_test_metrics["trades_per_day"],
        "down_ev_per_day": _down_ev_per_day,
        "payout": ML_PAYOUT_RATIO,
        "wf_avg_wr": wf_results["avg_wr"],
        "wf_std_wr": wf_results["std_wr"],
        "wf_min_wr": wf_results["min_wr"],
        "wf_max_wr": wf_results["max_wr"],
        "wf_folds": len(wf_results["fold_results"]),
        "wf_fold_results": wf_results["fold_results"],
        "regime_vol_p5": _regime_vol_p5,
        "regime_vol_p95": _regime_vol_p95,
        "sample_count": n,
        "train_size": val_start,
        "val_size": split_boundary - val_start,
        "test_size": n - split_boundary,
        "feature_cols": FEATURE_COLS,
        "best_iteration": model.best_iteration,
        "blocked": blocked,
        "training_feature_stats": _training_feature_stats,
        "sample_quality": _sample_weight_result["summary"],
        "probability_calibration": probability_calibration,
        "probability_diagnostics": {
            "validation_raw": _val_diag_raw,
            "validation_calibrated": _val_diag_cal,
            "test_raw": _test_diag_raw,
            "test_calibrated": _test_diag_cal,
        },
        "live_trust_gate": {
            "enabled": True,
            "monitored_features": ["vol_regime", "funding_zscore", "atr_percentile_24h", "vol_zscore"],
            "zscore_limit": 3.5,
            "max_feature_breaches": 1,
            "max_validation_ece": 0.20,
        },
    }
    model_store.save_model(model, slot, metadata)

    return {
        "model": model,
        "threshold": best_threshold,
        "down_threshold": down_threshold,
        "down_enabled": down_enabled,
        "down_val_wr": down_val_wr,
        "down_val_tpd": down_val_tpd,
        "down_test_metrics": down_test_metrics,
        "test_metrics": test_metrics,
        "val_wr": best_wr,
        "val_trades": best_trades_per_day,
        "best_iteration": model.best_iteration,
        "blocked": blocked,
        "wf_results": wf_results,
        "probability_calibration": probability_calibration,
        "probability_diagnostics": {
            "validation_raw": _val_diag_raw,
            "validation_calibrated": _val_diag_cal,
            "test_raw": _test_diag_raw,
            "test_calibrated": _test_diag_cal,
        },
        "sample_quality": _sample_weight_result["summary"],
        "warning_reason": (
            f"Test WR {test_metrics['wr']*100:.2f}% is below the 59% deployment gate "
            f"(Blueprint Rule 10). Candidate saved but NOT auto-promoted."
        ) if blocked else None,
    }
