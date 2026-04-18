from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_EPS = 1e-6


def _safe_log_loss(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    p = clip_probs(probs)
    y = np.asarray(y_true, dtype=float)
    if len(p) == 0 or len(np.unique(y)) < 2:
        return None
    try:
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    except Exception:
        return None


def _compact_bucket_summary(diagnostics: dict[str, Any] | None) -> str:
    if not isinstance(diagnostics, dict):
        return "na"
    buckets = diagnostics.get("buckets") or []
    if not buckets:
        return "none"
    picked = []
    for idx in sorted({0, len(buckets) // 2, len(buckets) - 1}):
        bucket = buckets[idx]
        picked.append(
            f"b{bucket.get('bucket')}:{bucket.get('prob_mean', 0.0):.3f}->{bucket.get('win_rate', 0.0):.3f}"
        )
    return ",".join(picked)


def _diagnostic_summary(diagnostics: dict[str, Any] | None, logloss_value: float | None = None) -> str:
    if not isinstance(diagnostics, dict):
        return "na"
    return (
        f"brier={diagnostics.get('brier_score', 0.0):.4f} "
        f"ece={diagnostics.get('ece', 0.0):.4f} "
        f"mean={diagnostics.get('prob_mean', 0.0):.4f} "
        f"logloss={(f'{logloss_value:.4f}' if logloss_value is not None else 'na')} "
        f"buckets={_compact_bucket_summary(diagnostics)}"
    )


def clip_probs(probs: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    return np.clip(arr, _EPS, 1.0 - _EPS)


class ProbabilityCalibrator:
    def __init__(self, kind: str, payload: dict[str, Any]):
        self.kind = kind
        self.payload = payload
        self._model = None
        if kind == "isotonic":
            self._model = (
                np.asarray(payload["x_thresholds"], dtype=float),
                np.asarray(payload["y_thresholds"], dtype=float),
            )
        elif kind == "platt":
            pass
        else:
            raise ValueError(f"Unsupported calibration kind: {kind}")

    def apply(self, probs: np.ndarray | list[float]) -> np.ndarray:
        p = clip_probs(probs)
        if self.kind == "isotonic":
            x_thr, y_thr = self._model
            return clip_probs(np.interp(p, x_thr, y_thr, left=y_thr[0], right=y_thr[-1]))
        if self.kind == "platt":
            a = float(self.payload["a"])
            b = float(self.payload["b"])
            logits = np.log(p / (1.0 - p))
            return clip_probs(1.0 / (1.0 + np.exp(-(a * logits + b))))
        return p

    def to_metadata(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.payload}


def _fit_platt(raw_probs: np.ndarray, y_true: np.ndarray) -> ProbabilityCalibrator | None:
    p = clip_probs(raw_probs)
    y = np.asarray(y_true, dtype=float)
    logits = np.log(p / (1.0 - p))
    a = 1.0
    b = 0.0
    lr = 0.05
    for _ in range(400):
        z = a * logits + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        err = pred - y
        grad_a = float(np.mean(err * logits))
        grad_b = float(np.mean(err))
        a -= lr * grad_a
        b -= lr * grad_b
        if abs(grad_a) < 1e-6 and abs(grad_b) < 1e-6:
            break
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    return ProbabilityCalibrator("platt", {"a": float(a), "b": float(b)})


def fit_probability_calibrator(raw_probs: np.ndarray, y_true: np.ndarray) -> dict[str, Any] | None:
    p = clip_probs(raw_probs)
    y = np.asarray(y_true, dtype=int)
    sample_count = len(p)
    if sample_count < 50:
        log.warning(
            "fit_probability_calibrator: skipped sample_count=%d reason=too_few_samples",
            sample_count,
        )
        return None
    if len(np.unique(y)) < 2:
        log.warning(
            "fit_probability_calibrator: skipped sample_count=%d reason=single_class",
            sample_count,
        )
        return None
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives < 10 or negatives < 10:
        log.warning(
            "fit_probability_calibrator: skipped sample_count=%d reason=class_imbalance pos=%d neg=%d",
            sample_count,
            positives,
            negatives,
        )
        return None

    raw_diag = compute_probability_diagnostics(p, y, "calibration_raw")
    raw_logloss = _safe_log_loss(y, p)
    log.info(
        "fit_probability_calibrator: attempting method=auto sample_count=%d pos=%d neg=%d raw=%s",
        sample_count,
        positives,
        negatives,
        _diagnostic_summary(raw_diag, raw_logloss),
    )

    candidates: list[tuple[str, ProbabilityCalibrator, float, dict[str, Any], float | None]] = []
    try:
        order = np.argsort(p)
        p_sorted = p[order]
        y_sorted = y[order].astype(float)
        uniq_x = []
        uniq_y = []
        start = 0
        while start < len(p_sorted):
            end = start + 1
            while end < len(p_sorted) and abs(p_sorted[end] - p_sorted[start]) < 1e-12:
                end += 1
            uniq_x.append(float(np.mean(p_sorted[start:end])))
            uniq_y.append(float(np.mean(y_sorted[start:end])))
            start = end
        uniq_x = np.asarray(uniq_x, dtype=float)
        uniq_y = np.asarray(uniq_y, dtype=float)
        if len(uniq_x) >= 2:
            blocks_x = []
            blocks_y = []
            blocks_n = []
            for x_val, y_val in zip(uniq_x, uniq_y):
                blocks_x.append([x_val])
                blocks_y.append(y_val)
                blocks_n.append(1)
                while len(blocks_y) >= 2 and blocks_y[-2] > blocks_y[-1]:
                    n1 = blocks_n[-2]
                    n2 = blocks_n[-1]
                    merged_n = n1 + n2
                    merged_y = (blocks_y[-2] * n1 + blocks_y[-1] * n2) / merged_n
                    merged_x = blocks_x[-2] + blocks_x[-1]
                    blocks_x[-2] = merged_x
                    blocks_y[-2] = merged_y
                    blocks_n[-2] = merged_n
                    blocks_x.pop()
                    blocks_y.pop()
                    blocks_n.pop()
            x_thresholds = [float(np.mean(xs)) for xs in blocks_x]
            y_thresholds = [float(yv) for yv in blocks_y]
            iso_cal = ProbabilityCalibrator(
                "isotonic",
                {
                    "x_thresholds": x_thresholds,
                    "y_thresholds": y_thresholds,
                },
            )
            iso_pred = iso_cal.apply(p)
            iso_diag = compute_probability_diagnostics(iso_pred, y, "calibration_isotonic")
            iso_logloss = _safe_log_loss(y, iso_pred)
            candidates.append(("isotonic", iso_cal, float(np.mean((iso_pred - y) ** 2)), iso_diag, iso_logloss))
    except Exception as exc:
        log.warning("fit_probability_calibrator: isotonic failed: %s", exc)

    try:
        platt = _fit_platt(p, y)
        if platt is not None:
            platt_pred = platt.apply(p)
            platt_diag = compute_probability_diagnostics(platt_pred, y, "calibration_platt")
            platt_logloss = _safe_log_loss(y, platt_pred)
            candidates.append(("platt", platt, float(np.mean((platt_pred - y) ** 2)), platt_diag, platt_logloss))
    except Exception as exc:
        log.warning("fit_probability_calibrator: platt failed: %s", exc)

    if not candidates:
        log.warning(
            "fit_probability_calibrator: failed sample_count=%d reason=no_candidate_succeeded",
            sample_count,
        )
        return None
    chosen_method, calibrator, brier, chosen_diag, chosen_logloss = min(candidates, key=lambda x: x[2])
    raw_brier = float(raw_diag.get("brier_score", 0.0))
    raw_ece = float(raw_diag.get("ece", 0.0))
    chosen_ece = float(chosen_diag.get("ece", 0.0))
    delta_brier = brier - raw_brier
    log.info(
        "fit_probability_calibrator: selected method=%s sample_count=%d before={%s} after={%s} delta_brier=%+.4f",
        chosen_method,
        sample_count,
        _diagnostic_summary(raw_diag, raw_logloss),
        _diagnostic_summary(chosen_diag, chosen_logloss),
        delta_brier,
    )
    return {
        "method": calibrator.kind,
        "fitted_on": "validation",
        "min_samples": len(p),
        "class_balance": float(np.mean(y)),
        "payload": calibrator.to_metadata(),
        "validation_brier": round(brier, 6),
        "raw_validation_brier": round(raw_brier, 6),
        "raw_validation_ece": round(raw_ece, 6),
        "validation_ece": round(chosen_ece, 6),
        "raw_validation_log_loss": round(raw_logloss, 6) if raw_logloss is not None else None,
        "validation_log_loss": round(chosen_logloss, 6) if chosen_logloss is not None else None,
        "diagnostics_summary": {
            "raw": raw_diag,
            "calibrated": chosen_diag,
        },
    }


def load_probability_calibrator(metadata: dict[str, Any] | None) -> ProbabilityCalibrator | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        log.warning("load_probability_calibrator: raw fallback reason=metadata_malformed type=%s", type(metadata).__name__)
        return None
    calib = metadata.get("probability_calibration")
    if calib is None:
        log.warning("load_probability_calibrator: raw fallback reason=calibration_metadata_missing")
        return None
    if not isinstance(calib, dict):
        log.warning("load_probability_calibrator: raw fallback reason=calibration_metadata_malformed type=%s", type(calib).__name__)
        return None
    payload = calib.get("payload")
    if not isinstance(payload, dict):
        log.warning("load_probability_calibrator: raw fallback reason=payload_missing_or_malformed")
        return None
    kind = payload.get("kind") or calib.get("method")
    if not isinstance(kind, str):
        log.warning("load_probability_calibrator: raw fallback reason=calibration_method_missing")
        return None
    try:
        return ProbabilityCalibrator(kind, payload)
    except Exception as exc:
        log.warning("load_probability_calibrator: invalid metadata ignored, raw fallback reason=%s", exc)
        return None


def apply_probability_calibration(
    probs: np.ndarray | list[float],
    metadata: dict[str, Any] | None,
) -> tuple[np.ndarray, bool, str | None]:
    p = clip_probs(probs)
    calibrator = load_probability_calibrator(metadata)
    if calibrator is None:
        return p, False, None
    try:
        return calibrator.apply(p), True, calibrator.kind
    except Exception as exc:
        log.warning("apply_probability_calibration: failed, falling back to raw probs: %s", exc)
        return p, False, None


def compute_probability_diagnostics(
    probs: np.ndarray | list[float],
    y_true: np.ndarray | list[int],
    label: str,
    bucket_count: int = 10,
) -> dict[str, Any]:
    p = clip_probs(probs)
    y = np.asarray(y_true, dtype=float)
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]

    bucket_edges = np.linspace(0, len(p_sorted), bucket_count + 1, dtype=int)
    buckets = []
    ece = 0.0
    for i in range(bucket_count):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]
        if hi <= lo:
            continue
        bp = p_sorted[lo:hi]
        by = y_sorted[lo:hi]
        conf = float(np.mean(bp))
        win = float(np.mean(by))
        gap = win - conf
        weight = len(bp) / max(len(p_sorted), 1)
        ece += abs(gap) * weight
        buckets.append(
            {
                "bucket": i + 1,
                "count": int(len(bp)),
                "prob_mean": round(conf, 6),
                "win_rate": round(win, 6),
                "gap": round(gap, 6),
            }
        )

    return {
        "label": label,
        "sample_count": int(len(p)),
        "positive_rate": round(float(np.mean(y)), 6) if len(y) else 0.0,
        "prob_mean": round(float(np.mean(p)), 6) if len(p) else 0.0,
        "brier_score": round(float(np.mean((p - y) ** 2)), 6) if len(p) else 0.0,
        "ece": round(float(ece), 6),
        "buckets": buckets,
    }


def derive_sample_weights(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        log.info("derive_sample_weights: disabled sample_count=%d reason=config_disabled", len(y))
        return {"weights": np.ones(len(y), dtype=float), "summary": {"enabled": False, "reason": "config_disabled"}}

    weights = np.ones(len(y), dtype=float)
    borderline_target = float(cfg.get("borderline_target_weight", 0.7))
    vol_tail_weight = float(cfg.get("vol_tail_weight", 0.85))
    funding_tail_weight = float(cfg.get("funding_tail_weight", 0.9))

    def _maybe_apply(feature: str, low_pct: float | None = None, high_pct: float | None = None, tail_weight: float = 1.0):
        if feature not in feature_names:
            return None
        idx = feature_names.index(feature)
        col = np.asarray(X[:, idx], dtype=float)
        finite = col[np.isfinite(col)]
        if len(finite) < 20:
            return None
        low = float(np.percentile(finite, low_pct)) if low_pct is not None else None
        high = float(np.percentile(finite, high_pct)) if high_pct is not None else None
        mask = np.ones(len(col), dtype=bool)
        if low is not None:
            mask &= col >= low
        if high is not None:
            mask &= col <= high
        tail_mask = ~mask & np.isfinite(col)
        weights[tail_mask] *= tail_weight
        return {
            "feature": feature,
            "low": round(low, 6) if low is not None else None,
            "high": round(high, 6) if high is not None else None,
            "affected": int(np.sum(tail_mask)),
            "tail_weight": tail_weight,
        }

    body_info = None
    if "body_ratio" in feature_names:
        idx = feature_names.index("body_ratio")
        body = np.abs(np.asarray(X[:, idx], dtype=float))
        finite = body[np.isfinite(body)]
        if len(finite) >= 20:
            cutoff = float(np.percentile(finite, 35))
            borderline_mask = np.isfinite(body) & (body <= cutoff)
            weights[borderline_mask] *= borderline_target
            body_info = {
                "feature": "body_ratio",
                "abs_cutoff": round(cutoff, 6),
                "affected": int(np.sum(borderline_mask)),
                "weight": borderline_target,
            }

    applied = [x for x in [
        body_info,
        _maybe_apply("vol_regime", 2.5, 97.5, vol_tail_weight),
        _maybe_apply("funding_zscore", 2.5, 97.5, funding_tail_weight),
    ] if x is not None]

    weights = np.clip(weights, 0.25, 1.0)
    affected = int(np.sum(weights < 0.999999))
    summary = {
        "enabled": True,
        "affected_samples": affected,
        "sample_count": int(len(y)),
        "min_weight": round(float(np.min(weights)), 6),
        "max_weight": round(float(np.max(weights)), 6),
        "mean_weight": round(float(np.mean(weights)), 6),
        "rules": applied,
    }
    if applied:
        log.info(
            "derive_sample_weights: enabled sample_count=%d affected=%d min=%.3f max=%.3f mean=%.3f rules=%s",
            len(y),
            affected,
            float(np.min(weights)),
            float(np.max(weights)),
            float(np.mean(weights)),
            [f"{rule['feature']}:{rule['affected']}@{rule.get('weight', rule.get('tail_weight'))}" for rule in applied],
        )
    else:
        log.warning(
            "derive_sample_weights: enabled sample_count=%d affected=0 reason=no_rules_applied",
            len(y),
        )
    return {
        "weights": weights,
        "summary": summary,
    }


def build_live_trust_report(
    feature_row: np.ndarray,
    feature_names: list[str],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    report = {
        "ok": True,
        "reasons": [],
        "details": {},
    }
    if not isinstance(metadata, dict):
        return report

    try:
        stats = metadata.get("training_feature_stats")
        trust_cfg = metadata.get("live_trust_gate") or {}
        report["details"]["config"] = {
            "enabled": bool(trust_cfg.get("enabled", True)),
            "max_validation_ece": float(trust_cfg.get("max_validation_ece", 0.20)),
        }
        if isinstance(stats, dict) and trust_cfg.get("enabled", True):
            monitored = trust_cfg.get("monitored_features") or ["vol_regime", "funding_zscore", "atr_percentile_24h", "vol_zscore"]
            sigma = float(trust_cfg.get("zscore_limit", 3.5))
            breaches = []
            for fname in monitored:
                if fname not in feature_names or fname not in stats:
                    continue
                idx = feature_names.index(fname)
                live_val = float(feature_row[0, idx])
                mean = stats[fname].get("mean")
                std = stats[fname].get("std")
                if mean is None or std in (None, 0):
                    continue
                z = (live_val - float(mean)) / float(std)
                report["details"][fname] = {
                    "live": round(live_val, 6),
                    "mean": round(float(mean), 6),
                    "std": round(float(std), 6),
                    "z": round(float(z), 3),
                }
                if not np.isfinite(z):
                    continue
                if abs(z) > sigma:
                    breaches.append(f"{fname}_z={z:.2f}")
            max_breaches = int(trust_cfg.get("max_feature_breaches", 1))
            report["details"]["feature_drift"] = {
                "monitored": monitored,
                "breach_count": len(breaches),
                "max_feature_breaches": max_breaches,
                "zscore_limit": sigma,
            }
            if len(breaches) > max_breaches:
                report["ok"] = False
                report["reasons"].append("feature_drift:" + ",".join(breaches[:4]))

        calib_diag = metadata.get("probability_diagnostics") or {}
        calib_eval = calib_diag.get("validation_calibrated") if isinstance(calib_diag, dict) else None
        calib_cfg = metadata.get("probability_calibration") or {}
        if isinstance(calib_eval, dict) and isinstance(calib_cfg, dict):
            ece_limit = float((metadata.get("live_trust_gate") or {}).get("max_validation_ece", 0.20))
            validation_ece = float(calib_eval.get("ece", 0.0))
            report["details"]["validation_calibration"] = {
                "method": calib_cfg.get("method"),
                "ece": round(validation_ece, 6),
                "ece_limit": ece_limit,
            }
            if validation_ece > ece_limit:
                report["ok"] = False
                report["reasons"].append(f"validation_ece={validation_ece:.3f}")
        elif metadata.get("live_trust_gate"):
            report["details"]["validation_calibration"] = {
                "method": (calib_cfg or {}).get("method") if isinstance(calib_cfg, dict) else None,
                "ece": None,
                "ece_limit": float((metadata.get("live_trust_gate") or {}).get("max_validation_ece", 0.20)),
            }
    except Exception as exc:
        log.warning("build_live_trust_report: failed open reason=%s", exc)
        return {"ok": True, "reasons": [], "details": {"failed_open": str(exc)}}

    return report
