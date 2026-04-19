"""Message formatters — every output shows UTC timeslots with emojis."""

from __future__ import annotations

import html as _html
from typing import Any


def _e(value: object) -> str:
    """Escape a value for safe inclusion in Telegram HTML messages."""
    return _html.escape(str(value))


# ---------------------------------------------------------------------------
# Risk card helper — shared by both retrain formatters.
#
# Design: key-value card layout using HTML bold labels and plain values.
# No <code> block, no monospace padding, no fixed-width columns.
# This is fully immune to Telegram font scaling on all clients (desktop,
# iOS, Android).  Val and Test appear inline on each row, clearly labeled.
# ---------------------------------------------------------------------------

# Thin divider line used between sections (20 em-dashes, renders cleanly
# in Telegram HTML without any monospace dependency).
_RISK_DIV = "\u2015" * 18


def _fmt_dd_dollar(v: float) -> str:
    """Format a drawdown dollar value.  e.g. -11.95 -> '-$11.95'"""
    if v < 0:
        return f"-${abs(v):.2f}"
    return "$0.00"


def _fmt_dd_pct(v: float) -> str:
    """Format a drawdown percentage.  e.g. -1.0 -> '-1.0%'"""
    if v < 0:
        return f"{v:.1f}%"
    return "0.0%"


def _fmt_streak(v: int) -> str:
    return str(int(v))


def _fmt_pf(v: float) -> str:
    return "\u221e" if v == float("inf") else f"{v:.2f}"


def _fmt_sharpe(v: float) -> str:
    return f"{v:.2f}"


def _risk_row(label: str, val_str: str, test_str: str | None = None) -> str:
    """Return one HTML line for a risk metric.

    If test_str is provided:   📌 <b>Label</b>  Val: X  │  Test: Y
    If test_str is None/empty: 📌 <b>Label</b>  X
    """
    if test_str:
        return f"\U0001f4cc <b>{label}</b>  Val: {val_str}  \u2502  Test: {test_str}\n"
    return f"\U0001f4cc <b>{label}</b>  {val_str}\n"


def _build_risk_table(meta: dict) -> str | None:
    """Return a key-value card risk message for a separate Telegram send.

    Returns None when neither val_risk nor test_risk is present in *meta*.
    The returned string is standalone HTML — send with parse_mode='HTML'.

    Layout (Option 2 — key-value card):
    ──────────────────
    ⚠️ Risk Metrics
    ──────────────────
    📉 Max Drawdown
       Val  →  -$18.30  (-195.5%)
       Test →  -$13.95  (-265.8%)
    📌 Loss streak   Val: 6  │  Test: 7
    📌 Win streak    Val: 12 │  Test: 18
    📌 Profit factor Val: 1.32 │ Test: 1.26
    📌 Sharpe        Val: 22.92 │ Test: 18.69
    ──────────────────
    Walk-Forward (worst)
    📌 DD $      -$14.69
    📌 DD %      -280.0%
    📌 Loss streak   7
    """
    val_risk  = meta.get("val_risk", {})
    test_risk = meta.get("test_risk", {})
    if not val_risk and not test_risk:
        return None

    # ── Drawdown dollar ──────────────────────────────────────────────────
    v_dd_d = _fmt_dd_dollar(val_risk.get("max_dd_dollar", 0.0))
    t_dd_d = _fmt_dd_dollar(test_risk.get("max_dd_dollar", 0.0))
    v_dd_p = _fmt_dd_pct(val_risk.get("max_dd_pct", 0.0))
    t_dd_p = _fmt_dd_pct(test_risk.get("max_dd_pct", 0.0))

    # ── Other per-split metrics ───────────────────────────────────────────
    v_ls = _fmt_streak(val_risk.get("max_loss_streak", 0))
    t_ls = _fmt_streak(test_risk.get("max_loss_streak", 0))
    v_ws = _fmt_streak(val_risk.get("max_win_streak", 0))
    t_ws = _fmt_streak(test_risk.get("max_win_streak", 0))
    v_pf = _fmt_pf(val_risk.get("profit_factor", 0.0))
    t_pf = _fmt_pf(test_risk.get("profit_factor", 0.0))
    v_sh = _fmt_sharpe(val_risk.get("sharpe", 0.0))
    t_sh = _fmt_sharpe(test_risk.get("sharpe", 0.0))

    # ── Walk-forward worst-case (top-level keys in meta) ──────────────────
    wf_dd_d  = meta.get("wf_worst_dd_dollar", 0.0)
    wf_dd_p  = meta.get("wf_worst_dd_pct", 0.0)
    wf_ls    = meta.get("wf_worst_loss_streak", 0)
    has_wf   = any([wf_dd_d, wf_dd_p, wf_ls])

    # ── Build message ─────────────────────────────────────────────────────
    lines: list[str] = [
        f"\u26a0\ufe0f <b>Risk Metrics</b>\n{_RISK_DIV}\n",
        # Drawdown — given its two sub-values ($ and %), use a dedicated block
        (
            f"\U0001f4c9 <b>Max Drawdown</b>\n"
            f"   Val  \u2192  {v_dd_d}  ({v_dd_p})\n"
            f"   Test \u2192  {t_dd_d}  ({t_dd_p})\n"
        ),
        _risk_row("Loss streak",   v_ls, t_ls),
        _risk_row("Win streak",    v_ws, t_ws),
        _risk_row("Profit factor", v_pf, t_pf),
        _risk_row("Sharpe",        v_sh, t_sh),
    ]

    if has_wf:
        lines += [
            f"{_RISK_DIV}\n\U0001f6e4 <b>Walk-Forward (worst)</b>\n",
            _risk_row("DD $",        _fmt_dd_dollar(wf_dd_d)),
            _risk_row("DD %",        _fmt_dd_pct(wf_dd_p)),
            _risk_row("Loss streak", _fmt_streak(wf_ls)),
        ]

    return "".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Consistent separator
# ---------------------------------------------------------------------------

SEP = "\u2501" * 20


# ---------------------------------------------------------------------------
# Live notifications (sent by scheduler)
# ---------------------------------------------------------------------------

def format_signal(
    side: str,
    entry_price: float,
    slot_start_str: str,
    slot_end_str: str,
    pattern: str | None = None,
) -> str:
    """Pure signal notification — market info only, no trade layer details."""
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"

    pattern_line = ""
    if pattern:
        pattern_line = f"\u2502 \U0001f522 Pattern: {_e(pattern)}\n"

    return (
        "\U0001f4e1 <b>Signal Fired!</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC\n"
        f"\u2502 {side_emoji} Side: {side}\n"
        f"\u2502 \U0001f4b2 Ask Price: ${entry_price:.2f}\n"
        f"{pattern_line}"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_skip(
    slot_start_str: str,
    slot_end_str: str,
    reason: str = "No pattern match",
    pattern: str | None = None,
) -> str:
    """Skip notification when the strategy does not generate a trade signal."""
    pattern_line = f" | Pattern: {_e(pattern)}" if pattern else ""
    return (
        "\u23ed\ufe0f <b>No Signal</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC\n"
        f"\u2502 {_e(reason)}{pattern_line}\n"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_ml_signal(
    side: str,
    entry_price: float,
    slot_start_str: str,
    slot_end_str: str,
    ml_p_up: float,
    ml_p_down: float,
    ml_up_threshold: float,
    ml_down_threshold: float,
    ml_down_enabled: bool = False,
) -> str:
    """ML signal notification — Option A card with confidence and edge."""
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"

    # Determine winning probability and threshold based on direction
    if side == "Up":
        win_prob = ml_p_up
        win_thr  = ml_up_threshold
        win_arrow = "\u2191"
        win_label = "UP  "
        los_arrow = "\u2193"
        los_label = "DOWN"
        los_prob  = ml_p_down
    else:
        win_prob = ml_p_down
        win_thr  = ml_down_threshold
        win_arrow = "\u2193"
        win_label = "DOWN"
        los_arrow = "\u2191"
        los_label = "UP  "
        los_prob  = ml_p_up

    edge = round((win_prob - win_thr) * 100, 1)
    edge_str = f"{edge:+.1f}%"

    # For the threshold row: if firing DOWN but DOWN is disabled, show "disabled"
    # (shouldn't normally fire if disabled, but guard defensively)
    if side == "Down" and not ml_down_enabled:
        thr_line = "\u2502  Threshold: \u2265 disabled\n"
    else:
        thr_line = f"\u2502  Threshold: \u2265 {win_thr*100:.1f}%\n"

    return (
        "\U0001f4e1 <b>Signal Fired!</b>  \U0001f916 ML\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u23f0 Slot:  {slot_start_str} \u2013 {slot_end_str} UTC\n"
        f"\u2502 {side_emoji} Side:  {side}\n"
        f"\u2502 \U0001f4b2 Price: ${entry_price:.2f}\n"
        "\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "\u2502 \U0001f9e0 ML Confidence\n"
        f"\u2502  {win_arrow} {win_label}   {win_prob*100:.1f}%  \u2705  edge {edge_str}\n"
        f"\u2502  {los_arrow} {los_label}   {los_prob*100:.1f}%\n"
        f"{thr_line}"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_ml_skip(
    slot_start_str: str,
    slot_end_str: str,
    ml_p_up: float,
    ml_p_down: float,
    ml_up_threshold: float,
    ml_down_threshold: float,
    ml_down_enabled: bool,
) -> str:
    """ML no-signal notification — Option C card with shortfall or disabled status."""
    # Shortfall is threshold - prob; positive means below threshold (normal skip),
    # negative means above threshold (that side passed but the other caused the skip).
    # Use :+.1f so the sign is always explicit — avoids "short −-1.2%" double-sign.
    up_short   = round((ml_up_threshold - ml_p_up) * 100, 1)
    down_short = round((ml_down_threshold - ml_p_down) * 100, 1)

    up_note   = f"short {up_short:+.1f}%"

    if ml_down_enabled:
        down_note = f"short {down_short:+.1f}%"
    else:
        down_note = "disabled"

    return (
        "\u23ed\ufe0f <b>No Signal</b>  \U0001f916 ML\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u23f0 Slot:  {slot_start_str} \u2013 {slot_end_str} UTC\n"
        "\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "\u2502 \U0001f9e0 ML Output\n"
        f"\u2502  \u2715 UP    {ml_p_up*100:.1f}%   {up_note}\n"
        f"\u2502  \u2715 DOWN  {ml_p_down*100:.1f}%   {down_note}\n"
        f"\u2502  UP thr \u2265 {ml_up_threshold*100:.1f}%  \u2502  DOWN thr \u2265 {ml_down_threshold*100:.1f}%\n"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_signal_resolution(
    is_win: bool,
    side: str,
    entry_price: float,
    slot_start_str: str,
    slot_end_str: str,
) -> str:
    """Signal outcome — always sent. No P&L (trade-layer concern)."""
    result_price = 1.00 if is_win else 0.00
    icon = "\u2705" if is_win else "\u274c"
    label = "WIN" if is_win else "LOSS"
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    return "\n".join([
        f"{icon} <b>Signal Result \u2014 {label}</b>",
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC",
        f"\u2502 {side_emoji} Side: {side}",
        f"\u2502 \U0001f4b2 Entry: ${entry_price:.2f} \u2192 Result: ${result_price:.2f}",
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    ])


def format_trade_resolution(
    is_win: bool,
    side: str,
    entry_price: float,
    slot_start_str: str,
    slot_end_str: str,
    pnl: float,
) -> str:
    """Real trade outcome — only sent when a real trade was placed."""
    icon = "\u2705" if is_win else "\u274c"
    label = "WIN" if is_win else "LOSS"
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    sign = "+" if pnl >= 0 else ""
    return "\n".join([
        f"{icon} <b>Trade Result \u2014 {label}</b>",
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC",
        f"\u2502 {side_emoji} Side: {side}",
        f"\u2502 \U0001f4b2 Entry: ${entry_price:.2f}",
        f"\u2502 \U0001f4b0 P&L: {sign}${pnl:.2f}",
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    ])


def format_demo_resolution(
    is_win: bool,
    side: str,
    entry_price: float,
    slot_start_str: str,
    slot_end_str: str,
    pnl: float,
    new_bankroll: float,
) -> str:
    """Demo trade outcome — only sent when a demo trade was placed."""
    icon = "\u2705" if is_win else "\u274c"
    label = "WIN" if is_win else "LOSS"
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    sign = "+" if pnl >= 0 else ""
    return "\n".join([
        f"{icon} <b>\U0001f9ea [DEMO] Trade Result \u2014 {label}</b>",
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC",
        f"\u2502 {side_emoji} Side: {side}",
        f"\u2502 \U0001f4b2 Entry: ${entry_price:.2f}",
        f"\u2502 \U0001f4b0 P&L: {sign}${pnl:.2f}",
        f"\u2502 \U0001f4b5 Bankroll: ${new_bankroll:.2f}",
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    ])


# ---------------------------------------------------------------------------
# Trade execution notifications (sent by scheduler during FOK retry loop)
# ---------------------------------------------------------------------------

def format_trade_filled(
    side: str,
    slot_label: str,
    ask_price: float,
    amount_usdc: float,
    shares: float | None,
    order_id: str | None,
    attempts: int,
) -> str:
    """Rich fill confirmation box sent when a FOK order is MATCHED."""
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    attempt_note = f" (attempt {attempts})" if attempts > 1 else ""
    shares_line = ""
    if shares is not None:
        shares_line = f"\u2502 \U0001f4ca Shares: {shares:.4f}\n"
    oid_line = ""
    if order_id:
        oid_short = (order_id[:10] + "..." + order_id[-6:]) if len(order_id) > 16 else order_id
        oid_line = f"\u2502 \U0001f9fe Order ID: <code>{_e(oid_short)}</code>\n"
    return (
        f"\u2705 <b>Trade FILLED{attempt_note}</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 {side_emoji} Side: {side}  \u23f0 {slot_label}\n"
        f"\u2502 \U0001f4b2 Ask Price: ${ask_price:.4f}\n"
        f"\u2502 \U0001f4b5 Amount: ${amount_usdc:.2f} USDC\n"
        f"{shares_line}"
        f"{oid_line}"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_trade_unmatched(
    side: str,
    slot_label: str,
    attempts: int,
    reason: str,
) -> str:
    """Rich failure box sent when all FOK retry attempts are exhausted."""
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    return (
        "\u274c <b>Trade UNMATCHED</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 {side_emoji} Side: {side}  \u23f0 {slot_label}\n"
        f"\u2502 \U0001f504 Attempts: {attempts}\n"
        f"\u2502 \U0001f4cb Reason: {_e(reason)}\n"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_trade_aborted(
    side: str,
    slot_label: str,
    reason: str,
) -> str:
    """Rich abort box sent when time fence or duplicate guard fires."""
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    return (
        "\u26d4 <b>Trade ABORTED</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 {side_emoji} Side: {side}  \u23f0 {slot_label}\n"
        f"\u2502 \U0001f4cb Reason: {_e(reason)}\n"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_trade_retrying(
    side: str,
    slot_label: str,
    attempt: int,
    max_attempts: int,
    reason: str,
) -> str:
    """Compact inline message sent before each retry attempt (no box)."""
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    return (
        f"\U0001f504 <b>Trade retrying</b> (attempt {attempt}/{max_attempts}) "
        f"{side_emoji} {side} {slot_label} \u2014 {_e(reason)}"
    )


# ---------------------------------------------------------------------------
# Redemption formatters
# ---------------------------------------------------------------------------

def format_redeem_preview(results: list[dict]) -> str:
    """Format /redeem dry-run scan results before user confirms."""
    if not results:
        return (
            "\U0001f4b0 <b>Redeem \u2014 No Positions Found</b>\n"
            + SEP + "\n"
            "No redeemable positions detected in your wallet.\n"
            "Positions only appear here once the market resolves on-chain."
        )

    won_count  = sum(1 for r in results if r.get("won"))
    lost_count = len(results) - won_count

    lines = [
        f"\U0001f4b0 <b>Redeem Preview ({len(results)} position(s) found)</b>",
        SEP,
    ]
    for i, r in enumerate(results, 1):
        title = _e((r.get("title") or r.get("condition_id", "Unknown"))[:60])
        size  = r.get("size", 0)
        label = "\u2705 WON" if r.get("won") else "\u274c LOST"
        lines.append(f"{i}. {label}  {title}")
        lines.append(f"   \U0001f4b0 Size: {size:.4f} shares")
    lines += [
        SEP,
        f"Won: <b>{won_count}</b>  Lost: <b>{lost_count}</b>",
        "Tap <b>Confirm Redeem</b> to execute all redemptions on-chain.\n"
        "<i>Lost positions are burned for $0 to clear them from your wallet.</i>",
    ]
    return "\n".join(lines)

def format_redeem_results(results: list[dict]) -> str:
    """Format the outcome after redemption transactions are sent."""
    if not results:
        return (
            "\U0001f4b0 <b>Redeem Complete</b>\n"
            + SEP + "\n"
            "No redeemable positions found \u2014 nothing to redeem."
        )

    success_count = sum(1 for r in results if r.get("success"))
    fail_count    = len(results) - success_count

    lines = [
        f"\U0001f4b0 <b>Redeem Complete</b>  \u2705 {success_count}  \u274c {fail_count}",
        SEP,
    ]
    for i, r in enumerate(results, 1):
        title = _e((r.get("title") or r.get("condition_id", "Unknown"))[:55])
        size  = r.get("size", 0)
        won   = r.get("won", True)   # default True for backwards-compat
        outcome_label = "WON" if won else "LOST"
        recovered     = f"${size:.2f}" if won else "$0.00"
        if r.get("success"):
            tx = r.get("tx_hash", "")
            short_tx = tx[:10] + "..." + tx[-6:] if tx and len(tx) > 16 else (tx or "N/A")
            gas = r.get("gas_used")
            gas_str = f"  gas={gas:,}" if gas else ""
            lines.append(f"\u2705 {i}. [{outcome_label}] {title}")
            lines.append(f"   {size:.4f} shares  recovered: {recovered}  tx: <code>{_e(short_tx)}</code>{gas_str}")
        else:
            err = _e((r.get("error") or "unknown error")[:200])
            lines.append(f"\u274c {i}. [{outcome_label}] {title}")
            lines.append(f"   Error: {err}")
    lines.append(SEP)
    return "\n".join(lines)

def format_auto_redeem_notification(results: list[dict]) -> str:
    """Compact notification sent by the auto-redeem scheduler job."""
    success = [r for r in results if r.get("success")]
    failed  = [r for r in results if not r.get("success")]

    lines = [
        f"\U0001f916 <b>Auto-Redeem Complete</b>  \u2705 {len(success)}  \u274c {len(failed)}",
        SEP,
    ]
    for r in success:
        title     = _e((r.get("title") or r.get("condition_id", "?"))[:55])
        won       = r.get("won", True)
        outcome_label = "WON" if won else "LOST"
        recovered     = f"${r.get('size', 0):.2f}" if won else "$0.00"
        tx        = r.get("tx_hash", "")
        short_tx  = tx[:10] + "..." + tx[-6:] if tx and len(tx) > 16 else (tx or "N/A")
        lines.append(f"\u2705 [{outcome_label}] {title}")
        lines.append(f"   recovered: {recovered}  tx: <code>{_e(short_tx)}</code>")
    for r in failed:
        title = _e((r.get("title") or r.get("condition_id", "?"))[:55])
        won   = r.get("won", True)
        outcome_label = "WON" if won else "LOST"
        err   = _e((r.get("error") or "unknown")[:200])
        lines.append(f"\u274c [{outcome_label}] {title}")
        lines.append(f"   {err}")
    lines.append(SEP)
    return "\n".join(lines)

def format_error_alert(context: str, error: str, detail: str | None = None) -> str:
    """Format a system-level error alert for Telegram.

    Parameters
    ----------
    context : str
        Where the error occurred (e.g. 'auto_redeem_job', 'fetch_positions').
    error : str
        Short error message.
    detail : str | None
        Optional full traceback or extended detail (truncated to 600 chars).
    """
    lines = [
        f"\u26a0\ufe0f <b>Error \u2014 {_e(context)}</b>",
        SEP,
        f"<b>Error:</b> {_e(error[:200])}",
    ]
    if detail:
        short_detail = detail[-600:] if len(detail) > 600 else detail
        lines.append(f"<pre>{_e(short_detail)}</pre>")
    lines.append(SEP)
    return "\n".join(lines)


def format_redemption_history(stats: dict, recent: list[dict]) -> str:
    """Format the /redemptions dashboard."""
    lines = [
        "\U0001f4b0 <b>Redemption History</b>",
        SEP,
        f"\U0001f4ca Total Redeemed: {stats['total']}",
        f"\u2705 Success: {stats['success']}  \u274c Failed: {stats['failed']}",
        f"\U0001f4b0 Total Size Redeemed: {stats['total_size']:.4f} shares",
        SEP,
    ]
    if not recent:
        lines.append("No redemptions recorded yet.")
        return "\n".join(lines)

    lines.append("\U0001f4cb <b>Recent Redemptions:</b>")
    for r in recent:
        ts = r.get("created_at", "")[:16]
        title = _e((r.get("title") or r.get("condition_id", "Unknown"))[:45])
        size = r.get("size", 0)
        status = r.get("status", "?")
        icon = "\u2705" if status == "success" else "\u274c"
        tx = r.get("tx_hash") or ""
        short_tx = tx[:8] + "..." if tx else "N/A"
        lines.append(f"{icon} {ts}  {title}")
        lines.append(f"   {size:.4f} sh  tx: <code>{_e(short_tx)}</code>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dashboards (requested via bot commands)
# ---------------------------------------------------------------------------

def format_signal_stats(stats: dict[str, Any], label: str = "All Time") -> str:
    streak_str = "0"
    if stats.get("current_streak") and stats.get("current_streak_type"):
        streak_str = f"{stats['current_streak']}{stats['current_streak_type']}"

    lines = [
        f"\U0001f4ca <b>Signal Performance ({label})</b>",
        SEP,
        f"\u26a1 Total Signals: {stats['total_signals']}",
        f"\u2705 Wins: {stats['wins']}  |  \u274c Losses: {stats['losses']}",
        f"\U0001f4c8 Win Rate: {stats['win_pct']}%",
        SEP,
        f"\U0001f525 Current Streak: {streak_str}",
        f"\U0001f3c6 Best Win Streak: {stats['best_win_streak']}",
        f"\U0001f480 Worst Loss Streak: {stats['worst_loss_streak']}",
        SEP,
        f"\u23ed\ufe0f Skipped (No Signal): {stats['skip_count']}",
    ]
    return "\n".join(lines)


def format_trade_stats(stats: dict[str, Any], label: str = "All Time") -> str:
    streak_str = "0"
    if stats.get("current_streak") and stats.get("current_streak_type"):
        streak_str = f"{stats['current_streak']}{stats['current_streak_type']}"

    sign = "+" if stats["net_pnl"] >= 0 else ""
    roi_sign = "+" if stats["roi_pct"] >= 0 else ""

    lines = [
        f"\U0001f4b0 <b>Trade Performance ({label})</b>",
        SEP,
        f"\U0001f4ca Total Trades: {stats['total_trades']}",
        f"\u2705 Wins: {stats['wins']}  |  \u274c Losses: {stats['losses']}",
        f"\U0001f4c8 Win Rate: {stats['win_pct']}%",
        SEP,
        f"\U0001f4b5 Total Deployed: ${stats['total_deployed']:.2f}",
        f"\U0001f4b0 Total Returned: ${stats['total_returned']:.2f}",
        f"\U0001f4c8 Net P&L: {sign}${stats['net_pnl']:.2f}",
        f"\U0001f4ca ROI: {roi_sign}{stats['roi_pct']}%",
        SEP,
        f"\U0001f525 Current Streak: {streak_str}",
        f"\U0001f3c6 Best Win Streak: {stats['best_win_streak']}",
    ]
    return "\n".join(lines)


def format_status(
    connected: bool,
    balance: float | None,
    autotrade: bool,
    trade_amount: float,
    open_positions: int,
    uptime_str: str,
    last_signal: str | None,
    auto_redeem: bool = False,
    demo_trade_enabled: bool = False,
    demo_bankroll: float | None = None,
    trade_mode: str = "fixed",
    trade_pct: float = 5.0,
) -> str:
    conn_icon = "\U0001f7e2" if connected else "\U0001f534"
    conn_text = "Connected" if connected else "Disconnected"
    at_text = "ON" if autotrade else "OFF"
    ar_text = "ON" if auto_redeem else "OFF"
    dt_text = "ON" if demo_trade_enabled else "OFF"
    bal_text = f"{balance:.2f} USDC" if balance is not None else "N/A"
    sig_text = last_signal or "None"

    if trade_mode == "pct":
        mode_line = f"\U0001f4b5 Trade Mode: PCT {trade_pct:.1f}%"
    else:
        mode_line = f"\U0001f4b5 Trade Mode: FIXED ${trade_amount:.2f}"

    lines = [
        "\U0001f916 <b>AutoPoly Status</b>",
        SEP,
        f"{conn_icon} Bot: Running",
        f"\U0001f517 Polymarket: {conn_text}",
        f"\U0001f4b0 Balance: {bal_text}",
        SEP,
        f"\U0001f916 AutoTrade: {at_text}",
        mode_line,
        f"\U0001f4ca Open Positions: {open_positions}",
        f"\U0001f4b0 Auto-Redeem: {ar_text}",
        SEP,
        f"\U0001f9ea Demo Trade: {dt_text}",
    ]
    if demo_bankroll is not None:
        lines.append(f"\U0001f4b5 Demo Bankroll: ${demo_bankroll:.2f}")
    lines += [
        SEP,
        f"\u23f0 Uptime: {uptime_str}",
        f"\U0001f4e1 Last Signal: {sig_text}",
    ]
    return "\n".join(lines)


def format_recent_signals(signals: list[dict[str, Any]]) -> str:
    if not signals:
        return "\nNo signals recorded yet."
    lines = ["\n\U0001f4cb <b>Recent Signals:</b>"]
    for s in signals:
        ss = s["slot_start"].split(" ")[-1] if " " in s["slot_start"] else s["slot_start"]
        se = s["slot_end"].split(" ")[-1] if " " in s["slot_end"] else s["slot_end"]
        if s["skipped"]:
            lines.append(f"\u23ed\ufe0f {ss}-{se} UTC \u2014 skipped")
        else:
            icon = "\u2705" if s.get("is_win") == 1 else ("\u274c" if s.get("is_win") == 0 else "\u23f3")
            lines.append(f"{icon} {ss}-{se} UTC  {s['side']}  ${s.get('entry_price', 0):.2f}")
    return "\n".join(lines)


def format_recent_trades(trades: list[dict[str, Any]]) -> str:
    if not trades:
        return "\nNo trades recorded yet."
    lines = ["\n\U0001f4cb <b>Recent Trades:</b>"]
    for t in trades:
        ss = t["slot_start"].split(" ")[-1] if " " in t["slot_start"] else t["slot_start"]
        se = t["slot_end"].split(" ")[-1] if " " in t["slot_end"] else t["slot_end"]
        icon = "\u2705" if t.get("is_win") == 1 else ("\u274c" if t.get("is_win") == 0 else "\u23f3")
        pnl_str = ""
        if t.get("pnl") is not None:
            sign = "+" if t["pnl"] >= 0 else ""
            pnl_str = f"  {sign}${t['pnl']:.2f}"
        lines.append(f"{icon} {ss}-{se} UTC  {t['side']}  ${t['amount_usdc']:.2f}{pnl_str}")
    return "\n".join(lines)


def format_help() -> str:
    return (
        "\u2753 <b>Help & Commands</b>\n"
        + SEP + "\n"
        "<b>Dashboard</b>\n"
        "/status  &middot; /signals  &middot; /trades  &middot; /patterns\n\n"
        "<b>Actions</b>\n"
        "/redeem  &middot; /redemptions\n\n"
        "<b>Config</b>\n"
        "/settings  &middot; /demo\n\n"
        "<b>Misc</b>\n"
        "/help  &middot; /start\n\n"
        "<b>ML Thresholds</b>\n"
        "/set_threshold &lt;val&gt;  — manually override the UP (LONG) inference threshold\n"
        "/set_down_threshold &lt;val&gt;  — manually override the DOWN (SHORT) inference threshold\n"
        + SEP + "\n"
        "<b>How it works:</b>\n"
        "Every 5 minutes the bot analyses the last six closed BTC-USD "
        "5-minute candles from Coinbase, builds a 6-character pattern "
        "(U=up, D=down), and looks it up in a pattern table. If the "
        "pattern matches, a signal fires for the predicted direction. "
        "If no match, the slot is skipped.\n\n"
        "<b>Auto-Redeem:</b>\n"
        "When enabled, the bot periodically scans your wallet for resolved "
        "winning positions and calls redeemPositions() on the Polygon CTF "
        "contract to collect your USDC.e. Use /redeem for a manual scan."
        + SEP
    )


# ---------------------------------------------------------------------------
# Demo Trade Formatters
# ---------------------------------------------------------------------------

def format_demo_stats(stats: dict, bankroll: float, label: str = "All Time") -> str:
    """Format the demo trade P&L dashboard."""
    sign = "+" if stats["net_pnl"] >= 0 else ""
    roi_sign = "+" if stats["roi_pct"] >= 0 else ""
    lines = [
        f"\U0001f9ea <b>Demo Trade Performance ({label})</b>",
        SEP,
        f"\U0001f4ca Total Trades: {stats['total_trades']}",
        f"\u2705 Wins: {stats['wins']}  |  \u274c Losses: {stats['losses']}",
        f"\U0001f4c8 Win Rate: {stats['win_pct']}%",
        SEP,
        f"\U0001f4b5 Total Deployed: ${stats['total_deployed']:.2f}",
        f"\U0001f4b0 Total Returned: ${stats['total_returned']:.2f}",
        f"\U0001f4c8 Net P&L: {sign}${stats['net_pnl']:.2f}",
        f"\U0001f4ca ROI: {roi_sign}{stats['roi_pct']}%",
        SEP,
        f"\U0001f4b0 Current Bankroll: ${bankroll:.2f}",
    ]
    return "\n".join(lines)


def format_demo_recent_trades(trades: list) -> str:
    """Format a list of recent demo trades."""
    if not trades:
        return "\nNo demo trades recorded yet."
    lines = ["\n\U0001f4cb <b>Recent Demo Trades:</b>"]
    for t in trades:
        ss = t["slot_start"].split(" ")[-1] if " " in t["slot_start"] else t["slot_start"]
        se = t["slot_end"].split(" ")[-1] if " " in t["slot_end"] else t["slot_end"]
        icon = "\u2705" if t.get("is_win") == 1 else ("\u274c" if t.get("is_win") == 0 else "\u23f3")
        pnl_str = ""
        if t.get("pnl") is not None:
            s = "+" if t["pnl"] >= 0 else ""
            pnl_str = f"  {s}${t['pnl']:.2f}"
        lines.append(
            f"\U0001f9ea {icon} {ss}-{se} UTC  {t['side']}  ${t['amount_usdc']:.2f}{pnl_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pattern analytics formatter
# ---------------------------------------------------------------------------

def format_pattern_stats(rows: list[dict[str, Any]]) -> str:
    """Format per-pattern performance stats as a Telegram HTML message."""
    if not rows:
        return (
            "\U0001f522 <b>Pattern Performance</b>\n"
            + SEP + "\n"
            "No resolved real trades recorded yet.\n"
            "Pattern stats appear once trades complete."
        )

    lines = [
        "\U0001f522 <b>Pattern Performance</b>",
        SEP,
    ]
    for i, r in enumerate(rows):
        if i > 0:
            lines.append(SEP)
        wl = f"{r['wl_ratio']}" if r["wl_ratio"] != float("inf") else "\u221e"
        roi_sign = "+" if r["roi_pct"] >= 0 else ""
        pnl_sign = "+" if r["net_pnl"] >= 0 else ""
        lines += [
            f"\U0001f4cc <b>{r['pattern']}</b>  "
            f"({r['total_trades']} trades \u2022 last: {str(r['last_seen'])[:16]})",
            f"   \u2705 {r['wins']}W  \u274c {r['losses']}L  "
            f"\U0001f4c8 {r['win_pct']}%  W/L: {wl}",
            f"   \U0001f4b5 ${r['total_deployed']:.2f} deployed  "
            f"P&L: {pnl_sign}${r['net_pnl']:.2f}  "
            f"ROI: {roi_sign}{r['roi_pct']}%",
        ]
    lines.append(SEP)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ML model formatters
# ---------------------------------------------------------------------------

def format_model_status(slot: str, meta: dict, threshold: float) -> str:
    """Show model status summary including DOWN validation results."""
    down_enabled = meta.get("down_enabled", False)
    down_thr     = meta.get("down_threshold", round(1.0 - threshold, 4))
    down_val_wr  = meta.get("down_val_wr")
    down_test_wr = meta.get("down_test_wr")
    down_tpd     = meta.get("down_test_tpd", meta.get("down_val_tpd", 0))

    up_gate_pct  = meta.get("test_wr", 0) * 100
    up_gate_icon = "\u2705" if up_gate_pct >= 58.0 else "\u274c"
    up_gate_lbl  = "PASS" if up_gate_pct >= 58.0 else "FAIL"

    if down_val_wr is not None and down_test_wr is not None:
        down_status_lbl = "ENABLED" if down_enabled else "DISABLED"
        down_status_icon = "\u2705" if down_enabled else "\u26d4"
        down_section = (
            f"\u2502 \u2193 DOWN Side              {down_status_icon} {down_status_lbl}\n"
            f"\u2502   Win Rate   val {down_val_wr*100:.1f}% / test {down_test_wr*100:.1f}%\n"
            f"\u2502   Threshold  \u2265 {down_thr*100:.1f}%\n"
            f"\u2502   Trades/day {down_tpd:.1f}\n"
        )
    else:
        down_section = (
            "\u2502 \u2193 DOWN Side              \u26d4 DISABLED\n"
            "\u2502   Not validated\n"
        )

    return (
        f"\U0001f916 <b>ML Model Status</b>  [{slot.upper()}]\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \U0001f4c5 Trained:  {str(meta.get('train_date', 'N/A'))[:16]} UTC\n"
        f"\u2502 \U0001f4ca Samples:  {meta.get('sample_count', 0):,}\n"
        "\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u2191 UP Side               {up_gate_icon} {up_gate_lbl}\n"
        f"\u2502   Win Rate   val {meta.get('val_wr', 0)*100:.1f}% / test {meta.get('test_wr', 0)*100:.1f}%\n"
        f"\u2502   Threshold  \u2265 {threshold*100:.1f}%\n"
        f"\u2502   Trades/day {meta.get('test_trades_per_day', 0):.1f}\n"
        "\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"{down_section}"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_model_compare(current_meta: dict, candidate_meta: dict) -> str:
    """Side by side comparison of current vs candidate model, including DOWN validation."""
    def _fmt(m: dict, label: str) -> list[str]:
        down_enabled = m.get("down_enabled", False)
        down_icon = "\u2705" if down_enabled else "\u274c"
        down_thr = m.get("down_threshold", round(1.0 - m.get("threshold", 0.535), 4))
        down_val_wr = m.get("down_val_wr")
        down_test_wr = m.get("down_test_wr")
        down_tpd = m.get("down_test_tpd", m.get("down_val_tpd", 0))

        rows = [
            f"<b>{label}</b>",
            f"  Trained  : {str(m.get('train_date', 'N/A'))[:19]}",
            f"  Samples  : {m.get('sample_count', 0):,}",
            f"  \u2191 UP   thr={m.get('threshold', 0):.3f}  val={m.get('val_wr', 0)*100:.2f}%  "
            f"test={m.get('test_wr', 0)*100:.2f}%  tpd={m.get('test_trades_per_day', 0):.1f}",
        ]
        if down_val_wr is not None and down_test_wr is not None:
            rows.append(
                f"  \u2193 DOWN {down_icon} thr={down_thr:.3f}  val={down_val_wr*100:.2f}%  "
                f"test={down_test_wr*100:.2f}%  tpd={down_tpd:.1f}"
            )
        else:
            rows.append(f"  \u2193 DOWN {down_icon} {'ENABLED' if down_enabled else 'not validated'}  thr={down_thr:.3f}")
        return rows

    lines = [SEP, "<b>Model Comparison</b>", SEP]
    lines.extend(_fmt(current_meta, "CURRENT"))
    lines.append("")
    lines.extend(_fmt(candidate_meta, "CANDIDATE"))
    lines.append(SEP)
    lines.append("Use /promote_model to deploy candidate as current.")
    return "\n".join(lines)


def format_retrain_started() -> str:
    """Notification that background retraining has started."""
    return (
        f"{SEP}\n"
        "<b>Retraining started...</b>\n"
        "Fetching 5 months of MEXC data and training LightGBM.\n"
        "This takes ~5-10 minutes. You will receive a report when done.\n"
        f"{SEP}"
    )


def format_retrain_blocked(meta: dict, threshold: float) -> tuple[str, str | None]:
    """Notification sent when retrain completes but fails the 59% deployment gate.

    The candidate IS saved — the user must decide to promote or discard.

    Returns a tuple of (main_message, risk_message).
    *risk_message* is None when no risk data is available.
    Both messages must be sent with parse_mode='HTML'.
    """
    _GATE = 0.58
    down_enabled = meta.get("down_enabled", False)
    down_thr     = meta.get("down_threshold", round(1.0 - threshold, 4))
    down_val_wr  = meta.get("down_val_wr")
    down_test_wr = meta.get("down_test_wr")
    down_tpd     = meta.get("down_test_tpd", meta.get("down_val_tpd", 0))

    up_test_wr    = meta.get("test_wr", 0)
    shortfall     = round((up_test_wr - _GATE) * 100, 1)   # always negative here
    shortfall_str = f"{shortfall:+.1f}%"                   # e.g. "-2.2%"

    # Data date range (ISO date strings, e.g. "2025-11-14")
    data_start = meta.get("data_start")
    data_end   = meta.get("data_end")
    if data_start and data_end:
        data_line = f"\u2502 \U0001f5d3 Data:     {_e(data_start)} \u2192 {_e(data_end)}\n"
    else:
        data_line = ""

    # Payout ratio
    payout     = meta.get("payout")
    payout_line = f"\u2502 \U0001f4b0 Payout:   {payout:.2f}  ({payout*100:.0f}\u00a2 per $1)\n" if payout is not None else ""

    # UP EV/day
    up_ev      = meta.get("up_ev_per_day", 0.0)
    up_ev_str  = f"{up_ev:+.2f}" if up_ev != 0.0 else "N/A"

    # DOWN section
    if down_val_wr is not None and down_test_wr is not None:
        down_status_lbl  = "ENABLED" if down_enabled else "DISABLED"
        down_status_icon = "\u2705" if down_enabled else "\u26d4"
        down_ev     = meta.get("down_ev_per_day", 0.0)
        down_ev_str = f"{down_ev:+.2f}" if down_ev != 0.0 else "N/A"
        down_section = (
            f"\u2502 \u2193 DOWN Side              {down_status_icon} {down_status_lbl}\n"
            f"\u2502   Val  {down_val_wr*100:.1f}%  /  Test  {down_test_wr*100:.1f}%\n"
            f"\u2502   Threshold \u2265 {down_thr*100:.1f}%  \u2022  {down_tpd:.1f} trades/day\n"
            f"\u2502   EV/day  {down_ev_str}\n"
        )
    else:
        down_section = (
            "\u2502 \u2193 DOWN Side              \u26d4 DISABLED\n"
            "\u2502   Not validated\n"
        )

    main_msg = (
        "\u26a0\ufe0f <b>Retrain \u2014 Gate NOT Passed</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \U0001f4c5 Trained:  {str(meta.get('train_date', 'N/A'))[:16]} UTC\n"
        f"\u2502 \U0001f4ca Samples:  {meta.get('sample_count', 0):,}\n"
        f"{data_line}"
        f"{payout_line}"
        "\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u2191 UP Side        \u274c FAILED  {shortfall_str} below gate\n"
        f"\u2502   Val  {meta.get('val_wr', 0)*100:.1f}%  /  Test  {up_test_wr*100:.1f}%  (need \u2265 {_GATE*100:.1f}%)\n"
        f"\u2502   Threshold \u2265 {threshold*100:.1f}%  \u2022  {meta.get('test_trades_per_day', 0):.1f} trades/day\n"
        f"\u2502   EV/day  {up_ev_str}\n"
        "\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"{down_section}"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "Candidate saved \u2014 not live.\n"
        "/model to compare  \u2022  /promote_model to override gate"
    )

    return main_msg, _build_risk_table(meta)


def format_retrain_complete(meta: dict, threshold: float) -> tuple[str, str | None]:
    """Show retrain results for candidate model — UP passed gate, DOWN shown separately.

    Returns a tuple of (main_message, risk_message).
    *risk_message* is None when no risk data is available.
    Both messages must be sent with parse_mode='HTML'.
    """
    _GATE = 0.58
    down_enabled = meta.get("down_enabled", False)
    down_thr     = meta.get("down_threshold", round(1.0 - threshold, 4))
    down_val_wr  = meta.get("down_val_wr")
    down_test_wr = meta.get("down_test_wr")
    down_tpd     = meta.get("down_test_tpd", meta.get("down_val_tpd", 0))

    up_test_wr    = meta.get("test_wr", 0)
    up_margin     = round((up_test_wr - _GATE) * 100, 1)
    up_margin_str = f"+{up_margin:.1f}%" if up_margin >= 0 else f"{up_margin:.1f}%"

    # Data date range (ISO date strings, e.g. "2025-11-14")
    data_start = meta.get("data_start")
    data_end   = meta.get("data_end")
    if data_start and data_end:
        data_line = f"\u2502 \U0001f5d3 Data:     {_e(data_start)} \u2192 {_e(data_end)}\n"
    else:
        data_line = ""

    # Payout ratio
    payout      = meta.get("payout")
    payout_line = f"\u2502 \U0001f4b0 Payout:   {payout:.2f}  ({payout*100:.0f}\u00a2 per $1)\n" if payout is not None else ""

    # UP EV/day
    up_ev     = meta.get("up_ev_per_day", 0.0)
    up_ev_str = f"{up_ev:+.2f}" if up_ev != 0.0 else "N/A"

    # DOWN section
    if down_val_wr is not None and down_test_wr is not None:
        down_status_lbl  = "ENABLED" if down_enabled else "DISABLED"
        down_status_icon = "\u2705" if down_enabled else "\u26d4"
        down_ev     = meta.get("down_ev_per_day", 0.0)
        down_ev_str = f"{down_ev:+.2f}" if down_ev != 0.0 else "N/A"
        down_section = (
            f"\u2502 \u2193 DOWN Side              {down_status_icon} {down_status_lbl}\n"
            f"\u2502   Val  {down_val_wr*100:.1f}%  /  Test  {down_test_wr*100:.1f}%\n"
            f"\u2502   Threshold \u2265 {down_thr*100:.1f}%  \u2022  {down_tpd:.1f} trades/day\n"
            f"\u2502   EV/day  {down_ev_str}\n"
        )
    else:
        down_section = (
            "\u2502 \u2193 DOWN Side              \u26d4 DISABLED\n"
            "\u2502   Not validated\n"
        )

    main_msg = (
        "\u2705 <b>Retrain Complete</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \U0001f4c5 Trained:  {str(meta.get('train_date', 'N/A'))[:16]} UTC\n"
        f"\u2502 \U0001f4ca Samples:  {meta.get('sample_count', 0):,}\n"
        f"{data_line}"
        f"{payout_line}"
        "\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u2191 UP Side        \u2705 Gate passed  {up_margin_str}\n"
        f"\u2502   Val  {meta.get('val_wr', 0)*100:.1f}%  /  Test  {up_test_wr*100:.1f}%  (min {_GATE*100:.1f}%)\n"
        f"\u2502   Threshold \u2265 {threshold*100:.1f}%  \u2022  {meta.get('test_trades_per_day', 0):.1f} trades/day\n"
        f"\u2502   EV/day  {up_ev_str}\n"
        "\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"{down_section}"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "Use /promote_model to deploy  \u2022  /model to compare"
    )

    return main_msg, _build_risk_table(meta)


def format_set_threshold(threshold: float) -> str:
    """Confirmation message after /set_threshold."""
    return (
        f"\u2705 <b>UP threshold set to {threshold:.3f}</b>.\n"
        "Active on next signal check.\n\n"
        f"Signals will only LONG when p_up \u2265 <b>{threshold:.3f}</b>."
    )


def format_set_down_threshold(threshold: float) -> str:
    """Confirmation message after /set_down_threshold."""
    return (
        f"\u2705 <b>DOWN threshold set to {threshold:.3f}</b>.\n"
        "Active on next signal check.\n\n"
        f"Signals will only SHORT when p_down \u2265 <b>{threshold:.3f}</b>."
    )


def format_drift_alert(drifted_features: list, records_analyzed: int) -> str:
    """Format a feature drift alert for Telegram."""
    lines = [
        "\u26a0\ufe0f <b>Feature Drift Alert</b>",
        "\u2501" * 20,
        f"\U0001f4ca Records analyzed: {records_analyzed}",
        f"\u26a0\ufe0f {len(drifted_features)} feature(s) drifted &gt; 2\u03c3:",
    ]
    for d in drifted_features[:10]:
        z = d["z_score"]
        sign = "+" if z >= 0 else ""
        lines.append(
            f"  \U0001f4cc <b>{d['feature']}</b>: "
            f"live={d['live_mean']:.4f} vs train={d['train_mean']:.4f} "
            f"(z={sign}{z:.2f})"
        )
    lines += [
        "\u2501" * 20,
        "\U0001f916 Model may need retraining. Use /retrain to update.",
    ]
    return "\n".join(lines)
