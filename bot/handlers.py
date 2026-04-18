"""Telegram command and callback-query handlers."""

from __future__ import annotations

import asyncio
import csv
import io
import html as _html
import logging
from datetime import datetime, timezone
from typing import Any

import openpyxl
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config as cfg
from bot.formatters import (
    format_demo_recent_trades,
    format_demo_stats,
    format_help,
    format_model_compare,
    format_model_status,
    format_pattern_stats,
    format_recent_signals,
    format_recent_trades,
    format_redeem_preview,
    format_redeem_results,
    format_redemption_history,
    format_retrain_blocked,
    format_retrain_complete,
    format_set_threshold,
    format_set_down_threshold,
    format_signal_stats,
    format_status,
    format_trade_stats,
)
from bot.keyboards import (
    back_to_menu,
    down_override_keyboard,
    main_menu,
    ml_menu,
    pattern_keyboard,
    redeem_confirm_keyboard,
    redeem_done_keyboard,
    retrain_blocked_keyboard,
    settings_keyboard,
    signal_filter_row,
    trade_filter_row,
)
from bot.middleware import auth_check
from db import queries
from polymarket import account as pm_account

log = logging.getLogger(__name__)

# Set at startup by main.py
_start_time: datetime = datetime.now(timezone.utc)
_poly_client: Any = None


def set_poly_client(client: Any) -> None:
    global _poly_client
    _poly_client = client


def set_start_time() -> None:
    global _start_time
    _start_time = datetime.now(timezone.utc)


def _uptime() -> str:
    delta = datetime.now(timezone.utc) - _start_time
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _parse_threshold_input(raw: str) -> float:
    """Parse a user-provided threshold value.

    Accepts any finite numeric input and rejects non-numeric or non-finite values.
    """
    value = float(raw.strip())
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("non-finite")
    return value


# ---------------------------------------------------------------------------
# Safe edit helper — silently ignores 'Message is not modified' errors
# ---------------------------------------------------------------------------

async def _safe_edit(query, text, reply_markup=None, parse_mode="HTML"):
    """Edit a message, silently ignoring 'not modified' errors."""
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except BadRequest as e:
        if "not modified" in str(e).lower():
            pass  # Content unchanged — not an error
        else:
            raise


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

@auth_check
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "\U0001f916 <b>Welcome to AutoPoly!</b>\n\n"
        "BTC Up/Down 5-min trading bot for Polymarket.\n"
        "Select an option below:"
    )
    await update.message.reply_text(text, reply_markup=main_menu(), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

@auth_check
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    connected = False
    balance = None
    positions = []
    if _poly_client:
        connected = await pm_account.get_connection_status(_poly_client)
        balance = await pm_account.get_balance(_poly_client)
        positions = await pm_account.get_open_positions(_poly_client)

    autotrade = await queries.is_autotrade_enabled()
    auto_redeem = await queries.is_auto_redeem_enabled()
    trade_amount = await queries.get_trade_amount()
    last_sig = await queries.get_last_signal()
    last_sig_str = None
    if last_sig:
        ss = last_sig["slot_start"].split(" ")[-1] if " " in last_sig["slot_start"] else last_sig["slot_start"]
        last_sig_str = f"{ss} UTC ({last_sig['side']})"

    demo_trade = await queries.is_demo_trade_enabled()
    demo_bankroll = await queries.get_demo_bankroll() if demo_trade else None
    trade_mode = await queries.get_trade_mode()
    trade_pct = await queries.get_trade_pct()
    text = format_status(
        connected=connected,
        balance=balance,
        autotrade=autotrade,
        trade_amount=trade_amount,
        open_positions=len(positions),
        uptime_str=_uptime(),
        last_signal=last_sig_str,
        auto_redeem=auto_redeem,
        demo_trade_enabled=demo_trade,
        demo_bankroll=demo_bankroll,
        trade_mode=trade_mode,
        trade_pct=trade_pct,
    )
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=back_to_menu())
    else:
        target = update.message
        if target is None:
            return
        await target.reply_text(text, reply_markup=back_to_menu(), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /signals
# ---------------------------------------------------------------------------

async def _render_signals(update: Update, limit: int | None, active: str) -> None:
    stats = await queries.get_signal_stats(limit=limit)
    label = {"10": "Last 10", "50": "Last 50", "all": "All Time"}[active]
    text = format_signal_stats(stats, label)
    recent = await queries.get_recent_signals(10)
    text += format_recent_signals(recent)
    kb = signal_filter_row(active)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


@auth_check
async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _render_signals(update, limit=None, active="all")


# ---------------------------------------------------------------------------
# /trades
# ---------------------------------------------------------------------------

async def _render_trades(update: Update, limit: int | None, active: str) -> None:
    stats = await queries.get_trade_stats(limit=limit)
    label = {"10": "Last 10", "50": "Last 50", "all": "All Time"}[active]
    text = format_trade_stats(stats, label)
    recent = await queries.get_recent_trades(10)
    text += format_recent_trades(recent)
    kb = trade_filter_row(active)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


@auth_check
async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _render_trades(update, limit=None, active="all")


# ---------------------------------------------------------------------------
# /settings
# ---------------------------------------------------------------------------

@auth_check
async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    autotrade = await queries.is_autotrade_enabled()
    auto_redeem = await queries.is_auto_redeem_enabled()
    trade_amount = await queries.get_trade_amount()
    trade_mode = await queries.get_trade_mode()
    trade_pct = await queries.get_trade_pct()
    demo_trade = await queries.is_demo_trade_enabled()
    demo_bankroll = await queries.get_demo_bankroll()
    invert_trades = await queries.is_invert_trades_enabled()
    at_text = "ON" if autotrade else "OFF"
    mode_summary = f"{trade_pct:.1f}%" if trade_mode == "pct" else f"${trade_amount:.2f}"
    dt_text = "ON" if demo_trade else "OFF"
    text = (
        f"\u2699\ufe0f <b>Settings</b>\n"
        f"AutoTrade: {at_text}  |  Mode: {mode_summary}  |  Demo: {dt_text}"
    )
    kb = settings_keyboard(autotrade, trade_amount, auto_redeem, demo_trade, demo_bankroll, trade_mode, trade_pct, invert_trades)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

@auth_check
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = format_help()
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=back_to_menu())
    else:
        await update.message.reply_text(text, reply_markup=back_to_menu(), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /redeem — manual redemption (dry-run preview then confirm)
# ---------------------------------------------------------------------------

@auth_check
async def cmd_redeem(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Step 1: dry-run scan — show what would be redeemed, await confirmation."""
    from core.redeemer import scan_and_redeem

    wallet = cfg.POLYMARKET_FUNDER_ADDRESS
    if not wallet:
        text = "\u274c <b>Redeem Error</b>\n\nPOLYMARKET_FUNDER_ADDRESS is not configured."
        if update.callback_query:
            await update.callback_query.answer()
            await _safe_edit(update.callback_query, text, reply_markup=back_to_menu())
        else:
            await update.message.reply_text(text, parse_mode="HTML", reply_markup=back_to_menu())
        return

    # Show scanning message first
    scanning_text = "\U0001f50d <b>Scanning wallet for redeemable positions...</b>"
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, scanning_text)
        sent = None  # edits are done via callback_query
    else:
        sent = await update.message.reply_text(scanning_text, parse_mode="HTML")

    try:
        results = await scan_and_redeem(wallet, dry_run=True)
    except Exception:
        log.exception("cmd_redeem: scan_and_redeem raised unexpectedly")
        error_text = "\u274c <b>Scan failed</b>\n\nCould not fetch positions. Please try again."
        if update.callback_query:
            await _safe_edit(update.callback_query, error_text, reply_markup=back_to_menu())
        else:
            await sent.edit_text(error_text, parse_mode="HTML", reply_markup=back_to_menu())
        return

    # Store dry-run results in user_data for the confirm step
    context.user_data["redeem_preview"] = results

    text = format_redeem_preview(results)
    kb = redeem_confirm_keyboard() if results else back_to_menu()

    if update.callback_query:
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await sent.edit_text(text, parse_mode="HTML", reply_markup=kb)


# ---------------------------------------------------------------------------
# /redemptions — history dashboard
# ---------------------------------------------------------------------------

@auth_check
async def cmd_redemptions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    stats = await queries.get_redemption_stats()
    recent = await queries.get_recent_redemptions(10)
    text = format_redemption_history(stats, recent)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=back_to_menu())
    else:
        await update.message.reply_text(text, reply_markup=back_to_menu(), parse_mode="HTML")


# ---------------------------------------------------------------------------
# Download handlers
# ---------------------------------------------------------------------------

@auth_check
async def cmd_download_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Export all signals as a CSV file (filter_blocked column removed — always was 0)."""
    query = update.callback_query
    await query.answer("Preparing CSV...")
    rows = await queries.get_all_signals_for_export()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "slot_start", "side", "entry_price", "is_win", "pattern"])
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)
    await query.message.reply_document(
        document=io.BytesIO(buf.getvalue().encode()),
        filename="signals.csv",
        caption="\U0001f4e5 All signals export (CSV)",
    )


@auth_check
async def cmd_download_excel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Export all signals as an Excel file (filter_blocked column removed — always was 0)."""
    query = update.callback_query
    await query.answer("Preparing Excel...")
    rows = await queries.get_all_signals_for_export()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Signals"
    ws.append(["id", "slot_start", "side", "entry_price", "is_win", "pattern"])
    for r in rows:
        ws.append([r["id"], r["slot_start"], r["side"], r["entry_price"], r["is_win"], r.get("pattern", "")])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    await query.message.reply_document(
        document=buf,
        filename="signals.xlsx",
        caption="\U0001f4e5 All signals export (Excel)",
    )



# ---------------------------------------------------------------------------
# /patterns — per-pattern performance dashboard
# ---------------------------------------------------------------------------

@auth_check
async def cmd_patterns(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = await queries.get_pattern_stats()
    text = format_pattern_stats(rows)
    kb = pattern_keyboard()
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


@auth_check
async def cmd_download_pattern_excel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Export per-pattern stats as an Excel file."""
    query = update.callback_query
    await query.answer("Preparing Excel...")
    rows = await queries.get_pattern_stats_for_export()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Patterns"
    ws.append([
        "Pattern", "Total Trades", "Wins", "Losses",
        "Win%", "W/L Ratio", "Deployed USDC", "Net PnL", "ROI%", "Last Seen",
    ])
    for r in rows:
        ws.append([
            r["pattern"], r["total_trades"], r["wins"], r["losses"],
            r["win_pct"],
            r["wl_ratio"] if r["wl_ratio"] != float("inf") else "inf",
            r["total_deployed"], r["net_pnl"], r["roi_pct"], r["last_seen"],
        ])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    await query.message.reply_document(
        document=buf,
        filename="pattern_performance.xlsx",
        caption="\U0001f4e5 Per-pattern stats export (Excel)",
    )


# ---------------------------------------------------------------------------
# Callback query router
# ---------------------------------------------------------------------------

@auth_check
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data

    if data == "cmd_menu":
        await query.answer()
        text = "\U0001f916 <b>AutoPoly Menu</b>\n\nSelect an option:"
        await _safe_edit(query, text, reply_markup=main_menu())

    elif data == "cmd_status":
        await cmd_status(update, context)

    elif data == "cmd_signals":
        await _render_signals(update, limit=None, active="all")

    elif data == "cmd_trades":
        await _render_trades(update, limit=None, active="all")

    elif data == "cmd_settings":
        await cmd_settings(update, context)

    elif data == "cmd_help":
        await cmd_help(update, context)

    elif data == "cmd_redeem":
        await cmd_redeem(update, context)

    elif data == "cmd_redemptions":
        await cmd_redemptions(update, context)

    # Signal filters
    elif data == "signals_10":
        await _render_signals(update, limit=10, active="10")
    elif data == "signals_50":
        await _render_signals(update, limit=50, active="50")
    elif data == "signals_all":
        await _render_signals(update, limit=None, active="all")

    # Trade filters
    elif data == "trades_10":
        await _render_trades(update, limit=10, active="10")
    elif data == "trades_50":
        await _render_trades(update, limit=50, active="50")
    elif data == "trades_all":
        await _render_trades(update, limit=None, active="all")

    # Settings toggles
    elif data == "toggle_autotrade":
        current = await queries.is_autotrade_enabled()
        await queries.set_setting("autotrade_enabled", "false" if current else "true")
        new_state = "OFF" if current else "ON"
        await query.answer(f"AutoTrade {new_state}")
        await cmd_settings(update, context)

    elif data == "toggle_auto_redeem":
        current = await queries.is_auto_redeem_enabled()
        await queries.set_setting("auto_redeem_enabled", "false" if current else "true")
        new_state = "ON" if not current else "OFF"
        await query.answer(f"Auto-Redeem {new_state}")
        await cmd_settings(update, context)

    elif data == "toggle_trade_mode":
        current_mode = await queries.get_trade_mode()
        new_mode = "pct" if current_mode == "fixed" else "fixed"
        await queries.set_setting("trade_mode", new_mode)
        await query.answer(f"Trade mode switched to {new_mode.upper()}")
        await cmd_settings(update, context)

    elif data == "change_amount":
        await query.answer()
        trade_mode = await queries.get_trade_mode()
        trade_pct = await queries.get_trade_pct()
        trade_amount = await queries.get_trade_amount()
        if trade_mode == "pct":
            await _safe_edit(
                query,
                f"\U0001f522 <b>Set Trade Percentage</b>\n\n"
                f"Current: <b>{trade_pct:.1f}%</b>\n\n"
                "Type the percentage to use per trade (e.g. <code>5</code> for 5%).\n"
                "<i>Minimum trade is always $1.00 (Polymarket limit).</i>",
            )
            context.user_data["awaiting_trade_pct"] = True
        else:
            await _safe_edit(
                query,
                f"\U0001f4b5 <b>Set Trade Amount</b>\n\n"
                f"Current: <b>${trade_amount:.2f}</b>\n\n"
                "Type the new amount in USDC (e.g. <code>2.50</code>):",
            )
            context.user_data["awaiting_amount"] = True

    elif data == "download_csv":
        await cmd_download_csv(update, context)

    elif data == "download_xlsx":
        await cmd_download_excel(update, context)

    # Redeem confirm / cancel
    elif data == "redeem_confirm":
        await _handle_redeem_confirm(update, context)

    elif data == "redeem_cancel":
        context.user_data.pop("redeem_preview", None)
        await query.answer("Cancelled.")
        await _safe_edit(
            query,
            "\u274c Redemption cancelled.",
            reply_markup=back_to_menu(),
        )

    elif data == "toggle_demo_trade":
        current = await queries.is_demo_trade_enabled()
        await queries.set_setting("demo_trade_enabled", "false" if current else "true")
        new_state = "OFF" if current else "ON"
        await query.answer(f"Demo Trade {new_state}")
        await cmd_settings(update, context)

    elif data == "set_demo_bankroll":
        await query.answer()
        demo_bankroll = await queries.get_demo_bankroll()
        await _safe_edit(
            query,
            f"\U0001f4b0 <b>Set Demo Bankroll</b>\n\n"
            f"Current balance: <b>${demo_bankroll:.2f}</b>\n\n"
            "Type the new bankroll amount in USDC (e.g. <code>500.00</code>):",
        )
        context.user_data["awaiting_demo_bankroll"] = True

    elif data == "reset_demo_bankroll":
        await queries.reset_demo_bankroll(1000.00)
        await query.answer("Demo bankroll reset to $1000.00")
        await cmd_settings(update, context)

    elif data == "toggle_invert_trades":
        current = await queries.is_invert_trades_enabled()
        await queries.set_setting("invert_trades_enabled", "false" if current else "true")
        new_state = "OFF" if current else "ON"
        await query.answer(f"Invert Trades {new_state}")
        await cmd_settings(update, context)

    elif data == "cmd_demo":
        await _render_demo_stats(update, active="all")

    elif data == "demo_10":
        await _render_demo_stats(update, limit=10, active="10")

    elif data == "demo_50":
        await _render_demo_stats(update, limit=50, active="50")

    elif data == "demo_all":
        await _render_demo_stats(update, limit=None, active="all")

    elif data == "cmd_patterns":
        await cmd_patterns(update, context)

    elif data == "download_pattern_xlsx":
        await cmd_download_pattern_excel(update, context)

    # ML Model submenu
    elif data == "cmd_ml":
        await query.answer()
        await _safe_edit(
            query,
            "\U0001f916 <b>ML Model</b>\n\nManage the ML inference model, compare versions, retrain, or adjust the signal threshold.",
            reply_markup=ml_menu(),
        )

    elif data == "ml_status":
        await query.answer()
        await cmd_model_status(update, context)

    elif data == "ml_compare":
        await query.answer()
        await cmd_model_compare(update, context)

    elif data == "ml_promote":
        await query.answer()
        await cmd_promote_model(update, context)

    elif data == "ml_retrain":
        await query.answer()
        await cmd_retrain(update, context)

    elif data == "ml_promote_anyway":
        # Answer immediately with cache_time to suppress Telegram re-fires on double-tap
        await query.answer("Promoting...", cache_time=10)
        from ml import model_store
        from core.strategies.ml_strategy import request_model_reload, set_model as _set_model
        if not model_store.has_model("candidate"):
            await query.message.reply_text(
                "&#x274C; No candidate model found. Please retrain first.\n"
                "<i>(If you already promoted, the candidate was consumed.)</i>",
                parse_mode="HTML",
                reply_markup=ml_menu(),
            )
        else:
            # Promote on disk first, then persist to DB so it survives container restarts
            model_store.promote_candidate()
            try:
                await model_store.promote_candidate_in_db()
            except Exception:
                log.exception("ml_promote_anyway: failed to persist promotion to DB (disk promote succeeded)")
            # Inject the newly promoted model into the strategy before requesting reload
            try:
                promoted = await model_store.load_model_from_db("current")
                if promoted:
                    _set_model(promoted)
            except Exception:
                log.exception("ml_promote_anyway: failed to preload promoted model into strategy (non-fatal)")
            request_model_reload()
            meta = model_store.load_metadata("current") or {}
            threshold = await queries.get_ml_threshold()
            text = format_model_status("current (force-promoted)", meta, threshold)
            await query.message.reply_text(
                f"{text}\n\n&#x26A0;&#xFE0F; Candidate promoted despite failing the 59% gate. "
                "Monitor live performance closely.",
                parse_mode="HTML",
                reply_markup=ml_menu(),
            )
            # If the DOWN side did not pass its own gate, ask the user whether to
            # enable it anyway via the override keyboard.
            if not meta.get("down_enabled", True):
                down_val_wr = meta.get("down_val_wr", 0.0)
                down_test_wr = meta.get("down_test_wr", 0.0)
                down_thr = meta.get("down_threshold", "n/a")
                await query.message.reply_text(
                    f"\u2139\ufe0f <b>DOWN signal did not pass the 59\u202f% gate</b>\n\n"
                    f"Val win-rate: <b>{down_val_wr:.1%}</b>  |  "
                    f"Test win-rate: <b>{down_test_wr:.1%}</b>  |  "
                    f"Threshold: <b>{down_thr}</b>\n\n"
                    "Do you want to enable the DOWN signal anyway, or keep it disabled?",
                    parse_mode="HTML",
                    reply_markup=down_override_keyboard(),
                )

    elif data == "ml_discard_candidate":
        await query.answer()
        # User chose to discard the blocked candidate
        from ml import model_store
        if model_store.has_model("candidate"):
            model_store.delete_model("candidate")
        await _safe_edit(
            query,
            "\U0001f5d1 <b>Candidate discarded.</b>\n\n"
            "The blocked candidate has been removed. "
            "The current production model is unchanged.",
            reply_markup=ml_menu(),
        )

    elif data == "ml_down_override_anyway":
        await query.answer("DOWN signal enabled.", cache_time=10)
        from ml import model_store
        # Patch the on-disk metadata so ml_strategy picks up down_override=True
        model_store.patch_metadata("current", {"down_override": True})
        # Also patch DB metadata so the flag survives container restarts
        try:
            import aiosqlite
            import config as cfg
            import json as _json
            async with aiosqlite.connect(cfg.DB_PATH) as _db:
                _cur = await _db.execute(
                    "SELECT metadata FROM model_blobs WHERE slot = ?", ("current",)
                )
                _row = await _cur.fetchone()
                if _row:
                    _meta = _json.loads(_row[0])
                    _meta["down_override"] = True
                    await _db.execute(
                        "UPDATE model_blobs SET metadata = ?, updated_at = CURRENT_TIMESTAMP "
                        "WHERE slot = ?",
                        (_json.dumps(_meta), "current"),
                    )
                    await _db.commit()
        except Exception:
            log.exception("ml_down_override_anyway: failed to patch DB metadata (non-fatal)")
        await query.message.reply_text(
            "\u2705 <b>DOWN signal override enabled.</b>\n\n"
            "The DOWN model will fire on the next signal check even though it did not "
            "pass the 59\u202f% gate. Monitor performance closely.",
            parse_mode="HTML",
            reply_markup=ml_menu(),
        )

    elif data == "ml_down_override_skip":
        await query.answer("DOWN signal kept disabled.")
        from ml import model_store
        # Explicitly mark down_override=False so intent is clear in metadata
        model_store.patch_metadata("current", {"down_override": False})
        await query.message.reply_text(
            "\u274c <b>DOWN signal remains disabled.</b>\n\n"
            "The UP model is live. The DOWN side will not fire until a retrain "
            "produces a candidate that passes its own 59\u202f% gate.",
            parse_mode="HTML",
            reply_markup=ml_menu(),
        )

    elif data == "ml_set_threshold":
        await query.answer()
        threshold = await queries.get_ml_threshold()
        await _safe_edit(
            query,
            f"\u2699\ufe0f <b>Set ML Threshold</b>\n\nCurrent threshold: <b>{threshold:.3f}</b>\n\n"
            "Type the new threshold value. Any numeric value is accepted.\n"
            "Example: <code>0.56</code>",
        )
        context.user_data["awaiting_ml_threshold"] = True

    elif data == "ml_set_down_threshold":
        await query.answer()
        threshold = await queries.get_ml_down_threshold()
        await _safe_edit(
            query,
            f"\u2699\ufe0f <b>Set ML DOWN Threshold</b>\n\nCurrent DOWN threshold: <b>{threshold:.3f}</b>\n\n"
            "Type the new DOWN threshold value. Any numeric value is accepted.\n"
            "Example: <code>0.55</code>",
        )
        context.user_data["awaiting_ml_down_threshold"] = True

    else:
        await query.answer("Unknown action")


async def _handle_redeem_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Execute redemptions after user confirmed the dry-run preview."""
    from core.redeemer import redeem_position

    query = update.callback_query
    await query.answer("Executing redemptions...")

    # Retrieve and immediately clear the stored preview
    preview = context.user_data.pop("redeem_preview", None)

    if not preview:
        await _safe_edit(
            query,
            "\u274c <b>Nothing to redeem</b>\n\nThe preview has expired or no positions were found. "
            "Run /redeem again to rescan.",
            reply_markup=back_to_menu(),
        )
        return

    won_count  = sum(1 for p in preview if p.get("won"))
    lost_count = len(preview) - won_count
    await _safe_edit(
        query,
        f"\u23f3 <b>Executing {len(preview)} redemption(s) on-chain...</b>\n"
        f"Won: {won_count}  Lost (burn): {lost_count}\n\nThis may take up to 2 minutes.",
    )

    wallet = cfg.POLYMARKET_FUNDER_ADDRESS
    if not wallet:
        await _safe_edit(
            query,
            "\u274c POLYMARKET_FUNDER_ADDRESS not configured.",
            reply_markup=back_to_menu(),
        )
        return

    results: list[dict] = []
    for pos in preview:
        result = await redeem_position(pos["condition_id"])
        merged = {**pos, **result, "dry_run": False}
        results.append(merged)

        # Persist each result to DB immediately (even if failed)
        try:
            await queries.insert_redemption(
                condition_id=pos["condition_id"],
                outcome_index=pos["outcome_index"],
                size=pos["size"],
                title=pos.get("title"),
                tx_hash=result.get("tx_hash"),
                status="success" if result.get("success") else "failed",
                error=result.get("error"),
                gas_used=result.get("gas_used"),
                dry_run=False,
            )
        except Exception:
            log.exception("Failed to persist redemption record for condition=%s", pos.get("condition_id"))

    text = format_redeem_results(results)
    await _safe_edit(query, text, reply_markup=redeem_done_keyboard())

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # -- Trade percentage input ----------------------------------------------------
    if context.user_data.get("awaiting_trade_pct"):
        context.user_data["awaiting_trade_pct"] = False
        raw = update.message.text.strip().replace("%", "")
        try:
            pct = float(raw)
            if pct <= 0 or pct > 100:
                raise ValueError("out of range")
        except ValueError:
            await update.message.reply_text(
                "\u274c Invalid percentage. Please enter a number between 0.1 and 100 (e.g. <code>5</code>).",
                parse_mode="HTML",
            )
            return
        pct = round(pct, 2)
        await queries.set_setting("trade_pct", str(pct))
        await update.message.reply_text(
            f"\u2705 Trade percentage set to <b>{pct:.2f}%</b>\n"
            f"<i>Minimum trade is always $1.00 (Polymarket limit).</i>",
            parse_mode="HTML",
        )
        # Refresh settings panel
        autotrade = await queries.is_autotrade_enabled()
        auto_redeem = await queries.is_auto_redeem_enabled()
        trade_amount = await queries.get_trade_amount()
        trade_mode = await queries.get_trade_mode()
        demo_trade = await queries.is_demo_trade_enabled()
        demo_bankroll = await queries.get_demo_bankroll()
        invert_trades = await queries.is_invert_trades_enabled()
        kb = settings_keyboard(autotrade, trade_amount, auto_redeem, demo_trade, demo_bankroll, trade_mode, pct, invert_trades)
        await update.message.reply_text(
            f"\u2699\ufe0f <b>Settings</b>\nAutoTrade: {'ON' if autotrade else 'OFF'}  |  Mode: {'PCT' if trade_mode == 'pct' else 'FIXED'} {pct}%  |  Demo: {'ON' if demo_trade else 'OFF'}",
            reply_markup=kb,
            parse_mode="HTML",
        )
        return

    # -- Demo bankroll input -------------------------------------------------------
    if context.user_data.get("awaiting_demo_bankroll"):
        context.user_data["awaiting_demo_bankroll"] = False
        raw = update.message.text.strip().replace("$", "")
        try:
            amount = float(raw)
            if amount < 0:
                raise ValueError("negative")
        except ValueError:
            await update.message.reply_text(
                "\u274c Invalid amount. Please enter a non-negative number (e.g. 500.00)."
            )
            return
        amount = round(amount, 2)
        await queries.set_demo_bankroll(amount)
        await update.message.reply_text(
            f"\u2705 Demo bankroll set to <b>${amount:.2f}</b>",
            parse_mode="HTML",
        )
        # Refresh settings panel
        autotrade = await queries.is_autotrade_enabled()
        auto_redeem = await queries.is_auto_redeem_enabled()
        trade_mode = await queries.get_trade_mode()
        trade_pct = await queries.get_trade_pct()
        demo_trade = await queries.is_demo_trade_enabled()
        trade_amount = await queries.get_trade_amount()
        invert_trades = await queries.is_invert_trades_enabled()
        kb = settings_keyboard(autotrade, trade_amount, auto_redeem, demo_trade, amount, trade_mode, trade_pct, invert_trades)
        mode_summary = f"{trade_pct:.1f}%" if trade_mode == "pct" else f"${trade_amount:.2f}"
        await update.message.reply_text(
            f"\u2699\ufe0f <b>Settings</b>\nAutoTrade: {'ON' if autotrade else 'OFF'}  |  Mode: {mode_summary}  |  Demo: {'ON' if demo_trade else 'OFF'}",
            reply_markup=kb,
            parse_mode="HTML",
        )
        return

    # -- ML threshold input -------------------------------------------------------
    if context.user_data.get("awaiting_ml_threshold"):
        context.user_data["awaiting_ml_threshold"] = False
        raw = update.message.text.strip()
        try:
            threshold = _parse_threshold_input(raw)
        except ValueError:
            await update.message.reply_text(
                "\u274c Invalid value. Enter any numeric threshold (e.g. <code>0.56</code>).",
                parse_mode="HTML",
            )
            return
        await queries.set_ml_threshold(threshold)
        await update.message.reply_text(
            f"\u2705 ML threshold set to <b>{threshold:.3f}</b>. Active on next signal check.",
            parse_mode="HTML",
            reply_markup=ml_menu(),
        )
        return

    # -- ML DOWN threshold input ---------------------------------------------------
    if context.user_data.get("awaiting_ml_down_threshold"):
        context.user_data["awaiting_ml_down_threshold"] = False
        raw = update.message.text.strip()
        try:
            threshold = _parse_threshold_input(raw)
        except ValueError:
            await update.message.reply_text(
                "\u274c Invalid value. Enter any numeric DOWN threshold (e.g. <code>0.55</code>).",
                parse_mode="HTML",
            )
            return
        await queries.set_ml_down_threshold(threshold)
        await update.message.reply_text(
            f"\u2705 ML DOWN threshold set to <b>{threshold:.3f}</b>. Active on next signal check.",
            parse_mode="HTML",
            reply_markup=ml_menu(),
        )
        return

    # -- Trade amount input --------------------------------------------------------
    if not context.user_data.get("awaiting_amount"):
        return

    context.user_data["awaiting_amount"] = False
    raw = update.message.text.strip().replace("$", "")
    try:
        amount = float(raw)
        if amount <= 0:
            raise ValueError("non-positive")
    except ValueError:
        await update.message.reply_text(
            "\u274c Invalid amount. Please enter a positive number (e.g. 2.50)."
        )
        return

    amount = round(amount, 2)
    await queries.set_setting("trade_amount_usdc", str(amount))
    await update.message.reply_text(
        f"\u2705 Trade amount updated to <b>${amount:.2f}</b>",
        parse_mode="HTML",
    )
    # Show settings panel again
    autotrade = await queries.is_autotrade_enabled()
    auto_redeem = await queries.is_auto_redeem_enabled()
    trade_mode = await queries.get_trade_mode()
    trade_pct = await queries.get_trade_pct()
    demo_trade = await queries.is_demo_trade_enabled()
    demo_bankroll = await queries.get_demo_bankroll()
    invert_trades = await queries.is_invert_trades_enabled()
    mode_summary = f"{trade_pct:.1f}%" if trade_mode == "pct" else f"${amount:.2f}"
    kb = settings_keyboard(autotrade, amount, auto_redeem, demo_trade, demo_bankroll, trade_mode, trade_pct, invert_trades)
    await update.message.reply_text(
        f"\u2699\ufe0f <b>Settings</b>\nAutoTrade: {'ON' if autotrade else 'OFF'}  |  Mode: {mode_summary}  |  Demo: {'ON' if demo_trade else 'OFF'}",
        reply_markup=kb,
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# /demo — Demo trade performance dashboard
# ---------------------------------------------------------------------------

async def _render_demo_stats(update: Update, limit: int | None = None, active: str = "all") -> None:
    from bot.keyboards import demo_filter_row
    stats = await queries.get_demo_trade_stats(limit=limit)
    bankroll = await queries.get_demo_bankroll()
    label = {"10": "Last 10", "50": "Last 50", "all": "All Time"}[active]
    text = format_demo_stats(stats, bankroll, label)
    recent = await queries.get_recent_demo_trades(10)
    text += format_demo_recent_trades(recent)
    kb = demo_filter_row(active)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


@auth_check
async def cmd_demo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _render_demo_stats(update, limit=None, active="all")


# ---------------------------------------------------------------------------
# Register all handlers
# ---------------------------------------------------------------------------

def register(application) -> None:
    """Attach all command and callback handlers to the Telegram Application."""
    application.add_handler(CommandHandler("start",       cmd_start))
    application.add_handler(CommandHandler("status",      cmd_status))
    application.add_handler(CommandHandler("signals",     cmd_signals))
    application.add_handler(CommandHandler("trades",      cmd_trades))
    application.add_handler(CommandHandler("settings",    cmd_settings))
    application.add_handler(CommandHandler("help",        cmd_help))
    application.add_handler(CommandHandler("redeem",      cmd_redeem))
    application.add_handler(CommandHandler("redemptions", cmd_redemptions))
    application.add_handler(CommandHandler("demo",        cmd_demo))
    application.add_handler(CommandHandler("patterns",    cmd_patterns))
    # ML model management commands
    application.add_handler(CommandHandler("set_threshold",      cmd_set_threshold))
    application.add_handler(CommandHandler("set_down_threshold", cmd_set_down_threshold))
    application.add_handler(CommandHandler("model_status",   cmd_model_status))
    application.add_handler(CommandHandler("model_compare",  cmd_model_compare))
    application.add_handler(CommandHandler("promote_model",  cmd_promote_model))
    application.add_handler(CommandHandler("retrain",        cmd_retrain))
    application.add_handler(CallbackQueryHandler(callback_router))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    async def _error_handler(update, context):
        import traceback
        err_text = "".join(traceback.format_exception(type(context.error), context.error, context.error.__traceback__))
        log.error("Unhandled Telegram error:\n%s", err_text)
        try:
            if cfg.TELEGRAM_CHAT_ID:
                short = err_text[-800:] if len(err_text) > 800 else err_text
                await context.bot.send_message(
                    chat_id=int(cfg.TELEGRAM_CHAT_ID),
                    text=f"&#x26A0;&#xFE0F; <b>Unhandled Bot Error</b>\n<pre>{_html.escape(short)}</pre>",
                    parse_mode="HTML",
                )
        except Exception:
            log.exception("Failed to send error notification to Telegram")

    application.add_error_handler(_error_handler)


# ---------------------------------------------------------------------------
# ML model management commands
# ---------------------------------------------------------------------------

@auth_check
async def cmd_set_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set ML inference threshold. Usage: /set_threshold 0.56"""
    if not context.args:
        await update.message.reply_text(
            "Usage: /set_threshold &lt;value&gt;\nExample: /set_threshold 0.56\nAny numeric value is accepted.",
            parse_mode="HTML",
        )
        return
    try:
        threshold = _parse_threshold_input(context.args[0])
    except (ValueError, IndexError):
        await update.message.reply_text(
            "Invalid value. Example: /set_threshold 0.56\nEnter any numeric threshold.",
            parse_mode="HTML",
        )
        return
    await queries.set_ml_threshold(threshold)
    await update.message.reply_text(
        format_set_threshold(threshold),
        parse_mode="HTML",
    )


@auth_check
async def cmd_set_down_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set ML DOWN inference threshold. Usage: /set_down_threshold 0.55"""
    if not context.args:
        await update.message.reply_text(
            "Usage: /set_down_threshold &lt;value&gt;\nExample: /set_down_threshold 0.55\nAny numeric value is accepted.",
            parse_mode="HTML",
        )
        return
    try:
        threshold = _parse_threshold_input(context.args[0])
    except (ValueError, IndexError):
        await update.message.reply_text(
            "Invalid value. Example: /set_down_threshold 0.55\nEnter any numeric DOWN threshold.",
            parse_mode="HTML",
        )
        return
    await queries.set_ml_down_threshold(threshold)
    await update.message.reply_text(
        format_set_down_threshold(threshold),
        parse_mode="HTML",
    )


@auth_check
async def cmd_model_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current ML model status."""
    from ml import model_store
    if update.callback_query:
        await update.callback_query.answer()
        send = update.callback_query.message.reply_text
    else:
        send = update.message.reply_text
    meta = model_store.load_metadata("current")
    if meta is None:
        await send("No model trained yet. Use /retrain to train one.", parse_mode="HTML")
        return
    threshold = await queries.get_ml_threshold()
    text = format_model_status("current", meta, threshold)
    await send(text, parse_mode="HTML", reply_markup=back_to_menu())


@auth_check
async def cmd_model_compare(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Compare current vs candidate model."""
    from ml import model_store
    if update.callback_query:
        await update.callback_query.answer()
        send = update.callback_query.message.reply_text
    else:
        send = update.message.reply_text
    current_meta = model_store.load_metadata("current")
    candidate_meta = model_store.load_metadata("candidate")
    if current_meta is None:
        await send("No current model. Use /retrain to train one.", parse_mode="HTML")
        return
    if candidate_meta is None:
        await send("No candidate model. Use /retrain to generate a candidate.", parse_mode="HTML")
        return
    text = format_model_compare(current_meta, candidate_meta)
    await send(text, parse_mode="HTML", reply_markup=back_to_menu())


@auth_check
async def cmd_promote_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Promote candidate model to current (disk + DB)."""
    from ml import model_store
    from core.strategies.ml_strategy import request_model_reload
    if update.callback_query:
        await update.callback_query.answer()
        send = update.callback_query.message.reply_text
    else:
        send = update.message.reply_text
    if not model_store.has_model("candidate"):
        await send(
            "No candidate model to promote. Use /retrain first.\n"
            "<i>(If you already promoted, the candidate was consumed.)</i>",
            parse_mode="HTML",
        )
        return
    # Promote on disk first, then persist to DB so it survives container restarts
    model_store.promote_candidate()
    try:
        await model_store.promote_candidate_in_db()
    except Exception:
        log.exception("cmd_promote_model: failed to persist promotion to DB (disk promote succeeded)")
    # Inject the newly promoted model into the strategy before requesting reload,
    # so _load_model() picks it up from memory rather than falling back to disk
    # (disk is ephemeral on Railway and may not have the model after a redeploy).
    try:
        promoted = await model_store.load_model_from_db("current")
        if promoted:
            from core.strategies.ml_strategy import set_model
            set_model(promoted)
    except Exception:
        log.exception("cmd_promote_model: failed to preload promoted model into strategy (non-fatal)")
    request_model_reload()
    meta = model_store.load_metadata("current")
    threshold = await queries.get_ml_threshold()
    text = format_model_status("current (promoted)", meta or {}, threshold)
    await send(
        f"{text}\n\nCandidate promoted to current. Model will reload on next signal check.",
        parse_mode="HTML",
    )


@auth_check
async def cmd_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Trigger async background retraining."""
    if update.callback_query:
        await update.callback_query.answer()
        send = update.callback_query.message.reply_text
    else:
        send = update.message.reply_text
    await send("Retraining started... estimated time ~5-8 min. I'll notify you when done.", parse_mode="HTML")
    asyncio.create_task(_retrain_background(context.application, cfg.TELEGRAM_CHAT_ID))


async def _retrain_background(application, chat_id) -> None:
    """Background retraining: fetch data, build features, train, save to DB, report."""
    import asyncio as _asyncio
    import html as _html
    from ml import data_fetcher, features as feat_eng, trainer, model_store

    async def notify(text: str, reply_markup=None) -> None:
        try:
            await application.bot.send_message(
                chat_id=int(chat_id),
                text=text,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
        except Exception as e:
            log.warning("_retrain_background: failed to send notification: %s", e)

    try:
        loop = _asyncio.get_event_loop()
        log.info("Retrain: fetching 9 months of training inputs...")
        data = await _asyncio.wait_for(
            loop.run_in_executor(None, lambda: data_fetcher.fetch_all(months=9)),
            timeout=1500,
        )
        log.info("Retrain: building features...")
        df_feat = await _asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: feat_eng.build_features(
                    data["df5"], data["df15"], data["df1h"], data["cvd"]
                )
            ),
            timeout=1500,
        )
        log.info("Retrain: training LightGBM (candidate slot)...")
        result = await _asyncio.wait_for(
            loop.run_in_executor(None, lambda: trainer.train(df_feat, slot="candidate")),
            timeout=1500,
        )
        meta = model_store.load_metadata("candidate") or {}
        threshold = result.get("threshold", 0.535)
        # down_threshold now comes from the real DOWN sweep in trainer.py,
        # NOT the arithmetic complement. Falls back to complement only for
        # models trained before Option B was implemented.
        down_threshold = result.get("down_threshold", round(1.0 - threshold, 4))
        down_enabled   = result.get("down_enabled", False)
        down_val_wr    = result.get("down_val_wr", 0.0)
        down_test_wr   = result.get("down_test_metrics", {}).get("wr", 0.0)

        # Persist up/down thresholds to DB
        try:
            await queries.set_ml_threshold(threshold)
            await queries.set_ml_down_threshold(down_threshold)
        except Exception as thr_exc:
            log.warning("Retrain: failed to persist thresholds to DB: %s", thr_exc)

        # Persist trained candidate model to DB
        try:
            await model_store.save_model_to_db(result["model"], "candidate", meta)
        except Exception as db_exc:
            log.warning("Retrain: failed to save candidate to DB: %s", db_exc)

        if result.get("blocked"):
            # UP side failed the 59% gate — saved to candidate, user decides
            log.warning(
                "Retrain: candidate blocked by deployment gate. "
                "val_wr=%.4f test_wr=%.4f threshold=%.3f | "
                "down_enabled=%s down_val_wr=%.4f down_test_wr=%.4f down_threshold=%.3f",
                result.get("val_wr", 0),
                result.get("test_metrics", {}).get("wr", 0),
                threshold,
                down_enabled, down_val_wr, down_test_wr, down_threshold,
            )
            main_msg, risk_msg = format_retrain_blocked(meta, threshold)
            await notify(main_msg, reply_markup=retrain_blocked_keyboard())
            if risk_msg:
                await notify(risk_msg)
        else:
            log.info(
                "Retrain complete. val_wr=%.4f test_wr=%.4f | "
                "down_enabled=%s down_val_wr=%.4f down_test_wr=%.4f down_threshold=%.3f",
                result.get("val_wr", 0),
                result.get("test_metrics", {}).get("wr", 0),
                down_enabled, down_val_wr, down_test_wr, down_threshold,
            )
            main_msg, risk_msg = format_retrain_complete(meta, threshold)
            await notify(main_msg)
            if risk_msg:
                await notify(risk_msg)
            # Auto-promote: UP passed the gate — promote candidate to current now.
            try:
                from ml import model_store as _ms
                from core.strategies.ml_strategy import (
                    request_model_reload as _req_reload,
                    set_model as _set_model,
                )
                _ms.promote_candidate()
                try:
                    await _ms.promote_candidate_in_db()
                except Exception:
                    log.exception("_retrain_background: failed to persist auto-promotion to DB (disk promote succeeded)")
                try:
                    _promoted = await _ms.load_model_from_db("current")
                    if _promoted:
                        _set_model(_promoted)
                except Exception:
                    log.exception("_retrain_background: failed to preload auto-promoted model into strategy (non-fatal)")
                _req_reload()
                log.info("_retrain_background: candidate auto-promoted to current")
            except Exception:
                log.exception("_retrain_background: auto-promotion failed (non-fatal)")
            # If DOWN side did not pass its own gate, ask user whether to override.
            if not down_enabled:
                from bot.keyboards import down_override_keyboard as _down_kb
                await notify(
                    f"\u2139\ufe0f <b>DOWN signal did not pass the 59\u202f% gate</b>\n\n"
                    f"Val win-rate: <b>{down_val_wr:.1%}</b>  |  "
                    f"Test win-rate: <b>{down_test_wr:.1%}</b>  |  "
                    f"Threshold: <b>{down_threshold}</b>\n\n"
                    "The UP model has been auto-promoted. "
                    "Do you want to enable the DOWN signal anyway, or keep it disabled?",
                    reply_markup=_down_kb(),
                )

    except _asyncio.TimeoutError:
        log.error("Retrain background task timed out after 25 min")
        await notify("Retrain timed out after 25 min. Try again or check Railway logs.")
    except Exception as exc:
        log.exception("Retrain background task failed: %s", exc)
        safe_msg = _html.escape(str(exc))
        await notify(f"\u274c <b>Retrain failed</b>\n\n{safe_msg}\n\nCheck Railway logs for details.")
