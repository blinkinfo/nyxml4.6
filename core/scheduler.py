"""APScheduler loop — syncs to 5-min slot boundaries, fires signals, trades, resolves, redeems."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta


from apscheduler.schedulers.asyncio import AsyncIOScheduler

import config as cfg
from core import strategy, trader, resolver
from core import pending_queue
from db import queries
from ml import inference_logger
import html as _html
from polymarket.markets import SLOT_DURATION

log = logging.getLogger(__name__)

SCHEDULER: AsyncIOScheduler | None = None

# Holds references so Telegram bot can send messages
_tg_app = None
_poly_client = None


def _next_check_time() -> datetime:
    """Calculate the next T-85s check time (slot_end - SIGNAL_LEAD_TIME).

    Slots align to :00, :05, :10 ... :55 of each hour.
    T-85s = slot_end - 85 seconds = slot_start + 300 - 85 = slot_start + 215 seconds.
    """
    now = datetime.now(timezone.utc)
    epoch = int(now.timestamp())
    current_slot_start = epoch - (epoch % SLOT_DURATION)
    check_epoch = current_slot_start + SLOT_DURATION - cfg.SIGNAL_LEAD_TIME

    if check_epoch <= epoch:
        # Already past this slot's check time — schedule for next slot
        check_epoch += SLOT_DURATION

    return datetime.fromtimestamp(check_epoch, tz=timezone.utc)


async def _send_telegram(text: str) -> None:
    """Send a message to the configured Telegram chat."""
    if _tg_app is None or cfg.TELEGRAM_CHAT_ID is None:
        return
    try:
        await _tg_app.bot.send_message(
            chat_id=int(cfg.TELEGRAM_CHAT_ID),
            text=text,
            parse_mode="HTML",
        )
    except Exception:
        log.exception("Failed to send Telegram message")


def _calculate_resolution_pnl(amount_usdc: float, entry_price: float, is_win: bool) -> float:
    """Return fee-adjusted PnL for a resolved binary trade."""
    if not is_win:
        return -amount_usdc

    gross_shares = amount_usdc / entry_price
    fee_usdc = gross_shares * 0.072 * entry_price * (1.0 - entry_price)
    return gross_shares - amount_usdc - fee_usdc


async def _resolve_and_notify(signal_id: int, slug: str, side: str, entry_price: float,
                               slot_start: str, slot_end: str, trade_id: int | None,
                               amount_usdc: float | None,
                               is_demo: bool = False) -> None:
    """Poll for resolution, update DB, notify Telegram.

    Always sends a signal result message.
    Also sends a separate trade/demo result message if a trade was placed.
    """
    from bot.formatters import (
        format_signal_resolution,
        format_trade_resolution,
        format_demo_resolution,
    )

    winner = await resolver.resolve_slot(slug)
    if winner is None:
        log.warning(
            "Could not resolve slot %s after all attempts — adding to persistent retry queue",
            slug,
        )
        await pending_queue.add_pending(
            signal_id=signal_id,
            slug=slug,
            side=side,
            entry_price=entry_price,
            slot_start=slot_start,
            slot_end=slot_end,
            trade_id=trade_id,
            amount_usdc=amount_usdc,
            is_demo=is_demo,
        )
        return

    is_win = winner == side
    await queries.resolve_signal(signal_id, winner, is_win)

    # Back-fill the outcome into the inference debug log (non-fatal if it fails)
    inference_logger.log_outcome(slug, winner=winner, is_win=is_win)

    # Extract HH:MM from slot_start/slot_end full strings
    s_start = slot_start.split(" ")[-1] if " " in slot_start else slot_start
    s_end = slot_end.split(" ")[-1] if " " in slot_end else slot_end

    # Always send signal result (pure market outcome, no P&L)
    await _send_telegram(format_signal_resolution(
        is_win=is_win,
        side=side,
        entry_price=entry_price,
        slot_start_str=s_start,
        slot_end_str=s_end,
    ))

    # Send trade result only if a trade was actually placed
    if trade_id is not None and amount_usdc is not None:
        pnl = round(_calculate_resolution_pnl(amount_usdc, entry_price, is_win), 4)
        await queries.resolve_trade(trade_id, winner, is_win, pnl)

        if is_demo:
            # Credit demo bankroll with the resolved payout on win; loss was already deducted at entry.
            if is_win:
                new_bankroll = await queries.adjust_demo_bankroll(amount_usdc + pnl)
            else:
                new_bankroll = await queries.get_demo_bankroll()
            await _send_telegram(format_demo_resolution(
                is_win=is_win,
                side=side,
                entry_price=entry_price,
                slot_start_str=s_start,
                slot_end_str=s_end,
                pnl=pnl,
                new_bankroll=new_bankroll,
            ))
        else:
            await _send_telegram(format_trade_resolution(
                is_win=is_win,
                side=side,
                entry_price=entry_price,
                slot_start_str=s_start,
                slot_end_str=s_end,
                pnl=pnl,
            ))


async def _reconcile_pending() -> None:
    """Retry resolution for all slots in the persistent pending queue.

    Called every 5 minutes by the scheduler. Tries check_resolution() once
    per pending slot. Resolved slots are removed from the queue and reported
    to Telegram. Unresolved slots remain for the next cycle.
    """
    from bot.formatters import format_signal_resolution, format_trade_resolution, format_demo_resolution

    pending = await pending_queue.list_pending()
    if not pending:
        return

    log.info("Reconciler: checking %d pending slot(s)...", len(pending))

    for item in pending:
        signal_id = item["signal_id"]
        slug = item["slug"]
        side = item["side"]
        entry_price = item["entry_price"]
        slot_start = item["slot_start"]
        slot_end = item["slot_end"]
        trade_id = item.get("trade_id")
        amount_usdc = item.get("amount_usdc")
        is_demo = item.get("is_demo", False)

        try:
            winner, resolved = await resolver.check_resolution(slug)
        except Exception:
            log.exception("Reconciler: error checking slug=%s", slug)
            continue

        if not resolved:
            log.debug("Reconciler: slot %s still unresolved — will retry next cycle", slug)
            continue

        # Resolved — update DB
        is_win = winner == side
        await queries.resolve_signal(signal_id, winner, is_win)

        # Back-fill the outcome into the inference debug log (non-fatal if it fails)
        inference_logger.log_outcome(slug, winner=winner, is_win=is_win)

        pnl: float | None = None
        if trade_id is not None and amount_usdc is not None:
            pnl = round(_calculate_resolution_pnl(amount_usdc, entry_price, is_win), 4)
            await queries.resolve_trade(trade_id, winner, is_win, pnl)

            # Credit demo bankroll with the resolved payout on win (mirrors _resolve_and_notify).
            if is_demo and pnl is not None:
                if is_win:
                    await queries.adjust_demo_bankroll(amount_usdc + pnl)

        # Remove from queue
        await pending_queue.remove_pending(signal_id)

        # Notify Telegram — signal result always, trade/demo result only if trade was placed
        s_start = slot_start.split(" ")[-1] if " " in slot_start else slot_start
        s_end = slot_end.split(" ")[-1] if " " in slot_end else slot_end
        await _send_telegram(format_signal_resolution(
            is_win=is_win,
            side=side,
            entry_price=entry_price,
            slot_start_str=s_start,
            slot_end_str=s_end,
        ))
        if trade_id is not None and pnl is not None:
            if is_demo:
                new_bankroll = await queries.get_demo_bankroll()
                await _send_telegram(format_demo_resolution(
                    is_win=is_win,
                    side=side,
                    entry_price=entry_price,
                    slot_start_str=s_start,
                    slot_end_str=s_end,
                    pnl=pnl,
                    new_bankroll=new_bankroll,
                ))
            else:
                await _send_telegram(format_trade_resolution(
                    is_win=is_win,
                    side=side,
                    entry_price=entry_price,
                    slot_start_str=s_start,
                    slot_end_str=s_end,
                    pnl=pnl,
                ))
        log.info(
            "Reconciler: resolved signal %d — winner=%s is_win=%s",
            signal_id, winner, is_win,
        )


async def _auto_redeem_job() -> None:
    """Scheduled auto-redeem scan — runs every AUTO_REDEEM_INTERVAL_MINUTES minutes."""
    from core.redeemer import scan_and_redeem
    from bot.formatters import format_auto_redeem_notification, format_error_alert

    # Guard: only run if auto-redeem is enabled
    enabled = await queries.is_auto_redeem_enabled()
    if not enabled:
        log.debug("auto_redeem_job: disabled — skipping")
        return

    wallet = cfg.POLYMARKET_FUNDER_ADDRESS
    if not wallet:
        log.warning("auto_redeem_job: POLYMARKET_FUNDER_ADDRESS not set — skipping")
        return

    if not cfg.POLYGON_RPC_URL:
        log.warning("auto_redeem_job: POLYGON_RPC_URL not set — skipping")
        return

    log.info("auto_redeem_job: scanning wallet %s...", wallet)

    try:
        results = await scan_and_redeem(wallet, dry_run=False)
    except Exception as exc:
        import traceback as _tb
        tb_str = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
        log.error("auto_redeem_job: scan_and_redeem raised an exception:\n%s", tb_str)
        await _send_telegram(
            format_error_alert("auto_redeem_job", f"{type(exc).__name__}: {exc}", tb_str)
        )
        return

    if not results:
        log.info("auto_redeem_job: no redeemable positions found")
        return

    # Deduplicate: skip conditions already successfully redeemed
    new_results: list[dict] = []
    for r in results:
        cid = r.get("condition_id", "")
        if await queries.redemption_already_recorded(cid):
            log.debug("auto_redeem_job: condition %s already redeemed — skipping", cid)
            continue
        new_results.append(r)

    if not new_results:
        log.info("auto_redeem_job: all positions already redeemed")
        return

    # Persist to DB
    for r in new_results:
        try:
            is_success = bool(r.get("success"))
            is_verified = is_success and bool(r.get("verified_zero_balance"))
            if is_verified:
                db_status = "verified"
            elif is_success:
                db_status = "success"
            else:
                db_status = "failed"
            await queries.insert_redemption(
                condition_id=r["condition_id"],
                outcome_index=r["outcome_index"],
                size=r["size"],
                title=r.get("title"),
                tx_hash=r.get("tx_hash"),
                status=db_status,
                error=r.get("error"),
                gas_used=r.get("gas_used"),
                dry_run=False,
                verified=is_verified,
            )
        except Exception:
            log.exception(
                "auto_redeem_job: failed to persist redemption for condition=%s",
                r.get("condition_id"),
            )

    # Notify Telegram — send individual alerts for failed redemptions
    for r in new_results:
        if not r.get("success"):
            err = r.get("error") or "unknown error"
            tb = r.get("error_detail", "")
            detail = tb[-600:] if tb else err[:200]
            title = (r.get("title") or r.get("condition_id", "?"))[:55]
            await _send_telegram(
                f"&#x26A0;&#xFE0F; <b>Redemption Failed</b>\n"
                f"{_html.escape(title)}\n<pre>{_html.escape(detail)}</pre>"
            )

    msg = format_auto_redeem_notification(new_results)
    await _send_telegram(msg)
    log.info(
        "auto_redeem_job: processed %d redemption(s) (%d success, %d failed)",
        len(new_results),
        sum(1 for r in new_results if r.get("success")),
        sum(1 for r in new_results if not r.get("success")),
    )


async def _check_and_trade() -> None:
    """Core loop body — called at T-85s for each slot.

    _schedule_next() is always called in the finally block, so the chain
    can never break — even if an unhandled exception escapes any inner code.
    """
    try:
        # --- Hour filter ---
        now_utc = datetime.now(timezone.utc)
        if now_utc.hour in cfg.BLOCKED_TRADE_HOURS_UTC:
            log.info(
                "Hour filter: skipping slot at %s UTC (blocked hours: %s)",
                now_utc.strftime("%H:%M"),
                sorted(cfg.BLOCKED_TRADE_HOURS_UTC),
            )
            return   # finally will still call _schedule_next()

        from bot.formatters import (
            format_signal,
            format_skip,
            format_ml_signal,
            format_ml_skip,
            format_trade_filled,
            format_trade_unmatched,
            format_trade_aborted,
            format_trade_retrying,
        )
        from core.trade_manager import TradeManager

        # 1. Check signal (delegated to active strategy via orchestrator)
        signal = await strategy.check_signal()
        if signal is None:
            log.error("Strategy returned None (hard error) — skipping this slot")
            await _send_telegram("\u274c Strategy error — could not fetch prices. Skipping slot.")
            return

        slot_start_full = signal["slot_n1_start_full"]
        slot_end_full = signal["slot_n1_end_full"]
        slot_start_str = signal["slot_n1_start_str"]
        slot_end_str = signal["slot_n1_end_str"]
        slot_ts = signal["slot_n1_ts"]

        # 2. Log signal to DB
        if signal["skipped"]:
            signal_id = await queries.insert_signal(
                slot_start=slot_start_full,
                slot_end=slot_end_full,
                slot_timestamp=slot_ts,
                side=None,
                entry_price=None,
                opposite_price=None,
                skipped=True,
            )
            if "ml_p_up" in signal:
                msg = format_ml_skip(
                    slot_start_str=slot_start_str,
                    slot_end_str=slot_end_str,
                    ml_p_up=signal["ml_p_up"],
                    ml_p_down=signal["ml_p_down"],
                    ml_up_threshold=signal["ml_up_threshold"],
                    ml_down_threshold=signal["ml_down_threshold"],
                    ml_down_enabled=signal["ml_down_enabled"],
                )
            else:
                msg = format_skip(
                    slot_start_str=slot_start_str,
                    slot_end_str=slot_end_str,
                    reason=signal.get("reason", "No pattern match"),
                    pattern=signal.get("pattern"),
                )
            await _send_telegram(msg)
            return

        side = signal["side"]
        entry_price = signal["entry_price"]
        opposite_price = signal["opposite_price"]
        token_id = signal["token_id"]
        pattern = signal.get("pattern")

        # Invert trades: swap side, prices, and token_id to bet opposite the signal.
        invert_trades = await queries.is_invert_trades_enabled()
        if invert_trades:
            side = "Down" if side == "Up" else "Up"
            entry_price, opposite_price = opposite_price, entry_price
            token_id = signal.get("opposite_token_id", token_id)
        slug = signal.get("slot_n1_slug", f"btc-updown-5m-{slot_ts}")

        signal_id = await queries.insert_signal(
            slot_start=slot_start_full,
            slot_end=slot_end_full,
            slot_timestamp=slot_ts,
            side=side,
            entry_price=entry_price,
            opposite_price=opposite_price,
            skipped=False,
            pattern=pattern,
        )

        # 3. TradeManager passthrough (filters removed — always allowed)
        demo_trade_enabled = await queries.is_demo_trade_enabled()
        _filter_result = await TradeManager.check(
            signal_side=side,
            current_slot_ts=slot_ts,
            is_demo=demo_trade_enabled,
        )
        # filter_result.allowed is always True — but keeping the check for future extensibility

        # 4. Check autotrade
        autotrade = await queries.is_autotrade_enabled()
        trade_amount, _amount_label = await queries.resolve_trade_amount(
            poly_client=_poly_client,
            is_demo=demo_trade_enabled,
        )

        # 5. Send signal notification
        if "ml_p_up" in signal:
            msg = format_ml_signal(
                side=side,
                entry_price=entry_price,
                slot_start_str=slot_start_str,
                slot_end_str=slot_end_str,
                ml_p_up=signal["ml_p_up"],
                ml_p_down=signal["ml_p_down"],
                ml_up_threshold=signal["ml_up_threshold"],
                ml_down_threshold=signal["ml_down_threshold"],
                ml_down_enabled=signal.get("ml_down_enabled", False),
            )
        else:
            msg = format_signal(
                side=side,
                entry_price=entry_price,
                slot_start_str=slot_start_str,
                slot_end_str=slot_end_str,
                pattern=pattern,
            )
        await _send_telegram(msg)

        # 6. Place trade if autotrade on (with robust retry logic)
        trade_id: int | None = None
        amount_usdc: float | None = None
        slot_label = f"{slot_start_str}-{slot_end_str}"

        if demo_trade_enabled:
            # -- Demo Trade Path --
            amount_usdc = round(trade_amount, 2)
            demo_bankroll = await queries.get_demo_bankroll()

            if demo_bankroll < amount_usdc:
                log.warning(
                    "Demo bankroll ($%.2f) insufficient for trade amount ($%.2f) — skipping demo trade",
                    demo_bankroll, amount_usdc,
                )
                msg = (
                    f"\U0001f9ea <b>[DEMO] Bankroll Insufficient</b>\n"
                    f"Bankroll: ${demo_bankroll:.2f}  |  Required: ${amount_usdc:.2f}\n"
                    f"Demo trade skipped. Top up via /settings."
                )
                await _send_telegram(msg)
            else:
                new_bankroll = await queries.adjust_demo_bankroll(-amount_usdc)

                trade_id = await queries.insert_trade(
                    signal_id=signal_id,
                    slot_start=slot_start_full,
                    slot_end=slot_end_full,
                    side=side,
                    entry_price=entry_price,
                    amount_usdc=amount_usdc,
                    status="filled",
                    is_demo=True,
                )
                log.info(
                    "Demo trade placed: signal=%d trade_id=%d side=%s amount=$%.2f bankroll=$%.2f",
                    signal_id, trade_id, side, amount_usdc, new_bankroll,
                )
                msg = (
                    f"\U0001f9ea <b>[DEMO] Trade Placed</b>\n"
                    f"{'\U0001f4c8' if side == 'Up' else '\U0001f4c9'} {side}  @${entry_price:.2f}  "
                    f"${amount_usdc:.2f}\n"
                    f"\U0001f4b0 Demo Bankroll: ${new_bankroll:.2f}"
                )
                await _send_telegram(msg)

        elif autotrade and _poly_client is not None and token_id:
            amount_usdc = round(trade_amount, 2)
            trade_id = await queries.insert_trade(
                signal_id=signal_id,
                slot_start=slot_start_full,
                slot_end=slot_end_full,
                side=side,
                entry_price=entry_price,
                amount_usdc=amount_usdc,
                status="pending",
            )

            # Compute slot end timestamp for time-fencing
            slot_end_ts = slot_ts + SLOT_DURATION

            # Wrap trader to inject retry notifications
            max_retries = cfg.FOK_MAX_RETRIES

            async def _place_with_notifications():
                """Thin wrapper: forwards retry-in-progress telegrams, then delegates
                to trader.place_fok_order_with_retry for the actual order logic."""
                sent_attempts: set[int] = set()

                async def _retry_watcher():
                    """Poll trade DB row; send a notification on each new attempt."""
                    import asyncio as _asyncio
                    for _ in range(max_retries * 10):  # generous upper bound
                        await _asyncio.sleep(0.8)
                        try:
                            row = await queries.get_active_trade_for_signal(signal_id)
                            if row is None:
                                continue
                            retry_count = row.get("retry_count", 0) or 0
                            status = row.get("status", "")
                            if status == "retrying" and retry_count not in sent_attempts:
                                sent_attempts.add(retry_count)
                                retry_msg = format_trade_retrying(
                                    side=side,
                                    slot_label=slot_label,
                                    attempt=retry_count + 1,
                                    max_attempts=max_retries,
                                    reason="FOK order not matched — retrying",
                                )
                                await _send_telegram(retry_msg)
                            if status in ("filled", "unmatched", "aborted", "duplicate_prevented"):
                                break
                        except Exception:
                            pass  # watcher is non-critical

                watcher_task = asyncio.create_task(_retry_watcher())

                result = await trader.place_fok_order_with_retry(
                    poly_client=_poly_client,
                    token_id=token_id,
                    amount_usdc=amount_usdc,
                    signal_id=signal_id,
                    trade_id=trade_id,
                    slot_end_ts=slot_end_ts,
                )

                watcher_task.cancel()
                try:
                    await watcher_task
                except asyncio.CancelledError:
                    pass

                return result

            result = await _place_with_notifications()

            trade_status = result["status"]
            attempts = result["attempts"]
            reason = result["reason"]
            order_id = result.get("order_id")

            if trade_status == "filled":
                log.info("Trade filled: order_id=%s (attempts=%d)", order_id, attempts)
                shares: float | None = result.get("shares")
                msg = format_trade_filled(
                    side=side,
                    slot_label=slot_label,
                    ask_price=entry_price,
                    amount_usdc=amount_usdc,
                    shares=shares,
                    order_id=order_id,
                    attempts=attempts,
                )
                await _send_telegram(msg)

            elif trade_status == "aborted":
                log.warning("Trade aborted: %s (attempts=%d)", reason, attempts)
                msg = format_trade_aborted(
                    side=side,
                    slot_label=slot_label,
                    reason=reason,
                )
                await _send_telegram(msg)
                trade_id = None  # don't resolve a non-filled trade

            else:
                # unmatched / failed
                log.warning("Trade %s: %s (attempts=%d)", trade_status, reason, attempts)
                msg = format_trade_unmatched(
                    side=side,
                    slot_label=slot_label,
                    attempts=attempts,
                    reason=reason,
                )
                await _send_telegram(msg)
                trade_id = None  # don't resolve a non-filled trade

        # 7. Schedule resolution after slot N+1 ends
        resolve_time = datetime.fromtimestamp(slot_ts + SLOT_DURATION + 30, tz=timezone.utc)
        if SCHEDULER is not None:
            SCHEDULER.add_job(
                _resolve_and_notify,
                trigger="date",
                run_date=resolve_time,
                kwargs={
                    "signal_id": signal_id,
                    "slug": slug,
                    "side": side,
                    "entry_price": entry_price,
                    "slot_start": slot_start_full,
                    "slot_end": slot_end_full,
                    "trade_id": trade_id,
                    "amount_usdc": amount_usdc,
                    "is_demo": demo_trade_enabled,
                },
                id=f"resolve_{signal_id}",
                replace_existing=True,
            )
            log.debug("Scheduled resolution for signal %d at %s", signal_id, resolve_time.isoformat())

    except Exception:
        log.exception("_check_and_trade: unhandled exception — rescheduling next check")
        await _send_telegram("\u274c Scheduler error in check_and_trade — see logs. Next check rescheduled.")

    finally:
        # Single authoritative reschedule — runs after every return, exception, and normal exit.
        _schedule_next()


def _schedule_next() -> None:
    """Add the next check_and_trade job to the scheduler."""
    if SCHEDULER is None:
        return
    next_time = _next_check_time()
    SCHEDULER.add_job(
        _check_and_trade,
        trigger="date",
        run_date=next_time,
        id="check_and_trade",
        replace_existing=True,
    )
    log.info("Next check: %s UTC", next_time.strftime("%H:%M:%S"))


async def recover_unresolved() -> None:
    """On startup, schedule resolution for any unresolved signals/trades."""
    signals = await queries.get_unresolved_signals()
    if not signals:
        log.debug("No unresolved signals to recover.")
    else:
        log.info("Recovering %d unresolved signal(s)...", len(signals))
        for sig in signals:
            slug = f"btc-updown-5m-{sig['slot_timestamp']}"
            trade = await queries.get_trade_by_signal(sig["id"])
            trade_id = trade["id"] if trade else None
            amount_usdc = trade["amount_usdc"] if trade else None

            resolve_time = datetime.now(timezone.utc) + timedelta(seconds=5)
            if SCHEDULER is not None:
                SCHEDULER.add_job(
                    _resolve_and_notify,
                    trigger="date",
                    run_date=resolve_time,
                    kwargs={
                        "signal_id": sig["id"],
                        "slug": slug,
                        "side": sig["side"],
                        "entry_price": sig["entry_price"],
                        "slot_start": sig["slot_start"],
                        "slot_end": sig["slot_end"],
                        "trade_id": trade_id,
                        "amount_usdc": amount_usdc,
                        "is_demo": bool(trade.get("is_demo", 0)) if trade else False,
                    },
                    id=f"recover_{sig['id']}",
                    replace_existing=True,
                )

    pending = await pending_queue.list_pending()
    if pending:
        log.info(
            "%d slot(s) remain in persistent retry queue — reconciler will handle them.",
            len(pending),
        )


async def _feature_drift_check_job() -> None:
    """Daily feature drift monitoring -- compares recent inference feature means
    to training distribution. Sends Telegram alert if drift is detected."""
    from ml.evaluator import check_feature_drift
    from ml import model_store, inference_logger

    log.info("feature_drift_check: starting daily drift check")

    meta = model_store.load_metadata("current")
    if meta is None:
        log.warning("feature_drift_check: no model metadata found -- skipping")
        return

    training_stats = meta.get("training_feature_stats")
    if not training_stats:
        log.warning(
            "feature_drift_check: no training_feature_stats in metadata -- "
            "skipping (retrain to enable drift monitoring)"
        )
        return

    log_path = inference_logger.get_log_path()
    if not log_path:
        log.warning("feature_drift_check: inference log path not configured -- skipping")
        return

    result = check_feature_drift(
        inference_log_path=log_path,
        training_feature_stats=training_stats,
        n_recent=500,
        z_threshold=2.0,
    )

    if result.get("error"):
        log.warning("feature_drift_check: check returned error: %s", result["error"])
        return

    n = result["records_analyzed"]
    drifted = result["drifted_features"]

    if not drifted:
        log.info("feature_drift_check: no drift detected (%d records analyzed)", n)
        return

    # Build alert message
    drift_lines = []
    for d in drifted[:10]:
        drift_lines.append(
            f"  <b>{d['feature']}</b>: live={d['live_mean']:.4f} "
            f"train={d['train_mean']:.4f} z={d['z_score']:+.2f}"
        )

    msg = (
        f"\u26a0\ufe0f <b>Feature Drift Detected</b>\n"
        f"\u2500" * 20 + "\n"
        f"\U0001f4ca Records analyzed: {n}\n"
        f"\u26a0\ufe0f Drifted features ({len(drifted)}):\n"
        + "\n".join(drift_lines) + "\n"
        + "\u2500" * 20 + "\n"
        f"\U0001f916 Model may be operating on out-of-distribution data.\n"
        f"Consider retraining with /retrain."
    )

    await _send_telegram(msg)
    log.warning(
        "feature_drift_check: drift alert sent -- %d features drifted: %s",
        len(drifted), [d["feature"] for d in drifted],
    )


def start_scheduler(tg_app, poly_client) -> AsyncIOScheduler:
    """Create, configure, and start the scheduler."""
    global SCHEDULER, _tg_app, _poly_client
    _tg_app = tg_app
    _poly_client = poly_client

    SCHEDULER = AsyncIOScheduler(timezone="UTC")
    SCHEDULER.start()

    # Reconciler: retry pending slots every 5 minutes
    SCHEDULER.add_job(
        _reconcile_pending,
        trigger="interval",
        minutes=5,
        id="reconcile_pending",
        replace_existing=True,
    )
    log.info("Reconciler job scheduled (every 5 minutes).")

    # Auto-redeem: scan for redeemable positions on a configurable interval
    redeem_interval = cfg.AUTO_REDEEM_INTERVAL_MINUTES
    SCHEDULER.add_job(
        _auto_redeem_job,
        trigger="interval",
        minutes=redeem_interval,
        id="auto_redeem",
        replace_existing=True,
    )
    log.info("Auto-redeem job scheduled (every %d minutes).", redeem_interval)

    # Schedule first signal check
    _schedule_next()

    # Daily feature drift monitoring at 06:00 UTC
    SCHEDULER.add_job(
        _feature_drift_check_job,
        trigger="cron",
        hour=6,
        minute=0,
        timezone="UTC",
        id="feature_drift_check",
        replace_existing=True,
    )
    log.info("Feature drift check job scheduled (daily at 06:00 UTC).")

    log.info("Scheduler started.")
    return SCHEDULER
