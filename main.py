"""AutoPoly entry point — init DB, start Telegram bot + scheduler."""

from __future__ import annotations

import asyncio
import logging
import sys

from telegram import BotCommand
from telegram.ext import Application

import config as cfg
from bot import handlers
from core.scheduler import recover_unresolved, start_scheduler
from db.models import init_db, migrate_db
from polymarket.client import PolymarketClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
log = logging.getLogger("autopoly")


def _validate_config() -> bool:
    ok = True
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                 "POLYMARKET_PRIVATE_KEY", "POLYMARKET_FUNDER_ADDRESS"):
        if not getattr(cfg, name, None):
            log.error("Missing required env var: %s", name)
            ok = False

    # Warn (but don't block startup) if POLYGON_RPC_URL is missing
    if not getattr(cfg, "POLYGON_RPC_URL", None):
        log.warning(
            "POLYGON_RPC_URL is not set — on-chain redemptions will fail. "
            "Set this env var to enable /redeem and auto-redeem."
        )

    return ok


def main() -> None:
    if not _validate_config():
        sys.exit(1)

    # 1. Init DB synchronously first (create tables, seed defaults)
    asyncio.run(init_db())
    log.info("Database initialised at %s", cfg.DB_PATH)

    # One-time cleanup: remove bad redemption records from sig-type-2 bug
    _deleted = asyncio.run(cleanup_bad_redemptions())
    if _deleted:
        log.info("Startup cleanup: deleted %d bad redemption record(s)", _deleted)
    else:
        log.debug("Startup cleanup: no bad redemption records found (already clean)")

    # 2. Init Polymarket client (synchronous — derives creds)
    poly_client: PolymarketClient | None = None
    try:
        poly_client = PolymarketClient(cfg)
        log.info("Polymarket client ready.")
    except Exception:
        log.exception("Failed to initialise Polymarket client — trading disabled")

    # 3. Build Telegram Application with post_init hook
    async def post_init(application: Application) -> None:
        """Called after the Application is initialised but before polling starts.

        Order matters:
          1. start_scheduler() first — creates and starts the AsyncIOScheduler,
             setting the module-level SCHEDULER global.
          2. recover_unresolved() second — reads unresolved signals from the DB
             and schedules immediate resolution jobs onto SCHEDULER.  If called
             before start_scheduler(), SCHEDULER is still None and every
             recovery job is silently dropped (the `if SCHEDULER is not None`
             guard in recover_unresolved fires False for every signal).
        """
        start_scheduler(application, poly_client)
        await recover_unresolved()

        # Register bot commands so they appear in Telegram menu
        await application.bot.set_my_commands([
            BotCommand("status",      "Portfolio overview & bot health"),
            BotCommand("signals",     "Recent trading signals"),
            BotCommand("trades",      "Open & recent trades"),
            BotCommand("redeem",      "Scan & redeem winning positions"),
            BotCommand("redemptions", "Redemption history"),
            BotCommand("settings",    "View/adjust bot settings"),
            BotCommand("demo",        "Demo trading & virtual bankroll"),
            BotCommand("help",        "Show available commands"),
        ])

    application = (
        Application.builder()
        .token(cfg.TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # 4. Register handlers & inject poly client
    handlers.set_poly_client(poly_client)
    handlers.set_start_time()
    handlers.register(application)

    # 5. Run polling (blocks until stopped)
    #    Both scheduler and bot run in the same async event loop.
    log.info("Starting Telegram bot polling...")
    try:
        application.run_polling(drop_pending_updates=True)
    except Exception as exc:
        log.critical("Telegram bot exited: %s", exc)


if __name__ == "__main__":
    main()
