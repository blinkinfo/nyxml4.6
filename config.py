"""AutoPoly configuration — loads from environment variables with sensible defaults."""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------
POLYMARKET_PRIVATE_KEY: str | None = os.getenv("POLYMARKET_PRIVATE_KEY")
POLYMARKET_FUNDER_ADDRESS: str | None = os.getenv("POLYMARKET_FUNDER_ADDRESS")
POLYMARKET_SIGNATURE_TYPE: int = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "2"))

CLOB_HOST: str = "https://clob.polymarket.com"
GAMMA_API_HOST: str = "https://gamma-api.polymarket.com"
CHAIN_ID: int = 137

# ---------------------------------------------------------------------------
# Polygon RPC (required for on-chain redemptions via web3.py)
# ---------------------------------------------------------------------------
POLYGON_RPC_URL: str = os.getenv(
    "POLYGON_RPC_URL",
    "https://polygon-rpc.com",  # public fallback — consider Alchemy/Infura for production
)

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str | None = os.getenv("TELEGRAM_CHAT_ID")

# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------
TRADE_AMOUNT_USDC: float = float(os.getenv("TRADE_AMOUNT_USDC", "1.0"))
TRADE_MODE: str = os.getenv("TRADE_MODE", "fixed")  # "fixed" or "pct"
TRADE_PCT: float = float(os.getenv("TRADE_PCT", "5.0"))

# ---------------------------------------------------------------------------
# FOK Retry Settings
# ---------------------------------------------------------------------------
FOK_MAX_RETRIES: int = int(os.getenv("FOK_MAX_RETRIES", "3"))
FOK_RETRY_DELAY_BASE: float = float(os.getenv("FOK_RETRY_DELAY_BASE", "2.0"))
FOK_RETRY_DELAY_MAX: float = float(os.getenv("FOK_RETRY_DELAY_MAX", "5.0"))
FOK_SLOT_CUTOFF_SECONDS: int = int(os.getenv("FOK_SLOT_CUTOFF_SECONDS", "30"))

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: str = os.getenv("DB_PATH", "autopoly.db")

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
STRATEGY_NAME: str = os.getenv("STRATEGY_NAME", "ml")  # active strategy
COINBASE_CANDLE_URL: str = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

# ---------------------------------------------------------------------------
# ML Strategy
# ---------------------------------------------------------------------------
ML_MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models")
ML_DEFAULT_THRESHOLD: float = 0.535  # Blueprint Section 9: recommended threshold (64.22% WR @ ~50 trades/day)

# Payout ratio: fraction of stake returned as profit on a winning trade.
# Polymarket binary markets typically pay $0.85 profit per $1 risked.
# Used by the threshold sweep to select the threshold that maximises
# real dollar EV/day rather than a payout-agnostic proxy metric.
# Override via ML_PAYOUT_RATIO env var if your broker payout changes.
ML_PAYOUT_RATIO: float = float(os.getenv("ML_PAYOUT_RATIO", "0.85"))

# ---------------------------------------------------------------------------
# Inference Debug Logging
# ---------------------------------------------------------------------------
# Path where per-inference structured JSONL logs are written.
# Each line is a self-contained JSON record for one check_signal() call.
# Outcome (win/loss) is back-filled when the slot resolves.
# Set to "" or "none" to disable (not recommended in production).
INFERENCE_LOG_PATH: str = os.getenv("INFERENCE_LOG_PATH", "inference_log.jsonl")

# ---------------------------------------------------------------------------
# Signal Timing
# ---------------------------------------------------------------------------
SIGNAL_LEAD_TIME: int = 85  # seconds before slot end to check signal

# ---------------------------------------------------------------------------
# Auto-Redeem
# ---------------------------------------------------------------------------
# Scheduler interval (minutes) between automatic redemption scans.
AUTO_REDEEM_INTERVAL_MINUTES: int = int(os.getenv("AUTO_REDEEM_INTERVAL_MINUTES", "5"))

# ---------------------------------------------------------------------------
# Hour Filter
# ---------------------------------------------------------------------------
# UTC hours during which trading is blocked (0-23). Comma-separated in env.
# Default: 3 and 17 (03:XX UTC and 17:XX UTC).
# To change without redeploying, set BLOCKED_TRADE_HOURS_UTC=3,17 in env vars.
BLOCKED_TRADE_HOURS_UTC: frozenset[int] = frozenset(
    int(h.strip())
    for h in os.getenv("BLOCKED_TRADE_HOURS_UTC", "3,17").split(",")
    if h.strip().isdigit()
)
