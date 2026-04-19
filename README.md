# 🤖 AutoPoly — Polymarket BTC Binary Options Trading Bot

AutoPoly is a fully automated trading bot for **Polymarket BTC 5-minute Up/Down binary options markets**. It combines a pluggable prediction strategy engine (ML or pattern-matching), an APScheduler-driven execution loop, Fill-or-Kill CLOB order placement, on-chain position redemption, and a full Telegram bot interface — all backed by a local SQLite database.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Directory Structure](#-directory-structure)
- [Trading Modes](#-trading-modes)
- [Strategy Engine](#-strategy-engine)
- [ML Model Pipeline](#-ml-model-pipeline)
- [Scheduler Loop](#-scheduler-loop)
- [Bot Commands](#-telegram-bot-commands)
- [Configuration](#-configuration--environment-variables)
- [Database Schema](#-database-schema)
- [Setup & Deployment](#-setup--deployment)
- [Key Constants & Thresholds](#-key-constants--thresholds)

---

## 🔍 Overview

AutoPoly monitors Polymarket's BTC-USDT 5-minute binary markets around the clock. Every slot (`:00`, `:05`, ... `:55`), it:

1. Fires a signal check **85 seconds before slot end** (T−85s)
2. Runs the active prediction strategy (ML or Pattern)
3. If a strong enough signal is found, places a **Fill-or-Kill market order** on the Polymarket CLOB
4. Records the trade in SQLite
5. After slot expiry, **resolves** the outcome and **redeems** any winning positions on-chain
6. Sends Telegram notifications for every key event

The system supports both **demo** (simulated P&L) and **live** (real USDC on-chain) trading.

---

## 🏗 Architecture

```
main.py
  ├── config.py                   # All env-var configuration
  ├── db/
  │   ├── models.py               # SQLite schema init, migrations
  │   └── queries.py              # Async aiosqlite query layer
  ├── polymarket/
  │   ├── client.py               # PolymarketClient (CLOB auth + order placement)
  │   └── markets.py              # Market discovery, slot timing, price fetch
  ├── core/
  │   ├── scheduler.py            # APScheduler loop — fires every 5-min slot
  │   ├── strategy.py             # Strategy orchestrator (selects ML or Pattern)
  │   ├── trader.py               # FOK order execution with retry + fill verification
  │   ├── trade_manager.py        # Pre-trade gate (demo balance check, live passthrough)
  │   ├── resolver.py             # Outcome resolution after slot expiry
  │   ├── redeemer.py             # On-chain redemption of winning positions
  │   ├── pending_queue.py        # Persistent queue for trades awaiting resolution
  │   └── strategies/
  │       ├── base.py             # BaseStrategy ABC — defines signal dict contract
  │       ├── ml_strategy.py      # LightGBM model inference
  │       └── pattern_strategy.py # Historical candlestick pattern matching
  ├── ml/
  │   ├── features.py             # 40-feature engineering (zero lookahead bias)
  │   ├── trainer.py              # LightGBM training with walk-forward validation
  │   ├── data_fetcher.py         # MEXC OHLCV + CVD data fetching (ccxt + REST)
  │   └── model_store.py          # Model serialization / staging / promotion
  └── bot/
      ├── handlers.py             # All Telegram command handlers
      └── formatters.py           # HTML message formatters for bot replies
```

---

## 📁 Directory Structure

```
nyxmlopp/
├── main.py                 # Entry point — starts bot + scheduler
├── config.py               # Environment config (all env vars read here)
├── requirements.txt        # Python dependencies
├── .env.example            # Template for required environment variables
├── autopoly.db             # SQLite database (created on first run)
├── models/                 # Trained LightGBM model files (.lgb)
│   ├── candidate.lgb       # Staged model (awaiting promotion)
│   └── production.lgb      # Active production model
├── core/                   # Trading engine
├── ml/                     # ML pipeline
├── bot/                    # Telegram interface
├── db/                     # Database layer
└── polymarket/             # Polymarket CLOB integration
```

---

## 💹 Trading Modes

### Live vs Demo

| Mode | Description |
|------|-------------|
| **Live** | Real USDC trades on Polymarket CLOB via `py-clob-client` |
| **Demo** | Simulated trades — no real orders placed, P&L tracked in DB |

Toggle via the `/demo` Telegram command or the `DEMO_MODE` environment variable.

### Demo Payout Structure

In demo mode, simulated P&L now uses the same simplified fee-adjusted binary model as live reporting:
- Estimate gross shares from entry price: `gross_shares = stake / entry_price`
- Estimate crypto taker fee: `fee = stake × 0.072 × (1 - entry_price)`
- **Win:** `gross_shares - stake - fee`
- **Lose:** `−1.0 × stake`

### Fixed vs PCT Stake Sizing

| Mode | Env Var | Default | Description |
|------|---------|---------|-------------|
| **fixed** | `TRADE_MODE=fixed` | — | Fixed USDC amount per trade (set via `FIXED_STAKE`) |
| **pct** | `TRADE_MODE=pct` | `TRADE_PCT=5.0` | Percentage of available balance per trade |

In PCT demo mode, the stake is sized from the demo bankroll first, then resolved with the same fee-adjusted binary P&L model described above.

---

## 🧠 Strategy Engine

The active strategy is selected by `STRATEGY_NAME` env var (default: `"ml"`). Strategies are loaded via `core/strategy.py` which instantiates the correct class. All strategies implement `BaseStrategy` and return a standardized signal dict.

### Signal Dictionary Schema

```python
{
    "action":      "up" | "down" | "skip",  # trade direction or skip
    "confidence":  float,                    # model probability (0–1)
    "market_id":   str,                      # Polymarket market token ID
    "slot_end_ts": int,                      # Unix timestamp of slot expiry
    "reason":      str,                      # human-readable explanation
    "price":       float | None,             # best ask price at signal time
    "pattern":     str | None,               # matched pattern string (pattern strategy)
    "win_rate":    float | None,             # historical WR of matched pattern
}
```

### ML Strategy (`core/strategies/ml_strategy.py`)

- Loads a pre-trained LightGBM model from `models/production.lgb`
- Fetches live MEXC OHLCV + Gate.io CVD data via `ml/data_fetcher.py`
- Builds 40 features via `ml/features.py`
- Produces a binary probability; fires a trade if confidence ≥ threshold
- **Separate thresholds for UP and DOWN** directions (`ML_UP_THRESHOLD`, `ML_DOWN_THRESHOLD`)
- Default threshold: **0.535** (targets ~64% WR at ~50 trades/day)
- Excludes the current in-progress candle from inference (uses N−1 candles to predict N+1)
- Supports hot-reload of the production model without restart (`/promote_model`)

### Pattern Strategy (`core/strategies/pattern_strategy.py`)

- Fetches the most recently **closed** 5-min BTC-USD candles from Coinbase Advanced Trade API
- Always drops the most-recent API candle (may be in-progress) for safety
- Reads the last **up to 10** fully closed candles and converts them to direction strings (`U`/`D`)
- Builds pattern strings at depth 10, then 9 (longest-first greedy matching)
- Looks up the pattern in the `patterns` DB table
- If matched: trades the predicted direction for the N+1 candle
- If no match: skips the slot

---

## 🔬 ML Model Pipeline

### Data Sources

Training uses MEXC OHLCV plus Gate.io taker-flow context:
- **OHLCV:** MEXC spot BTC/USDT 5-minute candles via `ccxt`
- **CVD (Cumulative Volume Delta):** Gate.io futures taker-flow data
- **Multi-timeframe context:** 15-minute and 1-hour aggregations computed from 5-min data

### Feature Engineering (`ml/features.py`)

40 features — zero lookahead bias (all features use `shift(k>=1)`):

| Group | Features |
|-------|----------|
| **Candle shape (7)** | `body_ratio_n1`, `body_ratio_n2`, `body_ratio_n3`, `upper_wick_n1`, `upper_wick_n2`, `lower_wick_n1`, `lower_wick_n2` |
| **Volume (2)** | `volume_ratio_n1`, `volume_ratio_n2` |
| **15m context (3)** | `body_ratio_15m`, `dir_15m`, `volume_ratio_15m` |
| **1h context (3)** | `body_ratio_1h`, `dir_1h`, `ema9_slope_1h` |
| **OHLCV pressure (5)** | `body_ratio`, `upper_wick_ratio`, `lower_wick_ratio`, `vol_zscore`, `vol_trend` |
| **Time-of-day (4)** | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` |
| **Volatility regime (2)** | `atr_percentile_24h`, `vol_regime` |
| **Momentum (4)** | `rsi14`, `candle_streak`, `price_in_range`, `ema_cross_5m` |
| **Structure (3)** | `body_vs_range5`, `range_expansion`, `vwap_dist_20` |
| **Gate.io CVD and OI flow (7)** | `cvd_ratio`, `cvd_delta_norm`, `cvd_cumulative_5`, `cvd_cumulative_20`, `cvd_trend_slope`, `cvd_divergence`, `oi_change_5bar` |

Target label: `1` = next candle closes UP, `0` = DOWN (`shift(-1)` — only used for training labels, never as a feature).

### Training (`ml/trainer.py`)

**Time-series split (no shuffling):**
- Train: first 60% of data
- Validation: 60%–75%
- Test (hold-out): 75%–100%

**Walk-forward cross-validation:**
- Initial fold: 60% train+val block, 8% test
- Each subsequent fold adds `WF_STEP_PCT` more data
- Train/val split within each fold: 80/20

**LightGBM Hyperparameters:**

```python
{
    "objective":         "binary",
    "metric":            "binary_logloss",
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 50,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_jobs":            1,
}
NUM_BOOST_ROUND       = 1000
EARLY_STOPPING_ROUNDS = 50
```

**Threshold sweep:** performed on validation set only (never on test set). The optimal threshold maximises win rate subject to a minimum trade count.

### Deployment Gate

> **Blueprint Rule 10:** A retrained model is only promoted to production if its **test-set win rate ≥ 58%**. If the gate is not met, `DeploymentBlockedError` is raised and the candidate model is NOT promoted.

Model lifecycle:
1. `/retrain` → trains a new model → saved as `models/candidate.lgb`
2. `/promote_model` → validates gate → copies candidate → `models/production.lgb`
3. ML strategy hot-reloads the new production model without restart

### Model Store (`ml/model_store.py`)

- `save_candidate(model)` — serializes to `models/candidate.lgb`
- `promote_candidate()` — validates WR gate, moves candidate → production
- `load_production()` — deserializes active model
- Metadata (WR, threshold, training date) stored in `model_metadata` DB table

---

## ⏰ Scheduler Loop

`core/scheduler.py` uses APScheduler (`AsyncIOScheduler`) to sync to 5-minute slot boundaries.

**Timing:**
- Slots align to `:00`, `:05`, `:10`, ..., `:55` of every hour (UTC)
- Signal check fires at **T−85s** = `slot_start + 215s` (85 seconds before slot end)
- At T−85s the current candle is still open; strategies use only confirmed-closed candles

**Per-slot flow:**

```
T−85s  →  strategy.get_signal()
            ↓ (if action != "skip")
         trade_manager.maybe_trade()   # demo balance check / live passthrough
            ↓
         trader.execute_trade()        # FOK CLOB order, retry with backoff
            ↓ (fill confirmed)
         queries.insert_trade()        # persist to DB
            ↓
         Telegram notification
            ↓
         pending_queue.add()           # queued for resolution

Slot expiry → resolver.resolve_pending()   # check Polymarket outcome API
                ↓ (win)
              redeemer.redeem()            # on-chain USDC redemption
```

**Additional scheduled jobs:**
- Periodic redemption scan (interval configurable via `REDEMPTION_SCAN_INTERVAL`)
- Startup recovery: `recover_unresolved()` re-queues any trades that were open at last shutdown

---

## 💬 Telegram Bot Commands

All commands are restricted to the configured `TELEGRAM_CHAT_ID`.

### General

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and system status |
| `/help` | Full command reference |
| `/status` | Current bot status: mode, strategy, balance, active markets |
| `/settings` | Show all current configuration settings |

### Trading

| Command | Description |
|---------|-------------|
| `/demo` | Toggle demo mode on/off |
| `/trades [N]` | Show last N trades (default 10) with P&L |
| `/signals` | Show the last signal generated |
| `/redeem` | Manually trigger on-chain redemption of winning positions |
| `/redemptions` | Show recent redemption history |

### ML Model

| Command | Description |
|---------|-------------|
| `/retrain` | Trigger a full model retrain (fetches fresh MEXC data, trains, saves as candidate) |
| `/promote_model` | Promote the candidate model to production (validates ≥58% WR gate) |
| `/model_status` | Show current model metadata: WR, threshold, training date, feature count |

### Strategy & Thresholds

| Command | Description |
|---------|-------------|
| `/set_threshold <value>` | Set the UP signal confidence threshold (e.g. `0.55`) |
| `/set_down_threshold <value>` | Set the DOWN signal confidence threshold separately |
| `/patterns` | List all stored candlestick patterns with win rates and trade counts |
| `/pattern_add` | Add a new pattern manually |
| `/pattern_delete` | Delete a stored pattern |

---

## ⚙️ Configuration — Environment Variables

Copy `.env.example` to `.env` and fill in all required values.

### Required

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Authorized Telegram chat ID (only this chat can control the bot) |
| `POLY_API_KEY` | Polymarket CLOB API key |
| `POLY_API_SECRET` | Polymarket CLOB API secret |
| `POLY_API_PASSPHRASE` | Polymarket CLOB API passphrase |
| `POLY_PRIVATE_KEY` | EVM wallet private key for on-chain redemption |

### Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `DEMO_MODE` | `true` | `true` = demo (simulated), `false` = live real trades |
| `TRADE_MODE` | `fixed` | `fixed` = fixed stake per trade, `pct` = percentage of balance |
| `FIXED_STAKE` | — | USDC amount per trade (used when `TRADE_MODE=fixed`) |
| `TRADE_PCT` | `5.0` | Percentage of balance per trade (used when `TRADE_MODE=pct`) |
| `DEMO_BALANCE` | — | Starting virtual balance for demo mode |

### Strategy & Model

| Variable | Default | Description |
|----------|---------|-------------|
| `STRATEGY_NAME` | `ml` | Active strategy: `ml` or `pattern` |
| `ML_DEFAULT_THRESHOLD` | `0.535` | Default ML confidence threshold (both UP and DOWN) |
| `ML_MODEL_DIR` | `./models` | Directory for LightGBM model files |

### System

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `autopoly.db` | Path to SQLite database file |
| `SIGNAL_LEAD_TIME` | `85` | Seconds before slot end to fire signal check |
| `REDEMPTION_SCAN_INTERVAL` | — | Minutes between automatic redemption scans |

---

## 🗄️ Database Schema

AutoPoly uses **aiosqlite** (async SQLite). Schema is auto-created and migrated on startup via `db/models.py`.

### Tables

| Table | Purpose |
|-------|---------|
| `trades` | All trade records: slot, direction, stake, fill price, outcome, P&L |
| `signals` | Signal history: strategy, action, confidence, reason, market ID |
| `patterns` | Pattern strategy table: pattern string, predicted direction, WR, trade count |
| `model_metadata` | ML model versioning: WR, threshold, training timestamp, feature list |
| `redemptions` | On-chain redemption records: trade ID, amount, tx hash, status |
| `pending_queue` | Trades awaiting outcome resolution |
| `demo_balance` | Virtual balance state for demo mode |
| `settings` | Persistent runtime settings (thresholds, mode flags) overriding env vars |

---

## 🚀 Setup & Deployment

### Prerequisites

- Python 3.11+
- MEXC account (public API, no auth needed for data)
- Polymarket account with CLOB API credentials
- EVM wallet funded with USDC on Polygon for live trading
- Telegram bot token and authorized chat ID

### Installation

```bash
git clone https://github.com/blinkinfo/nyxmlopp.git
cd nyxmlopp
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

### Running

```bash
python main.py
```

On startup, `main.py`:
1. Validates all required env vars
2. Initializes and migrates the SQLite DB
3. Loads the production LightGBM model (if strategy is `ml`)
4. Recovers any unresolved trades from the previous session
5. Starts the Telegram bot (polling)
6. Starts the APScheduler loop

### Initial Model Training

Before running with the ML strategy, train an initial model:

```
/retrain    → trains and saves candidate model
/promote_model → promotes to production (only if test WR >= 58%)
```

### Deployment on Railway / VPS

- Set all env vars in Railway's environment panel
- Use `DB_PATH=/data/autopoly.db` for persistent disk
- Use `ML_MODEL_DIR=/data/models` for model persistence across deploys
- The scheduler is single-process; no worker coordination needed

---

## 📐 Key Constants & Thresholds

| Constant | Value | Location | Description |
|----------|-------|----------|-------------|
| `SIGNAL_LEAD_TIME` | 85s | `config.py` | Fire signal check 85s before slot end |
| `ML_DEFAULT_THRESHOLD` | 0.535 | `config.py` | Default ML confidence threshold |
| `WR_GATE` | 58% | `trainer.py` | Min test-set WR required to promote a new model |
| `NUM_BOOST_ROUND` | 1000 | `trainer.py` | Max LightGBM boosting rounds |
| `EARLY_STOPPING_ROUNDS` | 50 | `trainer.py` | Early stopping patience |
| `WF_INITIAL_PCT` | 0.60 | `trainer.py` | Walk-forward initial train+val fraction |
| Train/Val/Test split | 60/15/25% | `trainer.py` | Time-series split (no shuffle) |
| Pattern depth | 10, 9 | `pattern_strategy.py` | Longest-first greedy pattern match |
| Candle exclusion | N (current) | live inference | Drop current open candle; predict N+1 from N−1 |
| Slot duration | 300s (5 min) | `polymarket/markets.py` | Market slot length |
| FOK retry backoff | exponential | `trader.py` | Price refresh on each attempt |
| LGBM `num_leaves` | 63 | `trainer.py` | Tree complexity |
| LGBM `learning_rate` | 0.05 | `trainer.py` | Step size |
| Feature count | 40 | `ml/features.py` | Total engineered features |
| Demo win payout | fee-adjusted binary payout | `core/scheduler.py` | Simplified Polymarket-style win P&L based on entry price and crypto taker fee |
| Demo loss | 1.0× stake | `core/scheduler.py` | Full stake loss on loss |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `python-telegram-bot>=20.0` | Telegram async bot framework |
| `apscheduler>=3.10.0` | Async cron scheduler |
| `py-clob-client>=0.34.0` | Polymarket CLOB API client |
| `aiosqlite>=0.19.0` | Async SQLite |
| `lightgbm>=4.3.0` | LightGBM ML model |
| `scikit-learn>=1.4.0` | Metrics (precision, recall, F1) |
| `ccxt>=4.3.0` | MEXC OHLCV data fetching |
| `httpx>=0.25.0` | Async HTTP client (MEXC CVD API) |
| `pandas>=2.1.0` | Data manipulation |
| `numpy>=1.26.0` | Numerical operations |
| `web3>=6.0.0` | On-chain redemption |
| `python-dotenv>=1.0.0` | `.env` file loading |
| `openpyxl>=3.1.0` | Excel export (trade reports) |

---

## ⚠️ Disclaimer

This software is for educational and research purposes. Binary options trading involves substantial risk of loss. Past performance of any ML model or pattern strategy is not indicative of future results. Use at your own risk.
