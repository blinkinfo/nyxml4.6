# 🤖 AutoPoly

AutoPoly is an automated Polymarket trading bot for short-duration BTC binary markets. It combines a Telegram control surface, a scheduled execution loop, a pluggable strategy layer, persistent SQLite state, optional live trading on the Polymarket CLOB, and an ML training pipeline for the model-driven strategy.

This README is based on the current codebase and is intended to reflect what the repository actually does today.

---

## ✨ What this project does

AutoPoly runs on a 5-minute trading cadence and manages the full lifecycle of a trade:

1. Fetch market and timing context for the current BTC slot
2. Ask the active strategy for a signal
3. Place a Fill-or-Kill order in live mode, or simulate the trade in demo mode
4. Store signals, trades, settings, and model metadata in SQLite
5. Resolve settled trades after expiry
6. Redeem eligible live winning positions on-chain
7. Report activity through a Telegram bot

The project supports two operating modes:
- **Live mode** for real Polymarket orders and redemptions
- **Demo mode** for simulated trading with tracked bankroll and PnL

---

## 🧱 Architecture at a glance

```text
main.py
├── config.py                  # Centralized environment-driven configuration
├── bot/                       # Telegram commands, keyboards, formatters, middleware
├── core/                      # Scheduler, execution, resolution, redemption, strategies
├── db/                        # SQLite schema and async query layer
├── polymarket/                # CLOB client and market discovery helpers
├── ml/                        # Feature engineering, training, evaluation, model storage
├── models/                    # Saved model artifacts and metadata JSON
├── scripts/                   # Utility scripts, including threshold reporting
└── tests/                     # ML and strategy-focused test coverage
```

### Runtime flow

```text
Telegram bot + APScheduler start
        ↓
Scheduler aligns to 5-minute slots
        ↓
Active strategy generates signal
        ↓
Trade manager validates trade sizing / mode
        ↓
Trader places live FOK order or records demo trade
        ↓
Signal + trade saved to SQLite
        ↓
Scheduler resolves matured trades
        ↓
Live winners become redeemable and can be redeemed
        ↓
Telegram bot reports status, trades, signals, settings, and model actions
```

---

## 📁 Repository structure

```text
.
├── .env.example
├── Procfile
├── README.md
├── config.py
├── main.py
├── reset_redemptions.py
├── run_training.py
├── bot/
│   ├── formatters.py
│   ├── handlers.py
│   ├── keyboards.py
│   └── middleware.py
├── core/
│   ├── pending_queue.py
│   ├── redeemer.py
│   ├── resolver.py
│   ├── scheduler.py
│   ├── strategy.py
│   ├── trade_manager.py
│   ├── trader.py
│   └── strategies/
│       ├── base.py
│       ├── ml_strategy.py
│       └── pattern_strategy.py
├── db/
│   ├── models.py
│   └── queries.py
├── ml/
│   ├── data_fetcher.py
│   ├── evaluator.py
│   ├── features.py
│   ├── inference_logger.py
│   ├── model_store.py
│   ├── probability.py
│   └── trainer.py
├── models/
│   ├── model_current.lgb
│   └── model_current_meta.json
├── polymarket/
│   ├── account.py
│   ├── client.py
│   └── markets.py
├── scripts/
│   └── model_threshold_report.py
└── tests/
    ├── test_ml_features.py
    └── test_ml_strategy.py
```

---

## 🚀 Entry points

### Application runtime

- `main.py` initializes the database, validates configuration, starts the Telegram bot, creates the Polymarket client in live mode, recovers unresolved work, and starts the scheduler.
- `Procfile` runs the app with:

```bash
python main.py
```

### Model training

- `run_training.py` is the training entrypoint for the ML pipeline.
- `scripts/model_threshold_report.py` generates a threshold-focused report for model evaluation.

### Maintenance utilities

- `reset_redemptions.py` is a utility for resetting redemption-related state.

---

## ⚙️ Core runtime components

### `main.py`

Responsible for startup orchestration:
- config validation
- SQLite initialization and migrations
- cleanup of bad redemption records
- Telegram bot command registration
- conditional Polymarket client initialization for live mode
- scheduler startup
- unresolved trade recovery

### `core/scheduler.py`

The scheduler is the heart of the runtime loop.

It is responsible for:
- aligning work to 5-minute market slots
- requesting signals from the active strategy
- dispatching trade execution
- resolving expired trades
- reconciling pending items
- triggering redemption checks
- sending formatted Telegram notifications

The scheduler also contains the shared runtime settlement logic used to compute fee-adjusted PnL for resolved trades.

### `core/strategy.py`

Selects and instantiates the active strategy based on configuration. The codebase currently supports:
- `ml`
- `pattern`

### `core/trade_manager.py`

Handles pre-trade operational decisions such as:
- demo vs live behavior
- balance-aware sizing behavior
- trade amount validation before execution

### `core/trader.py`

Handles order execution behavior, including live Fill-or-Kill retries and execution verification.

### `core/resolver.py`

Resolves market outcomes once a slot has matured.

### `core/redeemer.py`

Handles live redemption of winning positions after settlement.

### `core/pending_queue.py`

Provides persistent handling for work that cannot be finalized immediately and must be revisited later.

---

## 📈 Trading modes

### Live mode

In live mode, AutoPoly uses the Polymarket CLOB client and interacts with real positions.

Live mode includes:
- market discovery
- order placement
- trade recording
- post-settlement redemption flow

### Demo mode

In demo mode, no live order is sent. The system still:
- generates signals
- records trades in the database
- tracks bankroll behavior
- resolves wins and losses using runtime settlement logic
- reports all of it through Telegram

### Demo PnL model

The current runtime demo settlement uses a fee-adjusted binary payout approximation:

- `gross_shares = stake / entry_price`
- `fee = stake × 0.072 × (1 - entry_price)`
- **Win PnL:** `gross_shares - stake - fee`
- **Lose PnL:** `-stake`

In percentage-based demo sizing, the stake is calculated from the demo bankroll first and then resolved using the same PnL logic.

---

## 🧠 Strategy layer

All strategies implement the shared strategy contract in `core/strategies/base.py` and return a normalized signal structure used by the rest of the runtime.

### ML strategy

`core/strategies/ml_strategy.py` drives model-based inference.

At a high level it:
- loads the current production model and metadata
- fetches market data needed for feature generation
- builds features using the ML feature pipeline
- applies probability handling and thresholds
- emits an UP, DOWN, or SKIP decision with metadata for logging and display

Notable behavior in the current codebase:
- separate UP and DOWN thresholds are supported
- model metadata is loaded from the model store
- inference logging is part of the ML pipeline
- probability calibration and trust-gate logic exist in the ML support modules

### Pattern strategy

`core/strategies/pattern_strategy.py` is the rule-based alternative.

It:
- reads recent closed candles
- builds directional patterns
- looks up matching historical patterns from the database
- emits a trade direction or skip decision

This provides a non-ML operating mode and also serves as a useful baseline / fallback strategy.

---

## 🧪 ML pipeline

The `ml/` package contains the training and evaluation stack used by the ML strategy.

### Key modules

- `ml/data_fetcher.py` - fetches raw market and flow data used for model training and inference
- `ml/features.py` - builds the feature matrix used by the model
- `ml/trainer.py` - trains the LightGBM model and computes metrics, thresholds, and metadata
- `ml/evaluator.py` - evaluates trained models and computes risk-style metrics
- `ml/probability.py` - handles probability calibration and live trust-gate logic
- `ml/inference_logger.py` - writes inference-time observability data
- `ml/model_store.py` - saves, loads, and promotes model artifacts and metadata

### Model lifecycle

The repository currently uses model slots on disk and in the database.

Common states include:
- `current` - active production model
- `candidate` - newly trained model awaiting review / promotion

The bot exposes commands to:
- retrain a model
- inspect model status
- compare current and candidate models
- promote the candidate model

### Training behavior

The current trainer includes:
- LightGBM-based binary classification
- walk-forward validation
- threshold sweeping
- separate threshold handling for UP and DOWN decisions
- metadata capture for deployment and inference
- a deployment gate via `DeploymentBlockedError`

### Saved artifacts

The repository currently includes:
- `models/model_current.lgb`
- `models/model_current_meta.json`

The model store module also supports candidate/current promotion and DB-backed model blob persistence.

---

## 🗄️ Database layer

The application uses SQLite with schema creation and migrations in `db/models.py` and an async query layer in `db/queries.py`.

### Main tables

Based on the current schema, the primary tables are:

- `signals`
- `trades`
- `settings`
- `redemptions`
- `ml_config`
- `model_registry`
- `model_blobs`

### What is stored

The database tracks:
- generated signals
- live and demo trades
- UI/runtime settings
- redemption attempts and verification state
- ML configuration values
- model metadata and serialized model blobs

---

## 🤝 Telegram bot

The Telegram bot is the main operator interface.

### Bot responsibilities

The bot can:
- show runtime status
- show recent signals and trades
- display and change settings
- toggle or inspect demo behavior
- trigger redemption workflows
- export trade data
- inspect pattern information
- manage ML thresholds and model lifecycle actions

### Commands currently registered

From `bot/handlers.py`, the command set currently includes:

- `/start`
- `/status`
- `/signals`
- `/trades`
- `/settings`
- `/help`
- `/redeem`
- `/redemptions`
- `/demo`
- `/patterns`
- `/set_threshold`
- `/set_down_threshold`
- `/model_status`
- `/model_compare`
- `/promote_model`
- `/retrain`

The bot also supports callbacks for inline controls and export-style actions.

---

## 🔌 Polymarket integration

The `polymarket/` package contains the exchange-facing integration layer.

### `polymarket/client.py`

Wraps `py-clob-client` and is responsible for creating the authenticated CLOB client from configured credentials.

### `polymarket/markets.py`

Provides market discovery and timing helpers used by the scheduler and strategies.

### `polymarket/account.py`

Contains account-level support functionality related to the Polymarket integration.

---

## 🛠️ Configuration

Configuration is centralized in `config.py` and driven primarily by environment variables.

### Required environment variables

These are required for the app to operate normally:

- `POLYMARKET_PRIVATE_KEY`
- `POLYMARKET_FUNDER_ADDRESS`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### Common runtime environment variables

The codebase currently reads these important runtime variables:

- `POLYMARKET_SIGNATURE_TYPE`
- `POLYGON_RPC_URL`
- `TRADE_AMOUNT_USDC`
- `TRADE_MODE`
- `TRADE_PCT`
- `FOK_MAX_RETRIES`
- `FOK_RETRY_DELAY_BASE`
- `FOK_RETRY_DELAY_MAX`
- `FOK_SLOT_CUTOFF_SECONDS`
- `DB_PATH`
- `STRATEGY_NAME`
- `ML_PAYOUT_RATIO`
- `INFERENCE_LOG_PATH`
- `AUTO_REDEEM_INTERVAL_MINUTES`
- `BLOCKED_TRADE_HOURS_UTC`

### Settings managed in the database instead of env vars

Some operational toggles are DB-managed rather than env-driven. The `.env.example` explicitly documents demo settings as Telegram-managed rather than environment-managed.

That means demo behavior is controlled operationally through the bot UI / settings flow instead of only through static env values.

---

## 💻 Local setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd nyxml4.6
```

### 2. Install dependencies

If you are using the current project setup:

```bash
uv sync
```

If you prefer a requirements-based install:

```bash
pip install -r requirements.txt
```

### 3. Create environment file

```bash
cp .env.example .env
```

Then fill in the required values.

### 4. Start the bot

```bash
python main.py
```

---

## 🧾 Deployment notes

The repository includes both `Procfile` and `railway.toml`, so it is already shaped for simple process-based deployment.

Operational notes:
- SQLite is used as the runtime database
- model files are stored locally under `models/`
- the app expects long-running process execution for the bot and scheduler
- live trading requires valid Polymarket credentials and RPC connectivity

---

## 🧰 Development notes

### Tests

Current test files include:

- `tests/test_ml_features.py`
- `tests/test_ml_strategy.py`

Run them with:

```bash
pytest
```

### Useful scripts

Train a model:

```bash
python run_training.py
```

Generate threshold report:

```bash
python scripts/model_threshold_report.py
```

---

## 📌 Important implementation notes

### Live vs ML payout terminology

There are two related but different concepts in the repository:

- **Runtime demo/live-style settlement logic** used by the scheduler when resolving trades
- **`ML_PAYOUT_RATIO`** used by the ML training/evaluation pipeline for threshold and EV-style calculations

Those are not the same code path and should not be confused when changing behavior.

### Current strategy names

The repository currently supports `ml` and `pattern` as strategy labels.

### Persistence model

The project uses both:
- on-disk model artifacts under `models/`
- database-backed model metadata / blob storage

This allows runtime loading plus promotion workflows without relying on only one storage path.

---

## 📚 Quick command reference

```text
/start              Start or initialize the Telegram interaction
/status             Show current bot/runtime status
/signals            Show recent signals
/trades             Show recent trades
/settings           Open settings and controls
/help               Show help
/redeem             Start redemption flow
/redemptions        Show redemption history/status
/demo               Show or manage demo behavior
/patterns           Show stored pattern information
/set_threshold      Update UP threshold
/set_down_threshold Update DOWN threshold
/model_status       Show model status
/model_compare      Compare current and candidate models
/promote_model      Promote candidate model
/retrain            Run retraining flow
```

---

## ✅ README scope

This document was rewritten to match the current repository layout and runtime behavior as reflected in:
- `main.py`
- `config.py`
- `core/`
- `db/`
- `bot/`
- `polymarket/`
- `ml/`
- `.env.example`
- deployment files and scripts

If you add new commands, schema changes, or strategy behavior, update the README alongside the code so the operational documentation stays accurate.
