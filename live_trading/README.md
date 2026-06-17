# decoder_tft live trading

This folder now contains the live-trading path for the current main-branch
`decoder_tft` setup on 5-minute bars.

## What is here

- `DecoderTftLiveTrade.py`
  - new live runner for the current 5-minute `decoder_tft` stack
  - uses the same feature pipeline from `src/feature_engineering.py`
  - uses the same threshold and execution logic family as the current backtest
  - converts normalized position into integer contracts using the bundle policy
- `export_current_decoder_tft_bundle.py`
  - exports a deployable bundle from the current main-branch config
  - writes config snapshot, feature schema, scaler, execution policy, and summary
- `bundle_utils.py`
  - bundle serialization and scaler helpers
- `dnse_broker.py`
  - DNSE execution adapter
- `SharpeHeadLiveTrade.py`
  - older JEPA-based trader, kept for reference only

## Bundle format

Each bundle lives under:

```text
live_trading/bundles/<bundle_name>/
```

Expected contents:

- `manifest.json`
- `config_snapshot.yaml`
- `feature_columns.json`
- `scaler.json`
- `execution_policy.json`
- `training_summary.json`
- `model.pt` if a deployable checkpoint has been supplied

## Prepare the current main-branch bundle

Run from the project root:

```bash
python live_trading/export_current_decoder_tft_bundle.py --config configs/default.yaml --bundle-name decoder_tft_main_branch_5m --outputs-run-label vn30f1m_outputs_version1
```

If you already have a final deployable checkpoint:

```bash
python live_trading/export_current_decoder_tft_bundle.py --config configs/default.yaml --bundle-name decoder_tft_main_branch_5m --outputs-run-label vn30f1m_outputs_version1 --checkpoint path/to/model.pt
```

## Environment variables for the live runner

Required for live data and broker access:

- `QUESTDB_HOST`
- `QUESTDB_PORT`
- `QUESTDB_USER`
- `QUESTDB_PASSWORD`
- `QUESTDB_DBNAME`
- `DNSE_API_KEY`
- `DNSE_API_SECRET`

Optional:

- `DNSE_TRADING_TOKEN`
- `DNSE_BROKER_SYMBOL`
- `DECODER_TFT_DATA_SYMBOL`
- `DECODER_TFT_LIVE_BUNDLE`
- `DECODER_TFT_POLL_INTERVAL_S`
- `DECODER_TFT_SLIPPAGE_ACCEPT`
- `DECODER_TFT_TICK_SIZE`
- `DECODER_TFT_DEVICE`
- `DECODER_TFT_DRY_RUN`

## Run in dry-run first

```bash
python live_trading/DecoderTftLiveTrade.py
```

Default behavior is dry-run unless `DECODER_TFT_DRY_RUN=false`.

## Important limitation

The current research pipeline does not yet save a final deployable
`decoder_tft` checkpoint automatically. The bundle exporter prepares everything
except the checkpoint unless `--checkpoint` is supplied.

That means the next operational step is:

1. save one final `decoder_tft` checkpoint,
2. place it into the bundle as `model.pt`,
3. dry-run the live trader against that bundle,
4. compare live replay decisions to backtest before real execution.
