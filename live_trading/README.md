# decoder_tft live trading

This folder contains the live-trading path for the current `decoder_tft`
setup on 5-minute bars.

## What is here

- `DecoderTftLiveTrade.py`
  - live runner for the current 5-minute `decoder_tft` stack
  - uses `src/feature_engineering.py`
  - uses the same threshold and execution logic family as the backtest
  - converts normalized position into integer contracts using the bundle policy
- `validate_decoder_tft_bundle.py`
  - preflight checker for the bundle and environment
- `export_current_decoder_tft_bundle.py`
  - exports a deployable bundle from the current config
- `bundle_utils.py`
  - bundle serialization and scaler helpers
- `dnse_broker.py`
  - DNSE execution adapter
- `SharpeHeadLiveTrade.py`
  - older JEPA-based trader, kept only as reference

## Current bundle

The current trained bundle is now expected at:

```text
live_trading/bundles/decoder_tft_main_branch_5m/
```

Required contents:

- `manifest.json`
- `config_snapshot.yaml`
- `feature_columns.json`
- `scaler.json`
- `execution_policy.json`
- `training_summary.json`
- `model.pt`

Useful extra files:

- `final_training_history.json`
- `final_training_report.json`

## Rebuild or refresh the bundle

Run from the project root:

```bash
python src/train_final_decoder_tft.py --config configs/default.yaml --bundle-name decoder_tft_main_branch_5m --outputs-run-label vn30f1m_outputs_version1
```

If you only need to rebuild metadata around an existing checkpoint:

```bash
python live_trading/export_current_decoder_tft_bundle.py --config configs/default.yaml --bundle-name decoder_tft_main_branch_5m --outputs-run-label vn30f1m_outputs_version1 --checkpoint path/to/model.pt
```

## Environment variables

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

See `.env.example` in this folder for a concrete template.

## Preflight check

Run this before the first dry-run:

```bash
python live_trading/validate_decoder_tft_bundle.py
```

It checks:

- required bundle files exist
- manifest and checkpoint agree
- the model can be loaded
- required environment variables are present

## Dry-run

Run the trader in dry-run first:

```bash
python live_trading/DecoderTftLiveTrade.py
```

Default behavior is dry-run unless `DECODER_TFT_DRY_RUN=false`.

## Next operational steps

1. populate `.env` from `.env.example`
2. run `validate_decoder_tft_bundle.py`
3. dry-run `DecoderTftLiveTrade.py`
4. compare live replay decisions against backtest behavior
5. only then enable real order submission
