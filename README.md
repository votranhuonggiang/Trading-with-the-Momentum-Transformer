# VN30F1M Momentum Transformer

Intraday VN30F1M trading research project based on the design in [instruction.md](instruction.md).

The current promoted model is a `decoder_tft` live-trading bundle trained on 5-minute VN30F1M bars. The repository includes the source code, notebooks, reproducibility outputs, and live-trading bundle needed to understand and rerun the research pipeline.

## Layout

- `configs/default.yaml`: main configuration
- `data/raw/vn30f1m.csv`: raw 1-minute input data
- `src/`: preprocessing, features, models, backtest, walk-forward pipeline
- `notebooks/`: data audit, feature check, baseline backtest, model training, and result analysis notebooks
- `outputs/vn30f1m_outputs_version1/`: generated metrics, predictions, plots, and tables for the promoted run
- `live_trading/`: deployable live-trading path and exported model bundle
- `reports/final_report.md`: final research report, if generated locally

## Current Model

The live bundle is stored at:

```text
live_trading/bundles/decoder_tft_main_branch_5m/
```

Model summary:

- Model family: decoder-only TFT-style neural network
- Input timeframe: 5-minute bars
- Sequence length: 156 bars
- Lookback context: 156 x 5 minutes = 780 minutes of recent intraday history
- Target horizon: 3 bars, about 15 minutes
- Hidden size: 64
- Attention heads: 4
- Dropout: 0.2
- Objective: net Sharpe
- Auxiliary tasks: future volatility and volatility-regime prediction
- Live execution: integer contracts, maximum 5 contracts, nearest rounding

The model flow is:

```text
feature sequence -> gated feature block -> LSTM -> causal decoder attention -> tanh position signal
```

The model outputs a normalized position in `[-1, +1]`. The live runner then applies confidence gating, volatility-regime exposure scaling, entry/exit/reversal thresholds, and integer contract rounding to produce final live exposure from `-5` to `+5` contracts.

Training summary from the exported bundle:

- Train period: `2017-11-07 14:15:00` to `2025-07-01 14:25:00`
- Validation period: `2025-07-01 14:30:00` to `2026-05-11 13:35:00`
- Train rows: `95,115`
- Validation rows: `10,569`
- Bundle headline total return: `3.6074`
- Bundle headline Sharpe ratio: `2.7581`
- Bundle headline max drawdown: `-0.4206`
- Bundle headline profit factor: `1.1876`

## Factors Used

The deployed model uses the feature list saved in:

```text
live_trading/bundles/decoder_tft_main_branch_5m/feature_columns.json
```

Feature groups:

- Time/session: day-of-week encodings, intraday time encodings, time to close
- Returns and momentum: `log_return`, returns over 1, 2, 3, 6, 12, 24, 48, and 78 bars
- Volatility: rolling and exponentially weighted volatility windows
- Volatility-normalized returns: normalized returns over the same short and medium horizons
- Trend: EMA levels, price-to-SMA ratios, EMA slopes
- MACD: MACD level, signal, histogram, slope, normalized values, sign, and cross flags
- Range/risk: ATR over 14 and 78 bars
- Trading mask: `trade_allowed`

The full feature set is intentionally stored with the live bundle so live inference uses the same feature order as training.

## Planned CLI

```bash
pip install -r requirements.txt
python src/data_preprocessing.py --config configs/default.yaml
python src/feature_engineering.py --config configs/default.yaml
python src/walk_forward.py --config configs/default.yaml
python src/evaluate.py --config configs/default.yaml
```

## Status

Research pipeline is active. Current promoted artifacts are in `outputs/vn30f1m_outputs_version1/` and `live_trading/bundles/decoder_tft_main_branch_5m/`.

## Live Trading

The live-trading path is in `live_trading/`.

Important files:

- `live_trading/DecoderTftLiveTrade.py`: current decoder-TFT live runner
- `live_trading/validate_decoder_tft_bundle.py`: preflight validator for the exported bundle
- `live_trading/dnse_broker.py`: DNSE execution adapter
- `live_trading/bundles/decoder_tft_main_branch_5m/model.pt`: trained model checkpoint
- `live_trading/bundles/decoder_tft_main_branch_5m/manifest.json`: bundle metadata
- `live_trading/logs/`: dry-run live-trading output logs from the current run

Before live trading, populate local environment variables for QuestDB and DNSE access. Credentials are intentionally not committed. Required variables include:

```text
QUESTDB_HOST
QUESTDB_PORT
QUESTDB_USER
QUESTDB_PASSWORD
QUESTDB_DBNAME
DNSE_API_KEY
DNSE_API_SECRET
```

Run the bundle preflight check:

```bash
python live_trading/validate_decoder_tft_bundle.py
```

Run dry-run live trading:

```bash
python live_trading/DecoderTftLiveTrade.py
```

Default behavior is dry-run unless `DECODER_TFT_DRY_RUN=false` is set.

## Recent Daily Live Dry-Run Performance

The committed live-trading logs cover recent daily dry-run sessions:

```text
2026-06-25 09:25:00 to 2026-07-02 14:25:00
```

These results are from `live_trading/logs/`, not from the research backtest. They should be read as daily live-run outcomes because the live runner is intended to be started each trading day:

```bash
python live_trading/DecoderTftLiveTrade.py
```

The daily PnL values below are not a single continuous weekly portfolio return. They summarize what happened on each dry-run trading day after that day's decisions, turnover, and estimated execution costs.

Operational totals across the logged daily sessions:

- Trading days: 6
- Processed bars: 250
- Logged decisions: 250
- Transactions: 94
- Total turnover: 139 contracts
- Total gross PnL points before costs: `+19.10`
- Total estimated trading costs: `39.33` points, about `3,933,250` VND
- Sum of daily net PnL points after costs: `-20.23`

Daily dry-run performance:

| Date | Bars | Transactions | Turnover | Gross PnL | Cost Points | Net PnL | Net PnL x 100k VND |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-25 | 42 | 18 | 28 | 21.00 | 7.73 | 13.27 | 1,327,000 |
| 2026-06-26 | 42 | 13 | 21 | -17.50 | 5.51 | -23.01 | -2,301,000 |
| 2026-06-29 | 39 | 12 | 16 | 2.80 | 4.64 | -1.84 | -184,000 |
| 2026-06-30 | 42 | 18 | 24 | 0.30 | 6.95 | -6.65 | -665,000 |
| 2026-07-01 | 41 | 10 | 16 | 28.60 | 4.64 | 23.96 | 2,396,000 |
| 2026-07-02 | 44 | 23 | 34 | -16.10 | 9.86 | -25.96 | -2,596,000 |

For the logged daily sessions, gross points were positive in aggregate, but estimated transaction costs more than offset them. This is useful operational evidence for checking turnover, threshold behavior, and execution cost assumptions before enabling real order submission.

## Reproducibility Notes

After pulling this branch, another user should be able to rerun the notebooks and regenerate the research outputs using the committed data, config, source code, and notebooks. They can also load the committed live bundle and run the live-trading code in dry-run mode after setting local environment variables.

Real broker execution requires valid DNSE credentials, valid QuestDB access, and local operational checks. Do not enable real order submission until dry-run decisions have been compared against expected behavior.
