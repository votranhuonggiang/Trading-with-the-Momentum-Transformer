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

Research pipeline is active. Current promoted scenario is `s10_validation_selection_stricter`.

## Reproduce `s10`

Configuration and code:
- Scenario summary: `outputs/scenario_summaries/s10_validation_selection_stricter.md`
- Scenario registry: `outputs/backtest_scenario_registry.csv`
- Main config: `configs/default.yaml`

Key `s10` settings:
- `features.model_feature_subset = clean_trend`
- `training.decoder_tft_valid_selection_turnover_lambda = 0.05`
- `sequence_length = 156`
- `hidden_size = 64`
- `dropout = 0.2`
- `learning_rate = 1e-4`
- `batch_size = 64`

Run order:
1. `notebooks/03_baseline_backtest.ipynb`
2. `notebooks/04_model_training.ipynb` or `notebooks/04_model_training_kaggle.ipynb`
3. `notebooks/05_result_analysis.ipynb`

Main output folder:
- `outputs/vn30f1m_outputs_version1`

Reference `decoder_tft` result for `s10`:
- `total_return = 0.7308`
- `sharpe_ratio = 4.0222`
- `max_drawdown = -0.0602`
- `turnover = 5295.84`

Daily aggregated metrics exported by `05_result_analysis.ipynb`:
- `outputs/vn30f1m_outputs_version1/tables/daily_metrics_summary.csv`

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

## Reproducibility Notes

After pulling this branch, another user should be able to rerun the notebooks and regenerate the research outputs using the committed data, config, source code, and notebooks. They can also load the committed live bundle and run the live-trading code in dry-run mode after setting local environment variables.

Real broker execution requires valid DNSE credentials, valid QuestDB access, and local operational checks. Do not enable real order submission until dry-run decisions have been compared against expected behavior.
