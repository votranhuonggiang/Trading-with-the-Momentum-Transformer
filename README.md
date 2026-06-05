# VN30F1M Momentum Transformer

Intraday VN30F1M trading research project based on the design in [instruction.md](C:/Users/votranhuonggiang/OneDrive/ドキュメント/Python/MiQuant/Training/MomentumTransformer/vn30f1m_momentum_transformer/instruction.md).

## Layout

- `configs/default.yaml`: main configuration
- `data/raw/vn30f1m.csv`: raw 1-minute input data
- `src/`: preprocessing, features, models, backtest, walk-forward pipeline
- `outputs/`: generated metrics, predictions, plots, and tables
- `reports/final_report.md`: final research report

## Planned CLI

```bash
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
