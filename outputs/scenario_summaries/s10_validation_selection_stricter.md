# Scenario Summary: s10_validation_selection_stricter

## 1. Scenario
- Scenario ID: `s10_validation_selection_stricter`
- Compare to: `s09_feature_subset_clean_trend`
- Output folder: `outputs/vn30f1m_outputs_version1`
- Commit: `142bea6`

## 2. Changes
- Keep `sequence_length = 156`
- Keep `hidden_size = 64`, `dropout = 0.2`, `learning_rate = 1e-4`, `batch_size = 64`
- Keep the `clean_trend` feature subset from `s09`
- Keep training turnover penalty disabled
- Change TFT checkpoint selection to:
  - `valid_score = valid_loss + 0.05 * valid_turnover_proxy`

## 3. Decoder TFT Focus
- `total_return = 0.7308`
- `sharpe = 4.0222`
- `max_drawdown = -0.0602`
- `turnover = 5295.84`

## 4. Other Deep Models
- `decoder_transformer`
  - `total_return = 0.1816`
  - `sharpe = 1.1685`

- `lstm_dmn`
  - `total_return = 0.0741`
  - `sharpe = 2.3659`
  - `turnover = 400.80`

## 5. Interpretation
- `s10` improved materially versus `s09` for `decoder_tft`.
- The gain came from better checkpoint selection, not lower turnover.
- This supports the idea that part of the prior TFT weakness was validation/model-selection noise, not only feature noise.

## 6. Decision
- Decision: `Promote`
- Reason:
  - `decoder_tft` improved on return, Sharpe, and drawdown versus `s09`.
  - Turnover stayed essentially flat, so the stricter validation rule improved signal quality rather than simply reducing activity.

## 7. Reproducibility
- Config path: `configs/default.yaml`
- Output path: `outputs/vn30f1m_outputs_version1`
- Run notebooks in order:
  - `notebooks/03_baseline_backtest.ipynb`
  - `notebooks/04_model_training.ipynb` or `notebooks/04_model_training_kaggle.ipynb`
  - `notebooks/05_result_analysis.ipynb`
- Compare these files after rerun:
  - `outputs/vn30f1m_outputs_version1/tables/model_comparison_metrics.csv`
  - `outputs/vn30f1m_outputs_version1/tables/trade_statistics.csv`
  - `outputs/vn30f1m_outputs_version1/tables/daily_metrics_summary.csv`
  - `outputs/vn30f1m_outputs_version1/metrics/walk_forward_results.csv`
