# Instruction.md  
# VN30F1M Momentum Transformer Trading Project

## 0. Project Objective

Build a real-life tradable intraday futures trading system for VN30F1M by adapting the Momentum Transformer framework from the paper *Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture*.

This is not a pure academic replication. The goal is to adapt the paper to a single-asset, intraday, Vietnam futures setting and produce a robust trading research pipeline that can be realistically evaluated after transaction costs.

The final system must:
1. Use VN30F1M continuous 1-minute OHLCV data.
2. Resample the data to 5-minute bars as the main trading frequency.
3. Generate direct position sizing signals in the range [-1, 1].
4. Trade long, short, or flat intraday.
5. Close all positions before the end of each trading day.
6. Avoid overnight holding.
7. Include realistic round-trip transaction cost assumptions.
8. Evaluate performance through walk-forward out-of-sample backtesting.
9. Compare deep learning models against simple trading baselines.
10. Produce a full project repository structure with code, configs, outputs, and reports.

The project should be designed for real trading research, not just for producing a visually good backtest.

---

## 1. Source Data

### 1.1 Input File

Use the file:

```text
data/raw/vn30f1m.csv
```

The available columns are:

```text
timestamp
open
high
low
close
volume
```

The data period is:

```text
2017-11-06 09:00:00 to 2026-05-11 14:45:00
```

The file is already a clean continuous VN30F1M series. Do not perform futures rollover adjustment unless severe data discontinuities are detected during validation.

### 1.2 Required Data Checks

Before modeling, run a full data audit:

1. Parse `timestamp` as datetime.
2. Sort data by timestamp.
3. Remove duplicated timestamps.
4. Check missing timestamps within trading sessions.
5. Check zero or negative OHLC values.
6. Check whether `high >= max(open, close)` and `low <= min(open, close)`.
7. Check extreme price jumps.
8. Check extreme volume outliers.
9. Check whether trading sessions match the expected VN30F1M trading hours.
10. Produce a short data quality report.

Save the report to:

```text
outputs/data_quality_report.md
```

### 1.3 Trading Session Handling

The system must treat VN30F1M as an intraday-only trading asset.

Rules:
1. No overnight position.
2. Force close any open position before the end of the trading day.
3. Do not allow signals to carry across days.
4. Reset intraday cumulative features at the start of each trading day.
5. Use timestamp-based session features.

Recommended practical filters:
1. Avoid opening new positions in the first 5 minutes of the day.
2. Avoid opening new positions in the last 5 minutes before end of day.
3. Always close positions before the end of day.
4. Do not open new positions during invalid or missing session periods.

The exact market microstructure schedule should be inferred from the timestamp distribution if not explicitly provided.

---

## 2. Main Timeframe

### 2.1 Main Trading Frequency

The main strategy must use 5-minute bars.

Resample the original 1-minute OHLCV data into 5-minute OHLCV bars:

```text
open: first
high: max
low: min
close: last
volume: sum
```

Save the processed file to:

```text
data/processed/vn30f1m_5min.csv
```

### 2.2 Optional Additional Timeframes

After the main 5-minute pipeline works, the system may optionally test:

```text
1-minute
15-minute
30-minute
```

However, 5-minute must remain the main version because it balances signal responsiveness and transaction cost robustness.

---

## 3. Trading Logic

### 3.1 Model Output

The main model output is a continuous position:

```text
z_t in [-1, 1]
```

Interpretation:

```text
z_t > 0: long exposure
z_t < 0: short exposure
z_t = 0: flat or no exposure
```

Use `tanh` activation in the final layer to constrain the output.

### 3.2 Position Holding Rule

The model holds a position until a new signal changes the position.

This means the strategy is not based on a fixed holding period. The next position is determined by the next signal after applying trading filters, thresholds, and end-of-day rules.

### 3.3 Thresholding and Trading Filter

Even though the model outputs a continuous value, the backtest must include practical filters to avoid overtrading.

Recommended initial thresholds:

```yaml
long_entry_threshold: 0.20
short_entry_threshold: 0.30
long_exit_threshold: 0.03
short_exit_threshold: 0.08
long_to_short_reverse_threshold: 0.50
short_to_long_reverse_threshold: 0.35
```

Rules:
1. If current position is flat and `z_t > long_entry_threshold`, enter long.
2. If current position is flat and `z_t < -short_entry_threshold`, enter short.
3. If current position is long and `z_t < long_exit_threshold`, exit to flat.
4. If current position is short and `z_t > -short_exit_threshold`, exit to flat.
5. If current position is long and `z_t < -long_to_short_reverse_threshold`, allow reversal from long to short.
6. If current position is short and `z_t > short_to_long_reverse_threshold`, allow reversal from short to long.
7. Reversal is allowed, but it requires a stronger threshold than normal entry.
8. If the signal is weak, keep the current position unless exit rules are triggered.

These thresholds must be configurable in `configs/default.yaml`.

### 3.4 Position Sizing

Use continuous position sizing by default, consistent with the paper.

The raw model output is:

```text
z_t_raw in [-1, 1]
```

The executed position is:

```text
z_t_executed in [-1, 1]
```

However, `z_t_executed` must be adjusted by:
1. Threshold rules.
2. No-trade periods.
3. Forced end-of-day flat rule.
4. Optional volatility scaling.
5. Optional maximum leverage constraint.

Do not assume that continuous position sizing means infinite divisibility in real trading. In the report, explicitly discuss that live execution may require converting continuous exposure into contract units.

---

## 4. Return and Cost Model

### 4.1 Return Definition

Use log returns for feature construction:

```text
r_t = log(close_t / close_{t-1})
```

For backtesting, both log and simple returns may be computed, but strategy PnL should be easy to interpret in index points and percentage terms.

### 4.2 Strategy Return

The strategy return before cost is:

```text
gross_pnl_t = position_{t-1} * price_change_t
```

where:

```text
price_change_t = close_t - close_{t-1}
```

Also compute a normalized return version:

```text
gross_return_t = position_{t-1} * simple_return_t
```

### 4.3 Transaction Cost

Use round-trip transaction cost:

```text
base_round_trip_cost = 0.60 to 0.75 point
```

The base assumption should use:

```text
round_trip_cost_points = 0.20
```

Convert to one-way cost:

```text
one_way_cost_points = round_trip_cost_points / 2
```

Transaction cost should be charged according to turnover.

For continuous position sizing:

```text
turnover_t = abs(position_t - position_{t-1})
cost_points_t = one_way_cost_points * turnover_t
net_pnl_t = gross_pnl_t - cost_points_t
```

Important:
1. Opening from 0 to 1 has turnover 1.
2. Closing from 1 to 0 has turnover 1.
3. Reversing from 1 to -1 has turnover 2.
4. Therefore, a full long-to-short reversal costs one full round-trip cost.

### 4.4 Cost Sensitivity

Use fixed transaction cost:

```text
0.10 point for sell side and 0.10 point for long side
```

Equivalent round-trip cost:

```text
0.20 point round-trip
```

For every model and baseline, report performance under this fixed cost setting.

### 4.5 Net Sharpe Optimization

The deep learning models should be trained to maximize net Sharpe, not raw Sharpe.

Define the loss as negative Sharpe of net strategy returns:

```text
loss = - annualized_sharpe(net_returns)
```

The cost model must be included inside the training objective whenever technically feasible.

If implementing cost-aware loss is unstable, first train with raw Sharpe and then fine-tune or evaluate with net Sharpe. Clearly report which version is used.

---

## 5. Feature Engineering

### 5.1 Core Price Features

Create multi-horizon return features on 5-minute bars:

```text
ret_1
ret_2
ret_3
ret_6
ret_12
ret_24
ret_48
ret_78
```

Approximate interpretation for 5-minute bars:

```text
1 bar = 5 minutes
3 bars = 15 minutes
6 bars = 30 minutes
12 bars = 1 hour
78 bars = approximately 1 trading day
```

Use log returns.

### 5.2 Volatility Features

Create volatility features:

```text
rolling_vol_12
rolling_vol_24
rolling_vol_48
rolling_vol_78
ewm_vol_78
ewm_vol_390
```

Use rolling or exponentially weighted standard deviation of returns.

### 5.3 Volatility-Normalized Returns

For each return horizon, create volatility-normalized versions:

```text
ret_h_norm = ret_h / (ewm_vol_reference * sqrt(h))
```

Use a small epsilon to avoid division by zero.

### 5.4 Trend and Momentum Features

Create:

```text
sma_6
sma_12
sma_24
sma_48
sma_78
ema_6
ema_12
ema_24
ema_48
ema_78
price_to_sma_24
price_to_sma_78
ema_slope_12
ema_slope_24
ema_slope_78
```

### 5.5 MACD Features

Implement MACD-style indicators inspired by the paper.

For 5-minute bars, use:

```text
MACD(8, 24)
MACD(16, 48)
MACD(32, 96)
```

Each MACD should be volatility-normalized where possible.

### 5.6 Intraday Features

Create time-based features:

```text
minute_of_day
bar_index_in_day
day_of_week
is_morning_session
is_afternoon_session
time_to_close
is_first_5min
is_last_5min
```

Use cyclical encoding for time variables:

```text
sin_time
cos_time
sin_day_of_week
cos_day_of_week
```

### 5.7 Volume Features

Create:

```text
volume
log_volume
volume_zscore_24
volume_zscore_78
volume_change
volume_ratio_to_intraday_average
```

### 5.8 Range and Volatility Proxy Features

Create:

```text
high_low_range
close_open_range
upper_wick
lower_wick
true_range
atr_14
atr_78
```

### 5.9 Intraday Reset Features

The following features must reset or be computed within each trading day:

```text
intraday_return_from_open
intraday_high_so_far
intraday_low_so_far
distance_from_intraday_high
distance_from_intraday_low
intraday_volume_cumsum
```

### 5.10 Feature Standardization

Use only training data to fit scalers.

Rules:
1. Fit scalers on training data only.
2. Apply the same scaler to validation and test.
3. Never use future information.
4. Use robust scaling or z-score scaling.
5. Save fitted scalers for each walk-forward window.

---

## 6. Target Construction

### 6.1 Main Learning Target

The main deep learning models should not use classification labels as the primary target.

The model should output position directly and be optimized using a trading objective.

### 6.2 Return Used in Loss

For each timestamp, the model output at time `t` should be applied to the next bar return:

```text
strategy_return_{t+1} = position_t * return_{t+1} - cost_{t+1}
```

Use no look-ahead bias.

### 6.3 Optional Auxiliary Direction Label

For analysis only, create an auxiliary direction label:

```text
future_return_sign = sign(return_{t+1})
```

Use this only for diagnostic direction accuracy, not as the main training objective.

---

## 7. Baselines

Implement simple baselines before deep learning.

### 7.1 Buy and Hold Benchmark

For VN30F1M, buy-and-hold is not a perfect benchmark because this is an intraday futures strategy. Still, compute:

```text
long-only intraday benchmark
```

Rules:
1. Long during tradable periods.
2. Flat at end of day.
3. Include transaction costs.

### 7.2 Time-Series Momentum Baseline

Create a TSMOM rule:

```text
if ret_lookback > threshold: long
if ret_lookback < -threshold: short
else: flat
```

Test lookbacks:

```text
12, 24, 48, 78 bars
```

### 7.3 Moving Average Crossover Baseline

Test:

```text
EMA(8) vs EMA(24)
EMA(16) vs EMA(48)
EMA(32) vs EMA(96)
```

### 7.4 MACD Baseline

Use MACD signal rules:

```text
MACD > signal: long
MACD < signal: short
else: flat
```

### 7.5 Optional Linear Baseline

Add a simple regularized regression or logistic model as a sanity check:

```text
Ridge regression
Logistic regression
```

This is optional but useful for ensuring deep learning is adding value.

---

## 8. Deep Learning Models

Implement the following models:

```text
1. LSTM Deep Momentum Network
2. Decoder-Only Transformer
3. Decoder-Only Temporal Fusion Transformer
```

The models should output direct position sizing.

### 8.1 LSTM Deep Momentum Network

Input:
```text
sequence of engineered features
```

Output:
```text
position z_t in [-1, 1]
```

Architecture:
1. LSTM layer or stacked LSTM.
2. Dropout.
3. Dense layer.
4. Tanh output.

Initial search space:

```yaml
sequence_length: [78, 156, 390]
hidden_size: [32, 64, 128]
num_layers: [1, 2]
dropout: [0.1, 0.2, 0.3]
learning_rate: [0.0001, 0.001]
batch_size: [64, 128, 256]
```

### 8.2 Decoder-Only Transformer

Input:
```text
sequence of engineered features
```

Output:
```text
position z_t in [-1, 1]
```

Architecture:
1. Linear feature embedding.
2. Positional encoding.
3. Causal masked self-attention.
4. Feedforward network.
5. Tanh output.

Initial search space:

```yaml
sequence_length: [78, 156, 390]
d_model: [64, 128]
num_heads: [4, 8]
num_layers: [2, 4]
dim_feedforward: [128, 256, 512]
dropout: [0.1, 0.2, 0.3]
learning_rate: [0.0001, 0.001]
batch_size: [64, 128]
```

### 8.3 Decoder-Only Temporal Fusion Transformer

This is the main paper-inspired model.

Required components:
1. Variable Selection Network or feature gating module.
2. LSTM local processing block.
3. Interpretable multi-head attention or causal self-attention.
4. Gated residual connections where feasible.
5. Tanh position output.

If full TFT is too complex, implement a simplified practical TFT:

```text
feature gating -> LSTM -> causal attention -> dense -> tanh position
```

Initial search space:

```yaml
sequence_length: [156, 390]
hidden_size: [32, 64, 128]
num_heads: [4]
dropout: [0.1, 0.2, 0.3]
learning_rate: [0.0001, 0.001]
batch_size: [32, 64, 128]
```

### 8.4 CPD Extension

Changepoint detection is not required in the first complete version.

After the baseline, LSTM, Transformer, and TFT models are working, implement CPD as an extension.

Potential CPD features:

```text
cpd_score_short
cpd_score_long
cpd_location_short
cpd_location_long
```

Recommended windows for 5-minute bars:

```text
short_window = 78 bars
long_window = 390 bars
```

Use CPD only if it does not introduce look-ahead bias.

---

## 9. Walk-Forward Backtesting

### 9.1 Main Backtesting Approach

Use expanding walk-forward validation.

Suggested windows:

```text
Train: 2017-11-06 to 2020-12-31
Validation: 2021-01-01 to 2021-06-30
Test: 2021-07-01 to 2021-12-31

Train: 2017-11-06 to 2021-06-30
Validation: 2021-07-01 to 2021-12-31
Test: 2022-01-01 to 2022-06-30

Train: 2017-11-06 to 2021-12-31
Validation: 2022-01-01 to 2022-06-30
Test: 2022-07-01 to 2022-12-31

Train: 2017-11-06 to 2022-06-30
Validation: 2022-07-01 to 2022-12-31
Test: 2023-01-01 to 2023-06-30

Train: 2017-11-06 to 2022-12-31
Validation: 2023-01-01 to 2023-06-30
Test: 2023-07-01 to 2023-12-31

Train: 2017-11-06 to 2023-06-30
Validation: 2023-07-01 to 2023-12-31
Test: 2024-01-01 to 2024-06-30

Train: 2017-11-06 to 2023-12-31
Validation: 2024-01-01 to 2024-06-30
Test: 2024-07-01 to 2024-12-31

Train: 2017-11-06 to 2024-06-30
Validation: 2024-07-01 to 2024-12-31
Test: 2025-01-01 to 2025-06-30

Train: 2017-11-06 to 2024-12-31
Validation: 2025-01-01 to 2025-06-30
Test: 2025-07-01 to 2025-12-31

Train: 2017-11-06 to 2025-06-30
Validation: 2025-07-01 to 2025-12-31
Test: 2026-01-01 to 2026-03-31

Train: 2017-11-06 to 2025-12-31
Validation: 2026-01-01 to 2026-03-31
Test: 2026-04-01 to 2026-05-11
```

If some windows have insufficient data, adjust dates while preserving chronological order.

### 9.2 No Data Leakage

Strictly enforce:

1. No future data in features.
2. No future data in scaling.
3. No future data in hyperparameter tuning.
4. Validation must occur after training.
5. Test must occur after validation.
6. Final reported performance must be test-only.

### 9.3 Model Selection

For each walk-forward window:

1. Train models on training set.
2. Tune hyperparameters on validation set.
3. Select best model based on validation net Sharpe and drawdown constraints.
4. Evaluate selected model once on test set.
5. Store test predictions, positions, trades, and metrics.

---

## 10. Evaluation Metrics

Report all metrics on both gross and net returns.

### 10.1 Core Metrics

Compute:

```text
total_return
annualized_return
annualized_volatility
sharpe_ratio
sortino_ratio
calmar_ratio
max_drawdown
hit_rate
profit_factor
average_trade_pnl
average_win
average_loss
win_loss_ratio
```

### 10.2 Trading Behavior Metrics

Compute:

```text
number_of_trades
average_trades_per_day
turnover
average_holding_period
long_trade_count
short_trade_count
long_pnl
short_pnl
percentage_time_long
percentage_time_short
percentage_time_flat
```

### 10.3 Risk Metrics

Compute:

```text
daily_pnl_distribution
worst_day
best_day
max_consecutive_losses
drawdown_duration
tail_loss_95
tail_loss_99
```

### 10.4 Cost Sensitivity Metrics

For each cost scenario, compute:

```text
net_total_return
net_sharpe
net_max_drawdown
net_profit_factor
net_average_trade_pnl
```

### 10.5 Yearly and Monthly Metrics

Produce:

```text
yearly_performance.csv
monthly_performance.csv
```

### 10.6 Regime Analysis

Evaluate performance in known market periods:

```text
COVID period: 2020
Bond shock period: 2022
Recent market period: 2023 to 2026
```

Exact regime dates may be configured in `configs/default.yaml`.

---

## 11. Model Interpretability

For the final model, provide interpretability outputs.

### 11.1 Feature Importance

For TFT or feature-gated models, extract variable selection weights if implemented.

If not available, use permutation importance on validation and test sets.

### 11.2 Attention Analysis

For Transformer and TFT models:

1. Save attention weights if technically feasible.
2. Plot attention patterns around major market periods.
3. Check whether the model attends to momentum turning points.
4. Compare attention behavior during trending and mean-reverting periods.

### 11.3 Signal Diagnostics

Plot:

```text
price vs position
price vs signal
position distribution
daily turnover
rolling Sharpe
rolling drawdown
```

---

## 12. Required Output Files

The agent must create a full project structure:

```text
vn30f1m_momentum_transformer/
│
├── README.md
├── instruction.md
├── requirements.txt
├── configs/
│   └── default.yaml
│
├── data/
│   ├── raw/
│   │   └── vn30f1m.csv
│   ├── processed/
│   │   └── vn30f1m_5min.csv
│   └── features/
│       └── features_5min.parquet
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── datasets.py
│   ├── costs.py
│   ├── backtest.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   ├── walk_forward.py
│   ├── visualization.py
│   └── models/
│       ├── __init__.py
│       ├── lstm_dmn.py
│       ├── decoder_transformer.py
│       └── decoder_tft.py
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_check.ipynb
│   ├── 03_baseline_backtest.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_result_analysis.ipynb
│
├── outputs/
│   ├── data_quality_report.md
│   ├── metrics/
│   ├── predictions/
│   ├── models/
│   ├── plots/
│   └── tables/
│
└── reports/
    └── final_report.md
```

---

## 13. Required Plots

Generate and save:

```text
equity_curve_gross.png
equity_curve_net.png
drawdown_curve.png
yearly_returns.png
monthly_returns_heatmap.png
position_vs_price.png
signal_distribution.png
turnover_by_day.png
cost_sensitivity.png
model_comparison.png
walk_forward_performance.png
```

---

## 14. Required Tables

Generate and save:

```text
model_comparison_metrics.csv
cost_sensitivity_table.csv
yearly_performance.csv
monthly_performance.csv
walk_forward_results.csv
trade_statistics.csv
long_short_breakdown.csv
regime_performance.csv
```

---

## 15. Configuration File

Create `configs/default.yaml` with at least:

```yaml
data:
  raw_path: "data/raw/vn30f1m.csv"
  processed_path: "data/processed/vn30f1m_5min.csv"
  timestamp_col: "timestamp"
  timeframe: "5min"

trading:
  intraday_only: true
  close_before_end_of_day: true
  no_overnight: true
  long_entry_threshold: 0.20
  short_entry_threshold: 0.30
  long_exit_threshold: 0.03
  short_exit_threshold: 0.08
  long_to_short_reverse_threshold: 0.50
  short_to_long_reverse_threshold: 0.35
  base_round_trip_cost_points: 0.20
  cost_scenarios_points: [0.20]
  # base_round_trip_cost_points and cost_scenarios_points are legacy compatibility
  # fields only. Active backtest cost model:
  # - buy/long side: fixed fee only
  # - sell/short side: fixed fee + transfer tax
  # transfer-tax formula:
  # tax_per_sell_side = close_price * 100000 * margin_rate * 0.001 / 2
  margin_rate: 0.1848
  transfer_tax_rate: 0.001

features:
  return_horizons: [1, 2, 3, 6, 12, 24, 48, 78]
  volatility_windows: [12, 24, 48, 78, 390]
  macd_pairs:
    - [8, 24]
    - [16, 48]
    - [32, 96]

models:
  sequence_lengths: [78, 156, 390]
  main_models:
    - "lstm_dmn"
    - "decoder_transformer"
    - "decoder_tft"

walk_forward:
  method: "expanding"
  validation_months: 6
  test_months: 6
  step_months: 6

training:
  objective: "net_sharpe"
  max_epochs: 100
  early_stopping_patience: 10
  device: "auto"
```

---

## 16. Implementation Standards

### 16.1 Code Quality

The code must be:

1. Modular.
2. Reproducible.
3. Config-driven.
4. Easy to run from command line.
5. Compatible with future live trading integration.
6. Documented with clear comments.
7. Written in Python.

### 16.2 Reproducibility

Set random seeds for:

```text
numpy
pandas-related sampling if any
torch
python random
```

Save:

```text
config used
model weights
scalers
feature columns
walk-forward split dates
predictions
positions
metrics
```

### 16.3 Command Line Usage

The project should support commands like:

```bash
python src/data_preprocessing.py --config configs/default.yaml
python src/feature_engineering.py --config configs/default.yaml
python src/walk_forward.py --config configs/default.yaml
python src/evaluate.py --config configs/default.yaml
```

---

## 17. Backtest Validity Requirements

The agent must explicitly check and avoid:

1. Look-ahead bias.
2. Survivorship bias in the available single-asset data context.
3. Leakage from scaling.
4. Leakage from feature engineering.
5. Using the same data for validation and test.
6. Ignoring transaction costs.
7. Ignoring turnover.
8. Ignoring forced intraday close.
9. Reporting only gross returns.
10. Selecting the best test result after repeated experiments.

If any issue is found, report it in:

```text
reports/final_report.md
```

---

## 18. Real-Life Trading Constraints

Since this project is intended for real trading research, the system must be conservative.

### 18.1 Minimum Performance Requirements

The final model should not be considered tradeable unless it satisfies all conditions on out-of-sample test data:

```text
net Sharpe > 1.0
positive net return after 0.75 point round-trip cost
max drawdown acceptable relative to return
profit factor > 1.1
reasonable number of trades per day
performance not driven by one single period
cost sensitivity remains acceptable up to 1.0 point round-trip
```

These are not guarantees of live trading profitability. They are minimum research filters.

### 18.2 Stability Checks

Check:

1. Performance by year.
2. Performance by month.
3. Performance under higher cost.
4. Performance during high volatility.
5. Performance during low volatility.
6. Long-side vs short-side performance.
7. Whether the model overtrades.
8. Whether the model collapses in recent test windows.

### 18.3 Live Trading Readiness

The final report must include a section:

```text
Live Trading Readiness Assessment
```

It should answer:

1. Is the model robust enough for paper trading?
2. What are the main failure modes?
3. What cost assumption is needed to remain profitable?
4. Does the model rely too much on short-term reversal?
5. How many trades per day does it generate?
6. Is execution realistic?
7. What should be monitored in live deployment?

---

## 19. Final Report Structure

Create:

```text
reports/final_report.md
```

Use this structure:

```text
# VN30F1M Momentum Transformer Trading Project

## 1. Objective
## 2. Data Description
## 3. Data Cleaning and Resampling
## 4. Feature Engineering
## 5. Trading Rules and Cost Assumptions
## 6. Baseline Strategies
## 7. Deep Learning Models
## 8. Walk-Forward Backtest Design
## 9. Main Results
## 10. Cost Sensitivity
## 11. Long vs Short Performance
## 12. Yearly and Monthly Stability
## 13. Regime Analysis
## 14. Model Interpretability
## 15. Live Trading Readiness Assessment
## 16. Limitations
## 17. Next Steps
```

---

## 20. Development Roadmap

Implement in stages.

### Stage 1: Data and Baseline

1. Load and audit data.
2. Resample to 5-minute bars.
3. Engineer basic features.
4. Implement cost model.
5. Implement TSMOM, moving average, and MACD baselines.
6. Produce baseline backtest results.

### Stage 2: LSTM DMN

1. Build dataset class.
2. Implement LSTM position model.
3. Train with Sharpe or net Sharpe loss.
4. Evaluate through expanding walk-forward.
5. Compare with baselines.

### Stage 3: Decoder-Only Transformer

1. Implement causal Transformer.
2. Train with the same feature set and walk-forward splits.
3. Evaluate against LSTM and baselines.

### Stage 4: Decoder-Only TFT

1. Implement simplified TFT.
2. Add feature gating or variable selection.
3. Add LSTM local processing.
4. Add causal attention.
5. Evaluate and compare.

### Stage 5: Cost and Robustness

1. Run cost sensitivity.
2. Run threshold sensitivity.
3. Analyze turnover.
4. Analyze long vs short.
5. Analyze regime performance.

### Stage 6: CPD Extension

1. Add changepoint features.
2. Retrain the best model.
3. Compare with non-CPD version.
4. Keep CPD only if it improves net out-of-sample performance.

---

## 21. Important Design Philosophy

Do not optimize for the highest possible backtest return.

Optimize for:

1. Realistic net performance.
2. Stability across time.
3. Low sensitivity to transaction costs.
4. Controlled turnover.
5. Transparent assumptions.
6. No look-ahead bias.
7. Practical deployment readiness.

A model with lower return but stable net Sharpe, lower drawdown, and lower turnover is preferable to a model with very high gross return but poor cost-adjusted performance.

---

## 22. Final Deliverables

The final deliverables must include:

1. Full source code.
2. Config file.
3. Processed data output.
4. Feature matrix.
5. Baseline results.
6. Deep learning model results.
7. Walk-forward results.
8. Cost sensitivity analysis.
9. Plots and tables.
10. Final report.
11. README with instructions to run the project.

The project is complete only when another user can clone/open the folder, place `vn30f1m.csv` into `data/raw/`, install requirements, and reproduce the full backtest pipeline.
