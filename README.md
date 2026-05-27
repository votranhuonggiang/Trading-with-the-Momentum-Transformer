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

Scaffold created. Implementation should follow `instruction.md` strictly.

