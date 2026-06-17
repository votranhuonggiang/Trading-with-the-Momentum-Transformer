# Bundle status

- bundle_dir: `C:\Users\votranhuonggiang\OneDrive\ドキュメント\Python\MiQuant\Training\MomentumTransformer\vn30f1m_momentum_transformer\live_trading\bundles\decoder_tft_main_branch_5m`
- config_snapshot: `config_snapshot.yaml`
- checkpoint_present: `False`
- timeframe: `5min`
- sequence_length: `156`
- max_contracts: `5`

## What to do next

1. Export or save a final `decoder_tft` checkpoint as `model.pt` into this bundle.
2. Run `DecoderTftLiveTrade.py` against this bundle in `dry_run=True` first.
3. Compare bundle replay output to backtest output before enabling real orders.
