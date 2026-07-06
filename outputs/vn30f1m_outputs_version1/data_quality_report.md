# Data Quality Report

## Summary
- Rows: 512906
- Date range: 2017-11-06 09:00:00 -> 2026-05-11 14:45:00
- Trading days: 2119

## Integrity Checks
- Non-positive OHLC rows: 0
- High/Low inconsistency rows: 0
- Extreme price jump rows (|log ret| > 3%): 21
- Extreme volume outlier rows (IQR rule): 3112
- Missing timestamps vs inferred session grid: 1410
- Inferred session time points per day: 240

## Notes
- Input data was sorted by timestamp and duplicate timestamps removed.
- Session grid was inferred from recurring intraday timestamps across days.