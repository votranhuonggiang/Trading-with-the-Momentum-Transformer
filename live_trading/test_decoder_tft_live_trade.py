from __future__ import annotations

import unittest

import pandas as pd

from DecoderTftLiveTrade import _completed_5m_bars, _gross_contract_return


class CompletedBarTests(unittest.TestCase):
    def test_current_bucket_is_excluded(self) -> None:
        bars = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-06-24 10:35:00",
                        "2026-06-24 10:40:00",
                        "2026-06-24 10:45:00",
                    ]
                ),
                "close": [1996.0, 1997.0, 1998.0],
            }
        )

        completed = _completed_5m_bars(
            bars,
            now=pd.Timestamp("2026-06-24 10:47:27", tz="Asia/Ho_Chi_Minh"),
        )

        self.assertEqual(
            completed["timestamp"].tolist(),
            [
                pd.Timestamp("2026-06-24 10:35:00"),
                pd.Timestamp("2026-06-24 10:40:00"),
            ],
        )

    def test_new_bucket_makes_previous_bucket_complete(self) -> None:
        bars = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-06-24 10:40:00", "2026-06-24 10:45:00"]
                ),
                "close": [1997.0, 1998.0],
            }
        )

        completed = _completed_5m_bars(
            bars,
            now=pd.Timestamp("2026-06-24 10:50:01", tz="Asia/Ho_Chi_Minh"),
        )

        self.assertEqual(len(completed), 2)


class ReturnAccountingTests(unittest.TestCase):
    def test_contract_return_matches_vnd_notional_return(self) -> None:
        result = _gross_contract_return(
            contracts=3,
            price_change=-1.1,
            close_price=1996.7,
        )

        self.assertAlmostEqual(result, -3.3 / 1996.7)


if __name__ == "__main__":
    unittest.main()
