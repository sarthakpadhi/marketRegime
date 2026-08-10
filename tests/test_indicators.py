"""
Reference-value tests for the indicator math in indicators.py (see CRITIQUE.md S4.5).

Deliberately uses degenerate synthetic series (flat / strictly monotonic) rather than
real market data: for these inputs the mathematically correct output is unambiguous,
so no "expected" value needs to be independently recomputed with the same formula
(which would just be testing the implementation against itself). These same edge
cases (flat windows, all-gain or all-loss runs) are exactly what the S3.2 div-by-zero
guards (+1e-8) were added to handle without producing NaN/inf.
"""
import numpy as np
import pandas as pd
import pytest

from indicators import adx_di, cci, rsi


def test_rsi_flat_series_is_zero_not_nan():
    # zero gain AND zero loss every day -> rs = 0/(0+eps) = 0 -> RSI = 0, not NaN.
    close = pd.Series([100.0] * 30)
    result = rsi(close, window_days=14)
    tail = result.iloc[20:]
    assert tail.notna().all()
    assert (tail.abs() < 1e-3).all()


def test_rsi_all_gains_approaches_100():
    # zero loss every day -> rs -> huge -> RSI -> 100, and finite (no div-by-zero blowup).
    close = pd.Series(np.arange(100.0, 130.0))  # +1 every day, never down
    result = rsi(close, window_days=14)
    tail = result.iloc[20:]
    assert np.isfinite(tail).all()
    assert (tail > 99.9).all()


def test_rsi_all_losses_approaches_0():
    close = pd.Series(np.arange(130.0, 100.0, -1.0))  # -1 every day, never up
    result = rsi(close, window_days=14)
    tail = result.iloc[20:]
    assert np.isfinite(tail).all()
    assert (tail < 0.1).all()


def test_cci_flat_series_is_zero_not_nan():
    # constant OHLC -> mean absolute deviation is exactly 0 -> guarded to CCI = 0.
    n = 30
    high = pd.Series([50.0] * n)
    low = pd.Series([50.0] * n)
    close = pd.Series([50.0] * n)
    result = cci(high, low, close, window=14)
    tail = result.iloc[20:]
    assert tail.notna().all()
    assert (tail.abs() < 1e-3).all()


def test_cci_finite_on_trending_series():
    n = 30
    close = pd.Series(np.arange(50.0, 50.0 + n))
    high, low = close + 1, close - 1
    result = cci(high, low, close, window=14)
    assert np.isfinite(result.iloc[20:]).all()


def test_adx_flat_series_is_zero_not_nan():
    # no price movement at all -> TR, +DM, -DM all 0 every day -> guarded to DI/ADX = 0.
    n = 20
    high = pd.Series([100.0] * n)
    low = pd.Series([100.0] * n)
    close = pd.Series([100.0] * n)
    plus_di, minus_di, adx = adx_di(high, low, close, window=3)
    tail = slice(10, None)
    assert plus_di.iloc[tail].notna().all()
    assert minus_di.iloc[tail].notna().all()
    assert (plus_di.iloc[tail].abs() < 1e-3).all()
    assert (minus_di.iloc[tail].abs() < 1e-3).all()
    assert (adx.iloc[tail].abs() < 1e-3).all()


def test_adx_finite_and_nonnegative_on_trending_series():
    n = 20
    close = pd.Series(np.arange(100.0, 100.0 + n))
    high, low = close + 1, close - 1
    plus_di, minus_di, adx = adx_di(high, low, close, window=3)
    tail = slice(10, None)
    assert np.isfinite(plus_di.iloc[tail]).all()
    assert np.isfinite(minus_di.iloc[tail]).all()
    assert (adx.iloc[tail] >= 0).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
