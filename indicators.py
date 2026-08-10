"""
Standalone, testable versions of the technical-indicator math used inline in
macro-regime.ipynb and micro-regime.ipynb (MacroFeatureExtractor._compute_rsi,
MicroFeatureExtractor._cci, MicroFeatureExtractor.process_nifty's ADX block).

The notebooks still carry their own inline copies rather than importing from
here -- wiring them up would mean another full re-run/relabel cascade (see
CRITIQUE.md S4.5) -- but the formulas below are kept in lockstep with the
notebook cells, so tests/test_indicators.py gives real coverage of the same
math actually running in the pipeline.
"""
import numpy as np
import pandas as pd

EPS = 1e-8


def rsi(close: pd.Series, window_days: int = 14) -> pd.Series:
    delta = close.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0), index=close.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0), index=close.index)

    avg_gain = gain.ewm(alpha=1 / window_days, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window_days, adjust=False).mean()

    rs = avg_gain / (avg_loss + EPS)
    return 100 - (100 / (1 + rs))


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (mad + EPS)


def adx_di(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 3):
    """Returns (plus_di, minus_di, adx) using the simplified TR/DM definitions
    from MicroFeatureExtractor.process_nifty."""
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr_sum = tr.rolling(window).sum()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(window).sum() / (tr_sum + EPS)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(window).sum() / (tr_sum + EPS)
    adx = (plus_di - minus_di).abs()
    return plus_di, minus_di, adx
