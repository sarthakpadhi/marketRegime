# Market Regime Detection for Indian Equities: does an HMM give a tradeable edge?

A walk-forward, leak-free study of Hidden Markov Model (HMM) and statistical jump-model regimes
on the Nifty 50 and India VIX, with replications on the S&P 500 and other asset sets.

**Result in one line:** the regimes are persistent and describe volatility well (they mark the
Covid crash and calm bull runs), but after removing look-ahead they did **not** beat standard
baselines (EWMA vol-targeting, GARCH, VIX, a 200-day moving average, equal weight) in any
pre-registered test. For the Indian market, HMM regimes carry little tradeable edge; their value
is descriptive.

> Research code, not investment advice.

---

## What happened

1. **First version (legacy).** Two 6-state HMMs (299 and 425 parameters on about 1,600 rows,
   Jan 2019 to Aug 2025) labelled the full history with Viterbi decoding, and regime statistics and a
   backtest were computed from those labels. They use future data, so those numbers are not
   knowable in real time. The old write-up is kept at [`docs/legacy_v1_README.md`](docs/legacy_v1_README.md)
   with a "superseded" banner.
2. **Fixes.** 3 states, diagonal covariance, 4 causal features (32 parameters), real random
   restarts, scaler fit on training data only, states chosen by out-of-sample likelihood.
3. **Causal evaluation.** Filtered probabilities (forward pass only) with a 3-year warm-up and a
   refit every 6 months, plus a long-history rerun (2009 to 2026).
4. **A series of tests**, each against simple baselines, from the third onward with the pass/fail
   rule written in the notebook's first cell before any result existed.

## Results (all out of sample, walk-forward)

| Test | Notebook | Out-of-sample period | Outcome |
|---|---|---|---|
| Index overlay: scale Nifty exposure by regime probability | [`regime-filtered`](regime-filtered.ipynb) | 2022-02-24 to 2025-08-21 | Sharpe HMM 0.21 vs EWMA vol-target 0.39 vs buy and hold 0.53. With Covid out of sample, max drawdown -24.6% vs -17.5% (vol-target) vs -37.2% |
| Multi-asset allocation (Nifty, Bank, Midcap 50, Gold, cash) | [`regime-allocation`](regime-allocation.ipynb) | 2012-03-06 to 2026-10-16 | Regime mean-variance Sharpe 0.37 vs static 0.50, equal weight 0.55; lost in 0 of 7 settings |
| Volatility forecasting vs EWMA, GARCH, HAR, VIX | [`regime-vol-test`](regime-vol-test.ipynb) | Nifty 2012-03 to 2026-09; S&P 2003-02 to 2026-09 | Not better than EWMA (21-day QLIKE, Diebold-Mariano p = 0.48) |
| Correlation regimes (HMM on joint returns) | [`regime-correlation-test`](regime-correlation-test.ipynb) | US SPY/TLT/GLD 2007-11 to 2026-10; India 2012-02 to 2026-10 | Correlations do not differ across regimes (p = 0.49 / 0.11); worse than EWMA covariance |
| Statistical jump model vs vol-targeting | [`regime-jump-model`](regime-jump-model.ipynb) | Nifty 2013-03 to 2026-10; S&P 2004-02 to 2026-10 | More persistent regimes and shallower drawdowns, Sharpe gain not significant |
| Jump model vs binary rules (200-day MA, VIX hysteresis) | [`regime-jump-vs-binary`](regime-jump-vs-binary.ipynb) | same | Wins on Nifty (Calmar 0.64 vs 0.32 / 0.21), ties on S&P; the rule needed both |
| Events, transitions, detection lag | [`regime-event-diagnostics`](regime-event-diagnostics.ipynb) | Nifty 2012 to 2026 | No link to Budget or election days; losses not concentrated in transitions; jump-model flag lags a drawdown peak by about 15 trading days (HMM stress flag about 66) |

The jump model follows [Shu, Yu and Mulvey (2024), arXiv 2402.05272](https://arxiv.org/abs/2402.05272).

## Repository map

```
regime_filter.py          walk-forward HMM, forward (filtered) algorithm, soft-exposure overlay, backtest, metrics
regime_allocation.py      regime-conditional multi-asset allocation, data download and outlier filter
vol_forecast_eval.py      volatility forecasters, QLIKE, Diebold-Mariano, incremental-information regressions
corr_regime_eval.py       joint-return HMM, EWMA covariance baselines, permutation test for correlation regimes
jump_model.py             statistical jump model, vol-target and binary-rule benchmarks, Calmar bootstrap
event_diagnostics.py      event coincidence test, loss concentration, drawdown episodes, detection lag
indicators.py             standalone, tested RSI / CCI / ADX math mirroring the legacy notebooks
tests/                    unit tests (causality, filter vs hmmlearn, DP vs brute force, permutation calibration)
regime-*.ipynb            the study, one notebook per question (executed, outputs saved)
macro-regime.ipynb, micro-regime.ipynb, regime_prediction.ipynb
                          legacy first-version notebooks (look-ahead; kept for history)
NIFTY.csv, VIX.csv        raw legacy input data
docs/legacy_v1_README.md  superseded first write-up
```

Market data for the newer tests is downloaded with `yfinance` on first run and cached as CSV
files (git-ignored). Prices are price indices (no dividends); cash earns a flat 6% for India and
2% for the US.

## Reproducing

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv/Scripts/python -m ipykernel install --user --name marketregime --display-name "marketRegime (.venv)"
.venv/Scripts/python -m pytest          # unit tests
```

Open the `regime-*.ipynb` notebooks with the `marketregime` kernel and run top to bottom. Each
takes roughly 5 to 30 minutes because of the walk-forward refits. The notebooks that depend on
cached downloads fetch them on the first run, so an internet connection is needed once.

## Method notes

- **Filtered, not smoothed.** Regime probabilities use data up to day *t* only.
- **Execution.** Signal at close *t* trades at close *t+1* and first earns day *t+2* (the early
  index-overlay test used a one-day lag); costs of 5 to 10 bps per unit traded.
- **Baselines.** Buy and hold, EWMA vol-targeting (lambda 0.94, 12% target, 100% cap), VIX
  vol-targeting, 200-day MA, VIX hysteresis, equal weight, inverse-vol, static mean-variance.
- **Statistics.** Moving-block bootstrap (Sharpe, Calmar), Diebold-Mariano with HAC errors
  (QLIKE, MSE, log-score), HAC regressions, circular-shift permutation tests.

## Limitations

About 13 to 17 years of data hold only a handful of independent stress episodes, so "not
distinguishable" is not "equal". Price indices without dividends, a flat cash rate, Gaussian
emissions, and price/VIX-only inputs. Covariate-driven transitions (FII flows, yield spreads) and
RBI-policy events were not tested.
