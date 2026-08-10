# Market Regime Detection — NIFTY 50 / India VIX

A two-layer, unsupervised framework for identifying market regimes in the Indian equity
market (NIFTY 50) using a slow "macro" layer (40-80 day features) and a fast "micro" layer
(3-14 day features), fit with Gaussian Hidden Markov Models. This document covers the data,
the modeling decisions and the reasoning behind them, and the final results.

---

## 1. Executive summary

- Built two independent unsupervised regime detectors on ~6.5 years of NIFTY/VIX daily data:
  a **macro** layer (6 regimes, slow-moving) and a **micro** layer (3 regimes: an ordinary
  "Normal" state, a real "Elevated_Vol" tier, and an extreme "Extreme_Crisis" outlier — the
  COVID crash).
- Regimes are **highly persistent** (macro regimes last weeks to months) but **not
  predictable ahead of time** — an XGBoost classifier, evaluated four different ways (single
  holdout, 5-fold walk-forward, macro alone, micro alone, combined), never beat the trivial
  "assume tomorrow looks like today" baseline.
- Despite that, **simply reacting to the current regime has real economic value**: a
  no-forecasting, regime-conditional position rule improved Sharpe from 0.78 to 1.15 and cut
  max drawdown from -38% to -11% versus buy-and-hold, at a small cost in CAGR (13.5%→12.3%),
  with only ~3.5 position changes per year.
- The regime labels themselves are only **moderately stable** to how much history informed
  them (Adjusted Rand Index ~0.39-0.44 between a partial-history fit and the full-history
  fit) — a genuine caveat, not a technicality, for anyone building on top of this.

---

## 2. Objective

Given daily NIFTY 50 (price) and India VIX (implied volatility) data, identify recurring
"regimes" — statistically distinct market states — at two timescales:

- **Macro regime**: the slow-moving backdrop (trend direction, drawdown depth, medium-term
  volatility) — the kind of state that matters for strategic asset allocation.
- **Micro regime**: the fast-moving texture of the last 1-3 weeks (choppiness, short-term
  realized vol, trend efficiency) — the kind of state that matters for tactical/execution
  decisions.

The end-to-end question this project asks: **can these regimes be usefully identified, are
they predictable in advance, and do they carry economic value if traded on?**

---

## 3. Data

| File | Rows | Columns | Coverage |
|---|---|---|---|
| `NIFTY.csv` | 1,637 | Date, Open, High, Low, Close | 2019-01-02 → 2025-12-08 |
| `VIX.csv` | 2,138 | Date, Open, High, Low, Close | 2017-01-02 → 2025-08-21 |

Both are daily OHLC series; VIX here refers to India VIX, NSE's implied-volatility index on
NIFTY options (not the CBOE VIX).

### 3.1 Date handling

`NIFTY.csv` and `VIX.csv` use different date conventions — `NIFTY.csv` is `DD/MM/YY`,
`VIX.csv` is `MM/DD/YY`. Both are parsed with their convention made explicit
(`dayfirst=True` / `dayfirst=False`) rather than relying on automatic inference. This matters:
for any date with day ≤ 12 the two conventions are ambiguous, and a wrong guess silently
produces a different but still valid-looking calendar date rather than an error — around 40%
of the rows in `NIFTY.csv` fall in that ambiguous range, so getting the convention right
per-file, rather than assuming one global default, is what keeps the two series correctly
aligned on the same timeline.

### 3.2 What the raw series looks like

NIFTY over the sample period covers a genuinely diverse range of conditions: the 2020 COVID
crash and V-shaped recovery, a multi-year bull run, and several shallower corrections —
important, because a regime detector trained on a single-regime sample can't be validated
against anything. India VIX ranges from the low-teens (calm) to the 80s (the COVID peak),
giving real separation between "calm" and "stressed" states for the models to find.

---

## 4. Methodology

### 4.1 Why two layers instead of one

A single feature set spanning both timescales would force one HMM to represent both "is this
a bull or bear market" and "is today choppy or trending" simultaneously — two questions with
different natural update frequencies. Keeping them separate lets each layer use window
lengths appropriate to its own timescale, and lets the two be recombined later (§5.3) without
forcing a single model to do both jobs at once.

### 4.2 Feature engineering

**Macro layer** (computed on a 7-day-smoothed NIFTY close, 40-80 day windows):

| Feature | What it captures |
|---|---|
| `RSI_40d` | Long-window overbought/oversold momentum |
| `short_40_long_80_term_diff` | Medium-term trend (40d MA vs 80d MA spread) |
| `max_drawdown_40d` | Depth of decline from a rolling peak |
| `sharpe_like_40d` | Risk-adjusted momentum (40d return / 40d vol) |
| `volatility_40d` | Realized volatility |
| `VIX_rolling_mean_30d`, `VIX_rolling_std_30d` | Implied-vol level and its own volatility |
| `nifty_vix_ratio` | Price relative to implied fear level |

**Micro layer** (raw NIFTY, 3-14 day windows):

| Feature | What it captures |
|---|---|
| `NIR` | Normalized daily range (intraday volatility proxy) |
| `RV` | Short-window realized volatility |
| `KER` | Kaufman efficiency ratio — net move / sum of absolute moves (trending vs. choppy) |
| `ADX`, `VIX_ADX` | Simplified directional-trend strength, and a fear-weighted version |
| `CCI` | Commodity Channel Index (short-term momentum/mean-reversion) |
| `VIX`, `VIX_Change`, `VIX_smooth`, `NIR_VIX` | Level, change, and smoothed India VIX, plus its own daily range |

Both extractors are careful about one thing that's easy to get wrong: RSI, CCI, and the
directional-index calculations all involve a division that can hit zero on a degenerate
window (a perfectly flat run of prices). Each is guarded with a small epsilon so a flat
window produces a well-defined 0 rather than `NaN`/`inf` propagating silently downstream —
verified with a small unit-test suite (`tests/test_indicators.py`) using exactly these
edge cases.

### 4.3 Unsupervised regime detection: Gaussian HMM

Both layers use a `GaussianHMM` (full covariance) fit on the engineered features, decoded via
Viterbi to assign each day a regime label. Three modeling decisions mattered enough to be
worth explaining:

**Scaling: expanding-window, not a single global scaler.** Fitting one `StandardScaler` on
the entire history and using it everywhere leaks every future mean/std into every historical
row's normalized value — a look-ahead-bias problem for anything meant to reflect what would
have been knowable at the time. Each feature is instead normalized using only data up to and
including that row (`expanding().mean()`/`.std()`), used only as the HMM's input; the
exported data stays in raw, real units throughout.

**Model selection: AIC/BIC with genuinely diverse restarts.** The number of HMM components
(regimes) was chosen by comparing AIC, BIC, and log-likelihood across candidate values 1
through 11, taking the best of 10 random restarts per candidate (EM is initialization-sensitive).
For macro, both AIC and BIC bottom out at **n=10** within the tested range and rise again at
n=11 — a real, if modest, minimum. **6 was still chosen for macro over the criterion-selected
10, on interpretability grounds** — worth saying plainly rather than implying the final
component count was purely statistics-driven.

**Merging near-duplicate clusters: a quantitative criterion, not eyeballing.** Fitting with 6
components sometimes produces clusters that are statistically indistinguishable (two
component means very close together in the normalized feature space) — an artifact of
requesting more components than the data cleanly supports. Rather than merging clusters that
"look similar" in a printed table, cluster centres are compared pairwise (Euclidean distance
in standardized space), and clusters connected by a distance below a stated threshold
(1.5) are merged via connected components. This is not perfectly objective — the threshold
value itself is a judgment call, and it produced a very different outcome for each layer (see
§5) — but it's a stated, reproducible rule instead of a visual impression.

### 4.4 Regime interpretation

Once regimes are assigned, each cluster is labeled by inspecting the **mean of the raw
(unscaled) features** for the days assigned to it — e.g. a cluster with high RSI, positive
trend, and low realized volatility gets labeled `Bullish_LowVol`. This keeps labels grounded
in real, interpretable units rather than opaque cluster indices.

---

## 5. Results

### 5.1 Macro regimes — 6 regimes

The quantitative merge check found no pair close enough to combine (closest pair sits at
1.69, just above the 1.5 threshold) — all 6 kept distinct.

| Regime | Share of days | RSI | 40d trend spread | Max drawdown | 40d "Sharpe" | Realized vol | VIX level |
|---|---|---|---|---|---|---|---|
| Neutral_VeryLowVol | 45.6% | 54.7 | +0.006 | -5.1% | 0.3 | 0.31% | 15.7 |
| Bullish_LowVol | 18.8% | 71.0 | +0.038 | -2.8% | 24.6 | 0.35% | 20.3 |
| Bullish_VeryLowVol | 18.2% | 78.0 | +0.022 | -1.7% | 33.7 | 0.21% | 14.1 |
| Neutral_LowVol | 11.2% | 55.6 | -0.009 | -9.6% | 8.5 | 0.46% | 20.6 |
| Bearish_HighVol | 3.6% | 24.1 | -0.080 | -27.1% | -17.6 | 1.16% | 40.3 |
| Bearish_VeryLowVol | 2.6% | 33.9 | -0.012 | -7.3% | -25.8 | 0.25% | 14.6 |

*(Vol figures are daily realized vol; VIX level is a 30-day rolling mean of India VIX.)*

The two bearish regimes are the most interesting pair: `Bearish_HighVol` is the crash regime
(deep drawdown, elevated implied vol — this is where the 2020 COVID crash lands).
`Bearish_VeryLowVol` is a **slow, grinding decline** — negative momentum with *low* realized
and implied volatility. Distinguishing these matters economically: a slow bleed and a panic
crash call for different responses, and collapsing them into a single "bearish" bucket would
lose that distinction entirely.

**Persistence** (average streak length): `Neutral_VeryLowVol` streaks average ~69 trading
days across 10 occurrences; `Bullish_LowVol` streaks average ~95 days across 3 occurrences;
`Bearish_HighVol` occurred once, lasting 54 days. Regimes behave like regimes — they don't
flip day to day.

### 5.2 Micro regimes — 3 regimes

Under expanding-window scaling, the quantitative merge check found that **4 of the original 6
candidate clusters are mutually close together** — a continuum of "ordinary market"
conditions (VIX roughly 12-20, mildly trending to moderately choppy) — while a fifth cluster
forms a distinct, moderately-frequent middle tier, and a sixth stands drastically apart from
everything:

| Regime | Share of days | RV | VIX level | VIX daily change |
|---|---|---|---|---|
| Normal | 87.5% | 1.2-2.2% | 12.3-20.5 | — |
| Elevated_Vol | 11.4% | 3.5% | 25.7 | — |
| Extreme_Crisis | 1.1% (18 days) | 10.1% | 53.0 | +8.9%/day |

`Extreme_Crisis`'s 18 days run **2020-02-28 through 2020-03-27** — the COVID crash, and
nothing else in 6.5 years of data comes close to it in this feature space. `Elevated_Vol` is
a real, distinct, non-trivial regime — a genuine middle tier between ordinary conditions and
outright crisis, not just a fuzzy boundary between the two.

### 5.3 Combined regime

Joining the two layers on date gives **12 combined regimes** — mostly macro's 6-way split
further divided by which of micro's 3 states co-occurs with it. Notably, 17 of 18
`Extreme_Crisis` days co-occur with macro's `Bearish_HighVol` (the 1 exception is 2020-02-28,
right at the crash's onset, before the slower macro layer had caught up), and most
`Elevated_Vol` days co-occur with either `Bearish_HighVol` or `Neutral_LowVol`: the two
independently-fit models (different features, different timescales, no shared inputs)
broadly agree on *when* conditions were worst, even though — as §5.4 shows — neither can
predict it coming.

### 5.4 Are regime labels stable?

Refitting the macro HMM on only the first 80% of history and comparing its regime
assignments to the full-history fit (via Adjusted Rand Index, which is invariant to
arbitrary cluster-index relabeling) gives **ARI = 0.39** for macro and **0.44** for micro,
where 1.0 would mean identical grouping and 0.0 means no better than random. This is a real,
non-trivial caveat: **the regime label assigned to a given historical day is not fixed** — it
depends on how much subsequent history the model was allowed to see when it was fit. These
labels should be read as "this is what a retrospective, full-sample analysis says," not as a
stable ground truth that a live system would reproduce day by day.

### 5.5 Can regimes be predicted in advance?

An XGBoost classifier was evaluated on whether tomorrow's regime can be predicted from
today's features — using a forward-looking target, a chronological/walk-forward split, and a
"naive: assume no change" baseline reported alongside for honest comparison. Tested four ways:

| Evaluation | Macro naive | Macro model | Macro lift | Micro naive | Micro model | Micro lift |
|---|---|---|---|---|---|---|
| Single chronological holdout | 98.4% | 71.7% | **-26.6pp** | 97.5% | 90.6% | **-6.9pp** |
| 5-fold walk-forward (mean) | 98.6% | 61.8% | **-36.8pp** | 98.9% | 76.4% | **-22.5pp** |

| Evaluation | Combined naive | Combined model | Combined lift |
|---|---|---|---|
| Single chronological holdout | 97.0% | 68.4% | **-28.6pp** |
| 5-fold walk-forward (mean) | 97.9% | 59.1% | **-38.9pp** |

**In every configuration tried, the model loses to trivially assuming no regime change.**
This isn't a fluke of one unlucky test window — the walk-forward result, averaged over 5
independent folds spanning the entire back 80% of the dataset, is worse than the
single-holdout result for every target. Micro's gap is particularly large in the
walk-forward evaluation (-22.5pp) relative to the single holdout (-6.9pp): once evaluated
against the recurring `Elevated_Vol` regime across several different time periods rather than
one test window, the classifier's difficulty becomes much more apparent. None of the
engineered technical/volatility features tested here carry detectable forward-looking
information about regime *transitions*, beyond what persistence alone already tells you.

*(One caveat specific to `Extreme_Crisis`: it occurs only in February-March 2020, entirely
inside the training period for every evaluation method used — a walk-forward test fold, by
construction, always comes after its training fold, so an early-only rare event can never
appear in any test set. This is a structural limit of evaluating on one non-repeating
historical series, not a flaw in the evaluation code. It doesn't affect `Elevated_Vol`, which
recurs throughout the dataset and is fully represented in later test folds — the -22.5pp
walk-forward gap above is a genuine test of the model against a recurring regime.)*

### 5.6 Does the regime signal have economic value anyway?

Given regimes are highly persistent but not predictable ahead of time, the natural test is
different: **not** "can I forecast the next regime," but "does reacting to the current,
already-known regime help?" A simple backtest answers this:

- Position sized by yesterday's macro regime (no forecasting — lagged by one day so the
  signal was genuinely knowable before the trading day it's applied to): `Bullish_* → 100%`
  exposure, `Neutral_* → 50%`, `Bearish_* → 0%`.
- Uses raw NIFTY close (not the 7-day-smoothed price used internally for macro features,
  which would understate real volatility and flatter the results).
- No transaction-cost modeling, but turnover is low enough (21 position changes over 6 years,
  ~3.5/year) that costs would barely move the numbers.

| | Buy & Hold | Regime strategy |
|---|---|---|
| CAGR | 13.48% | 12.32% |
| Annualized volatility | 18.29% | 10.62% |
| Sharpe ratio | 0.78 | **1.15** |
| Max drawdown | -38.44% | **-10.67%** |

A meaningful risk-adjusted improvement, for a small cost in raw return — consistent with
§5.5's finding rather than contradicting it: this strategy doesn't predict anything, it
reacts to information that's free to observe (today's regime) and relies on persistence
(§5.1) to keep that reaction valid for the next day. The one caveat carried over from §5.4:
because the regime *taxonomy* was fit in-sample on the full history, this shows what a
retrospectively-labeled system would have achieved, not a guarantee for a version whose
regimes are refit walk-forward in production (see §7, Limitations and future work).

---

## 6. Key findings

1. **Regimes are real and persistent, but not predictable ahead of time.** Every prediction
   framing tried failed the same way, robustly. The economically useful move is reacting to
   the current regime, not forecasting the next one.
2. **"How many regimes exist" is a modeling choice, not just a fact read off a statistical
   criterion.** Macro's AIC/BIC curve does have a real minimum within the tested range — at
   n=10, not n=6. 6 was still used, chosen for interpretability over the criterion-selected
   value. Treat regime *counts*, in general, as provisional modeling choices shaped by what's
   useful to interpret, not pure discoveries.
3. **Regime labels depend on how much future data the model has seen** (§5.4) — a genuine
   epistemic limitation on treating HMM-derived regimes as objective market states.
4. **A single event can dominate a regime's identity.** Micro's `Extreme_Crisis` regime is,
   in effect, a detector for "is this as extreme as March 2020" — informative, but built from
   one historical instance.
5. **The two independently-fit layers agree on timing, not just structure.** Macro and micro
   use completely different features and timescales with no shared inputs, yet 17 of 18
   `Extreme_Crisis` days (micro) coincide with `Bearish_HighVol` days (macro) — independent
   evidence that both are capturing something real about market stress, even though neither
   can forecast it in advance.

---

## 7. Limitations and future work

- **Regimes are fit in-sample, once.** The single most important next step for any live use:
  refit the HMM walk-forward (expanding window, periodically) rather than once on the full
  history, and re-run §5.4's stability check and §5.6's backtest against those labels
  specifically.
- **Backtest position sizes were hand-picked** (1.0 / 0.5 / 0.0), not fit. A regime-conditional
  sizing rule derived from each regime's own historical forward-return/volatility profile
  (a regime-aware vol-targeting or Kelly-style overlay) is a natural, likely-superior next
  step.
- **Micro's `Extreme_Crisis` flag isn't used in the backtest at all.** Layering it in as a
  tail-risk override on top of the macro-driven position (e.g., force exposure to zero
  whenever it fires) is untested but straightforward to add.
- **No transaction costs or slippage modeled**, though turnover is low enough that this is a
  minor concern for the current strategy specifically.
- **Single asset, single market.** Everything here is NIFTY/India VIX; nothing about the
  approach is India-specific, but no other market was tested.
- **The notebooks carry their own copies of the indicator math** rather than importing from
  `indicators.py` (the tested, standalone version) — consolidating them into a single source
  of truth is a natural next step.

---

## 8. Repository structure

```
NIFTY.csv, VIX.csv                     raw daily OHLC data
macro-regime.ipynb                     macro feature engineering + HMM regime detection
micro-regime.ipynb                     micro feature engineering + HMM regime detection
regime_prediction.ipynb                predictability tests, combined regime, backtest
macro_features_with_regimes.csv        macro-regime notebook output (generated)
micro_regimes_classified.csv           micro-regime notebook output (generated)
indicators.py                          standalone, tested indicator math (RSI/CCI/ADX)
tests/test_indicators.py               reference-value tests for indicators.py
requirements.txt                       pinned dependencies
```

## 9. Reproducing this

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv/Scripts/python -m ipykernel install --user --name marketregime --display-name "marketRegime (.venv)"
```

Run the notebooks in order — `macro-regime.ipynb` and `micro-regime.ipynb` first (either
order, they're independent) to regenerate the two CSVs, then `regime_prediction.ipynb`.

```bash
.venv/Scripts/python -m pytest tests/ -v
```
