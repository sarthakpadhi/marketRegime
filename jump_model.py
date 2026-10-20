"""
Statistical jump model (SJM), following Shu, Yu & Mulvey (arXiv 2402.05272) as closely as
the available history allows.

  minimise   sum_t 0.5 * ||x_t - mu_{s_t}||^2  +  lambda * sum_t 1[s_t != s_{t-1}]

fit by coordinate descent: (i) states by dynamic programming given centroids,
(ii) centroids = state means; best of `n_init` random starts.

ONLINE (causal) inference: with centroids fixed, the regime on day t is the LAST state of
the optimal path over data ending at t. That is exactly argmin_k V_t(k) of the forward
recursion V_t(k) = d_t(k) + min(V_{t-1}(k), min_j V_{t-1}(j) + lambda), so one forward pass
yields every day's online signal (tested against re-running the full DP on each prefix).
Because V_t(k) - min_j V_t(j) <= lambda, the start of the window has bounded influence,
which is why a finite lookback (the paper's 3,000 days) and a full expanding pass agree
up to the usual edge effects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import regime_filter as rf

TRADING_DAYS = rf.TRADING_DAYS


# ----------------------------------------------------------------------------- features
def sjm_features(ret: pd.Series, rf_annual: float) -> pd.DataFrame:
    """Paper's three return-only features (all causal EWMAs of excess returns):
    downside deviation (half-life 10), Sortino ratios (half-lives 20 and 60)."""
    rf_d = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    ex = ret - rf_d
    neg2 = ex.clip(upper=0) ** 2

    def dd(h):
        return np.sqrt(neg2.ewm(halflife=h).mean())

    def sortino(h):
        return ex.ewm(halflife=h).mean() / (dd(h) + 1e-8)

    return pd.DataFrame({"dd_10": dd(10), "sortino_20": sortino(20), "sortino_60": sortino(60)})


# ---------------------------------------------------------------------------- core maths
def sq_dist(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """(T,K) of 0.5*||x_t - mu_k||^2."""
    return 0.5 * ((X[:, None, :] - mu[None, :, :]) ** 2).sum(2)


def dp_viterbi(D: np.ndarray, lam: float) -> np.ndarray:
    """Optimal state path for loss matrix D (T,K) and jump penalty lam."""
    T, K = D.shape
    V = D[0].copy()
    back = np.zeros((T, K), dtype=bool)
    arg = np.zeros(T, dtype=np.int64)
    for t in range(1, T):
        m = int(V.argmin())
        jump = V[m] + lam
        use_jump = jump < V
        arg[t] = m
        back[t] = use_jump
        V = D[t] + np.where(use_jump, jump, V)
    s = np.empty(T, dtype=np.int64)
    s[-1] = int(V.argmin())
    for t in range(T - 1, 0, -1):
        s[t - 1] = arg[t] if back[t, s[t]] else s[t]
    return s


def online_states(D: np.ndarray, lam: float) -> np.ndarray:
    """Causal signal: state_t = argmin_k V_t(k) of the forward recursion."""
    T, K = D.shape
    V = D[0].copy()
    out = np.empty(T, dtype=np.int64)
    out[0] = int(V.argmin())
    for t in range(1, T):
        m = V.argmin()
        V = D[t] + np.minimum(V, V[m] + lam)
        out[t] = int(V.argmin())
    return out


def objective(D: np.ndarray, s: np.ndarray, lam: float) -> float:
    return float(D[np.arange(len(s)), s].sum() + lam * (s[1:] != s[:-1]).sum())


def fit_jm(X: np.ndarray, lam: float, K: int = 2, n_init: int = 10, seed: int = 42, max_iter: int = 50):
    """Best-of-n_init coordinate descent. Returns (centroids, train states, objective)."""
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf)
    for _ in range(n_init):
        mu = X[rng.choice(len(X), K, replace=False)].copy()
        s_prev = None
        for _ in range(max_iter):
            s = dp_viterbi(sq_dist(X, mu), lam)
            if s_prev is not None and np.array_equal(s, s_prev):
                break
            for k in range(K):
                m = s == k
                mu[k] = X[m].mean(0) if m.any() else X[rng.integers(len(X))]
            s_prev = s
        D = sq_dist(X, mu)
        s = dp_viterbi(D, lam)
        obj = objective(D, s, lam)
        if obj < best[2]:
            best = (mu.copy(), s, obj)
    return best


def bull_state(states: np.ndarray, excess_ret: np.ndarray, K: int = 2) -> int:
    """Bull = state with the higher mean excess return on the training sample."""
    means = [excess_ret[states == k].mean() if (states == k).any() else -np.inf for k in range(K)]
    return int(np.argmax(means))


# --------------------------------------------------------------------------- strategies
def net_sharpe(exposure: pd.Series, ret: pd.Series, rf_annual: float, cost_bps: float, extra_delay=1) -> float:
    """Sharpe (excess of rf) of a lagged, cost-charged strategy; used for lambda validation."""
    r = rf.backtest(ret, exposure.shift(extra_delay), rf_annual, cost_bps).dropna()
    if len(r) < 30 or r.std() == 0:
        return -np.inf
    rf_d = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    return float((r - rf_d).mean() / r.std() * np.sqrt(TRADING_DAYS))


@dataclass
class JMConfig:
    lam_grid: tuple = (5, 10, 25, 50, 100, 200)
    warmup_days: int = 1008  # 4y: 2y train + 2y validation at the first fold
    val_days: int = 504  # 2y validation window for lambda selection
    refit_days: int = 126
    n_init: int = 10
    seed: int = 42
    rf_annual: float = 0.06
    cost_bps: float = 10.0
    K: int = 2


def jm_walk_forward(X_df: pd.DataFrame, ret: pd.Series, cfg: JMConfig = JMConfig()):
    """Walk-forward SJM. At each refit: choose lambda by validation Sharpe of the 0/1 strategy
    (centroids fit on data before the validation window), refit on all data before the fold,
    emit causal bull/bear signals for the fold. Returns dict with bull-signal Series (1=bull),
    chosen lambda per fold, and the raw online state path."""
    idx = X_df.index
    Xr = X_df.to_numpy()
    r = ret.reindex(idx)
    rf_d = (1 + cfg.rf_annual) ** (1 / TRADING_DAYS) - 1
    ex = (r - rf_d).to_numpy()
    T = len(idx)
    sig, lams, folds = [], [], []

    for f, s in enumerate(range(cfg.warmup_days, T, cfg.refit_days)):
        e = min(s + cfg.refit_days, T)
        s_tr = s - cfg.val_days

        # ---- choose lambda on the validation window (no data from the fold itself)
        m0, sd0 = Xr[:s_tr].mean(0), Xr[:s_tr].std(0) + 1e-12
        Z0 = (Xr[:s] - m0) / sd0
        best_lam, best_sh = cfg.lam_grid[0], -np.inf
        for lam in cfg.lam_grid:
            mu, st, _ = fit_jm(Z0[:s_tr], lam, cfg.K, cfg.n_init, cfg.seed)
            bull = bull_state(st, ex[:s_tr], cfg.K)
            online = online_states(sq_dist(Z0, mu), lam)
            expo = pd.Series((online == bull).astype(float), index=idx[:s])
            sh = net_sharpe(expo.iloc[s_tr:], r.iloc[s_tr:s], cfg.rf_annual, cfg.cost_bps)
            if sh > best_sh:
                best_sh, best_lam = sh, lam

        # ---- refit on everything before the fold with the chosen lambda
        m1, sd1 = Xr[:s].mean(0), Xr[:s].std(0) + 1e-12
        Z1 = (Xr[:e] - m1) / sd1
        mu, st, _ = fit_jm(Z1[:s], best_lam, cfg.K, cfg.n_init, cfg.seed)
        bull = bull_state(st, ex[:s], cfg.K)
        online = online_states(sq_dist(Z1, mu), best_lam)
        sig.append(pd.Series((online[s:e] == bull).astype(float), index=idx[s:e]))
        lams.append(best_lam)
        folds.append(pd.Series(f, index=idx[s:e]))
    return {"bull": pd.concat(sig), "lams": lams, "fold": pd.concat(folds)}


# ------------------------------------------------------------------------- benchmarks
def ewma_vol_target_exposure(ret: pd.Series, target_vol=0.12, lam=0.94) -> pd.Series:
    """Plain RiskMetrics vol targeting; exposure at close t uses returns <= t."""
    f = np.empty(len(ret))
    v = np.var(ret.to_numpy()[:20])
    for i, x in enumerate(ret.to_numpy()):
        v = lam * v + (1 - lam) * x**2
        f[i] = v
    vol = np.sqrt(f * TRADING_DAYS)
    return pd.Series(np.minimum(1.0, target_vol / vol), index=ret.index)


def vix_vol_target_exposure(ret: pd.Series, vix: pd.Series, target_vol=0.12, calib_window=756) -> pd.Series:
    """Vol targeting off the implied-vol index, rescaled by a causal rolling ratio of
    EWMA realised vol to VIX/100 (VIX embeds a variance premium)."""
    f = np.empty(len(ret))
    v = np.var(ret.to_numpy()[:20])
    for i, x in enumerate(ret.to_numpy()):
        v = 0.94 * v + 0.06 * x**2
        f[i] = v
    realised = pd.Series(np.sqrt(f * TRADING_DAYS), index=ret.index)
    iv = vix.reindex(ret.index) / 100
    c = (realised / iv).rolling(calib_window, min_periods=252).mean()
    return (target_vol / (c * iv)).clip(upper=1.0)


# ------------------------------------------------------------ binary-rule benchmarks
def ma_exposure(close: pd.Series, window: int = 200) -> pd.Series:
    """1 if close > its trailing simple moving average (incl. today), else 0. NaN before `window` days."""
    ma = close.rolling(window).mean()
    out = (close > ma).astype(float)
    out[ma.isna()] = np.nan
    return out


def vix_hysteresis_exposure(vix: pd.Series, window=756, hi_q=0.80, lo_q=0.60, min_periods=252) -> pd.Series:
    """Cash when VIX exceeds its trailing `hi_q` percentile, back in when it falls below the
    trailing `lo_q` percentile, otherwise keep the previous state. Percentiles are causal
    (rolling window including today), so the rule needs no market-specific level."""
    hi = vix.rolling(window, min_periods=min_periods).quantile(hi_q)
    lo = vix.rolling(window, min_periods=min_periods).quantile(lo_q)
    state, out = 1.0, np.full(len(vix), np.nan)
    for i, (v, h, l) in enumerate(zip(vix.to_numpy(), hi.to_numpy(), lo.to_numpy())):
        if np.isnan(h):
            continue
        if v > h:
            state = 0.0
        elif v < l:
            state = 1.0
        out[i] = state
    return pd.Series(out, index=vix.index)


def calmar(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    years = len(r) / TRADING_DAYS
    cagr = eq[-1] ** (1 / years) - 1
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return cagr / abs(mdd) if mdd < 0 else np.inf


def block_bootstrap_calmar_diff(a: pd.Series, b: pd.Series, block=20, n_boot=2000, seed=0):
    """Paired moving-block bootstrap of Calmar(a) - Calmar(b). Returns (point, lo, hi, P(diff<=0))."""
    df = pd.concat([a, b], axis=1).dropna().to_numpy()
    T = len(df)
    rng = np.random.default_rng(seed)
    point = calmar(df[:, 0]) - calmar(df[:, 1])
    n_blocks = int(np.ceil(T / block))
    d = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T - block + 1, n_blocks)
        z = np.concatenate([df[s : s + block] for s in starts])[:T]
        d[i] = calmar(z[:, 0]) - calmar(z[:, 1])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return point, lo, hi, float((d <= 0).mean())
