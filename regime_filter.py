"""
Walk-forward, causal (filtered) HMM regime probabilities + a confidence-weighted
risk overlay for the Nifty.

Design rules (why this differs from macro-regime / micro-regime notebooks):
  * Regime probabilities are FILTERED: P(state_t | x_1..x_t) from the forward pass
    only. Viterbi / predict_proba (smoothed) use data after t and are look-ahead.
  * Parameters are re-estimated walk-forward. At each refit date the scaler and the
    HMM see only data strictly before the fold's out-of-sample window.
  * State labels are not stable across refits, so states are ordered by their
    (training-estimated) return volatility: state 0 = calmest ... state n-1 = most
    stressed.
  * The signal known at close t is traded at close t+1 (one-day lag), with costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

EPS = 1e-12
TRADING_DAYS = 252


# --------------------------------------------------------------------------- data
def load_data(nifty_path="NIFTY.csv", vix_path="VIX.csv") -> pd.DataFrame:
    """Date-aligned Nifty close + VIX close. Date formats differ between the files
    (see CRITIQUE.md S1.1): NIFTY is DD/MM/YY, VIX is MM/DD/YY."""
    nifty = pd.read_csv(nifty_path)
    vix = pd.read_csv(vix_path)
    nifty["Date"] = pd.to_datetime(nifty["Date"], dayfirst=True)
    vix["Date"] = pd.to_datetime(vix["Date"], dayfirst=False)
    nifty = nifty.set_index("Date").sort_index()
    vix = vix.set_index("Date").sort_index()
    df = pd.DataFrame({"close": nifty["Close"], "vix": vix["Close"]}).dropna()
    return df[~df.index.duplicated(keep="last")]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compact, causal feature set. Every value at t uses data <= t only."""
    log_close = np.log(df["close"])
    lr = log_close.diff()
    log_vix = np.log(df["vix"])
    feats = pd.DataFrame(
        {
            "ret_5d": log_close.diff(5),
            "log_rv_20d": np.log(lr.rolling(20).std() + EPS),
            "log_vix": log_vix,
            "vix_chg_5d": log_vix.diff(5),
        }
    )
    return feats.dropna()


# ------------------------------------------------------------------ HMM utilities
def fit_hmm(X, n_states, cov_type="diag", n_init=10, seed=42, n_iter=300, tol=1e-4):
    """Best-of-n_init GaussianHMM (distinct seed per restart, degenerate fits skipped)."""
    best, best_score = None, -np.inf
    for i in range(n_init):
        try:
            m = GaussianHMM(
                n_components=n_states,
                covariance_type=cov_type,
                n_iter=n_iter,
                tol=tol,
                random_state=seed + i,
            ).fit(X)
            s = m.score(X)
        except ValueError:
            continue
        if np.isfinite(s) and s > best_score:
            best, best_score = m, s
    if best is None:
        raise RuntimeError("All HMM restarts degenerated")
    return best


def emission_loglik(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """(T, n_states) log N(x_t | mu_k, Sigma_k). Works for diag/full covariances."""
    covs = model.covars_
    out = np.empty((len(X), model.n_components))
    for k in range(model.n_components):
        out[:, k] = multivariate_normal.logpdf(X, mean=model.means_[k], cov=covs[k])
    return out


def forward_filter(model: GaussianHMM, X: np.ndarray):
    """Causal forward pass.

    Returns
    -------
    alpha : (T, n) filtered probabilities P(s_t | x_1..x_t)
    ll    : (T,)  one-step predictive log-density log p(x_t | x_1..x_{t-1})
            (sum over a window = out-of-sample log-likelihood of that window)
    """
    logB = emission_loglik(model, X)
    with np.errstate(divide="ignore"):
        logA = np.log(model.transmat_)
        logpi = np.log(model.startprob_)
    T, n = logB.shape
    log_alpha = np.empty((T, n))
    ll = np.empty(T)

    a = logpi + logB[0]
    ll[0] = logsumexp(a)
    log_alpha[0] = a - ll[0]
    for t in range(1, T):
        pred = logsumexp(log_alpha[t - 1][:, None] + logA, axis=0)
        a = pred + logB[t]
        ll[t] = logsumexp(a)
        log_alpha[t] = a - ll[t]
    return np.exp(log_alpha), ll


# ------------------------------------------------------------------- walk-forward
@dataclass
class WFConfig:
    n_states: int = 3
    cov_type: str = "diag"
    warmup_days: int = 756  # ~3y
    refit_days: int = 126  # ~6m
    n_init: int = 10
    seed: int = 42


@dataclass
class WFResult:
    probs: pd.DataFrame  # filtered P(state), columns ordered calm -> stressed
    smoothed: pd.DataFrame  # same-fold smoothed P (hindsight), for diagnostics only
    state_vol: pd.DataFrame  # annualised return vol per state, from the fold's TRAIN data
    oos_ll: pd.Series  # one-step predictive log-density
    fold_id: pd.Series
    transmats: list  # per-fold transition matrix, ordered calm -> stressed
    folds: list = None  # per-fold dict: s, e, gamma_tr (smoothed train posteriors, ordered)


def _state_vol(gamma, ret):
    """Posterior-weighted daily return std per state (annualised)."""
    w = gamma.sum(0)
    mu = gamma.T @ ret / np.maximum(w, EPS)
    var = (gamma * (ret[:, None] - mu[None, :]) ** 2).sum(0) / np.maximum(w, EPS)
    vol = np.sqrt(var) * np.sqrt(TRADING_DAYS)
    # a state with almost no training mass: fall back to the unconditional vol
    return np.where(w < 5, ret.std() * np.sqrt(TRADING_DAYS), vol)


def walk_forward(feats: pd.DataFrame, ret: pd.Series, cfg: WFConfig = WFConfig()) -> WFResult:
    """Expanding-window walk-forward. `ret` = daily simple returns aligned to feats."""
    ret = ret.reindex(feats.index)
    X_raw = feats.to_numpy()
    r = ret.to_numpy()
    T, n = len(feats), cfg.n_states
    if cfg.warmup_days >= T:
        raise ValueError("warmup_days >= available rows")

    P, S, V, LL, FID = [], [], [], [], []
    transmats, folds = [], []
    for fold, s in enumerate(range(cfg.warmup_days, T, cfg.refit_days)):
        e = min(s + cfg.refit_days, T)

        scaler = StandardScaler().fit(X_raw[:s])  # train-only statistics
        X = scaler.transform(X_raw[:e])
        model = fit_hmm(X[:s], n, cfg.cov_type, cfg.n_init, cfg.seed)

        # order states calm -> stressed by training return-vol (in-sample smoothed
        # posterior is fine for *parameter estimation*; it never reaches the signal)
        gamma_tr = model.predict_proba(X[:s])
        vol_k = _state_vol(gamma_tr, r[:s])
        order = np.argsort(vol_k)

        alpha, ll = forward_filter(model, X)  # causal; includes the burn-in rows
        smoothed = model.predict_proba(X)  # hindsight within [0, e), diagnostics only

        idx = feats.index[s:e]
        P.append(pd.DataFrame(alpha[s:e][:, order], index=idx))
        S.append(pd.DataFrame(smoothed[s:e][:, order], index=idx))
        V.append(pd.DataFrame(np.tile(vol_k[order], (e - s, 1)), index=idx))
        LL.append(pd.Series(ll[s:e], index=idx))
        FID.append(pd.Series(fold, index=idx))
        transmats.append(model.transmat_[np.ix_(order, order)])
        folds.append({"s": s, "e": e, "gamma_tr": gamma_tr[:, order]})

    cols = [f"S{k}" for k in range(n)]
    probs, smoothed, state_vol = pd.concat(P), pd.concat(S), pd.concat(V)
    probs.columns = smoothed.columns = state_vol.columns = cols
    return WFResult(probs, smoothed, state_vol, pd.concat(LL), pd.concat(FID), transmats, folds)


# ------------------------------------------------------------- risk overlay + eval
def ema_probs(probs: pd.DataFrame, span: int) -> pd.DataFrame:
    """Causal EMA of the probability vector (stays on the simplex). span<=1 = off."""
    if span <= 1:
        return probs
    return probs.ewm(span=span, adjust=False).mean()


def soft_exposure(probs, state_vol, target_vol=0.12, cap=1.0, hard=False) -> pd.Series:
    """exposure = sum_k p_k * min(cap, target_vol / vol_k).  hard=True uses argmax only."""
    state_expo = (target_vol / state_vol).clip(upper=cap)
    if hard:
        onehot = pd.DataFrame(
            np.eye(probs.shape[1])[probs.to_numpy().argmax(1)], index=probs.index, columns=probs.columns
        )
        return (onehot * state_expo).sum(1)
    return (probs * state_expo).sum(1)


def vol_target_exposure(feats: pd.DataFrame, index, target_vol=0.12, cap=1.0) -> pd.Series:
    """Plain trailing-realised-vol targeting (no HMM): the benchmark the HMM must beat."""
    rv = np.exp(feats["log_rv_20d"]) * np.sqrt(TRADING_DAYS)
    return (target_vol / rv).clip(upper=cap).reindex(index)


def backtest(ret, exposure, rf_annual=0.06, cost_bps=5.0) -> pd.Series:
    """exposure_t is known at close t and held over day t+1.  Starts on the 2nd day."""
    ret = ret.reindex(exposure.index)
    pos = exposure.shift(1)
    rf_d = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    turnover = pos.diff().abs()
    out = pos * ret + (1 - pos) * rf_d - turnover * cost_bps / 1e4
    return out.iloc[2:]  # drop the NaN lag day and the NaN first turnover


def perf(strategy_ret: pd.Series, exposure: pd.Series | None = None, rf_annual=0.06) -> pd.Series:
    r = strategy_ret.dropna()
    rf_d = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    ex = r - rf_d
    sharpe = ex.mean() / r.std() * np.sqrt(TRADING_DAYS)
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = ex.mean() * TRADING_DAYS / downside
    mdd = (eq / eq.cummax() - 1).min()
    d = {
        "CAGR": cagr,
        "AnnVol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": mdd,
        "Calmar": cagr / abs(mdd),
    }
    if exposure is not None:
        e = exposure.reindex(r.index)
        d["AvgExposure"] = e.mean()
        d["AnnTurnover"] = e.diff().abs().mean() * TRADING_DAYS
    return pd.Series(d)


def block_bootstrap_sharpe_diff(a, b, rf_annual=0.06, block=20, n_boot=2000, seed=0):
    """Paired moving-block bootstrap of Sharpe(a) - Sharpe(b). Returns (point, lo, hi, p_le0)."""
    df = pd.concat([a, b], axis=1).dropna()
    rf_d = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    x = df.to_numpy() - rf_d
    T = len(x)
    rng = np.random.default_rng(seed)

    def sharpe_diff(z):
        s = z.mean(0) / z.std(0, ddof=1) * np.sqrt(TRADING_DAYS)
        return s[0] - s[1]

    point = sharpe_diff(x)
    n_blocks = int(np.ceil(T / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T - block + 1, n_blocks)
        z = np.concatenate([x[s : s + block] for s in starts])[:T]
        diffs[i] = sharpe_diff(z)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi, float((diffs <= 0).mean())


# --------------------------------------------------------------------- diagnostics
def entropy(probs: pd.DataFrame) -> pd.Series:
    """Normalised Shannon entropy in [0,1]: 0 = certain, 1 = uniform."""
    p = probs.clip(lower=EPS)
    return -(p * np.log(p)).sum(1) / np.log(probs.shape[1])


def run_lengths(labels: pd.Series) -> pd.DataFrame:
    grp = labels.ne(labels.shift()).cumsum()
    runs = labels.groupby(grp).agg(["first", "size"])
    return runs.rename(columns={"first": "state", "size": "duration"})
