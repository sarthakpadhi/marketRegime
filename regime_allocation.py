"""
Regime-conditional multi-asset allocation (India: Nifty / Bank / Midcap / Gold / cash).

Uses the walk-forward FILTERED regime probabilities from regime_filter.py. For each
refit fold, per-state return moments are estimated from that fold's TRAIN window only
(posterior-weighted), shrunk toward the pooled moments, then blended with the current
filtered probabilities:

    mu_t    = sum_k p_k mu_k
    Sigma_t = sum_k p_k (Sigma_k + mu_k mu_k') - mu_t mu_t'      (mixture covariance)

and a long-only, capped mean-variance problem is solved. Execution is strict: a
decision made with information through close d trades at close d+1 and first earns
the return of day d+2; trading costs are charged on traded notional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import regime_filter as rf

TICKERS = {
    "nifty": "^NSEI",
    "bank": "^NSEBANK",
    "midcap": "^NSEMDCP50",
    "gold": "GOLDBEES.NS",
    "vix": "^INDIAVIX",
}
RISKY = ["nifty", "bank", "midcap", "gold"]


# --------------------------------------------------------------------------- data
def download_assets(cache="multiasset_prices.csv", start="2007-01-01", refresh=False) -> pd.DataFrame:
    import os

    if os.path.exists(cache) and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import yfinance as yf

    cols = {}
    for name, tk in TICKERS.items():
        d = yf.download(tk, start=start, progress=False, auto_adjust=True)
        s = d["Close"].squeeze()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        cols[name] = s
    px = pd.DataFrame(cols).dropna()
    px = px[~px.index.duplicated(keep="last")]
    px = drop_bad_prints(px, cols=RISKY)
    px.to_csv(cache)
    return px


def drop_bad_prints(px: pd.DataFrame, window=7, tol=0.5, cols=None) -> pd.DataFrame:
    """Drop dates where ANY series sits more than `tol` (50%) away from its centred
    rolling median. Catches vendor errors like GOLDBEES printing 0.336 vs ~33.6 on two
    days in Dec-2019 (a x100 error that would otherwise create a +9,900% 'return').
    Genuine large one-day moves (~10-20%) are far inside this band.

    Apply to PRICE series only (`cols`): a volatility index can legitimately jump >50%
    in a day (VIX on 2024-08-05, +65%), and the filter would wrongly delete that."""
    chk = px[cols] if cols is not None else px
    med = chk.rolling(window, center=True, min_periods=3).median()
    bad = ((chk / med - 1).abs() > tol).any(axis=1)
    if bad.any():
        print(f"drop_bad_prints: removed {int(bad.sum())} dates: {[d.date().isoformat() for d in px.index[bad]]}")
    return px[~bad]


# ------------------------------------------------------------ regime-conditional moments
def state_moments(R: np.ndarray, gamma: np.ndarray, lam_mean=0.7, lam_cov=0.2):
    """Posterior-weighted per-state mean/cov of the (T,A) return matrix `R`, shrunk
    toward the pooled moments. Returns mu_k (K,A), Sig_k (K,A,A), mu_pool, Sig_pool."""
    A = R.shape[1]
    mu_pool = R.mean(0)
    Sig_pool = np.cov(R.T)
    K = gamma.shape[1]
    mus, sigs = np.empty((K, A)), np.empty((K, A, A))
    for k in range(K):
        w = gamma[:, k]
        sw = w.sum()
        if sw < 20:  # state basically unseen in training: fall back to pooled
            mus[k], sigs[k] = mu_pool, Sig_pool
            continue
        mu = (w[:, None] * R).sum(0) / sw
        X = R - mu
        cov = (w[:, None, None] * (X[:, :, None] * X[:, None, :])).sum(0) / sw
        mus[k] = (1 - lam_mean) * mu + lam_mean * mu_pool
        sigs[k] = (1 - lam_cov) * cov + lam_cov * Sig_pool
    return mus, sigs, mu_pool, Sig_pool


def mixture_moments(p, mus, sigs):
    mu = p @ mus
    second = sum(p[k] * (sigs[k] + np.outer(mus[k], mus[k])) for k in range(len(p)))
    return mu, second - np.outer(mu, mu)


def solve_mv(mu, Sig, risk_aversion=4.0, cap=0.6, cash_idx=None, w0=None):
    """max mu'w - (gamma/2) w'Sigma w  s.t. sum w = 1, 0 <= w <= cap (cash cap = 1)."""
    A = len(mu)
    S = Sig + 1e-10 * np.eye(A)
    bounds = [(0.0, 1.0 if i == cash_idx else cap) for i in range(A)]
    w0 = np.full(A, 1.0 / A) if w0 is None else w0
    res = minimize(
        lambda w: -(mu @ w) + 0.5 * risk_aversion * w @ S @ w,
        w0,
        jac=lambda w: -mu + risk_aversion * S @ w,
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(A)}],
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0, None)
    return w / w.sum()


# ------------------------------------------------------------------------ target weights
def build_targets(res, R_risky: pd.DataFrame, rf_annual=0.06, rebal_every=5, risk_aversion=4.0,
                  cap=0.6, lam_mean=0.7, lam_cov=0.2, inv_vol_window=252):
    """Target weights on each decision day for every strategy.

    Returns dict[name] -> DataFrame(index=decision dates, columns=risky + ['cash']).
    """
    P = res.probs
    idx = P.index
    rf_d = (1 + rf_annual) ** (1 / rf.TRADING_DAYS) - 1
    cols = list(R_risky.columns) + ["cash"]
    A = len(cols)
    cash_i = A - 1
    R_all = R_risky.copy()
    R_all["cash"] = rf_d
    Rn = R_all.to_numpy()
    pos_of = {d: i for i, d in enumerate(R_all.index)}

    names = ["Regime MV", "Regime risk-only", "Static MV", "Inverse vol", "Equal weight"]
    out = {n: [] for n in names}
    dates = []

    fold_of_day = res.fold_id.to_numpy()
    fold_cache = {}
    for j in range(0, len(idx), rebal_every):
        d = idx[j]
        f = fold_of_day[j]
        if f not in fold_cache:
            info = res.folds[f]
            s = info["s"]
            Rtr = Rn[:s]
            mus, sigs, mu_pool, Sig_pool = state_moments(Rtr, info["gamma_tr"], lam_mean, lam_cov)
            fold_cache[f] = (mus, sigs, mu_pool, Sig_pool)
        mus, sigs, mu_pool, Sig_pool = fold_cache[f]
        p = P.iloc[j].to_numpy()
        mu_t, Sig_t = mixture_moments(p, mus, sigs)

        w_rm = solve_mv(mu_t, Sig_t, risk_aversion, cap, cash_i)
        mu_ro = mu_pool.copy()
        w_rr = solve_mv(mu_ro, Sig_t, risk_aversion, cap, cash_i)
        w_sm = solve_mv(mu_pool, Sig_pool, risk_aversion, cap, cash_i)

        i_abs = pos_of[d]
        win = R_risky.iloc[max(0, i_abs - inv_vol_window + 1): i_abs + 1]
        iv = 1.0 / win.std().to_numpy()
        w_iv = np.append(iv / iv.sum(), 0.0)
        w_ew = np.append(np.full(len(cols) - 1, 1.0 / (len(cols) - 1)), 0.0)

        for n, w in zip(names, [w_rm, w_rr, w_sm, w_iv, w_ew]):
            out[n].append(w)
        dates.append(d)
    return {n: pd.DataFrame(v, index=dates, columns=cols) for n, v in out.items()}


# --------------------------------------------------------------------------- simulation
def simulate(targets: pd.DataFrame, R_risky: pd.DataFrame, rf_annual=0.06, cost_bps=10.0, index=None):
    """Drifting-weights simulation with strict execution lag.

    Decision at close d -> trade at close d+1 -> first return earned is day d+2.
    Cost = cost_bps * traded notional (sum |w_target - w_drifted|), charged on the trade day.
    Returns (daily net returns Series, realised weights DataFrame, annual turnover).
    """
    index = R_risky.index if index is None else index
    rf_d = (1 + rf_annual) ** (1 / rf.TRADING_DAYS) - 1
    R = R_risky.reindex(index)
    Rn = np.column_stack([R.to_numpy(), np.full(len(R), rf_d)])
    A = Rn.shape[1]
    pos = {d: i for i, d in enumerate(index)}
    exec_at = {}
    for d in targets.index:
        i = pos[d] + 1  # trade at the next close
        if i < len(index):
            exec_at[i] = targets.loc[d].to_numpy()

    w = None
    rets = np.full(len(index), np.nan)
    W = np.full((len(index), A), np.nan)
    traded = 0.0
    for t in range(len(index)):
        if w is not None:
            r_p = w @ Rn[t]
            rets[t] = r_p
            W[t] = w
            w = w * (1 + Rn[t]) / (1 + r_p)
        if t in exec_at:
            tgt = exec_at[t]
            if w is None:
                w = tgt.copy()  # first allocation: pay cost from cash
                cost = cost_bps / 1e4 * np.abs(tgt[:-1]).sum()
            else:
                cost = cost_bps / 1e4 * np.abs(tgt - w).sum()
                traded += np.abs(tgt - w).sum()
            if not np.isnan(rets[t]):
                rets[t] -= cost
            w = tgt.copy()
    s = pd.Series(rets, index=index).dropna()
    Wd = pd.DataFrame(W, index=index, columns=list(R_risky.columns) + ["cash"]).dropna()
    years = max(len(s) / rf.TRADING_DAYS, 1e-9)
    return s, Wd, traded / years
