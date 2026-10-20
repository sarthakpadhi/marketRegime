"""
Do HMM regimes capture something other than variance? (correlation / joint-distribution test)

The HMM here is fit DIRECTLY on the joint daily return vector of a small asset set
(not on Nifty+VIX volatility features), with full-covariance Gaussian emissions, so a
regime is free to differ in correlation as well as in variance and drift.

Everything is walk-forward and causal, scored on identical out-of-sample days:
  * HMM            : filtered one-step-ahead predictive mixture density
  * EWMA-cov       : RiskMetrics covariance (smoothly time-varying correlation)
  * EWMA-vol+constR: EWMA vols with a constant correlation matrix (train-window)
Score = one-step-ahead predictive log density of the return vector.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

import regime_filter as rf
import vol_forecast_eval as vf

LAM = 0.94
SCALE = 100.0  # work in percent returns for numerical comfort (same for every model)


# ------------------------------------------------------------------------------ data
def load_set(tickers: dict, cache: str, start="2000-01-01", price_cols=None) -> pd.DataFrame:
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import yfinance as yf

    from regime_allocation import drop_bad_prints

    cols = {}
    for name, tk in tickers.items():
        d = yf.download(tk, start=start, progress=False, auto_adjust=True)["Close"].squeeze()
        d.index = pd.to_datetime(d.index).tz_localize(None)
        cols[name] = d
    px = pd.DataFrame(cols).dropna()
    px = px[~px.index.duplicated(keep="last")]
    px = drop_bad_prints(px, cols=price_cols or list(px.columns))
    px.to_csv(cache)
    return px


# ------------------------------------------------------------------------- baselines
def ewma_cov_pred(R: np.ndarray, lam=LAM, init_n=60) -> np.ndarray:
    """S[t] = covariance used to forecast R[t], built only from R[:t] (zero-mean RiskMetrics)."""
    T, d = R.shape
    S = np.empty((T, d, d))
    cur = np.cov(R[:init_n].T)
    for t in range(T):
        S[t] = cur
        cur = lam * cur + (1 - lam) * np.outer(R[t], R[t])
    return S


def const_corr_cov_pred(S_ewma: np.ndarray, R: np.ndarray, s: int) -> np.ndarray:
    """EWMA vols + constant correlation estimated from standardised residuals on rows < s."""
    sd = np.sqrt(np.einsum("tii->ti", S_ewma))
    Z = R / sd
    corr = np.corrcoef(Z[:s].T)
    return sd[:, :, None] * corr[None] * sd[:, None, :]


def gauss_logpdf(R: np.ndarray, mu: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Row-wise log N(R[t]; mu, S[t]) with per-row covariance S[t]."""
    d = R.shape[1]
    X = R - mu
    sign, logdet = np.linalg.slogdet(S)
    sol = np.linalg.solve(S, X[:, :, None])[:, :, 0]
    quad = (X * sol).sum(1)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


# --------------------------------------------------------------------------- the HMM
def filter_with_prediction(model, X: np.ndarray):
    """Forward pass returning (filtered alpha, predicted prior pi, ll) per row.
    pi[t] = P(s_t | x_<t) (used to condition on regime BEFORE seeing x_t)."""
    logB = rf.emission_loglik(model, X)
    with np.errstate(divide="ignore"):
        logA = np.log(model.transmat_)
        logpi0 = np.log(model.startprob_)
    T, n = logB.shape
    alpha = np.empty((T, n))
    prior = np.empty((T, n))
    ll = np.empty(T)
    la_prev = None
    for t in range(T):
        lp = logpi0 if t == 0 else logsumexp(la_prev[:, None] + logA, axis=0)
        prior[t] = np.exp(lp)
        a = lp + logB[t]
        ll[t] = logsumexp(a)
        la_prev = a - ll[t]
        alpha[t] = np.exp(la_prev)
    return alpha, prior, ll, logB


@dataclass
class CorrWFConfig:
    n_states: int = 3
    warmup_days: int = 756
    refit_days: int = 126
    n_init: int = 10
    seed: int = 42


def run_models(ret: pd.DataFrame, cfg: CorrWFConfig = CorrWFConfig()):
    """ret: daily simple returns (T,d). Returns dict with per-day scores and regime info on OOS rows."""
    idx = ret.index
    R = np.log1p(ret.to_numpy()) * SCALE
    T, d = R.shape
    S3_all = ewma_cov_pred(R)
    out = {k: [] for k in ["HMM", "EWMA-cov", "EWMA-vol+constR"]}
    marg = {k: [] for k in out}
    prior_l, label_l, alpha_l, rows_l, fold_l = [], [], [], [], []

    for f, s in enumerate(range(cfg.warmup_days, T, cfg.refit_days)):
        e = min(s + cfg.refit_days, T)
        model = rf.fit_hmm(R[:s], cfg.n_states, "full", cfg.n_init, cfg.seed)
        order = np.argsort(np.trace(model.covars_, axis1=1, axis2=2))  # calm -> stress
        alpha, prior, ll, _ = filter_with_prediction(model, R[:e])
        rows = np.arange(s, e)
        mu_tr = R[:s].mean(0)

        # --- joint scores
        out["HMM"].append(ll[rows])
        S3 = S3_all[rows]
        out["EWMA-cov"].append(gauss_logpdf(R[rows], mu_tr, S3))
        S2 = const_corr_cov_pred(S3_all, R, s)[rows]
        out["EWMA-vol+constR"].append(gauss_logpdf(R[rows], mu_tr, S2))

        # --- per-asset marginal scores (to decompose joint = marginals + dependence)
        pri = prior[rows]
        mus, covs = model.means_, model.covars_
        m_h = np.zeros((len(rows), d))
        for i in range(d):
            comp = np.column_stack([norm.logpdf(R[rows, i], mus[k, i], np.sqrt(covs[k, i, i])) for k in range(cfg.n_states)])
            m_h[:, i] = logsumexp(np.log(np.maximum(pri, 1e-300)) + comp, axis=1)
        marg["HMM"].append(m_h)
        marg["EWMA-cov"].append(norm.logpdf(R[rows], mu_tr, np.sqrt(np.einsum("tii->ti", S3))))
        marg["EWMA-vol+constR"].append(norm.logpdf(R[rows], mu_tr, np.sqrt(np.einsum("tii->ti", S2))))

        # --- regimes (ordered calm->stress): label = previous day's filtered argmax
        prev_alpha = alpha[rows - 1] if s > 0 else alpha[rows]
        prior_l.append(pri[:, order])
        label_l.append(np.argmax(prev_alpha[:, order], axis=1))
        alpha_l.append(alpha[rows][:, order])
        rows_l.append(rows)
        fold_l.append(np.full(len(rows), f))

    rows_all = np.concatenate(rows_l)
    oidx = idx[rows_all]
    scores = pd.DataFrame({k: np.concatenate(v) for k, v in out.items()}, index=oidx)
    marginals = {k: pd.DataFrame(np.vstack(v), index=oidx, columns=ret.columns) for k, v in marg.items()}
    return {
        "scores": scores,
        "marginals": marginals,
        "labels": pd.Series(np.concatenate(label_l), index=oidx, name="label"),
        "filtered": pd.DataFrame(np.vstack(alpha_l), index=oidx),
        "fold": pd.Series(np.concatenate(fold_l), index=oidx),
        "R": R,
        "S_ewma": S3_all,
        "idx": idx,
    }


# ------------------------------------------------------------- step 1: permutation test
def regime_corr_table(Z: np.ndarray, labels: np.ndarray, n_states: int, min_days=100):
    """Correlation matrices of standardised returns within each regime (NaN if too few days)."""
    out = {}
    for k in range(n_states):
        m = labels == k
        out[k] = np.corrcoef(Z[m].T) if m.sum() >= min_days else None
    return out


def corr_range_stat(Z, labels, n_states, min_days=100):
    tabs = regime_corr_table(Z, labels, n_states, min_days)
    ok = [t for t in tabs.values() if t is not None]
    if len(ok) < 2:
        return 0.0, np.zeros(Z.shape[1] * (Z.shape[1] - 1) // 2)
    d = Z.shape[1]
    iu = np.triu_indices(d, 1)
    per_pair = np.array([max(t[iu][j] for t in ok) - min(t[iu][j] for t in ok) for j in range(len(iu[0]))])
    return per_pair.max(), per_pair


def permutation_pvalue(Z, labels, n_states, n_perm=2000, min_shift=63, seed=0):
    """Circular-shift permutation of the regime label sequence (preserves label persistence)."""
    rng = np.random.default_rng(seed)
    obs, per_pair = corr_range_stat(Z, labels, n_states)
    T = len(labels)
    null = np.empty(n_perm)
    null_pair = np.empty((n_perm, len(per_pair)))
    for b in range(n_perm):
        sh = rng.integers(min_shift, T - min_shift)
        st, pp = corr_range_stat(Z, np.roll(labels, sh), n_states)
        null[b], null_pair[b] = st, pp
    p_max = (1 + (null >= obs).sum()) / (n_perm + 1)
    p_pair = (1 + (null_pair >= per_pair).sum(0)) / (n_perm + 1)
    return obs, per_pair, p_max, p_pair


def standardised_residuals(bundle, rows_idx):
    """z = r / EWMA one-step-ahead vol, restricted to the OOS rows."""
    R, S = bundle["R"], bundle["S_ewma"]
    pos = bundle["idx"].get_indexer(rows_idx)
    sd = np.sqrt(np.einsum("tii->ti", S[pos]))
    return R[pos] / sd


# ----------------------------------------------------------------- step 2: log-score DM
def score_comparison(bundle):
    sc = bundle["scores"]
    rows = []
    for m in ["EWMA-cov", "EWMA-vol+constR"]:
        # loss = -log score; d = loss_HMM - loss_m ; negative => HMM better
        dm, st, p = vf.diebold_mariano(-sc["HMM"].to_numpy(), -sc[m].to_numpy(), 1)
        rows.append({"benchmark": m, "mean_logscore_HMM": sc["HMM"].mean(), "mean_logscore_bench": sc[m].mean(),
                     "HMM_minus_bench (logscore)": -dm, "DM_stat": st, "DM_p": p})
    return pd.DataFrame(rows).set_index("benchmark")


def decomposition(bundle):
    """joint log score = sum of marginal log scores + dependence term (per model, mean/day)."""
    sc, mg = bundle["scores"], bundle["marginals"]
    rows = {}
    for m in sc.columns:
        marg_sum = mg[m].sum(1)
        rows[m] = {"joint": sc[m].mean(), "marginals": marg_sum.mean(), "dependence": (sc[m] - marg_sum).mean()}
    return pd.DataFrame(rows).T
