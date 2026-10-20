import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import corr_regime_eval as cr  # noqa: E402
import regime_filter as rf  # noqa: E402


def test_gauss_logpdf_matches_scipy():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(20, 3))
    S = np.stack([np.eye(3) * (1 + i / 10) + 0.2 for i in range(20)])
    mu = np.array([0.1, -0.2, 0.0])
    ref = np.array([multivariate_normal.logpdf(R[i], mu, S[i]) for i in range(20)])
    np.testing.assert_allclose(cr.gauss_logpdf(R, mu, S), ref, atol=1e-9)


def test_filter_with_prediction_matches_base_filter():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-1, 1, (200, 2)), rng.normal(2, 0.5, (200, 2))])
    m = rf.fit_hmm(X, 2, "full", n_init=2, seed=0)
    a0, ll0 = rf.forward_filter(m, X)
    a1, prior, ll1, _ = cr.filter_with_prediction(m, X)
    np.testing.assert_allclose(a0, a1, atol=1e-10)
    np.testing.assert_allclose(ll0, ll1, atol=1e-10)
    np.testing.assert_allclose(prior[1:], a1[:-1] @ m.transmat_, atol=1e-10)  # prior = alpha_{t-1} A


def test_ewma_cov_pred_is_causal():
    rng = np.random.default_rng(2)
    R = rng.normal(size=(300, 3))
    a = cr.ewma_cov_pred(R)
    R2 = R.copy(); R2[200:] *= 5
    b = cr.ewma_cov_pred(R2)
    np.testing.assert_allclose(a[:201], b[:201])  # S[t] only uses R[:t]


def _planted(T=3000, seed=3, rho=(0.0, 0.8)):
    rng = np.random.default_rng(seed)
    lab = np.zeros(T, int)
    for t in range(1, T):
        lab[t] = lab[t - 1] if rng.random() < 0.98 else 1 - lab[t - 1]
    Z = np.empty((T, 2))
    for k in (0, 1):
        m = lab == k
        C = np.array([[1, rho[k]], [rho[k], 1]])
        Z[m] = rng.multivariate_normal([0, 0], C, m.sum())
    return Z, lab


def test_permutation_test_detects_planted_correlation_regimes():
    Z, lab = _planted()
    _, _, p, _ = cr.permutation_pvalue(Z, lab, 2, n_perm=300, seed=1)
    assert p < 0.02


def test_permutation_test_is_calibrated_under_null():
    ps = []
    for seed in range(15):
        Z, lab = _planted(T=1500, seed=seed, rho=(0.4, 0.4))  # same corr in both regimes
        _, _, p, _ = cr.permutation_pvalue(Z, lab, 2, n_perm=150, seed=seed)
        ps.append(p)
    assert np.mean(np.array(ps) < 0.05) <= 0.2  # nominal 5%, loose bound for 15 draws
