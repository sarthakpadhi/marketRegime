import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import regime_allocation as ra  # noqa: E402


def test_solve_mv_respects_constraints():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    Sig = A @ A.T / 100 + 1e-4 * np.eye(5)
    mu = rng.normal(0.0005, 0.0005, 5)
    w = ra.solve_mv(mu, Sig, risk_aversion=4, cap=0.4, cash_idx=4)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-12).all() and (w[:4] <= 0.4 + 1e-9).all()


def test_mixture_moments_degenerate_prob_returns_state_moments():
    mus = np.array([[0.1, 0.2], [0.3, 0.4]])
    sigs = np.array([np.eye(2), 2 * np.eye(2)])
    mu, S = ra.mixture_moments(np.array([0.0, 1.0]), mus, sigs)
    np.testing.assert_allclose(mu, mus[1])
    np.testing.assert_allclose(S, sigs[1])


def test_mixture_covariance_includes_between_state_spread():
    mus = np.array([[0.0], [1.0]])
    sigs = np.array([[[0.0]], [[0.0]]])
    _, S = ra.mixture_moments(np.array([0.5, 0.5]), mus, sigs)
    assert S[0, 0] == pytest.approx(0.25)  # variance of a fair coin on {0,1}


def test_state_moments_full_shrinkage_returns_pooled():
    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.01, size=(400, 3))
    g = np.column_stack([np.r_[np.ones(200), np.zeros(200)], np.r_[np.zeros(200), np.ones(200)]])
    mus, sigs, mu_p, S_p = ra.state_moments(R, g, lam_mean=1.0, lam_cov=1.0)
    np.testing.assert_allclose(mus[0], mu_p)
    np.testing.assert_allclose(sigs[1], S_p)


def test_state_moments_unseen_state_falls_back_to_pooled():
    rng = np.random.default_rng(2)
    R = rng.normal(0, 0.01, size=(100, 2))
    g = np.column_stack([np.ones(100), np.zeros(100)])
    mus, _, mu_p, _ = ra.state_moments(R, g)
    np.testing.assert_allclose(mus[1], mu_p)


def test_simulate_execution_lag_and_costs():
    idx = pd.bdate_range("2020-01-01", periods=8)
    R = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    # decide 100% 'a' on idx[0] -> trade at close idx[1] -> first return earned: idx[2]
    tgt = pd.DataFrame({"a": [1.0], "cash": [0.0]}, index=[idx[0]])
    s, W, _ = ra.simulate(tgt, R, rf_annual=0.0, cost_bps=0.0, index=idx)
    assert s.index[0] == idx[2]
    assert s.loc[idx[3]] == pytest.approx(0.10)

    # a decision on idx[2] must NOT capture the idx[3] jump (trade lands at close idx[3])
    tgt2 = pd.DataFrame({"a": [1.0], "cash": [0.0]}, index=[idx[2]])
    s2, _, _ = ra.simulate(tgt2, R, rf_annual=0.0, cost_bps=0.0, index=idx)
    assert s2.loc[idx[3]:].sum() == pytest.approx(0.0)


def test_drop_bad_prints_removes_x100_error_keeps_real_move():
    idx = pd.bdate_range("2020-01-01", periods=30)
    p = pd.Series(np.linspace(100, 110, 30), index=idx)
    p.iloc[10] = 1.0  # vendor error
    p.iloc[20] = p.iloc[19] * 0.9  # real -10% day
    out = ra.drop_bad_prints(pd.DataFrame({"x": p}))
    assert idx[10] not in out.index and idx[20] in out.index


def test_drop_bad_prints_cols_leaves_vol_index_spike():
    idx = pd.bdate_range("2024-01-01", periods=30)
    df = pd.DataFrame({"close": np.linspace(100, 110, 30), "vix": 16.0}, index=idx)
    df.loc[idx[15], "vix"] = 38.0  # genuine +140% VIX spike
    out = ra.drop_bad_prints(df, cols=["close"])
    assert idx[15] in out.index
