import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import regime_filter as rf  # noqa: E402


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    # two well-separated regimes with sticky transitions
    z = np.zeros(600, dtype=int)
    for t in range(1, 600):
        z[t] = z[t - 1] if rng.random() < 0.97 else 1 - z[t - 1]
    X = rng.normal(loc=np.where(z[:, None] == 0, -1.5, 1.5), scale=1.0, size=(600, 3))
    model = rf.fit_hmm(X, 2, "diag", n_init=3, seed=1)
    return X, model


@pytest.mark.parametrize("cov", ["diag", "full"])
def test_emission_matches_hmmlearn(cov):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    m = rf.fit_hmm(X, 2, cov, n_init=2, seed=0)
    ours = rf.emission_loglik(m, X)
    theirs = m._compute_log_likelihood(X)
    np.testing.assert_allclose(ours, theirs, atol=1e-6)


def test_filter_last_step_equals_smoothed(synthetic):
    X, m = synthetic
    alpha, _ = rf.forward_filter(m, X)
    smoothed = m.predict_proba(X)
    np.testing.assert_allclose(alpha[-1], smoothed[-1], atol=1e-6)


def test_filter_loglik_matches_hmmlearn_score(synthetic):
    X, m = synthetic
    _, ll = rf.forward_filter(m, X)
    assert ll.sum() == pytest.approx(m.score(X), rel=1e-9)


def test_filter_rows_are_distributions(synthetic):
    X, m = synthetic
    alpha, _ = rf.forward_filter(m, X)
    np.testing.assert_allclose(alpha.sum(1), 1.0, atol=1e-9)
    assert (alpha >= 0).all()


def test_filter_is_causal(synthetic):
    X, m = synthetic
    full, _ = rf.forward_filter(m, X)
    X2 = X.copy()
    X2[400:] += 50.0  # wreck the future
    corrupted, _ = rf.forward_filter(m, X2)
    np.testing.assert_allclose(full[:400], corrupted[:400], atol=1e-12)


def test_smoothed_is_not_causal(synthetic):
    # sanity check that the test above has teeth: smoothed probs DO change
    X, m = synthetic
    X2 = X.copy()
    X2[400:] += 50.0
    assert not np.allclose(m.predict_proba(X)[:400], m.predict_proba(X2)[:400], rtol=1e-9, atol=0)


def test_backtest_uses_lagged_exposure():
    idx = pd.date_range("2020-01-01", periods=6)
    ret = pd.Series([0.0, 0.01, -0.02, 0.03, 0.0, 0.0], index=idx)
    expo = pd.Series([1.0, 0.0, 1.0, 1.0, 1.0, 1.0], index=idx)
    out = rf.backtest(ret, expo, rf_annual=0.0, cost_bps=0.0)
    # day idx[2] holds expo[1]=0 -> 0 ; day idx[3] holds expo[2]=1 -> 0.03
    assert out.loc[idx[2]] == pytest.approx(0.0)
    assert out.loc[idx[3]] == pytest.approx(0.03)


def test_walk_forward_no_leakage_and_ordering():
    rng = np.random.default_rng(3)
    n = 900
    idx = pd.bdate_range("2018-01-01", periods=n)
    z = (np.arange(n) // 150) % 2
    ret = pd.Series(rng.normal(0, np.where(z == 0, 0.005, 0.02)), index=idx)
    feats = pd.DataFrame(
        {"a": rng.normal(z * 2, 1), "log_rv_20d": rng.normal(z, 0.3), "c": rng.normal(size=n)}, index=idx
    )
    cfg = rf.WFConfig(n_states=2, warmup_days=400, refit_days=250, n_init=3)
    res = rf.walk_forward(feats, ret, cfg)
    assert res.probs.index[0] == idx[400]
    assert (res.state_vol["S0"] <= res.state_vol["S1"]).all()  # calm -> stressed

    # changing data strictly after the last fold start must not alter earlier folds
    feats2 = feats.copy()
    feats2.iloc[800:] += 100
    res2 = rf.walk_forward(feats2, ret, cfg)
    pd.testing.assert_frame_equal(res.probs.iloc[:250], res2.probs.iloc[:250])
