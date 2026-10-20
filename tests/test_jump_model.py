import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jump_model as jm  # noqa: E402


def _brute(D, lam):
    T, K = D.shape
    best, bs = np.inf, None
    for seq in itertools.product(range(K), repeat=T):
        s = np.array(seq)
        c = D[np.arange(T), s].sum() + lam * (s[1:] != s[:-1]).sum()
        if c < best:
            best, bs = c, s
    return best, bs


@pytest.mark.parametrize("seed,lam", [(0, 0.5), (1, 2.0), (2, 5.0), (3, 0.0)])
def test_dp_matches_brute_force_objective(seed, lam):
    rng = np.random.default_rng(seed)
    D = rng.uniform(0, 3, size=(8, 3))
    s = jm.dp_viterbi(D, lam)
    best, _ = _brute(D, lam)
    assert jm.objective(D, s, lam) == pytest.approx(best)


def test_online_state_equals_last_state_of_prefix_dp():
    rng = np.random.default_rng(4)
    D = rng.uniform(0, 3, size=(60, 2))
    for lam in (0.0, 1.0, 4.0):
        on = jm.online_states(D, lam)
        for t in (0, 5, 17, 33, 59):
            assert on[t] == jm.dp_viterbi(D[: t + 1], lam)[-1]


def test_lambda_zero_is_nearest_centroid_and_large_lambda_never_jumps():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 2)); mu = np.array([[-1.0, 0], [1.0, 0]])
    D = jm.sq_dist(X, mu)
    np.testing.assert_array_equal(jm.dp_viterbi(D, 0.0), D.argmin(1))
    assert len(np.unique(jm.dp_viterbi(D, 1e9))) == 1


def test_online_signal_is_causal():
    rng = np.random.default_rng(6)
    D = rng.uniform(0, 3, size=(300, 2))
    D2 = D.copy(); D2[200:] = rng.uniform(0, 3, size=(100, 2))
    np.testing.assert_array_equal(jm.online_states(D, 3.0)[:200], jm.online_states(D2, 3.0)[:200])


def test_fit_recovers_planted_persistent_regimes_and_reduces_jumps():
    rng = np.random.default_rng(7)
    T = 1200
    lab = (np.arange(T) // 200) % 2
    X = rng.normal(np.where(lab[:, None] == 0, -1.0, 1.0), 1.0, size=(T, 2))
    mu, s, _ = jm.fit_jm(X, lam=25.0, K=2, n_init=5)
    acc = max((s == lab).mean(), (s != lab).mean())
    assert acc > 0.95
    s0 = jm.dp_viterbi(jm.sq_dist(X, mu), 0.0)
    assert (s[1:] != s[:-1]).sum() < (s0[1:] != s0[:-1]).sum()


def test_features_are_causal():
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0, 0.01, 400))
    a = jm.sjm_features(r, 0.06)
    r2 = r.copy(); r2.iloc[300:] *= 10
    b = jm.sjm_features(r2, 0.06)
    pd.testing.assert_frame_equal(a.iloc[:300], b.iloc[:300])


def test_bull_state_picks_higher_return_state():
    s = np.array([0, 0, 1, 1]); ex = np.array([-0.01, -0.02, 0.01, 0.02])
    assert jm.bull_state(s, ex) == 1


def test_vol_target_exposures_are_causal_and_capped():
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    r = pd.Series(rng.normal(0, 0.01, 1500), index=idx)
    vix = pd.Series(16 + rng.normal(0, 1, 1500), index=idx)
    e1, e2 = jm.ewma_vol_target_exposure(r), jm.vix_vol_target_exposure(r, vix)
    assert e1.max() <= 1.0 and e2.dropna().max() <= 1.0
    r2 = r.copy(); r2.iloc[1200:] *= 8
    np.testing.assert_allclose(e1.iloc[:1200], jm.ewma_vol_target_exposure(r2).iloc[:1200])
    np.testing.assert_allclose(e2.iloc[:1200].dropna(), jm.vix_vol_target_exposure(r2, vix).iloc[:1200].dropna())


def test_ma_exposure_is_causal_and_nan_before_window():
    close = pd.Series(np.linspace(100, 200, 300))
    e = jm.ma_exposure(close, 200)
    assert e.iloc[:199].isna().all() and e.iloc[250] == 1.0
    c2 = close.copy(); c2.iloc[250:] = 1.0
    pd.testing.assert_series_equal(e.iloc[:250], jm.ma_exposure(c2, 200).iloc[:250])


def test_vix_hysteresis_holds_state_in_band_and_is_causal():
    rng = np.random.default_rng(0)
    v = pd.Series(16 + rng.normal(0, 1, 1200))
    v.iloc[1000:1010] = 60.0  # spike -> cash
    e = jm.vix_hysteresis_exposure(v)
    assert e.iloc[1005] == 0.0
    assert e.iloc[1100] == 1.0  # recovered after spike
    v2 = v.copy(); v2.iloc[1100:] = 5.0
    pd.testing.assert_series_equal(e.iloc[:1100], jm.vix_hysteresis_exposure(v2).iloc[:1100])


def test_calmar_and_bootstrap_sign():
    rng = np.random.default_rng(1)
    good = pd.Series(rng.normal(0.0008, 0.005, 1500))
    bad = pd.Series(rng.normal(0.0000, 0.010, 1500))
    pt, lo, hi, p = jm.block_bootstrap_calmar_diff(good, bad, n_boot=300)
    assert pt > 0 and p < 0.1
