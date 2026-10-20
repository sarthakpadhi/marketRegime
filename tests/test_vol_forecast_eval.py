import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vol_forecast_eval as vf  # noqa: E402


def test_forward_mean_sq_alignment():
    r2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = vf.forward_mean_sq(r2, 2)
    assert out[0] == pytest.approx(2.5)  # mean(r2[1], r2[2])
    assert out[2] == pytest.approx(4.5)
    assert np.isnan(out[3]) and np.isnan(out[4])


def test_ewma_forecast_uses_only_past():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 200)
    a = vf.ewma_var(r)
    r2 = r.copy(); r2[150:] *= 10
    b = vf.ewma_var(r2)
    np.testing.assert_allclose(a[:150], b[:150])


def test_qlike_minimised_at_truth():
    y = 2.0
    grid = np.linspace(0.5, 5, 400)
    assert grid[np.argmin(vf.qlike(grid, y))] == pytest.approx(y, abs=0.02)


def test_dm_detects_clearly_better_model_and_not_equal_ones():
    rng = np.random.default_rng(1)
    good = rng.normal(1.0, 0.1, 500)
    bad = good + 0.3 + rng.normal(0, 0.1, 500)
    _, _, p = vf.diebold_mariano(good, bad, 1)
    assert p < 1e-6
    x = rng.normal(0, 1, 500)
    _, _, p_eq = vf.diebold_mariano(x, x + rng.normal(0, 0.01, 500), 1)
    assert p_eq > 0.05


def test_hmm_forecast_reverts_toward_stationary_mix():
    A = np.array([[0.9, 0.1], [0.2, 0.8]])
    v = np.array([1.0, 4.0])
    P = np.array([[1.0, 0.0]])
    f1 = vf.hmm_variance_forecast(P, A, v, 1)[0]
    f200 = vf.hmm_variance_forecast(P, A, v, 200)[0]
    pi = np.array([2 / 3, 1 / 3])
    assert f1 == pytest.approx(0.9 * 1 + 0.1 * 4)
    assert f200 == pytest.approx(pi @ v, abs=0.05)
