import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import event_diagnostics as ed  # noqa: E402


def test_map_to_trading_days_next_open_and_drops_outside():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    out = ed.map_to_trading_days(["2020-02-01", "2020-02-05", "2010-01-01"], idx)  # Sat -> Mon 3rd
    assert pd.Timestamp("2020-02-03") in out and pd.Timestamp("2020-02-05") in out and len(out) == 2


def _starts(T=2500, p=0.03, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=T)
    return pd.Series(rng.random(T) < p, index=idx)


def test_coincidence_detects_planted_alignment():
    s = _starts()
    ev = s.index[s.to_numpy()][::2][:25]
    r = ed.coincidence_test(s, ev, window=(0, 2), n_perm=800)
    assert r["hits"] == r["n_events"] and r["p"] < 0.01


def test_coincidence_not_significant_for_random_events():
    ps = []
    for seed in range(12):
        s = _starts(seed=seed)
        rng = np.random.default_rng(100 + seed)
        ev = s.index[np.sort(rng.choice(len(s), 20, replace=False))]
        ps.append(ed.coincidence_test(s, ev, n_perm=300, seed=seed)["p"])
    assert np.mean(np.array(ps) < 0.05) <= 0.25


def test_drawdown_episodes_and_detection_lag():
    idx = pd.bdate_range("2020-01-01", periods=120)
    c = np.r_[np.linspace(100, 120, 30), np.linspace(120, 90, 30), np.linspace(90, 130, 60)]
    close = pd.Series(c, index=idx)
    eps = ed.drawdown_episodes(close, 0.10)
    assert len(eps) == 1 and eps.loc[0, "depth"] == pytest.approx(90 / 120 - 1)
    flag = pd.Series(False, index=idx); flag.iloc[40:] = True  # path touches 120 at idx 29 and 30 (tie): peak = idx 30, flag at 40
    lag = ed.detection_lag(close, flag, eps, delay=2)
    assert lag.loc[0, "lag_days"] == 10
    assert 0 < lag.loc[0, "share_of_fall_at_flag"] < lag.loc[0, "share_of_fall_at_exec"] <= 1.0


def test_loss_concentration_detects_planted_losses_in_windows():
    rng = np.random.default_rng(1)
    T = 3000
    idx = pd.bdate_range("2005-01-01", periods=T)
    change = pd.Series(False, index=idx); change.iloc[rng.choice(np.arange(10, T - 10), 40, replace=False)] = True
    m = ed.transition_mask(change, 5)
    strat = pd.Series(rng.normal(0.0003, 0.005, T), index=idx)
    strat[m] -= 0.01  # big losses inside windows
    r = ed.loss_concentration(strat, pd.Series(0.0, index=idx), m, n_perm=500)
    assert r["loss_share_excess"] > 0.1 and r["p_loss"] < 0.01 and r["p_rel"] < 0.01
