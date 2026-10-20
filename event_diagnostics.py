"""
Event-study and diagnostic tools for regime signals.

  * coincidence_test   : do dated events (Budget, election results) coincide with regime starts
                         more often than chance? (circular-shift permutation null)
  * loss_concentration : does the strategy's loss concentrate in windows around regime switches?
  * drawdown_episodes / detection_lag : how late does a regime flag fire relative to a
                         price-defined drawdown, and how much of the fall had already happened?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ dated events
# Union Budget presentation dates: Wikipedia "Union budget of India" table (checked), with ONE
# correction: the page lists the 2024-25 full Budget as 22 July 2024; the Budget speech was on
# 23 July 2024 (22 July was the Economic Survey). Dates falling on a non-trading day are mapped
# to the next trading day in `map_to_trading_days`.
BUDGET_DATES = [
    "2009-02-16", "2009-07-06", "2010-02-26", "2011-02-28", "2012-03-16", "2013-02-28",
    "2014-02-17", "2014-07-10", "2015-02-28", "2016-02-29", "2017-02-01", "2018-02-01",
    "2019-02-01", "2019-07-05", "2020-02-01", "2021-02-01", "2022-02-01", "2023-02-01",
    "2024-02-01", "2024-07-23", "2025-02-01", "2026-02-01",
]
# Lok Sabha general-election result days (counting day).
ELECTION_RESULT_DATES = ["2009-05-16", "2014-05-16", "2019-05-23", "2024-06-04"]


def map_to_trading_days(dates, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Each event -> first trading day >= the event date (reaction day); drop events outside the index."""
    out = []
    for d in pd.to_datetime(list(dates)):
        pos = index.searchsorted(d)
        if pos < len(index) and (index[pos] - d).days <= 5:
            out.append(index[pos])
    return pd.DatetimeIndex(sorted(set(out)))


# --------------------------------------------------------------- event coincidence
def regime_starts(label: pd.Series) -> pd.Series:
    """Boolean Series: True on the first day of each new regime label (first row excluded)."""
    s = label.ne(label.shift())
    s.iloc[0] = False
    return s


def coincidence_test(starts: pd.Series, events: pd.DatetimeIndex, window=(0, 2), n_perm=5000, min_shift=63, seed=0):
    """Hits = number of events with >=1 regime start in [event+window[0], event+window[1]] trading days.
    Null: circularly shift the start indicator by a random offset (keeps start clustering/spacing)."""
    idx = starts.index
    ev = [idx.get_loc(e) for e in events if e in idx]
    if not ev:
        return dict(n_events=0, hits=0, expected=np.nan, p=np.nan, rate=np.nan)
    a = starts.to_numpy().astype(int)
    T = len(a)
    lo, hi = window

    def hits(arr):
        c = 0
        for i in ev:
            j0, j1 = max(i + lo, 0), min(i + hi, T - 1)
            c += arr[j0 : j1 + 1].any()
        return c

    obs = hits(a)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = hits(np.roll(a, int(rng.integers(min_shift, T - min_shift))))
    return dict(n_events=len(ev), hits=int(obs), expected=float(null.mean()), p=float((1 + (null >= obs).sum()) / (n_perm + 1)), rate=obs / len(ev))


# ------------------------------------------------------------- loss concentration
def transition_mask(effective_change: pd.Series, post_days=5) -> pd.Series:
    """True on the change day and the next `post_days` trading days."""
    ch = effective_change.fillna(False).to_numpy().astype(bool)
    m = np.zeros(len(ch), dtype=bool)
    for i in np.flatnonzero(ch):
        m[i : i + post_days + 1] = True
    return pd.Series(m, index=effective_change.index)


def loss_concentration(strat: pd.Series, bench: pd.Series, mask: pd.Series, n_perm=5000, min_shift=63, seed=0):
    """Two statistics, both vs a circular-shift null of the transition mask:
    (1) share of total negative-return mass falling in transition windows minus share of days there;
    (2) mean daily (strategy - benchmark) inside windows minus outside."""
    df = pd.concat([strat.rename("s"), bench.rename("b"), mask.rename("m")], axis=1).dropna()
    s, b, m = df["s"].to_numpy(), df["b"].to_numpy(), df["m"].to_numpy().astype(bool)
    neg = np.maximum(-s, 0.0)
    rel = s - b
    T = len(s)

    def stats(mm):
        sh = neg[mm].sum() / neg.sum() - mm.mean()
        dm = rel[mm].mean() - rel[~mm].mean() if mm.any() and (~mm).any() else 0.0
        return sh, dm

    obs = stats(m)
    rng = np.random.default_rng(seed)
    null = np.array([stats(np.roll(m, int(rng.integers(min_shift, T - min_shift)))) for _ in range(n_perm)])
    return dict(
        days_in_windows=float(m.mean()),
        loss_share_excess=float(obs[0]),
        p_loss=float((1 + (null[:, 0] >= obs[0]).sum()) / (n_perm + 1)),
        rel_mean_diff_bp=float(obs[1] * 1e4),
        p_rel=float((1 + (null[:, 1] <= obs[1]).sum()) / (n_perm + 1)),  # one-sided: underperformance in windows
    )


# ----------------------------------------------------------- drawdown episodes / lag
def drawdown_episodes(close: pd.Series, threshold=0.10) -> pd.DataFrame:
    """Peak-to-trough episodes whose depth reaches `threshold` (non-overlapping, until a new high)."""
    c = close.to_numpy()
    idx = close.index
    eps = []
    peak_i, peak_v = 0, c[0]
    in_ep, trough_i = False, 0
    for i in range(1, len(c)):
        if c[i] >= peak_v:
            if in_ep:
                eps.append((peak_i, trough_i, i))
                in_ep = False
            peak_i, peak_v = i, c[i]
            continue
        dd = c[i] / peak_v - 1
        if dd <= -threshold and not in_ep:
            in_ep, trough_i = True, i
        if in_ep and c[i] < c[trough_i]:
            trough_i = i
    if in_ep:
        eps.append((peak_i, trough_i, None))
    rows = []
    for p, t, r in eps:
        rows.append({"peak": idx[p], "trough": idx[t], "recovery": idx[r] if r is not None else pd.NaT,
                     "depth": c[t] / c[p] - 1, "peak_to_trough_days": t - p})
    return pd.DataFrame(rows)


def detection_lag(close: pd.Series, flag: pd.Series, episodes: pd.DataFrame, delay=2) -> pd.DataFrame:
    """For each episode: first day in [peak, trough] on which `flag` (risk-off) is True.
    Reports lag in trading days, fall already realised at the flag (and when trading after `delay` days)."""
    idx = close.index
    rows = []
    for _, e in episodes.iterrows():
        seg = flag.loc[e["peak"] : e["trough"]]
        seg = seg[seg.astype(bool)]
        rec = {"peak": e["peak"].date(), "trough": e["trough"].date(), "depth": e["depth"]}
        if len(seg) == 0:
            rec.update(flag_date=None, lag_days=np.nan, fall_at_flag=np.nan, share_of_fall_at_flag=np.nan,
                       share_of_fall_at_exec=np.nan, active_at_peak=bool(flag.get(e["peak"], False)))
        else:
            fd = seg.index[0]
            i_p, i_f = idx.get_loc(e["peak"]), idx.get_loc(fd)
            i_x = min(i_f + delay, len(idx) - 1)
            fall_f = close.iloc[i_f] / close.iloc[i_p] - 1
            fall_x = close.iloc[i_x] / close.iloc[i_p] - 1
            rec.update(flag_date=fd.date(), lag_days=i_f - i_p, fall_at_flag=fall_f,
                       share_of_fall_at_flag=fall_f / e["depth"], share_of_fall_at_exec=fall_x / e["depth"],
                       active_at_peak=bool(flag.get(e["peak"], False)))
        rows.append(rec)
    return pd.DataFrame(rows)
