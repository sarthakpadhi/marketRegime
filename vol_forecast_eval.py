"""
Does the walk-forward filtered HMM forecast volatility better than simple models,
and does it carry information beyond plain vol tracking?

Everything is evaluated on identical out-of-sample rows. All fitted components
(HMM, GARCH, HAR, VIX calibration) are refit on the same expanding-window schedule as
the HMM, using only data whose *forecast target is fully observed before the fold
starts* (no overlap leakage for h > 1).

Forecast object at close t, horizon h: the average daily variance over t+1..t+h.
Realised proxy: mean of squared daily log returns over t+1..t+h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import regime_filter as rf

HORIZONS = (1, 5, 21)
EPS = 1e-12


# --------------------------------------------------------------------------- targets
def forward_mean_sq(r2: np.ndarray, h: int) -> np.ndarray:
    """target[i] = mean(r2[i+1 .. i+h]); NaN where not fully observed."""
    T = len(r2)
    c = np.concatenate([[0.0], np.cumsum(r2)])
    out = np.full(T, np.nan)
    idx = np.arange(T - h)
    out[idx] = (c[idx + 1 + h] - c[idx + 1]) / h
    return out


# ------------------------------------------------------------------------ forecasters
def ewma_var(r: np.ndarray, lam=0.94, init_n=20) -> np.ndarray:
    """f[i] = forecast of day i+1 variance made at close i."""
    f = np.empty(len(r))
    v = np.var(r[:init_n])
    for i in range(len(r)):
        v = lam * v + (1 - lam) * r[i] ** 2
        f[i] = v
    return f


def garch_filter(r: np.ndarray, omega, alpha, beta, v0):
    """sigma2_{t+1|t} recursion for t = 0..T-1 with fixed parameters."""
    f = np.empty(len(r))
    v = v0
    for i in range(len(r)):
        v = omega + alpha * r[i] ** 2 + beta * v
        f[i] = v
    return f


def fit_garch(r_train: np.ndarray):
    from arch import arch_model

    am = arch_model(r_train * 100, mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
    res = am.fit(disp="off", show_warning=False)
    p = res.params
    omega, alpha, beta = p["omega"] / 1e4, p["alpha[1]"], p["beta[1]"]
    return omega, alpha, beta


def har_design(r2: np.ndarray):
    s = pd.Series(r2)
    return np.column_stack(
        [np.ones(len(r2)), r2, s.rolling(5, min_periods=1).mean().to_numpy(), s.rolling(22, min_periods=1).mean().to_numpy()]
    )


def hmm_variance_forecast(P: np.ndarray, A: np.ndarray, v_state: np.ndarray, h: int) -> np.ndarray:
    """Average over j=1..h of sum_k (p_t A^j)_k * v_k  (persistence-aware)."""
    out = np.zeros(len(P))
    Pj = P.copy()
    for _ in range(h):
        Pj = Pj @ A
        out += Pj @ v_state
    return out / h


# ------------------------------------------------------------------------- the harness
def build_forecasts(df: pd.DataFrame, cfg: rf.WFConfig, horizons=HORIZONS):
    """df: columns close, vix (daily, aligned). Returns dict with forecasts/targets
    (DataFrames on the OOS index, one per horizon) plus the regime result."""
    feats = rf.build_features(df)
    logr_all = np.log(df["close"]).diff()
    r = logr_all.reindex(feats.index)
    ret = df["close"].pct_change().reindex(feats.index)
    res = rf.walk_forward(feats, ret, cfg)

    rv = r.to_numpy()
    r2 = rv**2
    vix_var = ((df["vix"].reindex(feats.index) / 100) ** 2 / 252).to_numpy()
    T = len(rv)
    idx = feats.index
    P_all = res.probs

    ewma_all = ewma_var(rv)
    targets = {h: forward_mean_sq(r2, h) for h in horizons}
    design = har_design(r2)

    store = {h: {} for h in horizons}
    oos_rows = []
    for k, info in enumerate(res.folds):
        s, e = info["s"], info["e"]
        rows = np.arange(s, e)
        oos_rows.append(rows)

        # --- GARCH: fit on returns[:s], filter forward with fixed params
        om, al, be = fit_garch(rv[:s])
        g1 = garch_filter(rv[:e], om, al, be, np.var(rv[:s]))

        # --- HMM state variances from the fold's train posteriors (ordered)
        gamma = info["gamma_tr"]
        v_state = (gamma * r2[:s, None]).sum(0) / np.maximum(gamma.sum(0), EPS)
        A = res.transmats[k]
        P_rows = P_all.loc[idx[s:e]].to_numpy()

        for h in horizons:
            ok = np.arange(0, s - h)  # train rows whose h-day target ends before s
            y_tr = targets[h][ok]
            # --- HAR: OLS on train rows
            beta_har, *_ = np.linalg.lstsq(design[ok], y_tr, rcond=None)
            har_f = np.clip(design[rows] @ beta_har, 1e-10, None)
            # --- VIX calibration
            c = np.nanmean(y_tr) / np.nanmean(vix_var[ok])
            rho = min(al + be, 0.9999)
            vbar = om / (1 - rho)
            garch_f = vbar + (g1[rows] - vbar) * np.mean(rho ** np.arange(h))
            d = {
                "HMM": hmm_variance_forecast(P_rows, A, v_state, h),
                "EWMA": ewma_all[rows],
                "GARCH": garch_f,
                "HAR": har_f,
                "VIX": vix_var[rows],
                "VIX_calibrated": c * vix_var[rows],
            }
            for name, v in d.items():
                store[h].setdefault(name, []).append(pd.Series(v, index=idx[rows]))
    out = {}
    for h in horizons:
        F = pd.DataFrame({n: pd.concat(v) for n, v in store[h].items()})
        F["target"] = pd.Series(targets[h], index=idx).reindex(F.index)
        out[h] = F.dropna()
    return {
        "forecasts": out,
        "res": res,
        "ewma_vol_ann": pd.Series(np.sqrt(ewma_all * 252), index=idx),
        "r": r,
        "vix": df["vix"].reindex(idx),
    }


# ----------------------------------------------------------------------------- losses
def qlike(f, y):
    f = np.maximum(f, EPS)
    return np.log(f) + y / f


def mse(f, y):
    return (f - y) ** 2


def diebold_mariano(loss_a, loss_b, h):
    """H0: equal expected loss.  d = loss_a - loss_b (negative => A better).
    Bartlett-HAC variance with lag h-1, Harvey-Leybourne-Newbold small-sample fix,
    t(T-1) reference distribution. Returns (mean_d, stat, two-sided p)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    T = len(d)
    dm = d.mean()
    lag = max(h - 1, 0)
    dc = d - dm
    lrv = (dc @ dc) / T
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        lrv += 2 * w * (dc[l:] @ dc[:-l]) / T
    lrv = max(lrv, EPS)
    stat = dm / np.sqrt(lrv / T)
    hln = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, EPS))
    stat *= hln
    p = 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))
    return dm, stat, p


def horse_race(forecasts: dict, challenger="HMM"):
    rows = []
    for h, F in forecasts.items():
        y = F["target"].to_numpy()
        models = [c for c in F.columns if c != "target"]
        for loss_name, fn in (("QLIKE", qlike), ("MSE", mse)):
            L = {m: fn(F[m].to_numpy(), y) for m in models}
            for m in models:
                rec = {"h": h, "loss": loss_name, "model": m, "mean_loss": L[m].mean() * (1e8 if loss_name == "MSE" else 1)}
                if m != challenger:
                    dm, st, p = diebold_mariano(L[challenger], L[m], h)
                    rec.update({"HMM_minus_model": dm * (1e8 if loss_name == "MSE" else 1), "DM_stat": st, "DM_p": p})
                rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ incremental information
def incremental_regressions(bundle, horizon=21, maxlags=21):
    import statsmodels.api as sm

    res = bundle["res"]
    P = res.probs
    idx = P.index
    r = bundle["r"].reindex(idx)
    rv = bundle["r"].to_numpy()
    fwd_rv = pd.Series(np.sqrt(forward_mean_sq(rv**2, horizon) * 252), index=bundle["r"].index).reindex(idx)
    # worst cumulative log return over the next `horizon` days
    cum = pd.Series(rv, index=bundle["r"].index).fillna(0)
    worst = np.full(len(cum), np.nan)
    arr = cum.to_numpy()
    for i in range(len(arr) - horizon):
        worst[i] = np.cumsum(arr[i + 1 : i + 1 + horizon]).min()
    worst = pd.Series(worst, index=cum.index).reindex(idx)

    X = pd.DataFrame(
        {
            "log_ewma_vol": np.log(bundle["ewma_vol_ann"].reindex(idx)),
            "P_stress": P.iloc[:, -1],
        }
    )
    X2 = X.assign(log_vix=np.log(bundle["vix"].reindex(idx)))
    out = {}
    for name, y, Xm in [
        ("fwd 21d log realised vol ~ EWMA + P(stress)  [PRIMARY]", np.log(fwd_rv), X),
        ("fwd 21d log realised vol ~ EWMA + log VIX + P(stress)  [secondary]", np.log(fwd_rv), X2),
        ("fwd 21d worst cum. return ~ EWMA + P(stress)  [secondary]", worst, X),
    ]:
        d = pd.concat([y.rename("y"), Xm], axis=1).dropna()
        m = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        out[name] = m
    return out
