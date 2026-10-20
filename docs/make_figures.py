"""
Build the README figures from the saved Nifty regime signals.

Run after regime-event-diagnostics.ipynb has produced nifty_regime_signals.csv:

    python docs/make_figures.py

Strategy returns use the same convention as the notebooks: signal at close t trades at t+1
and first earns day t+2, 10 bps one-way cost, cash at 6%, window 2013-03-21 to 2026-10-16.
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jump_model as jm  # noqa: E402
import regime_allocation as ra  # noqa: E402
import regime_filter as rf  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical slots 1-3 (blue, orange, aqua) + neutrals
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, LIGHT_GREY = "#6b6a66", "#b9b8b2"
COLORS = {"Jump model (0/1)": BLUE, "HMM soft overlay": ORANGE, "EWMA vol-target": AQUA,
          "Buy & hold": GREY, "VIX vol-target": LIGHT_GREY, "200-day MA": LIGHT_GREY}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})

# ------------------------------------------------------------------ data
px = ra.download_assets()
close, vix = px["nifty"], px["vix"]
ret_full = close.pct_change()
sig = pd.read_csv(ROOT / "nifty_regime_signals.csv", index_col=0, parse_dates=True)
bull = sig["sjm_bull"].dropna()
idx = bull.index
expo = {
    "Jump model (0/1)": bull,
    "HMM soft overlay": sig["hmm_exposure"].reindex(idx),
    "EWMA vol-target": jm.ewma_vol_target_exposure(ret_full.dropna(), 0.12).reindex(idx),
    "VIX vol-target": jm.vix_vol_target_exposure(ret_full.dropna(), vix, 0.12).reindex(idx),
    "200-day MA": jm.ma_exposure(close).reindex(idx),
    "Buy & hold": pd.Series(1.0, index=idx),
}
rets = {k: rf.backtest(ret_full.reindex(idx), v.shift(1), 0.06, 10.0).dropna() for k, v in expo.items()}
common = rets["Jump model (0/1)"].index
for r in rets.values():
    common = common.intersection(r.index)
rets = {k: r.loc[common] for k, r in rets.items()}
perf = pd.DataFrame({k: rf.perf(r, expo[k], 0.06) for k, r in rets.items()}).T
print(perf[["CAGR", "Sharpe", "MaxDD", "AnnTurnover"]].round(3))


def spans(mask: pd.Series):
    """Contiguous True runs as (start, end) timestamps."""
    m = mask.fillna(False).to_numpy().astype(bool)
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            out.append((mask.index[i], mask.index[j]))
            i = j + 1
        else:
            i += 1
    return out


# -------------------------------------------------------- figure 1: Nifty + regimes
start = sig.index.min()
nifty = close.loc[start:]
fig, (a0, a1) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]})
a0.plot(nifty, color=INK, lw=1.4)
a0.set_yscale("log")
a0.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a0.set_yticks([5000, 10000, 20000]); a0.set_yticklabels(["5,000", "10,000", "20,000"])
a0.set_ylabel("Nifty 50 (log scale)")
for s, e in spans(sig["hmm_state"].eq("S2")):
    a0.axvspan(s, e, color=ORANGE, alpha=0.28, lw=0)
a0.set_title("Nifty 50 with the HMM stress state shaded (filtered, out of sample)")
covid = pd.Timestamp("2020-03-23")
a0.annotate("Covid crash, Mar 2020", xy=(covid, nifty.loc[covid]), xytext=(pd.Timestamp("2016-08-01"), 6600),
            color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a0.plot([], [], color=ORANGE, alpha=0.5, lw=8, label="HMM stress state (highest-volatility regime)")
a0.legend(loc="upper left")

rows = [("HMM stress state", sig["hmm_state"].eq("S2"), ORANGE), ("Jump model bear state", bull.eq(0), BLUE)]
for k, (name, mask, c) in enumerate(rows):
    for s, e in spans(mask):
        a1.fill_between([s, e + pd.Timedelta(days=1)], k + 0.15, k + 0.85, color=c, lw=0)
a1.set_ylim(0, 2); a1.set_yticks([0.5, 1.5]); a1.set_yticklabels([r[0] for r in rows]); a1.grid(False)
a1.tick_params(axis="y", length=0)
a1.set_xlim(start, nifty.index.max())
fig.text(0.01, 0.005, "HMM: 3-state, walk-forward refit every 6 months from 2012-03. Jump model: from 2013-03. Both use only past data.",
         color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig(OUT / "nifty_regimes.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------- figure 2: equity and drawdown
show = ["Buy & hold", "EWMA vol-target", "HMM soft overlay", "Jump model (0/1)"]
fig, (a0, a1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})
ends = {}
for k in show:
    eq = (1 + rets[k]).cumprod()
    a0.plot(eq, color=COLORS[k], lw=1.7, label=k)
    dd = eq / eq.cummax() - 1
    a1.plot(dd, color=COLORS[k], lw=1.4)
    ends[k] = eq.iloc[-1]
# end labels with a minimum vertical gap (in log space) so near-identical series stay readable
short = {"Buy & hold": "Buy & hold", "EWMA vol-target": "EWMA", "HMM soft overlay": "HMM", "Jump model (0/1)": "Jump"}
ys = {k: np.log(v) for k, v in ends.items()}
order_ = sorted(ys, key=ys.get)
gap = 0.055
for i in range(1, len(order_)):
    if ys[order_[i]] - ys[order_[i - 1]] < gap:
        ys[order_[i]] = ys[order_[i - 1]] + gap
for k in show:
    a0.text(eq.index[-1] + pd.Timedelta(days=40), np.exp(ys[k]), f"{short[k]} {ends[k]:.1f}x", color=INK, fontsize=9, va="center")
a0.set_yscale("log"); a0.set_yticks([1, 2, 3, 4]); a0.set_yticklabels(["1x", "2x", "3x", "4x"])
a0.set_ylabel("Growth of 1 (log scale)")
a0.set_title("Out-of-sample growth of 1, net of 10 bps costs, Nifty 2013 to 2026")
a0.legend(loc="upper left", ncol=2)
a1.set_ylabel("Drawdown"); a1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
a1.set_title("Drawdown from the previous peak")
a1.set_xlim(common.min(), common.max() + pd.Timedelta(days=330))
fig.text(0.01, 0.005, "Signal at close t trades at t+1, first earns t+2. Cash earns 6% (the jump model's straight segments are periods in cash). Price index, no dividends.",
         color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig(OUT / "strategy_equity.png", dpi=160)
plt.close(fig)

# ----------------------------------------------------- figure 3: scorecard bars
order = perf.sort_values("Sharpe", ascending=True).index.tolist()  # bottom -> top
panels = [("Sharpe ratio", perf["Sharpe"], "{:.2f}", "higher is better"),
          ("Worst drawdown", -perf["MaxDD"], "{:.0%}", "lower is better"),
          ("Annual turnover", perf["AnnTurnover"], "{:.1f}x", "portfolio traded per year; lower is cheaper")]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
for ax, (title, vals, fmt, note) in zip(axes, panels):
    y = np.arange(len(order))
    ax.barh(y, [vals[k] for k in order], color=[COLORS[k] for k in order], height=0.62)
    for yi, k in zip(y, order):
        ax.text(vals[k] + vals.max() * 0.015, yi, fmt.format(vals[k]), va="center", fontsize=9, color=INK)
    ax.set_title(title, fontsize=11); ax.set_xlim(0, vals.max() * 1.22)
    ax.set_xlabel(note, color=INK2, fontsize=8.5)
    ax.grid(axis="y", visible=False); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False); ax.spines["left"].set_color(GRID)
for ax in axes[1:]:
    ax.tick_params(axis='y', left=False)
axes[0].set_yticks(np.arange(len(order))); axes[0].set_yticklabels(order)
fig.suptitle("Nifty 2013 to 2026, net of 10 bps: the jump model protects best, no rule beats buy and hold on return",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "strategy_scorecard.png", dpi=160)
plt.close(fig)
print("saved:", sorted(p.name for p in OUT.glob("*.png")))
