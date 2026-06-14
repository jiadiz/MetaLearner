"""
Diagnose why macd_line / macd_signal / macd_hist and res_streak diverge from
the ground-truth notebook for the 252 / 504 windows.

Hypothesis
----------
In create_mean_reversion_variants, the MACD EMAs and res_streak are computed as

    resid.iloc[window:].ewm(...).reindex(resid.index)
    resid.iloc[window:].expanding().apply(streak).reindex(resid.index)

But `resid` is built from the per-ticker dict, which only contains the dates
that actually have a residual (positions >= window). It has NO leading-NaN
warmup region. So `.iloc[window:]` drops a SECOND `window` of residuals and
seeds the EMA `window` rows too late vs the ground truth, which seeds the
incremental EMA at the very first residual (loop `for i in range(window, n)`).

This script isolates that by feeding the SAME residual series into:
  * the ground-truth incremental EMA loop (update_ema), and
  * the current module consumption (`resid.iloc[window:].ewm(...)`), and
  * the proposed fix (`resid.ewm(...)`).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, r"C:\Users\zhan5\OneDrive\Desktop\Quant")

from MetaLearner.Data.MeanReversionFeatures import (  # noqa: E402
    fill_missing_mean_reversion_features,
    create_mean_reversion_variants,
    update_ema,
    residual_streak_length,
)

DATA = r"C:\Users\zhan5\OneDrive\Desktop\Quant\MetaLearner\Datafiles\Experimental_data.csv"
TICKER = "CHRW"
WINDOWS = [126, 252, 504]

df = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date")
p = df[f"{TICKER}_Close"]
etf_p = df["XLI_Close"]
sp500_p = df["sp500_Close"]

store: dict = {}
for w in WINDOWS:
    fill_missing_mean_reversion_features(store, p, etf_p, sp500_p, TICKER, "return", w)


def get_resid(econ_or_sector: str, window: int) -> pd.Series:
    d = store[TICKER][f"mean_reversion_{econ_or_sector}_return_d{window}"]
    return pd.Series(d).apply(lambda x: x["resid_today"]).dropna()


def gt_macd(resid: pd.Series) -> pd.DataFrame:
    """Ground-truth incremental MACD: seed EMA at the first residual."""
    a_f, a_s, a_sig = 2 / 13, 2 / 27, 2 / 10
    v = resid.values
    n = len(v)
    line = np.full(n, np.nan)
    signal = np.full(n, np.nan)
    hist = np.full(n, np.nan)
    pf = ps = psig = np.nan
    for i in range(n):
        pf = update_ema(pf, v[i], a_f)
        ps = update_ema(ps, v[i], a_s)
        if not (np.isnan(pf) or np.isnan(ps)):
            line[i] = pf - ps
        psig = update_ema(psig, line[i], a_sig)
        signal[i] = psig
        if not (np.isnan(line[i]) or np.isnan(signal[i])):
            hist[i] = line[i] - signal[i]
    return pd.DataFrame({"line": line, "signal": signal, "hist": hist}, index=resid.index)


def module_macd(resid: pd.Series, window: int) -> pd.DataFrame:
    """Current module behavior: resid.iloc[window:].ewm(...).reindex(...)."""
    ema_fast = resid.iloc[window:].ewm(span=12, adjust=False).mean().reindex(resid.index)
    ema_slow = resid.iloc[window:].ewm(span=26, adjust=False).mean().reindex(resid.index)
    line = ema_fast - ema_slow
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return pd.DataFrame({"line": line, "signal": signal, "hist": hist})


def fixed_macd(resid: pd.Series) -> pd.DataFrame:
    """Proposed fix: run ewm over the full residual series (seed at first)."""
    ema_fast = resid.ewm(span=12, adjust=False).mean()
    ema_slow = resid.ewm(span=26, adjust=False).mean()
    line = ema_fast - ema_slow
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return pd.DataFrame({"line": line, "signal": signal, "hist": hist})


def gt_streak(resid: pd.Series) -> pd.Series:
    return resid.expanding().apply(residual_streak_length)


def module_streak(resid: pd.Series, window: int) -> pd.Series:
    return resid.iloc[window:].expanding().apply(residual_streak_length).reindex(resid.index)


def fixed_streak(resid: pd.Series) -> pd.Series:
    return resid.expanding().apply(residual_streak_length)


def cmp(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    j = pd.concat([a, b], axis=1).dropna()
    if len(j) < 2:
        return np.nan, np.nan
    return r2_score(j.iloc[:, 0], j.iloc[:, 1]), float((j.iloc[:, 0] - j.iloc[:, 1]).abs().max())


print(f"Ticker={TICKER}  n={len(p)}\n")
for eos in ["sector", "econ"]:
    for w in WINDOWS:
        resid = get_resid(eos, w)
        gt = gt_macd(resid)
        mod = module_macd(resid, w)
        fix = fixed_macd(resid)
        print(f"=== {eos}  window={w}  (#resid={len(resid)}) ===")
        for col in ["line", "signal", "hist"]:
            r2_mod, mx_mod = cmp(gt[col], mod[col])
            r2_fix, mx_fix = cmp(gt[col], fix[col])
            print(f"  macd_{col:6s}  current r2={r2_mod:8.4f} (maxdiff={mx_mod:.2e})   "
                  f"fixed r2={r2_fix:10.6f} (maxdiff={mx_fix:.2e})")
        r2_mod, mx_mod = cmp(gt_streak(resid), module_streak(resid, w))
        r2_fix, mx_fix = cmp(gt_streak(resid), fixed_streak(resid))
        print(f"  res_streak    current r2={r2_mod:8.4f} (maxdiff={mx_mod:.2e})   "
              f"fixed r2={r2_fix:10.6f} (maxdiff={mx_fix:.2e})")
        print()

# End-to-end: the ACTUAL create_mean_reversion_variants (post-fix) vs ground truth.
print("\n########## END-TO-END: real create_mean_reversion_variants vs ground truth ##########\n")
variants = create_mean_reversion_variants(store, TICKER, WINDOWS, is_price_series=False)
for eos in ["sector", "econ"]:
    for w in WINDOWS:
        resid = get_resid(eos, w)
        gt = gt_macd(resid)
        gts = gt_streak(resid)
        print(f"=== {eos} window={w} ===")
        for col in ["line", "signal", "hist"]:
            r2v, mxv = cmp(gt[col], variants[f"{eos}_r_macd_{col}_{w}"])
            print(f"  macd_{col:6s} r2={r2v:10.6f} (maxdiff={mxv:.2e})")
        r2v, mxv = cmp(gts, variants[f"{eos}_r_res_streak_{w}"])
        print(f"  res_streak  r2={r2v:10.6f} (maxdiff={mxv:.2e})")
        print()
