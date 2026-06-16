"""
Validate MeanReversionFeaturesV2 produces identical outputs to MeanReversionFeatures
for the legacy sector/econ two-predictor setup.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, r"C:\Users\zhan5\OneDrive\Desktop\Quant")

import MetaLearner.Data.MeanReversionFeatures as v1
import MetaLearner.Data.MeanReversionFeaturesV2 as v2

DATA = r"C:\Users\zhan5\OneDrive\Desktop\Quant\MetaLearner\Datafiles\Experimental_data.csv"
TICKER = "CHRW"
WINDOWS = [126, 252, 504]
SECTOR_TYPES = ["sector", "econ"]

df = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date")
p = df[f"{TICKER}_Close"]
etf_p = df["XLI_Close"]
sp500_p = df["sp500_Close"]

p_r = np.log(p / p.shift(1)).copy()
etf_r = np.log(etf_p / etf_p.shift(1)).copy()
sp500_r = np.log(sp500_p / sp500_p.shift(1)).copy()

predictor_configs = [
    (etf_p, "sector", "price"),
    (etf_r, "sector", "return"),
    (sp500_p, "econ", "price"),
    (sp500_r, "econ", "return"),
]

# --- V1 path ---
store_v1: dict = {}
for w in WINDOWS:
    v1.fill_missing_mean_reversion_features(store_v1, p, etf_p, sp500_p, TICKER, "price", w)
    v1.fill_missing_mean_reversion_features(store_v1, p, etf_p, sp500_p, TICKER, "return", w)

_, store_v1_slice, vanilla_v1 = v1.compute_mean_reversion_for_ticker(
    TICKER, p, etf_p, sp500_p, WINDOWS, existing_ticker_data=None)

price_var_v1 = v1.create_mean_reversion_variants(
    {TICKER: store_v1_slice}, TICKER, WINDOWS, is_price_series=True)
return_var_v1 = v1.create_mean_reversion_variants(
    {TICKER: store_v1_slice}, TICKER, WINDOWS, is_price_series=False)

# --- V2 path ---
store_v2: dict = {}
_, store_v2_slice, vanilla_v2 = v2.compute_mean_reversion_for_ticker(
    TICKER, p, p_r, predictor_configs, WINDOWS, existing_ticker_data=None)

price_var_v2 = v2.create_mean_reversion_variants(
    {TICKER: store_v2_slice}, TICKER, WINDOWS, SECTOR_TYPES, is_price_series=True)
return_var_v2 = v2.create_mean_reversion_variants(
    {TICKER: store_v2_slice}, TICKER, WINDOWS, SECTOR_TYPES, is_price_series=False)


def compare_dicts(d1: dict, d2: dict, label: str):
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())
    assert keys1 == keys2, f"{label}: cache keys differ {keys1 ^ keys2}"
    for k in sorted(keys1):
        dates1 = set(d1[k].keys())
        dates2 = set(d2[k].keys())
        assert dates1 == dates2, f"{label}/{k}: date keys differ (n1={len(dates1)}, n2={len(dates2)})"
        fields = ["y_today", "beta", "adf_p", "z_score_today", "resid_std",
                  "resid_today", "resid_forecast", "resid_forecasted_change", "resid_forecast_sum"]
        for dt in dates1:
            for f in fields:
                a = d1[k][dt][f]
                b = d2[k][dt][f]
                if np.isnan(a) and np.isnan(b):
                    continue
                assert np.isclose(a, b, rtol=0, atol=1e-12), (
                    f"{label}/{k}/{dt}/{f}: {a} != {b}")


def compare_frames(df1: pd.DataFrame, df2: pd.DataFrame, label: str):
    assert set(df1.columns) == set(df2.columns), (
        f"{label}: column mismatch\nonly_v1={set(df1.columns)-set(df2.columns)}\n"
        f"only_v2={set(df2.columns)-set(df1.columns)}")
    cols = sorted(df1.columns)
    j = df1[cols].join(df2[cols], lsuffix="_v1", rsuffix="_v2", how="inner")
    worst = ("", -np.inf)
    for col in cols:
        a = j[f"{col}_v1"]
        b = j[f"{col}_v2"]
        mask = ~(a.isna() | b.isna())
        if mask.sum() < 2:
            continue
        r2 = r2_score(a[mask], b[mask])
        mx = float((a[mask] - b[mask]).abs().max())
        if r2 < worst[1]:
            worst = (col, r2)
        assert r2 > 0.999999 or mx < 1e-10, f"{label}/{col}: r2={r2:.8f} maxdiff={mx:.3e}"
    print(f"  {label}: {len(cols)} columns, all match (worst r2={worst[1]:.10f} on {worst[0]})")


print("=== Cache parity (fill_missing path) ===")
compare_dicts(store_v1[TICKER], store_v2_slice, "fill_missing")
print("  OK")

print("\n=== Cache parity (compute_mean_reversion_for_ticker path) ===")
compare_dicts(store_v1_slice, store_v2_slice, "compute_worker")
print("  OK")

print("\n=== Vanilla features parity ===")
common = vanilla_v1.index.intersection(vanilla_v2.index)
for col in vanilla_v1.columns:
    a = vanilla_v1.loc[common, col]
    b = vanilla_v2.loc[common, col]
    mask = ~(a.isna() | b.isna())
    if mask.sum() < 2:
        continue
    r2 = r2_score(a[mask], b[mask])
    assert r2 > 0.999999, f"vanilla/{col}: r2={r2}"
print("  OK")

print("\n=== Variants parity ===")
compare_frames(price_var_v1, price_var_v2, "price_variants")
compare_frames(return_var_v1, return_var_v2, "return_variants")

print("\nAll parity checks passed.")
