from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def reference_momentum_features(
    price_series: pd.Series,
    lookback: int,
    holddays: int = 21,
    min_n_indep: int = 3,
) -> pd.DataFrame:
    """
    Original notebook method:
    - train correlation strictly before current trade date i
    - forecast_ret_{lookback} = corr * ret_lag_{lookback}
    """
    p = price_series.astype(float)
    ret_lag = (p - p.shift(lookback)) / p.shift(lookback)
    ret_fut = (p.shift(-holddays) - p) / p

    n_total = len(p)
    corr_out = np.full(n_total, np.nan, dtype=float)
    pval_out = np.full(n_total, np.nan, dtype=float)

    for i in range(n_total):
        ret_lag_sub = ret_lag.iloc[:i]
        ret_fut_sub = ret_fut.iloc[:i]

        good = (~ret_lag_sub.isna()) & (~ret_fut_sub.isna())
        ret_lag_train = ret_lag_sub[good]
        ret_fut_train = ret_fut_sub[good]

        step = holddays if holddays <= lookback else lookback
        idx = np.arange(0, len(ret_lag_train), step)
        ret_lag_train = ret_lag_train.iloc[idx]
        ret_fut_train = ret_fut_train.iloc[idx]

        if len(ret_lag_train) < min_n_indep:
            continue
        if ret_lag_train.std(ddof=1) == 0 or ret_fut_train.std(ddof=1) == 0:
            continue

        r, pv = pearsonr(ret_lag_train.to_numpy(), ret_fut_train.to_numpy())
        corr_out[i] = r
        pval_out[i] = pv

    corr_s = pd.Series(corr_out, index=p.index, name=f"corr_expanding_indep_{lookback}")
    pval_s = pd.Series(pval_out, index=p.index, name=f"pval_expanding_indep_{lookback}")
    ret_lag_s = ret_lag.rename(f"ret_lag_{lookback}")
    forecast_s = (corr_s * ret_lag_s).rename(f"forecast_ret_{lookback}")
    adjusted_s = (forecast_s * (1 - pval_s)).rename(f"forecast_s_times_1_minus_pval_{lookback}")

    return pd.concat([ret_lag_s, corr_s, pval_s, forecast_s, adjusted_s], axis=1)


def load_experimental_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("momentum_exp", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def module_momentum_features(
    module,
    price_series: pd.Series,
    ticker: str,
    lookbacks: list[int],
    holddays: int = 21,
) -> pd.DataFrame:
    db: dict = {}
    for lb in lookbacks:
        module.fill_missing_correlation_coefficients(db, price_series, ticker, lb, holddays)

    out = pd.DataFrame(index=price_series.index)
    for lb in lookbacks:
        combo = f"lag_{lb}_hold_{holddays}"
        series = pd.Series(db[ticker][combo])
        forecast = series.apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else np.nan)
        pval = series.apply(lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else np.nan)
        out[f"forecast_ret_{lb}"] = forecast
        out[f"pval_expanding_indep_{lb}"] = pval
        out[f"forecast_s_times_1_minus_pval_{lb}"] = forecast * (1 - pval)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_path = root / "MetaLearner" / "Datafiles" / "Experimental_data.csv"
    module_path = Path(__file__).resolve().parent / "MomentumFeaturesExperimental.py"

    price_data = pd.read_csv(data_path)
    price_data["Date"] = pd.to_datetime(price_data["Date"])
    price_data = price_data.set_index("Date")

    sampled_ticker = "CHRW"
    price_col = f"{sampled_ticker}_Close"
    price_series = price_data[price_col].astype(float)
    lookbacks = [1, 5, 10, 25, 60, 120, 250]

    module = load_experimental_module(module_path)
    module_df = module_momentum_features(module, price_series, sampled_ticker, lookbacks)

    checks = []
    for lb in lookbacks:
        ref_df = reference_momentum_features(price_series, lb, holddays=21, min_n_indep=3)
        merged = pd.concat(
            [
                ref_df[f"forecast_ret_{lb}"].rename("ref_forecast"),
                ref_df[f"pval_expanding_indep_{lb}"].rename("ref_pval"),
                ref_df[f"forecast_s_times_1_minus_pval_{lb}"].rename("ref_adjusted"),
                module_df[f"forecast_ret_{lb}"].rename("mod_forecast"),
                module_df[f"pval_expanding_indep_{lb}"].rename("mod_pval"),
                module_df[f"forecast_s_times_1_minus_pval_{lb}"].rename("mod_adjusted"),
            ],
            axis=1,
        )
        checks.append(
            {
                "lookback": lb,
                "forecast_identical": bool(np.allclose(merged["ref_forecast"], merged["mod_forecast"], equal_nan=True)),
                "pval_identical": bool(np.allclose(merged["ref_pval"], merged["mod_pval"], equal_nan=True)),
                "adjusted_identical": bool(np.allclose(merged["ref_adjusted"], merged["mod_adjusted"], equal_nan=True)),
                "forecast_max_abs_diff": float((merged["ref_forecast"] - merged["mod_forecast"]).abs().max(skipna=True) or 0.0),
                "pval_max_abs_diff": float((merged["ref_pval"] - merged["mod_pval"]).abs().max(skipna=True) or 0.0),
                "adjusted_max_abs_diff": float((merged["ref_adjusted"] - merged["mod_adjusted"]).abs().max(skipna=True) or 0.0),
            }
        )

    result_df = pd.DataFrame(checks)
    result_path = Path(__file__).resolve().parent / "momentum_alignment_results.csv"
    result_df.to_csv(result_path, index=False)

    print("Sampled ticker:", sampled_ticker)
    print("Results saved to:", result_path)
    print(result_df)


if __name__ == "__main__":
    main()
