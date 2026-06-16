# MetaLearner

MetaLearner is a modular stock-selection system that predicts **21-business-day forward log returns** from a large cross-sectional feature set, ranks names on each rebalance date, and applies a correlation filter to build a diversified portfolio. It is a production-oriented refactor of research notebooks: incremental feature stores, parallel ticker processing, universe-aware backtests, and a clear path from feature engineering to deployment.

## What it does

On each **test date** (a trading day whose close is available):

1. **Train** a regression model (default: Elastic Net) on historical panel data up to `test_date - 21 BDays`.
2. **Predict** forward 21-day returns for every ticker in the eligible universe on that date.
3. **Select** up to `k_top` names with the highest positive predictions that are not highly correlated with each other (using training-window price history).
4. **Evaluate** realized returns over the hold period in rolling backtests, or export picks for live deployment.

The target `y` is the log return from the test-date close to the close 21 business days later. Training rows with missing `y` (e.g. near the end of the sample) are dropped.

## Repository layout

```
MetaLearner/
├── Data/                  # Ingestion, universe, and feature engineering
├── StockSelection/        # Model training, ranking, correlation filter, deployment
├── Backtest/              # Rolling experiment orchestration and portfolio evaluation
├── Notebooks/             # Main pipeline notebook (data_process_script.ipynb)
├── Datafiles/             # Cached pickles and CSV feature stores
├── Experiment field/      # Parity checks and diagnostic scripts
├── OldFunctions/          # Legacy notebook code kept for validation
├── BackTest.py            # Compatibility import alias
└── requirement.txt
```

The parent `Quant/` folder also holds historical **deploy snapshots** (`deploy_MM_DD_YYYY.csv`) used to reconstruct membership universes for backtesting against past production runs.

## Feature pipeline

Features are built per ticker and cached in pickle databases so reruns only fill missing keys.

| Module | Role |
|--------|------|
| `fetch_universe_tickers.py` | S&P 500 / MidCap 400 historical membership from Wikipedia; sector ETFs, SPY, ^GSPC |
| `create_price_series.py` | Price/return series for stocks, sector ETFs, and macro inputs |
| `BuildBasicFeatures.py` | Momentum, volatility, RSI, MACD, drawdown, trend, volume variants |
| `MomentumFeatures.py` | Expanding-window lag/correlation forecast features (parallel via `joblib` / `loky`) |
| `MeanReversionFeatures.py` | Original OLS residual mean-reversion features (sector/econ price & return) |
| `MeanReversionFeaturesV2.py` | Flexible API: univariate/multivariate predictors, parallel variant workers |
| `fama_factors.py` | Fama-French factors; RF from Treasury 1-month yield (default) or compounded daily |
| `create_regime_data.py` | Macro regime inputs (rates, CPI, gold, VIX, etc.) |
| `create_sector_and_industries.py` | Sector / industry labels |
| `SaveLoadDictionaryDB.py` | Incremental pickle load/save for feature caches |
| `examine_data_quality.py` | Data-quality helpers |

**Mean reversion (V2)** fits rolling OLS of a stock series against one or more predictors (sector ETF, SP500, Fama-French), then engineers residual z-scores, ADF p-values, MACD, RSI, momentum/reversal, and multi-step AR(1) residual forecasts. Predictor configs are tuples of `(X, sector_type, series_type)` — e.g. `('sector', 'return')`, `('fama', 'return')`.

The main orchestration notebook is `Notebooks/data_process_script.ipynb`.

## Stock selection

`StockSelection/selection_engine.py` is the core.

### `SelectionConfig`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `horizon_bd` | 21 | Forward return horizon and train/test gap |
| `model_type` | `"elastic_net"` | Also `"linear"`, `"ridge"`, `"lasso"` |
| `train_data_density` | `"soft"` | `"soft"` keeps every 21st row per ticker; `"dense"` keeps all |
| `k_top` | 8 | Max picks per date |
| `corr_thresh` | 0.9 | Reject candidate if \|corr\| with any pick exceeds threshold; `None` or `≥1` disables |
| `nan_corr` | `"reject"` | `"reject"` drops picks when correlation is undefined; `"allow"` skips the check |
| `exclude_tickers` | `("sp500",)` | Tickers never selected |

### Key functions

- **`run_experiment_single_day`** — train + predict + select for one test date.
- **`run_rolling_backtest_selection`** — thin wrapper; prefer `Backtest.run_selection_backtest`.
- **`identify_stocks_for_deployment`** — train on full labeled history, score a deploy feature frame, apply the same correlation filter.

### Correlation filter

Candidates are sorted by `y_pred` descending. A name is kept only if its training-window close series is not highly correlated with any already-selected ticker. With `nan_corr='reject'`, missing price columns or NaN correlations also reject the candidate — a conservative setting that can leave fewer than `k_top` picks.

## Universe / membership filtering

When `membership_df` is passed to backtests, train and test rows are restricted to tickers in **S&P 500 ∪ S&P MidCap 400** as of that test date (snapshot membership, not per-row point-in-time).

Build membership with:

```python
from MetaLearner.Data.fetch_universe_tickers import get_all_universe_tickers

membership_df = get_all_universe_tickers(years=10)
```

For historical deploy universes, build a timeline from `deploy_*.csv` files in the `Quant/` folder (one `Date` and ticker list per file).

## Backtesting

`Backtest/portfolio_backtest.py` orchestrates rolling experiments and portfolio metrics.

```python
from MetaLearner.Backtest import run_selection_backtest, run_30_backtests
from MetaLearner.StockSelection.selection_engine import SelectionConfig

cfg = SelectionConfig(k_top=8, corr_thresh=0.9, nan_corr="reject")

results = run_selection_backtest(
    df_all=feature_df,          # panel with Date, Ticker, features, y
    price_df=price_df,          # wide close prices: columns like AAPL_Close
    membership_df=membership_df,
    config=cfg,
    test_dates=["2025-05-04", "2025-02-20"],  # optional; overrides auto scheduling
)

equity = run_30_backtests(price_df, results, hold_days=21)
```

- **`choose_selection_test_dates`** — auto-pick spaced dates from the feature tail, or accept an explicit `test_dates` list.
- **`run_30_backtests`** — simulate equal-weight holds from each test date's picks; compute equity curve, drawdown, Sharpe.
- **`summarize_backtest_metrics`** / **`plot_equity_curves`** — reporting helpers.

**Test-date semantics:** `test_date` is the as-of date (features computed from data through that close). `TRADE_DAY` in portfolio backtests snaps to the next trading day when needed.

## Deployment workflow

1. Build features through the latest available close (`data_process_script.ipynb`).
2. Assemble a deploy row per ticker (same schema as training features, without `y`).
3. Run `identify_stocks_for_deployment(train_df, deploy_df, price_df, config=cfg)`.
4. Export to `deploy_MM_DD_YYYY.csv` for record-keeping.

## Install

From the `MetaLearner` directory:

```bash
pip install -r requirement.txt
```

Core dependencies: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `yfinance`, `joblib`. Feature modules also use `statsmodels`, `requests`, and `joblib` parallel backends.

Run notebooks and scripts from the repo root with `MetaLearner` on `PYTHONPATH`, or install the package in editable mode if you add a `setup.py` / `pyproject.toml`.

## Validation

Scripts under `Experiment field/` verify notebook parity after refactors:

- `mean_reversion_v2_parity.py` — V1 vs V2 mean-reversion features
- `macd_streak_diagnosis.py` — MACD / residual streak alignment with ground truth

Use these when changing feature logic; target R² = 1.0 against the reference notebook outputs.

## Design notes

- **Incremental caches** — Feature dicts are keyed by `(ticker, series_type, lookback, …)` so partial reruns are cheap.
- **Parallelism** — Mean-reversion variants and momentum features use process-based workers (`loky`) to avoid GIL contention on large dicts.
- **Train leakage guard** — Training ends at `test_date - horizon_bd`; prices for correlation filtering use the same cutoff.
- **Sparse backtest days** — Non-trading test dates, empty universes, or all-NaN `y` on a date return minimal result dicts (metrics become NaN). Filter `test_dates` to valid trading days with sufficient forward data.
- **Legacy code** — `OldFunctions/` and `BackTest.py` remain for backward compatibility with older notebooks; new work should use `Data/`, `StockSelection/`, and `Backtest/` directly.

## Typical end-to-end flow

```
Download universe & prices
        ↓
Build basic / momentum / mean-reversion / regime features  (Data/)
        ↓
Merge into panel feature_df with Date, Ticker, y
        ↓
run_selection_backtest(...)  →  per-date picks & metrics
        ↓
run_30_backtests(...)        →  portfolio equity curve
        ↓
identify_stocks_for_deployment(...)  →  live picks
```

For questions about a specific parameter or module, start with `SelectionConfig` and `Notebooks/data_process_script.ipynb` — they wire together everything above.
