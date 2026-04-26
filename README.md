# MetaLearner

MetaLearner is a modularized implementation of the stock-selection workflow originally developed in research notebooks.

## What this package includes

- `Data/`: data ingestion, quality checks, feature engineering, and preprocessing utilities
- `StockSelection/`: model training, per-date prediction, and stock-picking logic
- `Backtest/`: date scheduling helpers, rolling selection backtest orchestration, portfolio backtest utilities, and visualization
- `BackTest.py`: compatibility import alias for backtest functions
- `OldFunctions/`: legacy functions kept for comparison and validation

## Install dependencies

From the `MetaLearner` directory:

```bash
pip install -r requirement.txt
```

## Typical workflow

1. Build/update processed features with functions in `Data/`.
2. Run stock-selection experiments using `StockSelection`.
3. Run rolling date-based orchestration with `Backtest.run_selection_backtest(...)`.
4. Evaluate strategy with `Backtest.run_30_backtests(...)` and plotting helpers.
5. Generate deployment picks with `StockSelection.identify_stocks_for_deployment(...)`.

## Notes

- Primary orchestration for test-date scheduling and rolling experiments is in `Backtest`.
- Notebook experiments can still call older wrappers in `StockSelection` for compatibility.
