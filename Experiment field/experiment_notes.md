# Momentum Feature Discrepancy Findings

Using `Experimental_data.csv` and sampled ticker `CHRW_Close`, we compared:

1. Reference momentum features from `Global_momentum_strategy_2_12_14_momentum_breakthrough_reboot_r_plus_p_mr`
2. Features produced by `MomentumFeaturesExperimental.py`

## Root Cause

The experimental module stored the wrong first value in the momentum database:

- It stored raw correlation `r`
- But downstream notebook code interpreted that first value as `forecast_return`

Reference notebook logic defines:

- `forecast_ret_{lookback} = corr_expanding_indep_{lookback} * ret_lag_{lookback}`

So the module had a semantic mismatch (correlation value mislabeled as forecast return).

## Fix Applied

In `MomentumFeaturesExperimental.py`, `fill_missing_correlation_coefficients(...)` now stores:

- `forecast_ret_today = r * ret_lag_today`
- and p-value as the second element

This exactly matches the reference notebook's forecast definition.

## Validation

`momentum_alignment_experiment.py` compares all lookbacks:

- `[1, 5, 10, 25, 60, 120, 250]`

Output file:

- `momentum_alignment_results.csv`

Result: all lookbacks match exactly (`max_abs_diff = 0.0` for forecast, p-value, and adjusted forecast signal).
