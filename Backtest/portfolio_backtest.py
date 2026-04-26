import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Callable, Sequence


HOLD_DAYS = 21
TRADING_DAYS_PER_YEAR = 252


def _to_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(x)
                return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
            except (ValueError, SyntaxError):
                return []
        return [x]
    if isinstance(x, (tuple, set)):
        return list(x)
    return [x]


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _risk_metrics(equity: pd.Series):
    if len(equity) < 2:
        return (np.nan, np.nan)
    rets = equity.pct_change().dropna()
    if rets.empty:
        return (0.0, np.nan)
    risk = float(rets.std(ddof=1))
    mean = float(rets.mean())
    if risk == 0.0 or np.isnan(risk):
        sharpe = np.nan
    else:
        sharpe = (mean / risk) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return (risk, float(sharpe))


def choose_selection_test_dates(
    df_all: pd.DataFrame,
    *,
    horizon_bd: int = 21,
    spacing_days: int = 11,
    lookback_windows: int = 8,
    test_dates: Sequence[str | pd.Timestamp] | None = None,
    newest_first: bool = True,
) -> list[pd.Timestamp]:
    """
    Choose test dates for cross-sectional stock-selection backtesting.

    If `test_dates` is provided, those dates are used (normalized and deduplicated).
    Otherwise, dates are auto-sampled from the tail of available feature dates.
    """
    if test_dates is not None:
        selected = pd.to_datetime(pd.Series(list(test_dates))).dropna().dt.normalize().drop_duplicates()
        selected = selected.sort_values(ascending=not newest_first)
        return selected.tolist()

    frame = df_all.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    unique_dates = frame["Date"].drop_duplicates().sort_values()
    candidate_tail = pd.Series(unique_dates[-(horizon_bd * lookback_windows + 2) :])
    if newest_first:
        candidate_tail = candidate_tail.iloc[::-1]
    eval_dates = candidate_tail.iloc[::spacing_days]
    return pd.to_datetime(eval_dates).dt.normalize().tolist()


def run_selection_backtest(
    df_all: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    config: Any = None,
    spacing_days: int = 11,
    lookback_windows: int = 8,
    test_dates: Sequence[str | pd.Timestamp] | None = None,
    newest_first: bool = True,
    single_day_runner: Callable[..., dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Run rolling single-day stock-selection experiments over chosen test dates.

    This lives in Backtest because date scheduling and experiment looping are
    backtesting orchestration concerns.
    """
    if single_day_runner is None:
        from MetaLearner.StockSelection.selection_engine import run_experiment_single_day

        single_day_runner = run_experiment_single_day

    horizon_bd = int(getattr(config, "horizon_bd", 21)) if config is not None else 21
    eval_dates = choose_selection_test_dates(
        df_all=df_all,
        horizon_bd=horizon_bd,
        spacing_days=spacing_days,
        lookback_windows=lookback_windows,
        test_dates=test_dates,
        newest_first=newest_first,
    )

    frame = df_all.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    price = price_df.copy()

    rows = []
    for dt in eval_dates:
        rows.append(single_day_runner(frame, price, dt, config=config))
    return pd.DataFrame(rows)


def run_30_backtests(price_df: pd.DataFrame, results_df: pd.DataFrame, hold_days: int = HOLD_DAYS):
    price_df = price_df.copy()
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()

    results = results_df.copy()
    results["TEST_DATE"] = pd.to_datetime(results["TEST_DATE"])
    results = results.sort_values("TEST_DATE").reset_index(drop=True)

    sel_col = "Selected tickers (rows mode)"
    results[sel_col] = results[sel_col].apply(_to_list)
    trading_days = price_df.index

    def snap_to_next_trading_day(d):
        pos = trading_days.searchsorted(d, side="left")
        if pos >= len(trading_days):
            return pd.NaT
        return trading_days[pos]

    results["TRADE_DAY"] = results["TEST_DATE"].apply(snap_to_next_trading_day)
    results = results.dropna(subset=["TRADE_DAY"])
    start_days = results["TRADE_DAY"].drop_duplicates().tolist()[:30]

    def close_cols_for(tickers):
        cols = [f"{t}_Close" for t in tickers]
        return [c for c in cols if c in price_df.columns]

    out_rows = []
    equities = []

    for i, start_day in enumerate(start_days, start=1):
        equity = pd.Series(index=trading_days[trading_days >= start_day], dtype=float)
        if equity.empty:
            out_rows.append(
                {
                    "backtest_id": i,
                    "start_trade_day": start_day,
                    "final_return": np.nan,
                    "max_drawdown": np.nan,
                    "risk_adjusted_return": np.nan,
                    "risk": np.nan,
                }
            )
            continue

        equity.iloc[0] = 1.0
        current_value = 1.0
        last_marked_day = equity.index[0]
        trade_schedule = results.loc[results["TRADE_DAY"] >= start_day, ["TRADE_DAY", sel_col]].copy()
        trade_schedule = trade_schedule.drop_duplicates(subset=["TRADE_DAY"]).sort_values("TRADE_DAY").reset_index(
            drop=True
        )

        j = 0
        while j < len(trade_schedule):
            buy_day = trade_schedule.at[j, "TRADE_DAY"]
            tickers = trade_schedule.at[j, sel_col]

            if buy_day < equity.index[0]:
                j += 1
                continue

            if last_marked_day < buy_day:
                flat_range = equity.index[(equity.index > last_marked_day) & (equity.index < buy_day)]
                if len(flat_range):
                    equity.loc[flat_range] = current_value
                if pd.isna(equity.loc[buy_day]):
                    equity.loc[buy_day] = current_value
                last_marked_day = buy_day

            buy_pos = trading_days.get_indexer([buy_day])[0]
            sell_pos = buy_pos + hold_days
            if sell_pos >= len(trading_days):
                remaining = equity.index[equity.index > last_marked_day]
                if len(remaining):
                    equity.loc[remaining] = current_value
                break

            sell_day = trading_days[sell_pos]
            cols = close_cols_for(tickers)
            if len(cols) == 0:
                hold_range = equity.index[(equity.index > buy_day) & (equity.index <= sell_day)]
                if len(hold_range):
                    equity.loc[hold_range] = current_value
                last_marked_day = sell_day
            else:
                prices = price_df.loc[buy_day:sell_day, cols].astype(float)
                buy_prices = prices.iloc[0]
                good = buy_prices.notna() & (buy_prices != 0)
                prices = prices.loc[:, good]

                if prices.shape[1] == 0:
                    hold_range = equity.index[(equity.index > buy_day) & (equity.index <= sell_day)]
                    if len(hold_range):
                        equity.loc[hold_range] = current_value
                    last_marked_day = sell_day
                else:
                    rel = prices.divide(prices.iloc[0], axis=1)
                    port_rel = rel.mean(axis=1)
                    hold_equity = current_value * port_rel
                    idx = hold_equity.index.intersection(equity.index)
                    equity.loc[idx] = hold_equity.loc[idx]
                    current_value = float(hold_equity.loc[sell_day])
                    last_marked_day = sell_day

            j += 1
            while j < len(trade_schedule) and trade_schedule.at[j, "TRADE_DAY"] <= sell_day:
                j += 1
            if sell_day in equity.index:
                equity.loc[sell_day] = current_value

        tail = equity.index[equity.index > last_marked_day]
        if len(tail):
            equity.loc[tail] = current_value

        final_return = float(equity.iloc[-1] - 1.0)
        mdd = _max_drawdown(equity)
        risk, sharpe = _risk_metrics(equity)
        out_rows.append(
            {
                "backtest_id": i,
                "start_trade_day": start_day,
                "final_return": final_return,
                "max_drawdown": mdd,
                "risk_adjusted_return": sharpe,
                "risk": risk,
            }
        )
        equities.append(equity)

    return pd.DataFrame(out_rows), equities


def summarize_backtest_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    if metrics_df.empty:
        return {"worst_drawdown": np.nan, "average_return": np.nan, "average_sharp": np.nan}
    return {
        "worst_drawdown": float(metrics_df["max_drawdown"].min()),
        "average_return": float(metrics_df["final_return"].mean()),
        "average_sharp": float(metrics_df["risk_adjusted_return"].mean()),
    }


def plot_equity_curves(equities: list[pd.Series], title: str = "Cumulative Return for Each Backtest") -> None:
    if not equities:
        return
    plt.figure(figsize=(12, 5))
    for i, equity in enumerate(equities, start=1):
        plt.plot(equity.index, equity - 1.0, label=f"backtest_{i}", alpha=0.6)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(alpha=0.2)
    plt.tight_layout()
