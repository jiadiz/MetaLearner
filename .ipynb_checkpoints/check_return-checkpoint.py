
from __future__ import annotations

from datetime import date, datetime
from typing import Sequence, Union, Optional, List, Dict, Any

import pandas as pd
import yfinance as yf


DateLike = Union[str, date, datetime, pd.Timestamp]


def yahoo_trading_day_return(
    tickers: Sequence[str],
    test_date: DateLike,
    *,
    trading_days: int = 21,
    use_adjusted: bool = True,
) -> pd.DataFrame:
    """
    Compute each ticker's return from the first trading day ON/AFTER `test_date`
    to the close `trading_days` trading days later. If fewer than `trading_days`
    exist, return from start to latest available close.

    Parameters
    ----------
    tickers : Sequence[str]
        List/tuple of ticker symbols (e.g. ["AAPL","MSFT"]).
    test_date : DateLike
        A date/datetime/string. Function uses first trading date on/after this date.
    trading_days : int
        Number of trading days after the start trading day (default 21).
    use_adjusted : bool
        If True, use "Adj Close" when available; otherwise use "Close".

    Returns
    -------
    pd.DataFrame
        Columns: ticker, start_date, end_date, trading_days_elapsed,
                 start_price, end_price, return_pct
        where return_pct is decimal (0.10 == +10%).
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty sequence of strings.")

    # Normalize tickers
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("tickers must contain at least one non-empty ticker string.")

    td: pd.Timestamp = pd.Timestamp(test_date).tz_localize(None).normalize()

    # Buffer the range a bit to survive weekends/holidays
    start_str: str = (td - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_str: str = (pd.Timestamp.today().tz_localize(None).normalize() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    raw: pd.DataFrame = yf.download(
        tickers=tickers,
        start=start_str,
        end=end_str,
        progress=False,
        group_by="column",
        auto_adjust=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError("No data returned from yfinance. Check tickers/date range/network.")

    # Choose price field
    preferred_field: str = "Adj Close" if use_adjusted else "Close"

    # Build a (date_index x tickers) price table
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0)
        field: str = preferred_field if preferred_field in fields else "Close"
        close: pd.DataFrame = raw[field].copy()
    else:
        # Single ticker edge-case
        field = preferred_field if preferred_field in raw.columns else "Close"
        close = raw[[field]].copy()
        close.columns = [tickers[0]]

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()

    rows: List[Dict[str, Any]] = []

    for t in close.columns:
        s: pd.Series = close[t].dropna()
        if s.empty:
            continue

        s_after: pd.Series = s[s.index >= td]
        if s_after.empty:
            continue

        start_dt: pd.Timestamp = s_after.index[0]
        start_i: int = int(s.index.get_loc(start_dt))

        end_i: int = min(start_i + trading_days, len(s) - 1)
        end_dt: pd.Timestamp = s.index[end_i]

        start_px: float = float(s.iloc[start_i])
        end_px: float = float(s.iloc[end_i])
        ret: float = (end_px / start_px) - 1.0

        rows.append({
            "ticker": t,
            "start_date": start_dt.date(),
            "end_date": end_dt.date(),
            "trading_days_elapsed": int(end_i - start_i),  # realized days (<= trading_days)
            "start_price": start_px,
            "end_price": end_px,
            "return_pct": ret,
        })

    out: pd.DataFrame = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "ticker", "start_date", "end_date", "trading_days_elapsed",
            "start_price", "end_price", "return_pct"
        ])

    return out
