"""
Download daily price series for a caller-supplied list of tickers and
save them as a single wide CSV (one *_Volume and *_Close column per ticker).

The ticker universe is no longer hardcoded here. Use
MetaLearner.Data.fetch_universe_tickers.get_all_universe_tickers() (or any
other source) and pass the result into create_price_series_csv(tickers=...).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def download_daily_data(ticker: str, start: str = "2018-01-01") -> pd.DataFrame | None:
    """Download daily OHLCV data for a ticker from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start, interval="1d", progress=False, threads=False)
        if data.empty:
            return None
        return data
    except Exception:
        return None


def create_price_series_csv(
    tickers: Iterable[str],
    output_path: str | Path | None = None,
    start_date: str = "2018-01-01",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download price data for each ticker in `tickers`, combine into a single
    wide DataFrame, and save to CSV.

    Parameters
    ----------
    tickers : iterable of str
        Tickers to download. Required. Typically produced by
        MetaLearner.Data.fetch_universe_tickers.get_all_universe_tickers().
        '^GSPC' is automatically renamed to 'sp500' in the output columns.
    output_path : str or Path or None
        Path for the output CSV file. If None, saves to
        MetaLearner/Datafiles/all_daily_adjusted_close.csv.
    start_date : str
        Start date for historical data (YYYY-MM-DD).
    verbose : bool
        If True, print progress for each ticker.

    Returns
    -------
    pd.DataFrame
        The combined price data (Volume + Close per ticker).
    """
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    if not tickers:
        raise ValueError("create_price_series_csv requires a non-empty `tickers` list.")

    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        if verbose:
            print(f"Downloading data for {ticker}...")
        df = download_daily_data(ticker, start=start_date)
        if df is not None and not df.empty:
            data[ticker] = df
        elif verbose:
            print(f"  Could not download data for {ticker}")

    close_data: dict[str, pd.DataFrame] = {}
    for ticker, df in data.items():
        stock_data = df[["Volume", "Close"]].copy()
        # The S&P 500 index quote (^GSPC) is referenced as 'sp500' downstream.
        if ticker == "^GSPC":
            ticker = "sp500"
        stock_data.columns = [f"{ticker}_Volume", f"{ticker}_Close"]
        close_data[ticker] = stock_data

    if not close_data:
        raise ValueError("No data was downloaded for any ticker.")

    # pd.concat with dict uses keys as column level; droplevel(0) yields ticker-prefixed names
    train_data = pd.concat(close_data, axis=1)
    train_data = train_data.reset_index()
    train_data.columns = train_data.columns.droplevel(0)
    train_data.columns.values[0] = "Date"

    if output_path is None:
        output_path = Path(__file__).resolve().parents[1] / "Datafiles" / "all_daily_adjusted_close.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(output_path, index=False)
    if verbose:
        print(f"\nSaved {len(train_data)} rows to {output_path}")

    return train_data


if __name__ == "__main__":
    # Convenience entry point: build the default universe and dump prices.
    from MetaLearner.Data.fetch_universe_tickers import get_all_universe_tickers

    universe = get_all_universe_tickers(years=2)
    create_price_series_csv(universe)
