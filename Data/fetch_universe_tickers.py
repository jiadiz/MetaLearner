"""
Fetch the full ticker universe MetaLearner needs to download:

  * S&P 500 historical membership (current constituents + every ticker that
    was added or removed within `years` years).
  * S&P MidCap 400 historical membership (same logic).
  * The sector SPDR ETFs + SPY.
  * The S&P 500 index quote symbol (^GSPC).

The historical-membership logic is adapted from
MetaLearner/Notebooks/Fetch_all_changes_in_sp_tickers.ipynb.
"""
from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

# Sector SPDR ETFs + the broad-market SPY ETF.
SECTOR_ETF_TICKERS: list[str] = [
    "XLK", "XLF", "XLI", "XLV", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLE",
    "XLB", "SPY",
]

# yfinance symbol for the S&P 500 index itself.
SP500_INDEX_TICKER: str = "^GSPC"

# Wikipedia rejects bare requests / generic User-Agents.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ----------------------------- Wikipedia helpers -----------------------------

def _read_wikipedia_tables(url: str) -> list[pd.DataFrame]:
    response = requests.get(url, headers=_BROWSER_HEADERS, timeout=30)
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) != "nan"]).strip()
            for col in df.columns
        ]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_change_table(url: str) -> pd.DataFrame:
    """Return the Wikipedia sub-table with Date / Added / Removed columns."""
    tables = _read_wikipedia_tables(url)
    candidates: list[pd.DataFrame] = []
    for table in tables:
        df = _flatten_columns(table)
        cols = [c.lower() for c in df.columns]
        if (
            any("date" in c for c in cols)
            and any("added" in c for c in cols)
            and any("removed" in c for c in cols)
        ):
            candidates.append(df)
    if not candidates:
        raise ValueError(f"No index-change table found at {url}")
    # If Wikipedia lists multiple change tables, the meaningful one is the longest.
    return max(candidates, key=len)


def _standardize_change_table(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df)
    rename_map: dict[str, str] = {}
    for col in df.columns:
        c = col.lower()
        if "date" in c:
            rename_map[col] = "Date"
        elif "added" in c and "ticker" in c:
            rename_map[col] = "Added_Ticker"
        elif "removed" in c and "ticker" in c:
            rename_map[col] = "Removed_Ticker"
    df = df.rename(columns=rename_map)
    keep = [c for c in ("Date", "Added_Ticker", "Removed_Ticker") if c in df.columns]
    df = df[keep].copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def _index_changes_within(url: str, years: int) -> pd.DataFrame:
    df = _standardize_change_table(_find_change_table(url))
    if "Date" in df.columns:
        cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
        df = df[df["Date"] >= cutoff].copy()
    return df.reset_index(drop=True)


def _current_constituent_symbols(url: str) -> list[str]:
    """
    Return the current-constituent ticker column for an index-constituents page.

    We can't trust `tables[0]` to be the constituents table: Wikipedia often
    renders an infobox/sidebar first (no headers, 2 columns) and the real
    constituents table only shows up further down. Instead we scan every
    table, pick the first one that has a Symbol/Ticker column AND enough
    rows to plausibly be a constituents list.
    """
    tables = _read_wikipedia_tables(url)
    symbol_col_candidates = ("Symbol", "Ticker", "Ticker symbol")
    MIN_CONSTITUENT_ROWS = 50  # safely below 400 / 500; safely above any infobox

    debug_columns: list[list[str]] = []
    for table in tables:
        df = _flatten_columns(table)
        debug_columns.append(df.columns.tolist())
        if len(df) < MIN_CONSTITUENT_ROWS:
            continue
        for candidate in symbol_col_candidates:
            if candidate in df.columns:
                symbols = df[candidate]
                return [s.strip() for s in symbols.tolist() if isinstance(s, str) and s.strip()]

    raise ValueError(
        f"Could not find a constituents table at {url}. "
        f"Scanned columns per table: {debug_columns}"
    )


# ------------------------------- Public API --------------------------------

def get_current_sp500_tickers() -> list[str]:
    """Current S&P 500 constituents from Wikipedia (typically ~503 rows)."""
    return _current_constituent_symbols(SP500_URL)


def get_current_sp_midcap_400_tickers() -> list[str]:
    """Current S&P MidCap 400 constituents from Wikipedia (~400 rows)."""
    return _current_constituent_symbols(SP400_URL)


def get_historical_index_tickers(url: str, years: int = 2) -> list[str]:
    """
    Return every ticker that was a member of the index at `url` at any
    point in the past `years` years: current constituents PLUS every ticker
    appearing in the Added_Ticker / Removed_Ticker columns of the change
    table within the window. Sorted, deduplicated.
    """
    current = _current_constituent_symbols(url)
    changes = _index_changes_within(url, years=years)

    universe: set[str] = set(current)
    for col in ("Added_Ticker", "Removed_Ticker"):
        if col in changes.columns:
            for t in changes[col].dropna().tolist():
                if isinstance(t, str) and t.strip():
                    universe.add(t.strip())
    return sorted(universe)


def get_all_universe_tickers(
    years: int = 2,
) -> tuple[list[str], list[str], list[str]]:
    """
    Single entry-point: return the deduplicated MetaLearner ticker universe
    split into three lists.

    Parameters
    ----------
    years : int
        How far back (in years) to look for historical S&P 500 and
        S&P MidCap 400 membership. Default: 2.

    Returns
    -------
    stock_tickers : list[str]
        Sorted, deduplicated union of current + past-`years` S&P 500
        and S&P MidCap 400 constituents.
    sector_etf_tickers : list[str]
        The 11 sector SPDR ETFs and SPY (see SECTOR_ETF_TICKERS).
    sp500_index_tickers : list[str]
        A one-element list containing the S&P 500 index quote symbol
        (^GSPC). Returned as a list so callers can concatenate uniformly.
    """
    stocks: set[str] = set()
    stocks.update(get_historical_index_tickers(SP500_URL, years=years))
    stocks.update(get_historical_index_tickers(SP400_URL, years=years))

    stock_tickers = sorted(stocks)
    sector_etf_tickers = list(SECTOR_ETF_TICKERS)
    sp500_index_tickers = [SP500_INDEX_TICKER]

    return stock_tickers, sector_etf_tickers, sp500_index_tickers


if __name__ == "__main__":
    stock_tickers, sector_etf_tickers, sp500_index_tickers = get_all_universe_tickers(years=2)
    print(f"Stock tickers: {len(stock_tickers)}")
    print(f"Sector ETF tickers: {len(sector_etf_tickers)} -> {sector_etf_tickers}")
    print(f"S&P 500 index tickers: {sp500_index_tickers}")
