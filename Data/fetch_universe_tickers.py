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


def _index_change_groups(url: str, years: int) -> list[tuple[pd.Timestamp, set, set]]:
    """
    Return the index-change events within the past `years` years, grouped by
    date and sorted ascending. Each element is
    ``(date, added_tickers_set, removed_tickers_set)``.

    Multiple rows sharing the same Wikipedia change date (e.g. several
    simultaneous additions/removals on a single rebalance day) are merged
    into one event so they can be applied / undone atomically.
    """
    changes = _index_changes_within(url, years)
    changes = changes.dropna(subset=["Date"])

    groups: list[tuple[pd.Timestamp, set, set]] = []
    for date, g in changes.groupby("Date", sort=True):  # ascending
        added: set[str] = set()
        removed: set[str] = set()
        if "Added_Ticker" in g.columns:
            added = {
                t.strip() for t in g["Added_Ticker"].dropna().tolist()
                if isinstance(t, str) and t.strip()
            }
        if "Removed_Ticker" in g.columns:
            removed = {
                t.strip() for t in g["Removed_Ticker"].dropna().tolist()
                if isinstance(t, str) and t.strip()
            }
        groups.append((date, added, removed))
    return groups


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


def reconstruct_index_membership_timeline(url: str, years: int = 2) -> pd.DataFrame:
    """
    Reconstruct an index's membership over the past `years` years by walking
    its Wikipedia change table backwards from today's constituents.

    The idea: today's constituents are known exactly. Each change event records
    a ticker that was *added* and a ticker that was *removed* on a given date.
    Replaying the events from newest to oldest and *undoing* them
    (i.e. removing the added ticker and re-adding the removed ticker) lets us
    rebuild the membership set as it stood just before each change.

    Parameters
    ----------
    url : str
        Wikipedia constituents/change page (SP500_URL or SP400_URL).
    years : int
        Look-back window in years.

    Returns
    -------
    pd.DataFrame
        Columns ``['Date', 'tickers']`` sorted ascending by Date, where Date is
        a ``datetime64`` and ``tickers`` is a sorted ``list[str]`` giving the
        membership effective *as of* that date (i.e. after that date's change,
        until the next change). A final row at the window cutoff captures the
        membership that held before the oldest change in the window.
    """
    current = set(_current_constituent_symbols(url))
    groups = _index_change_groups(url, years)  # ascending
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)

    running = set(current)
    snapshots: dict[pd.Timestamp, set] = {}

    # Walk events newest -> oldest. At each change date, the *running* set is
    # the membership effective on that date (it already reflects this change
    # and every later one). Record it, then undo this change to step backwards.
    for date, added, removed in reversed(groups):
        snapshots[date] = set(running)
        running -= added
        running |= removed

    # `running` is now the membership before the oldest in-window change, which
    # holds from the cutoff up to that oldest change.
    if cutoff not in snapshots:
        snapshots[cutoff] = set(running)

    rows = [(date, sorted(members)) for date, members in snapshots.items()]
    rows.sort(key=lambda r: r[0])  # ascending by Date
    return pd.DataFrame(rows, columns=["Date", "tickers"])


def build_membership_timeline(years: int = 2) -> pd.DataFrame:
    """
    Build a single timeline of S&P 500 and S&P MidCap 400 membership over the
    past `years` years.

    Returns
    -------
    pd.DataFrame
        Columns ``['Date', 'sp500_tickers', 'spmidcap400_tickers']`` sorted
        newest-first. ``Date`` is a string (``YYYY-MM-DD``). Each ticker column
        holds the ``list[str]`` of that index's constituents effective as of the
        row's date. Because the two indices change on different dates, every
        distinct change date from either index gets a row, and each column is
        filled with that index's membership as of that date (a backward as-of
        lookup).
    """
    sp500_tl = (
        reconstruct_index_membership_timeline(SP500_URL, years=years)
        .rename(columns={"tickers": "sp500_tickers"})
        .sort_values("Date")
        .reset_index(drop=True)
    )
    sp400_tl = (
        reconstruct_index_membership_timeline(SP400_URL, years=years)
        .rename(columns={"tickers": "spmidcap400_tickers"})
        .sort_values("Date")
        .reset_index(drop=True)
    )

    all_dates = sorted(set(sp500_tl["Date"]).union(sp400_tl["Date"]))
    base = pd.DataFrame({"Date": pd.to_datetime(all_dates)})

    # Backward as-of join: for each timeline date, pick each index's most recent
    # snapshot on or before it. Both right frames carry a cutoff snapshot, so no
    # in-window date is ever left without a membership list.
    base = pd.merge_asof(base, sp500_tl, on="Date", direction="backward")
    base = pd.merge_asof(base, sp400_tl, on="Date", direction="backward")

    base = base.sort_values("Date", ascending=False).reset_index(drop=True)
    base["Date"] = base["Date"].dt.strftime("%Y-%m-%d")
    return base[["Date", "sp500_tickers", "spmidcap400_tickers"]]


def get_all_universe_tickers(
    years: int = 2,
) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
    """
    Single entry-point: return the deduplicated MetaLearner ticker universe
    split into three lists, plus a membership-timeline DataFrame.

    Parameters
    ----------
    years : int
        How far back (in years) to look for historical S&P 500 and
        S&P MidCap 400 membership. Default: 2.

    Returns
    -------
    stock_tickers : list[str]
        Sorted, deduplicated union of every ticker that was an S&P 500 or
        S&P MidCap 400 member at any point in the window (derived directly
        from ``membership_df`` so the two are always consistent).
    sector_etf_tickers : list[str]
        The 11 sector SPDR ETFs and SPY (see SECTOR_ETF_TICKERS).
    sp500_index_tickers : list[str]
        A one-element list containing the S&P 500 index quote symbol
        (^GSPC). Returned as a list so callers can concatenate uniformly.
    membership_df : pd.DataFrame
        Columns ``['Date', 'sp500_tickers', 'spmidcap400_tickers']`` (Date as
        str, the two ticker columns as ``list[str]``), reconstructing both
        indices' membership as of every change date in the window. See
        ``build_membership_timeline``.
    """
    membership_df = build_membership_timeline(years=years)

    stocks: set[str] = set()
    for col in ("sp500_tickers", "spmidcap400_tickers"):
        for members in membership_df[col].dropna():
            stocks.update(members)

    stock_tickers = sorted(stocks)
    sector_etf_tickers = list(SECTOR_ETF_TICKERS)
    sp500_index_tickers = [SP500_INDEX_TICKER]

    return stock_tickers, sector_etf_tickers, sp500_index_tickers, membership_df


if __name__ == "__main__":
    stock_tickers, sector_etf_tickers, sp500_index_tickers, membership_df = get_all_universe_tickers(years=2)
    print(f"Stock tickers: {len(stock_tickers)}")
    print(f"Sector ETF tickers: {len(sector_etf_tickers)} -> {sector_etf_tickers}")
    print(f"S&P 500 index tickers: {sp500_index_tickers}")
    print(f"Membership timeline rows: {len(membership_df)}")
    with pd.option_context("display.max_colwidth", 60):
        print(membership_df.head())
