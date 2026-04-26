
from typing import Callable, Sequence
import pandas as pd
import yfinance as yf

def extract_price_and_volume_columns_and_tickers(price_data: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    price_columns = [i for i in price_data.columns if 'Close' in i]
    volume_columns = [i for i in price_data.columns if 'Volume' in i]
    tickers = [i.split('_')[0] for i in price_columns]
    return price_columns, volume_columns, tickers


def fetch_sector_industry(ticker: str) -> tuple[str, str]:
    """Fetch the industry and sector of a stock ticker."""
    try:
        stock_info = yf.Ticker(ticker).info
        return stock_info.get('sector', 'N/A'), stock_info.get('industry', 'N/A')
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return 'N/A', 'N/A'

def create_ticker_sector_industry_df(tickers: str, fetch_sector_industry: Callable[[str], tuple[str, str]]) -> pd.DataFrame:
    """
    Creates a DataFrame that maps tickers to their sectors and industries.

    Parameters:
    - tickers: list of tickers
    - fetch_sector_industry: function that returns sector and industry for a given ticker

    Returns:
    - DataFrame with columns ['Ticker', 'Sector', 'Industry']
    """
    data = []
    for ticker in tickers:
        sector, industry = fetch_sector_industry(ticker)
        print(ticker)
        data.append({'Ticker': ticker, 'Sector': sector, 'Industry': industry})

    return pd.DataFrame(data)
