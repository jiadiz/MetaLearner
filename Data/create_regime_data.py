

import yfinance as yf
from datetime import datetime
import pandas as pd

def get_gold_n_intest_data():
    # Download U.S. 10-year Treasury yield (^TNX) and Gold price (GC=F)
    today = datetime.today().strftime('%Y-%m-%d')

    # Download data until today
    gold_n_interest_data = yf.download(["^TNX", "GC=F"], start="2008-01-01", end=today)

    # View the data
    print(gold_n_interest_data)


    gold_n_treasury_prices = gold_n_interest_data.loc[:, pd.IndexSlice['Close', ['GC=F', '^TNX']]]

    # Renaming the columns for clarity
    gold_n_treasury_prices.columns = ['gold_close', 'treasury_yield_close']
    gold_n_treasury_prices = gold_n_treasury_prices.reset_index()

    return gold_n_treasury_prices 

def get_cpi_data():

    from fredapi import Fred
    # Replace 'your_api_key' with your actual FRED API key
    fred = Fred(api_key='a7d4814d655efcd211d13841fced1b80')

    # Get CPI data (e.g., US CPI)
    cpi_data = fred.get_series('CPIAUCSL')
    print(cpi_data)
    # Get U.S. Federal Funds Rate (or other interest rate series)
    interest_rate_data = fred.get_series('FEDFUNDS')
    print(fred.get_series_info('FEDFUNDS'))

    # Print the latest data
    print(cpi_data.tail())
    print(interest_rate_data.tail())

    interest_rate_df = pd.DataFrame(interest_rate_data).reset_index()
    # interest_rate_df.columns = ['month', 'interest_rate']
    interest_rate_df.columns = ['month', 'prev_month_interest_rate']
    interest_rate_df['month'] = pd.to_datetime(interest_rate_df['month'])
    interest_rate_df['pred_month'] = interest_rate_df['month'] + pd.DateOffset(months=1)
    interest_rate_df = interest_rate_df.drop('month', axis = 1)

    cpi_df = pd.DataFrame(cpi_data).reset_index()
    # cpi_df.columns = ['month', 'cpi']
    cpi_df.columns = ['month', 'prev_month_cpi']
    cpi_df['month'] = pd.to_datetime(cpi_df['month'])
    cpi_df['pred_month'] = cpi_df['month'] + pd.DateOffset(months=1)
    cpi_df = cpi_df.drop('month', axis = 1)

    return cpi_df, interest_rate_df

def get_VIX_data():
    today = datetime.today().strftime('%Y-%m-%d')
    vix = yf.download('^VIX', start="2008-01-01", end= today).reset_index()[['Date', 'Close']]
    print(vix.columns)
    vix.columns = [ 'Date','VIX_Close']
    return  vix

def pull_sp500_data():
    def download_daily_data(ticker):
        data = yf.download(ticker, period="max", interval="1d")
        return data

    # #download sp500 data
    sp500 = download_daily_data('^GSPC')
    sp500.columns = ['sp500_' + i[0] for i in sp500.columns]

    sp500.reset_index(inplace = True)
    # sp500.reset_index(inplace = True)
    sp500 = sp500[sp500['Date'] >= pd.to_datetime('2016-01-01')]
    sp500_columns = ['sp500_Adj Close', 'sp500_Volume']

    sp500 = sp500[['Date', 'sp500_Close', 'sp500_Volume']].copy()
    return sp500

def create_regime_data(
    sp500_df: pd.DataFrame,
    interest_rate_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    gold_n_treasury_prices_dropped: pd.DataFrame,
    vix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a daily 'regime' dataset by merging:
      - S&P 500 daily data (base)
      - Interest rate (monthly, joined by YYYY-MM)
      - CPI (monthly, joined by YYYY-MM)
      - Gold & Treasury yield (daily, joined by date)
      - VIX (daily, joined by date)

    Returns a DataFrame with columns:
      Date, sp500_Close, sp500_Volume, prev_month_interest_rate, prev_month_cpi,
      gold_close, treasury_yield_close, VIX_Close
    """

    # ----------------------------
    # 1) Make copies + normalize keys
    # ----------------------------
    sp = sp500_df.copy()
    sp["Date"] = pd.to_datetime(sp["Date"])
    sp["month_key"] = sp["Date"].dt.to_period("M").astype(str)      # e.g. "2026-02"
    sp["date_key"] = sp["Date"].dt.normalize()                      # midnight datetime

    ir = interest_rate_df.copy()
    ir["pred_month"] = pd.to_datetime(ir["pred_month"])
    ir["month_key"] = ir["pred_month"].dt.to_period("M").astype(str)

    cpi = cpi_df.copy()
    cpi["pred_month"] = pd.to_datetime(cpi["pred_month"])
    cpi["month_key"] = cpi["pred_month"].dt.to_period("M").astype(str)

    gold = gold_n_treasury_prices_dropped.copy()
    gold["Date"] = pd.to_datetime(gold["Date"])
    gold["date_key"] = gold["Date"].dt.normalize()

    v = vix.copy()
    v["Date"] = pd.to_datetime(v["Date"])
    v["date_key"] = v["Date"].dt.normalize()

    # ----------------------------
    # 2) Merge monthly tables (on month_key)
    # ----------------------------
    # Keep only needed columns from monthly sources to prevent collisions
    ir_keep = [c for c in ["month_key", "prev_month_interest_rate"] if c in ir.columns]
    cpi_keep = [c for c in ["month_key", "prev_month_cpi"] if c in cpi.columns]

    temp = sp.merge(ir[ir_keep], on="month_key", how="left")
    temp = temp.merge(cpi[cpi_keep], on="month_key", how="left")

    # ----------------------------
    # 3) Merge daily tables (on date_key)
    # ----------------------------
    # Drop their original Date columns to avoid Date_x / Date_y
    gold_daily = gold.drop(columns=["Date"], errors="ignore")
    vix_daily  = v.drop(columns=["Date"], errors="ignore")

    temp = temp.merge(gold_daily, on="date_key", how="left")
    temp = temp.merge(vix_daily, on="date_key", how="left")

    # ----------------------------
    # 4) Select final columns
    # ----------------------------
    regime_columns = [
        "Date",
        "sp500_Close",
        "sp500_Volume",
        "prev_month_interest_rate",
        "prev_month_cpi",
        "gold_close",
        "treasury_yield_close",
        "VIX_Close",
    ]

    missing = [c for c in regime_columns if c not in temp.columns]
    if missing:
        raise KeyError(f"Missing required columns after merge: {missing}")

    regime_data = temp.loc[:, regime_columns].copy()
    number_of_missing_data_points = regime_data.isna().sum()
    print('Number of missing data points:',
         number_of_missing_data_points)
    print('Forward filled')
    return regime_data.ffill()
